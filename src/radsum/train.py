"""QLoRA fine-tuning of Qwen2-VL-2B-Instruct for findings -> impression.

Custom training loop (rather than TRL) for explicit control over the
multimodal inputs: loss is computed on response tokens only.

Task format (chat template):
    user: [image] + "Findings: <findings>\n[indication]\nSummarize into a
           concise radiology impression."
    assistant: <impression>

Run:
    python -m radsum.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_messages(findings: str, indication: str) -> list:
    user_text = (
        "You are a radiologist. Given the chest X-ray image and its findings "
        "section, write a concise impression (summary) in standard radiology "
        "style.\n\n"
        f"Findings: {findings}"
    )
    if indication:
        user_text += f"\nIndication: {indication}"
    user_text += "\n\nImpression:"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},  # filled per-sample
                {"type": "text", "text": user_text},
            ],
        }
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out_dir", default="results/qlora_run")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="cap training samples (smoke tests)")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="override total optimizer steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(open(args.config).read())

    from PIL import Image
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # With accelerate device_map="auto" + 4-bit, model.device is unreliable;
    # inputs go to cuda:0 explicitly (single-GPU training).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model_id = cfg["model_id"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=cfg.get("min_pixels", 28 * 28 * 64),
        max_pixels=cfg.get("max_pixels", 28 * 28 * 512),
    )

    manifest = pd.read_parquet(cfg["manifest_path"])
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    if args.max_train_samples:
        train_df = train_df.iloc[: args.max_train_samples]
    print(f"train samples: {len(train_df)}", flush=True)

    # ---- data encoding -------------------------------------------------
    def encode_row(row) -> dict | None:
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            return None
        messages = build_messages(row["findings"], row.get("indication", ""))
        messages[0]["content"][0]["image"] = image
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full = prompt + row["impression"] + processor.tokenizer.eos_token
        enc_full = processor(
            text=[full], images=[image], padding=True, return_tensors="pt"
        )
        enc_prompt = processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        )
        input_ids = enc_full["input_ids"][0]
        prompt_len = enc_prompt["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # loss on response tokens only
        # Batch dims: text tensors get a batch dim of 1 in the loop, but
        # pixel_values (num_patches, 3, 14, 14) and image_grid_thw (1, 3)
        # must NOT gain one — the model matches patches via image_grid_thw.
        out = {
            "input_ids": input_ids,                      # (seq,)
            "attention_mask": enc_full["attention_mask"][0],  # (seq,)
            "labels": labels,                            # (seq,)
            "pixel_values": enc_full["pixel_values"],    # (patches, 3, 14, 14)
        }
        if "image_grid_thw" in enc_full:
            out["image_grid_thw"] = enc_full["image_grid_thw"]  # (1, 3)
        return out

    # ---- training loop -------------------------------------------------
    lr = cfg.get("learning_rate", 2e-4)
    grad_accum = cfg.get("grad_accum", 16)
    epochs = cfg.get("epochs", 3)
    total_samples = len(train_df) * epochs
    steps_per_epoch = math.ceil(len(train_df) / grad_accum)
    total_steps = args.max_steps or steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.get("warmup_ratio", 0.03) * total_steps),
        num_training_steps=total_steps,
    )

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a")
    model.train()
    global_step = 0
    t_start = time.time()
    indices = list(range(len(train_df)))

    for epoch in range(epochs):
        random.Random(cfg.get("seed", 42) + epoch).shuffle(indices)
        optimizer.zero_grad()
        acc_loss, acc_count = 0.0, 0
        for pos, idx in enumerate(indices):
            if global_step >= total_steps:
                break
            batch = encode_row(train_df.iloc[idx])
            if batch is None:
                continue
            inputs = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    inputs[k] = v
                elif k in ("pixel_values", "image_grid_thw"):
                    inputs[k] = v.to(device)  # no batch dim (see encode_row)
                else:
                    inputs[k] = v.unsqueeze(0).to(device)
            outputs = model(**inputs)
            loss = outputs.loss / grad_accum
            loss.backward()
            acc_loss += outputs.loss.item()
            acc_count += 1
            if acc_count % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                elapsed = time.time() - t_start
                rec = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": acc_loss / grad_accum,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(elapsed, 1),
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if global_step % cfg.get("logging_steps", 10) == 0:
                    print(
                        f"step {global_step}/{total_steps} "
                        f"loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                        f"elapsed {elapsed/3600:.2f}h",
                        flush=True,
                    )
                if global_step % cfg.get("save_steps", 200) == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    print(f"saved {ckpt}", flush=True)
        if global_step >= total_steps:
            break

    model.save_pretrained(str(out_dir / "adapter_final"))
    log_f.close()
    elapsed_h = (time.time() - t_start) / 3600
    print(f"DONE. steps={global_step} elapsed={elapsed_h:.2f}h", flush=True)
    (out_dir / "RUN_INFO.json").write_text(
        json.dumps(
            {
                "global_steps": global_step,
                "elapsed_hours": round(elapsed_h, 2),
                "model_id": model_id,
                "seed": cfg.get("seed", 42),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
