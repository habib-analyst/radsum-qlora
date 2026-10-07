"""Evaluation harness: generate impressions, score with CheXbert F1,
finding-hallucination rate, omission rate, ROUGE-L.

Modes (one model per run; compare runs with compare.py or the notebook):
  --adapter DIR     load QLoRA adapter on top of the base model
  --blank_image     text-only ablation: feed a blank image at inference
  (no --adapter)    zero-shot base model

Run:
    python -m radsum.evaluate --config configs/eval.yaml --split test \\
        --out results/eval_qlora.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from .chexbert_labeler import CheXbertLabeler
from .metrics import (
    CHEXBERT_CONDITIONS,
    KEY_CONDITIONS,
    chexbert_f1,
    finding_hallucination_rate,
    omission_rate,
    rouge_l_f1,
)
from .train import build_messages


def load_model_and_processor(cfg, adapter: str | None):
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

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
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded adapter: {adapter}", flush=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=cfg.get("eval_min_pixels", 28 * 28 * 64),
        max_pixels=cfg.get("eval_max_pixels", 28 * 28 * 256),
    )
    return model, processor


@torch.no_grad()
def generate_one(model, processor, image, findings, indication, max_new_tokens=128):
    from PIL import Image as PILImage

    messages = build_messages(findings, indication)
    messages[0]["content"][0]["image"] = image
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(
        target_device
    )
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--blank_image", action="store_true",
                    help="text-only ablation: replace image with blank")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, processor = load_model_and_processor(cfg, args.adapter)
    labeler = CheXbertLabeler(cfg["chexbert_weights"])

    manifest = pd.read_parquet(cfg["manifest_path"])
    df = manifest[manifest["split"] == args.split].reset_index(drop=True)
    if args.max_samples:
        df = df.iloc[: args.max_samples]
    print(f"eval samples ({args.split}): {len(df)}", flush=True)

    from PIL import Image

    preds, refs, findings_list = [], [], []
    t0 = time.time()
    for i, row in df.iterrows():
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue
        if args.blank_image:
            image = Image.new("RGB", image.size, (0, 0, 0))
        gen = generate_one(
            model, processor, image, row["findings"], row.get("indication", ""),
            max_new_tokens=cfg.get("max_new_tokens", 128),
        )
        preds.append(gen)
        refs.append(row["impression"])
        findings_list.append(row["findings"])
        if (i + 1) % 25 == 0:
            el = (time.time() - t0) / 60
            print(f"  {i+1}/{len(df)} generated ({el:.1f} min)", flush=True)

    gen_hours = (time.time() - t0) / 3600
    print("labeling with CheXbert ...", flush=True)
    pred_labels = labeler.label(preds)
    ref_labels = labeler.label(refs)
    src_labels = labeler.label(findings_list)

    f1_all = chexbert_f1(pred_labels, ref_labels)
    f1_key = chexbert_f1(pred_labels, ref_labels, conditions=KEY_CONDITIONS)
    fhr = finding_hallucination_rate(pred_labels, src_labels)
    om = omission_rate(pred_labels, ref_labels)
    rouge = sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / max(len(preds), 1)

    results = {
        "config": {"adapter": args.adapter, "blank_image": args.blank_image,
                   "split": args.split, "n": len(preds)},
        "chexbert_f1_all14": f1_all,
        "chexbert_f1_key5": f1_key,
        "finding_hallucination_rate": fhr,
        "omission_rate": om,
        "rouge_l_f1_mean": rouge,
        "gen_hours": round(gen_hours, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)

    # Save generations for audit / human spot-checks.
    audit = pd.DataFrame(
        {"findings": findings_list, "reference": refs, "generated": preds}
    )
    audit_path = out_path.with_name(out_path.stem + "_generations.parquet")
    audit.to_parquet(audit_path, index=False)
    print(f"wrote {out_path} and {audit_path}", flush=True)


if __name__ == "__main__":
    main()
