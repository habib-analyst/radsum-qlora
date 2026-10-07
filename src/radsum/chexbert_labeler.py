"""CheXbert report labeler (inference only).

Loads the public CheXbert weights (StanfordAIMI/RRG_scorers, chexbert.pth --
a BERT-base backbone with 14 linear heads: 13 heads x {blank, positive,
negative, uncertain} + 1 head x {no, yes} for "No Finding") from a LOCAL
weights file, so evaluation never needs a network call at runtime.

Reference: Smit et al., "Combining Automatic Labelers and Expert Annotations
for Accurate Radiology Report Labeling Using BERT" (CheXbert).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence

import torch

from .metrics import CHEXBERT_CONDITIONS


class _BertLabeler(torch.nn.Module):
    """BERT-base + 14 linear heads. Mirrors the CheXbert architecture."""

    def __init__(self, bert_path: str | Path | None = None):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(
            str(bert_path) if bert_path else "bert-base-uncased"
        )
        self.dropout = torch.nn.Dropout(0.1)
        hidden = self.bert.pooler.dense.in_features
        self.linear_heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, 4) for _ in range(13)]
            + [torch.nn.Linear(hidden, 2)]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        cls_hidden = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        return [head(cls_hidden) for head in self.linear_heads]


class CheXbertLabeler:
    """Inference wrapper. `.label(reports)` -> List[List[int]] of 14 class ids."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        bert_path: str | Path | None = None,
    ):
        from transformers import BertTokenizer

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        weights_path = Path(weights_path)
        # Default: sibling bert-base-uncased dir next to the weights file.
        if bert_path is None:
            sib = weights_path.parent / "bert-base-uncased"
            bert_path = sib if sib.exists() else None
        tok_src = str(bert_path) if bert_path else "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(tok_src)
        model = _BertLabeler(bert_path)
        ckpt = torch.load(str(weights_path), map_location="cpu")
        state = ckpt["model_state_dict"]
        # Strip the DataParallel "module." prefix if present.
        state = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in state.items()
        )
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").strip().replace("\n", " ")
        return re.sub(r"\s+", " ", text)

    def label(self, reports: Sequence[str], batch_size: int = 32) -> List[List[int]]:
        """Label reports -> one 14-class-id vector per report."""
        cleaned = [self._clean(r) for r in reports]
        results: List[List[int]] = []
        with torch.no_grad():
            for i in range(0, len(cleaned), batch_size):
                batch = cleaned[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                heads = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
                preds = [h.argmax(dim=1).cpu().tolist() for h in heads]  # 14 x B
                for b in range(len(batch)):
                    results.append([preds[j][b] for j in range(14)])
        return results
        results: List[List[int]] = []
        with torch.no_grad():
            for i in range(0, len(reports), batch_size):
                batch = reports[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                heads = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
                preds = [h.argmax(dim=1).cpu().tolist() for h in heads]  # 14 x B
                for b in range(len(batch)):
                    results.append([preds[j][b] for j in range(14)])
        return results
