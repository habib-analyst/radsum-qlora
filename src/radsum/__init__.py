"""radsum-qlora: QLoRA fine-tuning of a 2B vision-language model for radiology
finding summarization (IU X-ray findings -> impression), with clinical
evaluation (CheXbert F1) and a finding-hallucination rate metric."""

__version__ = "0.1.0"

from .metrics import (
    CHEXBERT_CONDITIONS,
    KEY_CONDITIONS,
    chexbert_f1,
    finding_hallucination_rate,
    omission_rate,
    rouge_l_f1,
)

__all__ = [
    "CHEXBERT_CONDITIONS",
    "KEY_CONDITIONS",
    "chexbert_f1",
    "finding_hallucination_rate",
    "omission_rate",
    "rouge_l_f1",
]
