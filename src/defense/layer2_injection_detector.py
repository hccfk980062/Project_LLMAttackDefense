"""Layer 2: Injection Detector - fine-tuned BERT/RoBERTa binary classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class DetectionResult:
    is_injection: bool
    confidence: float       # probability of being an injection
    label: str              # "INJECTION" | "BENIGN"


# Document scanning window. mBERT tokenises CJK roughly one char per token, so a
# 512-char RAG chunk would hit max_length=512 and silently truncate a payload
# appended at the end. 320 chars stays well under the limit, and stride 160
# guarantees any span of <= 160 chars (every poison payload we use) lies wholly
# inside at least one window.
DOC_WINDOW_CHARS = 320
DOC_STRIDE_CHARS = 160


def split_windows(
    text: str,
    window_chars: int = DOC_WINDOW_CHARS,
    stride_chars: int = DOC_STRIDE_CHARS,
) -> list[str]:
    """Overlapping character windows covering the whole text."""
    if len(text) <= window_chars:
        return [text]
    starts = list(range(0, len(text) - window_chars + 1, stride_chars))
    if starts[-1] + window_chars < len(text):
        starts.append(len(text) - window_chars)
    return [text[s:s + window_chars] for s in starts]


class InjectionDetector:
    """Wraps a fine-tuned multilingual BERT classifier for injection detection."""

    LABEL_MAP = {0: "BENIGN", 1: "INJECTION"}

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.7,
        device: str | None = None,
    ):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained_base(cls, threshold: float = 0.7) -> "InjectionDetector":
        """Fallback: load the base multilingual model (before fine-tuning)."""
        base = "bert-base-multilingual-cased"
        return cls._new_from_name(base, threshold)

    @classmethod
    def _new_from_name(cls, model_name: str, threshold: float) -> "InjectionDetector":
        obj = object.__new__(cls)
        obj.threshold = threshold
        obj.device = "cuda" if torch.cuda.is_available() else "cpu"
        obj.tokenizer = AutoTokenizer.from_pretrained(model_name)
        obj.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        obj.model.to(obj.device)
        obj.model.eval()
        return obj

    @torch.no_grad()
    def detect(self, text: str) -> DetectionResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        injection_prob = probs[1].item()

        is_injection = injection_prob >= self.threshold
        label = self.LABEL_MAP[int(is_injection)]

        return DetectionResult(
            is_injection=is_injection,
            confidence=injection_prob,
            label=label,
        )

    @torch.no_grad()
    def _injection_probs(self, texts: list[str], batch_size: int = 32) -> list[float]:
        probs: list[float] = []
        for i in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                texts[i:i + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            logits = self.model(**inputs).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        return probs

    def _result(self, injection_prob: float, threshold: float) -> DetectionResult:
        is_injection = injection_prob >= threshold
        return DetectionResult(
            is_injection=is_injection,
            confidence=injection_prob,
            label=self.LABEL_MAP[int(is_injection)],
        )

    def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
        return [self._result(p, self.threshold) for p in self._injection_probs(texts)]

    def scan_documents(
        self,
        texts: list[str],
        threshold: float | None = None,
        window_chars: int = DOC_WINDOW_CHARS,
        stride_chars: int = DOC_STRIDE_CHARS,
    ) -> list[DetectionResult]:
        """Score retrieved documents; each gets the max probability over its windows.

        All windows of all documents go through the model together, so a
        top-k retrieval costs one or two forward passes rather than k.
        """
        threshold = self.threshold if threshold is None else threshold
        windows: list[str] = []
        owner: list[int] = []
        for doc_idx, text in enumerate(texts):
            for w in split_windows(text, window_chars, stride_chars):
                windows.append(w)
                owner.append(doc_idx)
        best = [0.0] * len(texts)
        for doc_idx, p in zip(owner, self._injection_probs(windows)):
            best[doc_idx] = max(best[doc_idx], p)
        return [self._result(p, threshold) for p in best]
