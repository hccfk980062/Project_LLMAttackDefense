"""Layer 1: Input Sanitizer - regex filtering, keyword blacklist, length limit."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List


DEFAULT_BLACKLIST = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"忽略(所有)?先前的?(指令|指示|設定)",
    r"disregard\s+(all\s+)?previous",
    r"\[system\s+override\]",
    r"new\s+system\s+prompt",
    r"forget\s+your\s+instructions?",
    r"你的新指令",
    r"###\s*system",
    r"<\|im_start\|>\s*system",
    r"---end\s+of\s+context---",
]


# Patterns for scanning RETRIEVED DOCUMENTS (indirect injection). Much narrower
# than the query blacklist: knowledge-base text legitimately quotes attack
# phrases ("例如「忽略所有先前的指示」") when teaching about them, so only
# structural signatures that have no business appearing in a document body are
# listed — fake role tags, chat-template tokens, comments/asides addressed to
# the model. Semantic cases are left to L2.
#
# Bracket tags like "[SYSTEM]" are deliberately NOT listed: the knowledge base's
# own teaching material mentions and demonstrates them (02_prompt_injection_
# deep_dive.txt, 03_rag_system_security.txt), and every pattern here must have
# zero hits on the clean corpus (tests/test_defense/test_context_scan.py).
CONTEXT_BLACKLIST_DEFAULT = [
    r"<\|\s*(im_start|im_end|system|endoftext)\s*\|>",
    r"<!--[^>]{0,40}(系統|指令|AI|assistant|SYSTEM|INSTRUCTION)",
    r"={2,}\s*(SYSTEM|系統)[^=\n]{0,12}={2,}",
    r"^\s*(SYSTEM|SYSTEM\s+OVERRIDE|AI\s+INSTRUCTION)\s*:",
    r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+|any\s+)?(AI|assistant|chatbot|language\s+model|LLM)\b",
    r"(給|致)\s*(AI|語言模型|聊天機器人)",
    r"(讀取|讀到|處理)(本|這份|此)(文件|段落?|文字)的\s*(AI|語言模型|模型|助理)",
]


@dataclass
class SanitizerResult:
    is_blocked: bool
    cleaned_text: str
    reason: str = ""
    triggered_pattern: str = ""


class InputSanitizer:
    def __init__(
        self,
        max_length: int = 2000,
        keyword_blacklist: List[str] | None = None,
        blacklist_path: str | Path | None = None,
        strip_html: bool = True,
        normalize_unicode: bool = True,
        context_patterns: List[str] | None = None,
    ):
        self.max_length = max_length
        self.strip_html = strip_html
        self.normalize_unicode = normalize_unicode
        self.context_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in (context_patterns if context_patterns is not None else CONTEXT_BLACKLIST_DEFAULT)
        ]

        patterns = list(DEFAULT_BLACKLIST)
        if keyword_blacklist:
            patterns.extend(keyword_blacklist)
        if blacklist_path and Path(blacklist_path).exists():
            lines = Path(blacklist_path).read_text(encoding="utf-8").splitlines()
            patterns.extend(
                stripped
                for line in lines
                if (stripped := line.strip()) and not stripped.startswith("#")
            )

        self.compiled_patterns = []
        for p in patterns:
            try:
                self.compiled_patterns.append(re.compile(p, re.IGNORECASE | re.MULTILINE))
            except re.error as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Skipping invalid blacklist pattern %r: %s", p, e
                )

    def sanitize(self, text: str) -> SanitizerResult:
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.strip_html:
            text = re.sub(r"<[^>]+>", "", text)

        if len(text) > self.max_length:
            return SanitizerResult(
                is_blocked=True,
                cleaned_text="",
                reason="input_too_long",
                triggered_pattern=f"length={len(text)}",
            )

        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                return SanitizerResult(
                    is_blocked=True,
                    cleaned_text="",
                    reason="blacklist_match",
                    triggered_pattern=match.group(),
                )

        return SanitizerResult(is_blocked=False, cleaned_text=text)

    def scan_context(self, text: str) -> SanitizerResult:
        """Check one retrieved document against CONTEXT_BLACKLIST_DEFAULT.

        No length limit and no HTML stripping (an HTML comment is itself a
        signal here); the text is returned unchanged when it passes.
        """
        normalized = unicodedata.normalize("NFKC", text) if self.normalize_unicode else text
        for pattern in self.context_patterns:
            match = pattern.search(normalized)
            if match:
                return SanitizerResult(
                    is_blocked=True,
                    cleaned_text="",
                    reason="context_pattern_match",
                    triggered_pattern=match.group(),
                )
        return SanitizerResult(is_blocked=False, cleaned_text=text)
