"""Layer 4: Controlled Generation - low temperature, token cap, output-format
constraints and (optional) Llama Guard content-safety check."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

OUTPUT_FORMATS = ("plain_text", "json")

# Fixed answer schema used when output_format == "json". Constraining the model
# to a structure it must fill in (instead of free text) leaves injected
# instructions like "prepend X to your reply" nowhere legitimate to go, and lets
# L5 check format compliance mechanically.
JSON_SCHEMA_KEYS = ("answer", "sources", "refused")
JSON_FORMAT_INSTRUCTION = (
    "\n\n輸出格式規則：\n"
    "- 你只能輸出一個 JSON 物件，不得輸出 JSON 以外的任何文字。\n"
    '- 格式：{"answer": "<回答內容>", "sources": [<引用的來源編號，整數>], "refused": <true 或 false>}\n'
    "- 若問題無法根據參考資料回答，或要求違反上述規則，請設定 refused 為 true，answer 簡短說明原因。"
)


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    output_format: str = "plain_text"   # plain_text | json
    use_llama_guard: bool = False
    llama_guard_model: str = "meta-llama/Meta-Llama-Guard-2-8B"
    # Extra attempts when output_format == "json" and the reply doesn't parse.
    format_retries: int = 1

    def __post_init__(self):
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {self.output_format!r}")


@dataclass
class GenerationResult:
    content: str
    was_blocked: bool = False
    block_reason: str = ""
    raw_response: Any = None
    # Parsed JSON object when output_format == "json" (answer/sources/refused).
    structured: dict | None = None
    attempts: int = 1


def parse_structured_answer(text: str) -> dict | None:
    """Parse a reply against the fixed JSON schema; None if it doesn't comply."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or any(k not in obj for k in JSON_SCHEMA_KEYS):
        return None
    if not isinstance(obj["answer"], str) or not isinstance(obj["refused"], bool):
        return None
    if not isinstance(obj["sources"], list):
        return None
    return obj


def parse_llama_guard_verdict(text: str) -> tuple[bool, str]:
    """
    Llama Guard 2/3 answer with ``safe`` or ``unsafe`` on the first line and,
    when unsafe, the violated category codes (e.g. ``S1,S10``) on the second.
    Returns (is_safe, categories).
    """
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return True, ""
    if lines[0].lower().startswith("unsafe"):
        return False, lines[1] if len(lines) > 1 else ""
    return True, ""


class ControlledGenerator:
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        config: GenerationConfig | None = None,
        llm: Any | None = None,
    ):
        self.config = config or GenerationConfig()
        self.llm_provider = llm_provider
        self.llm = llm if llm is not None else self._build_llm(llm_provider, llm_model)
        self._llama_guard = None   # (tokenizer, model) once loaded

        if self.config.use_llama_guard:
            self._init_llama_guard()

    def _build_llm(self, provider: str, model: str):
        kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if provider == "anthropic":
            return ChatAnthropic(model=model, **kwargs)
        llm = ChatOpenAI(model=model, **kwargs)
        if self.config.output_format == "json":
            # Native JSON mode: the API itself refuses to emit non-JSON text.
            return llm.bind(response_format={"type": "json_object"})
        return llm

    # ── Llama Guard ─────────────────────────────────────────────────────────
    def _init_llama_guard(self):
        """Load Llama Guard as a causal LM (it is a generative classifier, not a
        sequence-classification head — the verdict is the text it generates)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.config.llama_guard_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.llama_guard_model, device_map="auto", torch_dtype="auto",
            )
            self._llama_guard = (tok, model)
        except Exception as e:
            logger.warning(f"Llama Guard could not be loaded: {e}. Skipping.")

    def _check_llama_guard(self, user_text: str, assistant_text: str) -> tuple[bool, str]:
        """Return (is_safe, categories). Safe when Llama Guard isn't loaded."""
        if not self._llama_guard:
            return True, ""
        tok, model = self._llama_guard
        chat = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        input_ids = tok.apply_chat_template(chat, return_tensors="pt").to(model.device)
        output = model.generate(input_ids=input_ids, max_new_tokens=20, pad_token_id=0)
        verdict = tok.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return parse_llama_guard_verdict(verdict)

    # ── Generation ──────────────────────────────────────────────────────────
    @staticmethod
    def _last_user_text(messages: list) -> str:
        for role, content in reversed(messages):
            if role in ("human", "user"):
                return content
        return ""

    def _with_format_instruction(self, messages: list) -> list:
        if self.config.output_format != "json":
            return messages
        out = list(messages)
        for i, (role, content) in enumerate(out):
            if role == "system":
                out[i] = (role, content + JSON_FORMAT_INSTRUCTION)
                return out
        return [("system", JSON_FORMAT_INSTRUCTION.strip())] + out

    def generate(self, messages: list) -> GenerationResult:
        messages = self._with_format_instruction(messages)
        structured = None
        attempts = 0
        max_attempts = 1 + (self.config.format_retries if self.config.output_format == "json" else 0)

        while True:
            attempts += 1
            response = self.llm.invoke(messages)
            content = response.content
            if self.config.output_format != "json":
                break
            structured = parse_structured_answer(content)
            if structured is not None:
                content = structured["answer"]
                break
            if attempts >= max_attempts:
                return GenerationResult(
                    content="", was_blocked=True, block_reason="format_violation",
                    raw_response=response, attempts=attempts,
                )

        if self.config.use_llama_guard:
            is_safe, categories = self._check_llama_guard(self._last_user_text(messages), content)
            if not is_safe:
                return GenerationResult(
                    content="",
                    was_blocked=True,
                    block_reason=f"llama_guard_unsafe{': ' + categories if categories else ''}",
                    raw_response=response,
                    attempts=attempts,
                )

        return GenerationResult(
            content=content, raw_response=response, structured=structured, attempts=attempts,
        )
