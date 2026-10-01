"""Prompt and context layout.

The model receives the question, at most one exact drive report, and up to three retrieved
knowledge documents. Nothing else: no labels, no table dumps, no other drives.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence

PROMPT_VERSION = "prompt_v1"

DRIVE_HEADINGS = ("Observed telemetry", "How the detector scored this drive", "General background", "Limits")
GENERAL_HEADINGS = ("Answer", "Limits")

_RULES = """You explain stored anomaly-detection results for a hard drive monitoring project.
Follow these rules exactly.
1. Use only the text in the DRIVE REPORT and KNOWLEDGE blocks. Do not use outside knowledge.
2. Cite every statement with the id of the block it came from, in square brackets, for example [attr:smart_197_raw].
3. Copy every number and date exactly as written in the context. Do not round, convert or compute new numbers.
4. Keep observed telemetry (from the DRIVE REPORT) separate from general background (from KNOWLEDGE blocks).
5. Say a drive was flagged as unusual by the detector. Do not state or guess a cause for any value, and do not say a drive is failing, will fail, or is likely to fail.
6. If the context does not contain something that was asked for, say it is not in the provided evidence.
7. Do not mention outcomes, labels or evaluation results."""

_DRIVE_FORMAT = """Write the answer with exactly these four sections, each starting with a line like "## Observed telemetry":
## Observed telemetry
## How the detector scored this drive
## General background (not specific to this drive)
## Limits
In "Observed telemetry" and "How the detector scored this drive" cite the DRIVE REPORT id.
In "General background" cite only KNOWLEDGE ids; if there are no KNOWLEDGE blocks write "No relevant background documents were retrieved."
If the DRIVE REPORT says the detector is isolation_forest, copy this sentence into "How the detector scored this drive": "It does not provide per-feature causal attribution."
In "Limits" write: "This describes stored detector output and observed telemetry. It is not a diagnosis, and it does not predict failure.\""""

_GENERAL_FORMAT = """Write the answer with exactly these two sections, each starting with a line like "## Answer":
## Answer
## Limits
In "Answer" cite the KNOWLEDGE ids you used.
In "Limits" write: "This describes background documents. It is not a diagnosis, and it does not predict failure.\""""


@dataclass
class Context:
    mode: str                       # "drive" or "general"
    question: str
    report: Optional[object]        # DriveReport for mode "drive"
    knowledge: List[Mapping] = field(default_factory=list)
    text: str = ""
    allowed_ids: List[str] = field(default_factory=list)

    @property
    def method(self) -> Optional[str]:
        return self.report.method if self.report else None

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def knowledge_block(doc: Mapping) -> str:
    return f"=== KNOWLEDGE [{doc['doc_id']}] {doc['title']} | source: {doc['source_name']} ===\n{doc['text']}"


def assemble_context(question: str, report=None, knowledge: Sequence[Mapping] = ()) -> Context:
    blocks = []
    allowed = []
    if report is not None:
        blocks.append(f"=== DRIVE REPORT [{report.doc_id}] ===\n{report.text}")
        allowed.append(report.doc_id)
    for doc in knowledge:
        blocks.append(knowledge_block(doc))
        allowed.append(doc["doc_id"])
    return Context(
        mode="drive" if report is not None else "general",
        question=question,
        report=report,
        knowledge=list(knowledge),
        text="\n\n".join(blocks),
        allowed_ids=allowed,
    )


def system_prompt(mode: str) -> str:
    return _RULES + "\n\n" + (_DRIVE_FORMAT if mode == "drive" else _GENERAL_FORMAT)


def build_messages(context: Context) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt(context.mode)},
        {"role": "user", "content": f"QUESTION: {context.question}\n\n{context.text}"},
    ]


def retry_messages(messages: List[dict], previous_answer: str, errors: Sequence) -> List[dict]:
    """Ask once more, listing exactly what was wrong with the first answer."""
    listing = "\n".join(f"- {e.code}: {e.detail}" for e in errors)
    return list(messages) + [
        {"role": "assistant", "content": previous_answer},
        {
            "role": "user",
            "content": "That answer broke the rules:\n" + listing + "\nWrite the answer again, following every rule and the required sections.",
        },
    ]


def messages_json(messages: Sequence[dict]) -> str:
    return json.dumps(list(messages))
