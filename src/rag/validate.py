"""Checks on a generated answer before it is accepted.

An answer that fails any check is regenerated at most once; if it still fails, the caller
returns the deterministic evidence-only fallback instead. Nothing here repairs an answer.

Beyond the wording and number checks, two things are verified against the evidence itself:

* telemetry claims: "SMART 197 z-score 6.7" must match the z-score of SMART 197 in the drive report's
  structured facts, not merely be a number that appears somewhere in the context;
* citation support: each sentence must cite a document that can actually support it. A SMART
  document cannot support a detector statement, the other detector's document cannot support this
  one, and a statement about the drive must cite the drive report.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.rag.prompt import DRIVE_HEADINGS, GENERAL_HEADINGS, Context

NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|nor|neither)\b|n't", re.IGNORECASE)
NEGATION_WINDOW = 60  # characters before a phrase, within the same sentence

CAUSAL_PATTERNS = (
    r"\bbecause\b",
    r"\bcaused? by\b",
    r"\bdue to\b",
    r"\bas a result of\b",
    r"\bleads? to\b",
    r"\bresult(?:s|ed|ing)? (?:in|from)\b",
    r"\bcaus(?:e|es|ed|ing)\b",
)
PREDICTION_PATTERNS = (
    r"\bwill (?:likely |soon |probably |eventually )?(?:fail|break|die)\b",
    r"\b(?:about|going|likely|expected|bound|set) to fail\b",
    r"\bimminent\b",
    r"\bis failing\b",
    r"\bpredict(?:s|ed|ing)? (?:a |the |drive )?failure\b",
    r"\bfailure is (?:likely|expected|coming|near)\b",
    r"\bat risk of failing\b",
)
OUTCOME_PATTERNS = (
    r"failure_within_30d",
    r"label_is_complete",
    r"\beval_\w+",
    r"\bfailed\b",
)
ATTRIBUTION_SENTENCE = "does not provide per-feature causal attribution"

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CITATION = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class ValidationError:
    code: str
    detail: str


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def numbers_in(text: str) -> set:
    """Digit sequences, with thousands separators removed so 31,968,003 equals 31968003."""
    return set(_NUMBER.findall(re.sub(r"(?<=\d),(?=\d)", "", text)))


def _answer_numbers(answer: str) -> set:
    body = _CITATION.sub(" ", answer)
    body = re.sub(r"(?m)^\s*\d+[.)]\s", " ", body)  # list numbering is not a telemetry number
    return numbers_in(body)


def _negated(text: str, start: int) -> bool:
    boundary = max(text.rfind(c, 0, start) for c in ".!?\n")
    return bool(NEGATION.search(text[boundary + 1:start][-NEGATION_WINDOW:]))


def _matches(answer: str, patterns: Sequence[str], allow_negated: bool) -> List[str]:
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, answer, flags=re.IGNORECASE):
            if allow_negated and _negated(answer, m.start()):
                continue
            found.append(m.group(0))
    return found


def split_sections(answer: str, headings: Sequence[str]) -> dict:
    """Section text by heading; a missing heading maps to None."""
    starts = {}
    for h in headings:
        m = re.search(rf"(?im)^\s*(?:#{{1,6}}\s*)?\**\s*{re.escape(h)}", answer)
        starts[h] = m.start() if m else None
    ordered = sorted((pos, h) for h, pos in starts.items() if pos is not None)
    sections = {h: None for h in headings}
    for i, (pos, h) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(answer)
        sections[h] = answer[pos:end]
    return sections


def _cited(text: Optional[str]) -> set:
    ids = set()
    for group in _CITATION.findall(text or ""):
        ids.update(part.strip() for part in re.split(r"[;,]", group) if part.strip())
    return ids


cited_ids = _cited

ALIAS_PREFIX = "KNOWLEDGE:"


def canonicalize_citations(answer: str, context: Context) -> str:
    """Rewrite `[KNOWLEDGE:<id>]` as `[<id>]`, but only when <id> is a knowledge document supplied in this context.

    Some models prefix the canonical ids they were given. Nothing else is rewritten: an unknown id, another
    prefix, a different letter case, or a report id behind the prefix is left as written, so the validator
    still rejects it. Report citations are never touched.
    """
    supplied = {k["doc_id"] for k in context.knowledge}

    def fix_part(part: str) -> str:
        token = part.strip()
        if token.startswith(ALIAS_PREFIX) and token[len(ALIAS_PREFIX):] in supplied:
            return part.replace(token, token[len(ALIAS_PREFIX):])
        return part

    def fix_group(match) -> str:
        return "[" + "".join(fix_part(p) if p not in ",;" else p for p in re.split(r"([;,])", match.group(1))) + "]"

    return _CITATION.sub(fix_group, answer)


# --- sentences ---------------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=\S)")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+")
_ONLY_CITATIONS = re.compile(r"^(?:\[[^\[\]]+\]\s*)+[.]?$")


def section_sentences(section: Optional[str]) -> List[str]:
    """The citation units of a section, without its heading.

    A line is split into sentences, and a citation covers the uncited sentences before it in the same
    line, so "A. B. [x]" is one unit supported by [x]. A lone citation joins the previous sentence, and
    sentences after the last citation in a line form an uncited unit. Lines never merge into each other.
    """
    if not section:
        return []
    section = section.lstrip()   # a section can start with the blank line before its heading
    body = section.split("\n", 1)[1] if "\n" in section else ""
    units: List[str] = []
    for line in body.split("\n"):
        line = _LIST_MARKER.sub("", line)      # "1. text": the number is list numbering, not a sentence
        pending: List[str] = []
        sentences: List[str] = []
        for piece in (p.strip() for p in _SENTENCE_BREAK.split(line)):
            if not piece:
                continue
            if _ONLY_CITATIONS.match(piece) and sentences:
                sentences[-1] += " " + piece
            else:
                sentences.append(piece)
        for sentence in sentences:
            pending.append(sentence)
            if _CITATION.search(sentence):
                units.append(" ".join(pending))
                pending = []
        if pending:
            units.append(" ".join(pending))
    return units


def _snippet(sentence: str, width: int = 90) -> str:
    text = _CITATION.sub("", sentence).strip()
    return text if len(text) <= width else text[:width] + "..."


# --- telemetry claims --------------------------------------------------------------------------

_ATTR = r"smart[\s_]*\d+(?!\d)(?:[\s_]*raw)?"      # (?!\d) stops backtracking from splitting "187" into "18" and "7"
_ATTR_RE = re.compile(_ATTR, re.IGNORECASE)
_FOR_ATTR = rf"(?:\s*(?:for|of|on)\s+(?P<attr2>{_ATTR}))?"
_FILLER = r"\s*(?:of|was|is|at|=|:)?\s*"
_NUM = r"(?P<num>[+-]?\d+(?:\.\d+)?)"
_DATE = r"\d{4}-\d{2}-\d{2}"

# Each cue must be followed directly by its number, so "a z-score above 3" is not read as a claim.
_FIELD_PATTERNS = {
    "z": re.compile(rf"\bz(?:[- ]?scores?)?\b{_FOR_ATTR}{_FILLER}{_NUM}", re.IGNORECASE),
    "mean": re.compile(rf"\b(?:7[- ]day |seven[- ]day |rolling )?mean\b{_FOR_ATTR}{_FILLER}{_NUM}", re.IGNORECASE),
    "delta": re.compile(
        rf"\b(?:change|changed|delta|rise|rose|rises|increase|increased|decrease|decreased)\b"
        rf"(?:\s+since\s+(?:the\s+)?previous\s+day)?{_FOR_ATTR}(?:\s+(?:was|is|of|by|to))?\s*{_NUM}",
        re.IGNORECASE,
    ),
    "raw": re.compile(rf"\b(?:raw\s+value|value|reading)\b{_FOR_ATTR}{_FILLER}{_NUM}", re.IGNORECASE),
}
_CELL = r"(?:not reported|[+-]?\d+(?:\.\d+)?)"
_TABLE_ROW = re.compile(
    rf"(?P<attr>{_ATTR})\s*:\s*(?P<raw>{_CELL})\s*\|\s*(?P<mean>{_CELL})\s*\|\s*(?P<delta>{_CELL})\s*\|\s*(?P<z>{_CELL})",
    re.IGNORECASE,
)
_TREND = re.compile(rf"(?P<date>{_DATE})\s*=\s*(?P<v>\d+|not recorded)", re.IGNORECASE)
_ATTR_EQUALS = re.compile(rf"(?P<attr>{_ATTR})\s*(?:=|:|was|is)\s*(?P<num>\d+)\b(?!\.\d)(?!\s*\|)", re.IGNORECASE)
_SCORE = re.compile(r"\bscor(?:e|ed|es)\b[^.\d]{0,25}?(?P<num>\d+)\s*(?:out of|of|/)\s*8\b", re.IGNORECASE)
_ANOMALY_SCORE = re.compile(r"\banomaly_score\s*(?:of|was|is|=|:)?\s*(?P<num>\d\.\d+)", re.IGNORECASE)
_RANK = re.compile(r"\brank(?:ed)?\s*(?:of|was|is|:)?\s*(?P<num>\d+)\b", re.IGNORECASE)
_DATE_RE = re.compile(_DATE)

# Precision of each displayed field: a claim may differ from the stored value by half a displayed unit,
# because the stored trigger text and the report formatter can round a tie differently.
_TOLERANCE = {"raw": 0.5, "delta": 0.5, "z": 0.0500001, "mean": 0.0500001, "anomaly_score": 0.00005}


@dataclass
class Claim:
    attribute: Optional[str]
    field: str
    value: Optional[float]      # None means the answer says "not reported" / "not recorded"
    text: str
    date: Optional[str] = None


def _key(attribute_text: str) -> str:
    number = re.search(r"\d+", attribute_text).group(0)
    return f"smart_{number}_raw"


def _value(text: str) -> Optional[float]:
    return float(text) if re.match(r"[+-]?\d", text) else None


def _only_date(sentence: str) -> Optional[str]:
    dates = set(_DATE_RE.findall(sentence))
    return dates.pop() if len(dates) == 1 else None


def extract_claims(sentence: str) -> List[Claim]:
    """Attribute / field / number claims in one sentence. Unowned numbers are left to the global check."""
    sentence = _CITATION.sub(" ", sentence)     # citation ids contain attribute and method names
    mentions = [(m.start(), _key(m.group(0))) for m in _ATTR_RE.finditer(sentence)]
    taken: List[tuple] = []
    claims: List[Claim] = []

    def free(match) -> bool:
        return not any(match.start() < end and start < match.end() for start, end in taken)

    def take(match) -> None:
        taken.append((match.start(), match.end()))

    def owner(position: int) -> Optional[str]:
        before = [key for start, key in mentions if start < position]
        return before[-1] if before else None

    for m in _TABLE_ROW.finditer(sentence):
        if free(m):
            take(m)
            for field in ("raw", "mean", "delta", "z"):
                claims.append(Claim(_key(m.group("attr")), field, _value(m.group(field)), m.group(0)))

    for m in _TREND.finditer(sentence):
        if free(m):
            take(m)
            claims.append(Claim(owner(m.start()), "raw_on_date", _value(m.group("v")), m.group(0), m.group("date")))

    date = _only_date(sentence)
    for field, pattern in _FIELD_PATTERNS.items():
        for m in pattern.finditer(sentence):
            if free(m):
                take(m)
                attribute = _key(m.group("attr2")) if m.group("attr2") else owner(m.start())
                claims.append(Claim(attribute, field, float(m.group("num")), m.group(0), date))

    for m in _ATTR_EQUALS.finditer(sentence):
        if free(m):
            take(m)
            claims.append(Claim(_key(m.group("attr")), "raw", float(m.group("num")), m.group(0), date))

    for field, pattern in (("score", _SCORE), ("anomaly_score", _ANOMALY_SCORE), ("rank", _RANK)):
        for m in pattern.finditer(sentence):
            claims.append(Claim(None, field, float(m.group("num")), m.group(0)))
    return claims


def _shown(field: str, value) -> str:
    if value is None:
        return "not reported"
    if field == "delta":
        return f"{int(round(value)):+d}"
    if field in ("raw", "raw_on_date", "score", "rank"):
        return str(int(round(value)))
    return f"{value:.4f}" if field == "anomaly_score" else f"{value:.1f}"


def _agrees(field: str, claimed: Optional[float], actual) -> bool:
    if claimed is None or actual is None:
        return claimed is None and actual is None
    return abs(claimed - float(actual)) <= _TOLERANCE.get(field, 0.0)


def check_claims(claims: Sequence[Claim], facts: Dict) -> List[ValidationError]:
    """Each claim must match the report's value for that attribute and field."""
    errors: List[ValidationError] = []
    for c in claims:
        if c.field in ("score", "anomaly_score", "rank"):
            actual = facts.get(c.field)
            if actual is not None and not _agrees(c.field, c.value, actual):
                errors.append(ValidationError(
                    "misattributed_number",
                    f"the {c.field} in the report is {_shown(c.field, actual)}, but the answer says {_shown(c.field, c.value)}",
                ))
            continue
        if c.attribute is None:
            continue
        values = facts["attributes"].get(c.attribute)
        if values is None:
            errors.append(ValidationError("misattributed_number", f"{c.attribute} is not in the drive report"))
            continue
        if c.field == "raw_on_date" or (c.field == "raw" and c.date):
            by_date = facts["raw_by_date"].get(c.attribute, {})
            day = next((d for d in by_date if str(d) == c.date), None)
            if day is None:
                errors.append(ValidationError("misattributed_number", f"the report has no {c.attribute} value for {c.date}"))
                continue
            actual, label, field = by_date[day], f"{c.attribute} on {c.date}", "raw"
        else:
            actual, label, field = values[c.field], f"{c.attribute} {c.field}", c.field
        if not _agrees(field, c.value, actual):
            errors.append(ValidationError(
                "misattributed_number",
                f"{label}: the report has {_shown(field, actual)}, but the answer says {_shown(field, c.value)} ('{c.text.strip()}')",
            ))
    return errors


# --- citation support --------------------------------------------------------------------------

_METHOD_WORDS = {
    "rule_baseline": re.compile(r"\brule[\s_]baseline\b", re.IGNORECASE),
    "isolation_forest": re.compile(r"\bisolation[\s_]forest\b|\bisolationforest\b", re.IGNORECASE),
}


_STOPWORDS = {"that", "this", "with", "from", "have", "were", "which", "their", "there", "about", "also", "into",
              "than", "then", "them", "they", "been", "being", "would", "could", "should", "such", "each", "only",
              "very", "when", "where", "what", "while", "does", "both", "more", "most", "some", "these", "those"}
_COVERAGE = 0.8      # share of a paraphrase's content words that must appear in the document it cites


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", _CITATION.sub(" ", text).lower()).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z][a-z0-9]{3,}", text.lower())      # identifiers split on "_": rule_baseline is two words
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words if w not in _STOPWORDS}


def _quotes(sentence: str, doc_text: str) -> bool:
    """The sentence appears word for word in the document (whitespace, case and citation marks aside)."""
    quoted = re.sub(r"^(?:[-*\u2022]+|\d+[.)])\s+", "", _normalize(sentence)).rstrip(".:;")     # a list marker is not part of the quote
    return bool(quoted) and quoted in _normalize(doc_text)


def _paraphrases(sentence: str, doc_text: str) -> bool:
    """Most of the sentence's content words, and every number in it, come from the document."""
    words = _content_words(sentence)
    if len(words) < 3 or not numbers_in(sentence) <= numbers_in(doc_text):
        return False
    return len(words & _content_words(doc_text)) / len(words) >= _COVERAGE


def _listed_doc(unit: str, knowledge: Sequence[Dict]) -> Optional[str]:
    """The id of the context document this unit lists as "- title (source: name)", if it is exactly that."""
    line = _normalize(unit).lstrip("-* ")
    for k in knowledge:
        if line == _normalize(f"{k['title']} (source: {k['source_name']})"):
            return k["doc_id"]
    return None


def _leading_label(text: str, supplied: set) -> Optional[tuple]:
    """(doc_id, text without the label) for a leading "**doc_id**:" or "doc_id:" label naming a supplied document."""
    match = _LABEL.match(text)
    if match and match.group(2) in supplied:
        return match.group(2), text[match.end():]
    return None


_LABEL = re.compile(r"^\s*(?:[-*\u2022]+\s+|\d+[.)]\s+)?(\*\*|__)?([A-Za-z]+:[A-Za-z0-9_]+(?::[A-Za-z0-9_]+)?)\1?:\s*")


def _background_verdict(unit: str, knowledge: Sequence[Dict]) -> str:
    """ok, missing (cites nothing) or wrong (cites documents that do not support what is said).

    Each sentence must be a verbatim quote of a cited document, or a paraphrase that cites the document of
    the attribute or detector it is about and is supported by that document's text.
    """
    # a document supports what its title says as well as what its body says
    texts = {k["doc_id"]: f"{k['title']}. {k['text']}" for k in knowledge}
    cites = _cited(unit)
    if not cites:
        return "missing"
    cited_docs = {i: texts[i] for i in cites if i in texts}
    if not cited_docs:
        return "wrong"
    if _listed_doc(unit, knowledge) in cited_docs:      # a fixed reference line for the very document it cites
        return "ok"
    body = _CITATION.sub(" ", unit)
    label = _leading_label(body, {k["doc_id"] for k in knowledge})
    if label:
        if label[0] not in cited_docs:      # a label naming a document the sentence does not cite
            return "wrong"
        body = label[1]                      # compared without the label; the label is not evidence
        if not body.strip(" .:;"):
            return "wrong"                   # nothing but a label
    for sentence in (p.strip() for p in _SENTENCE_BREAK.split(body) if p.strip()):
        if any(_quotes(sentence, t) for t in cited_docs.values()):
            continue
        methods = [m for m, pattern in _METHOD_WORDS.items() if pattern.search(sentence)]
        if methods:
            if not all(f"method:{m}" in cited_docs for m in methods):
                return "wrong"
            relevant = [cited_docs[f"method:{m}"] for m in methods]
        else:
            mentioned = {f"attr:{_key(m.group(0))}" for m in _ATTR_RE.finditer(sentence)}
            # one matching attribute document is enough: SMART 197's own text mentions SMART 198
            matching = mentioned & set(cited_docs)
            if mentioned and not matching:
                return "wrong"
            relevant = [cited_docs[i] for i in matching] or list(cited_docs.values())
        if not any(_paraphrases(sentence, t) for t in relevant):
            return "wrong"
    return "ok"


# a short introduction to a list ("Telemetry on the flagged date (...):") carries no claim of its own
_WINDOW_LABEL = re.compile(r"\b\d+[- ]days?\b", re.IGNORECASE)


def _is_lead_in(unit: str) -> bool:
    text = _CITATION.sub("", unit).strip()
    if not text.endswith(":") or len(text.split()) > 25 or len([p for p in _SENTENCE_BREAK.split(text) if p.strip()]) != 1:
        return False
    return not _ATTR_RE.search(text) and not re.search(r"\d", _WINDOW_LABEL.sub("", text))


def check_citation_support(sections: Dict, context: Context) -> List[ValidationError]:
    """Each sentence must cite a document that can support it."""
    errors: List[ValidationError] = []

    def add(heading: str, rule: str, missing: List[str], wrong: List[str]) -> None:
        for code, offenders in (("missing_citation", missing), ("unsupported_citation", wrong)):
            if offenders:
                shown = "; ".join(f"'{_snippet(s)}' cites {sorted(_cited(s)) or 'nothing'}" for s in offenders[:3])
                more = f" (and {len(offenders) - 3} more)" if len(offenders) > 3 else ""
                errors.append(ValidationError(code, f"in '{heading}', {rule}: {shown}{more}"))

    if context.mode == "drive":
        report_id, own_method = context.report.doc_id, f"method:{context.method}"
        other_methods = {f"method:{m}" for m in _METHOD_WORDS} - {own_method}

        missing, wrong = [], []
        observed = section_sentences(sections.get("Observed telemetry"))
        for i, sentence in enumerate(observed):
            cites = _cited(sentence)
            if report_id in cites:
                continue
            # a bare lead-in is fine when the very next line is cited evidence from the drive report
            if not cites and _is_lead_in(sentence) and i + 1 < len(observed) and report_id in _cited(observed[i + 1]):
                continue
            (wrong if cites else missing).append(sentence)
        add("Observed telemetry", f"every sentence about the drive must cite [{report_id}]", missing, wrong)

        missing, wrong = [], []
        for sentence in section_sentences(sections.get("How the detector scored this drive")):
            cites = _cited(sentence)
            if not cites:
                missing.append(sentence)
            elif not cites & {report_id, own_method} or cites & other_methods:
                wrong.append(sentence)
        add("How the detector scored this drive",
            f"a detector statement must cite [{report_id}] or [{own_method}], never a SMART document or the other detector's document",
            missing, wrong)

        heading, rule = "General background", ("every background sentence must cite the knowledge document it came from "
                                                "(a SMART attribute needs its own attr:* document, a detector needs its method document)")
        sentences = [s for s in section_sentences(sections.get(heading)) if not re.match(r"no relevant background", s, re.IGNORECASE)]
    else:
        heading, rule = "Answer", "every sentence must cite the knowledge document it came from"
        sentences = section_sentences(sections.get(heading))

    verdicts = [(s, _background_verdict(s, context.knowledge)) for s in sentences]
    add(heading, rule, [s for s, v in verdicts if v == "missing"], [s for s, v in verdicts if v == "wrong"])
    return errors


def validate_answer(answer: str, context: Context) -> List[ValidationError]:
    errors: List[ValidationError] = []
    headings = DRIVE_HEADINGS if context.mode == "drive" else GENERAL_HEADINGS
    sections = split_sections(answer, headings)

    for h, body in sections.items():
        if body is None:
            errors.append(ValidationError("missing_section", f"the section '{h}' is missing"))

    allowed = set(context.allowed_ids)
    for cited in sorted(_cited(answer) - allowed):
        errors.append(ValidationError("unknown_citation", f"[{cited}] is not one of the provided ids {sorted(allowed)}"))

    present = {h: body for h, body in sections.items() if body is not None}
    errors += check_citation_support(present, context)
    if context.mode == "drive" and present.get("General background") is not None and not context.knowledge:
        if "no relevant background" not in present["General background"].lower():
            errors.append(ValidationError("missing_citation", "'General background' must say no relevant background documents were retrieved"))

    unsupported = sorted(_answer_numbers(answer) - numbers_in(context.text), key=lambda n: (len(n), n))
    if unsupported:
        errors.append(ValidationError("unsupported_number", f"numbers not found in the context: {unsupported}"))

    if context.mode == "drive" and context.report.facts:
        claims = [c for body in present.values() for sentence in section_sentences(body) for c in extract_claims(sentence)]
        errors += check_claims(claims, context.report.facts)

    causal = _matches(answer, CAUSAL_PATTERNS, allow_negated=True)
    if causal:
        errors.append(ValidationError("causal_wording", f"unsupported causal wording: {sorted(set(causal))}"))
    predicted = _matches(answer, PREDICTION_PATTERNS, allow_negated=True)
    if predicted:
        errors.append(ValidationError("failure_prediction", f"failure prediction wording: {sorted(set(predicted))}"))
    outcomes = _matches(answer, OUTCOME_PATTERNS, allow_negated=False)
    if outcomes:
        errors.append(ValidationError("outcome_reference", f"mentions outcomes or labels: {sorted(set(outcomes))}"))

    squashed = _squash(answer)
    if context.method == "isolation_forest" and ATTRIBUTION_SENTENCE not in squashed:
        errors.append(ValidationError("missing_attribution_statement", "an isolation_forest answer must say it does not provide per-feature causal attribution"))
    if context.method == "rule_baseline" and ATTRIBUTION_SENTENCE in squashed:
        errors.append(ValidationError(
            "unsupported_detector_claim",
            "the no-per-feature-attribution statement describes the isolation_forest detector; this drive report is from rule_baseline, whose triggers are listed in the report",
        ))
    return errors
