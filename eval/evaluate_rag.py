"""Retrieval evaluation for the knowledge documents.

    python -m eval.evaluate_rag --set eval       # the 15 evaluation questions
    python -m eval.evaluate_rag --set dev        # the dev questions, for choosing the minimum similarity

This embeds the questions locally and ranks the document vectors in memory. It calls no language model
and does not touch Snowflake.
With 10 documents the top 3 covers 30% of the corpus, so hit@3 is an easy bar; hit@1 and the
abstention check are reported next to it.
"""
import argparse
import json
import pathlib
from typing import Dict, List, Optional, Sequence

from src.rag import retrieve

HERE = pathlib.Path(__file__).parent
QUESTION_FILES = {"eval": HERE / "rag_questions.json", "dev": HERE / "rag_dev_questions.json"}


def load_questions(path: pathlib.Path) -> List[Dict]:
    return json.loads(pathlib.Path(path).read_text())["questions"]


def hit_at_k(retrieved_ids: Sequence[str], expected: Sequence[str], k: int) -> bool:
    return any(doc_id in expected for doc_id in list(retrieved_ids)[:k])


def reciprocal_rank(retrieved_ids: Sequence[str], expected: Sequence[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def evaluate(results: Sequence[tuple], min_similarity: float) -> Dict:
    """results: (question dict, hits sorted best first). Retrieval questions get hit@1, hit@3 and MRR."""
    rows, retrieval, abstain = [], [], []
    for question, hits in results:
        ids = [h.doc_id for h in hits]
        top = hits[0].score if hits else None
        if question["abstain"]:
            passed = not hits or hits[0].score < min_similarity
            abstain.append(passed)
            rows.append({"id": question["id"], "kind": "abstain", "top_score": top, "passed": passed, "top_id": ids[:1]})
        else:
            h1, h3 = hit_at_k(ids, question["expected_doc_ids"], 1), hit_at_k(ids, question["expected_doc_ids"], 3)
            retrieval.append((h1, h3, reciprocal_rank(ids, question["expected_doc_ids"])))
            rows.append({"id": question["id"], "kind": "retrieval", "top_score": top, "hit1": h1, "hit3": h3, "top_ids": ids[:3]})
    n = len(retrieval)
    return {
        "retrieval_questions": n,
        "hit_at_1": sum(r[0] for r in retrieval) / n if n else None,
        "hit_at_3": sum(r[1] for r in retrieval) / n if n else None,
        "mrr": sum(r[2] for r in retrieval) / n if n else None,
        "abstention_questions": len(abstain),
        "abstention_passed": sum(abstain),
        "min_similarity": min_similarity,
        "rows": rows,
    }


def format_report(report: Dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    lines = [
        f"retrieval questions: {report['retrieval_questions']} | hit@1 {pct(report['hit_at_1'])} | hit@3 {pct(report['hit_at_3'])} | MRR {pct(report['mrr'])}",
        f"abstention: {report['abstention_passed']}/{report['abstention_questions']} passed at minimum similarity {report['min_similarity']}",
        "note: the corpus has 10 documents, so hit@3 covers 30% of it; compare hit@1 and the abstention result",
    ]
    for r in report["rows"]:
        score = "n/a" if r["top_score"] is None else f"{r['top_score']:.3f}"
        if r["kind"] == "retrieval":
            lines.append(f"  {r['id']}: hit@1 {r['hit1']} | hit@3 {r['hit3']} | top score {score} | top 3 {r['top_ids']}")
        else:
            lines.append(f"  {r['id']}: abstain {'ok' if r['passed'] else 'FAILED'} | top score {score} | top {r['top_id']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate knowledge retrieval.")
    parser.add_argument("--set", choices=sorted(QUESTION_FILES), default="eval")
    parser.add_argument("--min-similarity", type=float, default=retrieve.DEFAULT_MIN_SIMILARITY)
    args = parser.parse_args(argv)

    questions = load_questions(QUESTION_FILES[args.set])
    index = retrieve.build_local_index()
    results = [(q, index.search(q["question"], top_k=retrieve.TOP_K)) for q in questions]
    print(format_report(evaluate(results, args.min_similarity)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
