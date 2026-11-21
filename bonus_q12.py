"""Bonus: question 12 needs two documents, so split it into one search per car and combine the results.

  python bonus_q12.py
"""
import json
from pathlib import Path

from eval_retrieval import build_index, is_hit, reciprocal_rank
from rag.search import rrf, search

SUB_QUESTIONS = ["מה טווח הנסיעה של אופק E5 החשמלי (OF-E5)?", "מה טווח הנסיעה של אופק E7 החשמלי (OF-E7)?"]
TOP_K = 3


def main():
    golden = json.loads(Path("golden.json").read_text(encoding="utf-8"))
    q = next(item for item in golden if item["id"] == 12)
    index = build_index("headings")
    position = {c.id: i for i, c in enumerate(index.chunks)}

    rankings = []
    for sub in SUB_QUESTIONS:
        found = search(index, sub, "hybrid", TOP_K, current_only=True)
        rankings.append([position[c.id] for c in found])
        print(f"  {sub}\n    {[c.id for c in found]}")

    merged = [index.chunks[i] for i in rrf(rankings)]
    for size in (TOP_K, len(merged)):  # the top k overall, then everything both searches returned
        combined = merged[:size]
        hit, rr = is_hit(q, combined), reciprocal_rank(q, combined)
        print(f"\n  combined top {size}: {[c.id for c in combined]}")
        print(f"  {'✓' if hit else '✗'} recall@{size}={int(hit)} MRR={rr:.2f}")


if __name__ == "__main__":
    main()
