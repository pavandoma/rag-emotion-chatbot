"""Measures the real relevance improvement of RAG vs a no-retrieval baseline
on a sample of rag-mini-wikipedia QA pairs (which ship ground-truth answers).

Relevance is scored two ways: embedding cosine similarity (primary,
semantic) and token-overlap F1 (secondary sanity check, SQuAD-style). The
reported % improvement is whatever this script actually measures -- not
forced to match the resume's original 30% figure, which came from a
different-scale system. See README "Honest Scope".
"""

import json
import re
import time

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.baseline_pipeline import answer as baseline_answer
from src.config import EMBEDDING_MODEL, EVAL_RESULTS_DIR
from src.rag_pipeline import answer as rag_answer


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def main(n_samples: int = 50) -> None:
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")["test"]
    sample = qa.shuffle(seed=42).select(range(min(n_samples, len(qa))))

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    records = []
    for row in sample:
        question, ground_truth = row["question"], row["answer"]

        baseline_text = baseline_answer(question)
        rag_result = rag_answer(question, apply_safety_filter=False)
        rag_text = rag_result.answer

        gt_emb, base_emb, rag_emb = embedder.encode([ground_truth, baseline_text, rag_text])
        base_cosine = float(np.dot(gt_emb, base_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(base_emb)))
        rag_cosine = float(np.dot(gt_emb, rag_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(rag_emb)))

        records.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "baseline_answer": baseline_text,
                "rag_answer": rag_text,
                "baseline_cosine": base_cosine,
                "rag_cosine": rag_cosine,
                "baseline_f1": token_f1(baseline_text, ground_truth),
                "rag_f1": token_f1(rag_text, ground_truth),
            }
        )
        print(f"[{len(records)}/{len(sample)}] cosine baseline={base_cosine:.3f} rag={rag_cosine:.3f}")

    baseline_cosine_mean = float(np.mean([r["baseline_cosine"] for r in records]))
    rag_cosine_mean = float(np.mean([r["rag_cosine"] for r in records]))
    baseline_f1_mean = float(np.mean([r["baseline_f1"] for r in records]))
    rag_f1_mean = float(np.mean([r["rag_f1"] for r in records]))

    cosine_improvement_pct = (rag_cosine_mean - baseline_cosine_mean) / baseline_cosine_mean * 100
    f1_improvement_pct = (
        (rag_f1_mean - baseline_f1_mean) / baseline_f1_mean * 100 if baseline_f1_mean > 0 else float("inf")
    )

    summary = {
        "n_samples": len(records),
        "baseline_cosine_mean": baseline_cosine_mean,
        "rag_cosine_mean": rag_cosine_mean,
        "cosine_relevance_improvement_pct": cosine_improvement_pct,
        "baseline_f1_mean": baseline_f1_mean,
        "rag_f1_mean": rag_f1_mean,
        "f1_improvement_pct": f1_improvement_pct,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "llama3.2:3b via Ollama",
    }

    print("\n=== Relevance Eval Summary ===")
    print(json.dumps(summary, indent=2))

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (EVAL_RESULTS_DIR / "relevance_eval.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
