"""Downloads/caches the three small HF datasets this project uses, and
prints row counts so you can sanity-check before running ingestion/training.
"""

from datasets import load_dataset


def main() -> None:
    wiki = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    wiki_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    print(f"rag-mini-wikipedia passages: {len(wiki['passages'])}")
    print(f"rag-mini-wikipedia QA test pairs: {len(wiki_qa['test'])}")

    emotion = load_dataset("dair-ai/emotion")
    print(
        f"dair-ai/emotion train={len(emotion['train'])} "
        f"val={len(emotion['validation'])} test={len(emotion['test'])}"
    )

    toxicity = load_dataset("allenai/real-toxicity-prompts", split="train")
    print(f"real-toxicity-prompts rows available: {len(toxicity)}")

    print("\nAll datasets downloaded/cached successfully.")


if __name__ == "__main__":
    main()
