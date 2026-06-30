# RAG Emotion-Aware Conversational Assistant

Retrieval-Augmented Conversational AI Systems (LLMs): a real, runnable RAG chatbot with emotion-aware tone, layered safety mitigation, a FastAPI service, and an AWS deployment design — built to run end-to-end on a laptop, with every claim backed by code and a measured number.

##  Scope

1. **Emotion "fine-tuning" of Llama** — Ollama serves quantized GGUF models and exposes no training API, so fine-tuning Llama's own weights isn't practical on a laptop. Instead: a real LoRA/PEFT fine-tune of `distilbert-base-uncased` on `dair-ai/emotion` backs the emotion-*recognition* accuracy claim (measured 91.0% vs. a 15.2% un-adapted baseline), and a prompt-conditioning layer (feeding the detected emotion into Ollama's system prompt) delivers the tone-adaptation claim. The fine-tune is genuine; it's just on a small classifier, not Llama itself.
2. **The 30% / 5% improvement figures** — the original numbers came from a different, larger-scale context. This rebuild faithfully reproduces the *technique* (semantic retrieval vs. no retrieval; fine-tuned vs. un-adapted classifier) and reports the real measured numbers on this scaled-down system: **+97.5%** relevance (embedding cosine) for RAG vs. baseline, and a **91.0% vs. 15.2%** accuracy gap for the emotion classifier. Neither figure is forced to match the original 30%/5%.
3. **AWS inference pipeline** — `docker compose up --build` has been run and verified end-to-end (Ollama pulled `llama3.2:3b` inside its container, the API container auto-ingests the knowledge base on first start via `infra/entrypoint.sh`, and `/health` + `/chat` both responded correctly with grounded, safety-filtered answers from inside Docker). `sam validate -t infra/aws/template.yaml --lint` also passes. `infra/aws/template.yaml` is still illustrative and was never deployed to a live AWS account, to avoid cost — but the containerized pipeline it describes is now confirmed working, not just reviewed.
4. **Safety eval came back at 0% before/after** — on the 40-prompt curated set, `llama3.2:3b` never produced an unsafe completion in the first place (its own RLHF alignment refused), so the bias filter measured nothing to catch end-to-end. The filter itself is verified separately against synthetic unsafe text (see the table below) to confirm the mechanism works — the 0%-delta result is an honest finding about this particular model and prompt set, not a sign the filter is non-functional.

Bias/safety mitigation, the RESTful API, and the core RAG retrieval mechanics are fully real, running code — no caveats there.

## Architecture

```
User query
   |
   v
[emotion classifier] --(detected emotion)--> [prompt conditioning] --(tone instruction)--+
   |                                                                                      |
   v                                                                                      v
[embed query] -> [Chroma top-k retrieval] -> [build grounded prompt] -> [Ollama: llama3.2:3b] -> [safety filter] -> response
```

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store**: ChromaDB (`PersistentClient`, local)
- **Generation**: Ollama (`llama3.2:3b`)
- **Emotion**: LoRA-adapted DistilBERT classifier + prompt conditioning
- **Safety**: keyword denylist + `unitary/toxic-bert` classifier gate
- **API**: FastAPI

## Quick Start (Windows / PowerShell)

```powershell
# 1. Environment
.\scripts\setup_env.ps1
. .\.venv\Scripts\Activate.ps1

# 2. Ollama + model
.\scripts\pull_ollama_model.ps1

# 3. Datasets
python scripts\download_datasets.py

# 4. Ingest knowledge base into Chroma
python -m src.ingest

# 5. Fine-tune the emotion classifier
python -m src.emotion.train_lora

# 6. Run evals (produces eval/results/*.json)
python eval\run_relevance_eval.py
python eval\run_emotion_eval.py
python eval\run_safety_eval.py

# 7. Run the API
uvicorn src.api.main:app --reload

# 8. Try it
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" `
  -Body '{"query": "What is the capital of France?", "use_rag": true}'
```

##  Bullet → Code → Evidence Mapping

| claim | Code | Evidence (real, measured) | Honesty note |
|---|---|---|---|
| RAG via vector DB, ~30% relevance improvement | [`src/rag_pipeline.py`](src/rag_pipeline.py), [`eval/run_relevance_eval.py`](eval/run_relevance_eval.py) | **+97.5%** mean embedding-cosine relevance (RAG 0.313 vs. baseline 0.158) and **+818%** token-overlap F1, on 50 sampled `rag-mini-wikipedia` QA pairs — `eval/results/relevance_eval.json` | Real measured number on this scaled-down system; methodology faithful to the original claim, figure is this system's own (not forced to match 30%) |
| Fine-tuned Llama with emotion data, 5% accuracy improvement, natural tone | [`src/emotion/train_lora.py`](src/emotion/train_lora.py), [`src/emotion/prompt_conditioning.py`](src/emotion/prompt_conditioning.py), [`eval/run_emotion_eval.py`](eval/run_emotion_eval.py) | LoRA fine-tuned: **91.0% accuracy / 0.862 F1-macro** vs. **15.2% accuracy / 0.079 F1-macro** for the un-adapted baseline, on the full 2,000-row `dair-ai/emotion` test split — `eval/results/emotion_eval.json` | LoRA on DistilBERT, not Llama weights; baseline is a randomly-initialized classification head (~chance for 6 classes), so the gap is large by construction — tone adaptation comes via prompt conditioning, not Llama fine-tuning — see Honest Scope |
| Bias mitigation, reduced unsafe outputs | [`src/safety/bias_filter.py`](src/safety/bias_filter.py), [`eval/run_safety_eval.py`](eval/run_safety_eval.py) | 0% unsafe rate before *and* after on the 40-prompt curated set (`eval/results/safety_eval.json`) — `llama3.2:3b`'s own RLHF alignment already refused every prompt, so the filter had nothing to catch on this set. Filter mechanics verified directly: it correctly blocks synthetic unsafe text (e.g. self-harm and weapon-making prompts, toxicity score 1.000 → replaced) while passing benign text through unchanged | Filter is fully real and live in the API; this particular eval set didn't surface a case where it needed to intervene, which is itself an honest (if anticlimactic) result — see Honest Scope |
| RESTful APIs for LLM interaction | [`src/api/main.py`](src/api/main.py) | curl/`Invoke-RestMethod` examples above | Fully real |
| Inference pipeline on AWS | [`infra/Dockerfile`](infra/Dockerfile), [`infra/docker-compose.yml`](infra/docker-compose.yml), [`infra/aws/`](infra/aws/) | `infra/aws/DEPLOYMENT.md` | IaC valid (Docker/compose run locally, SAM template validates), not provisioned to avoid cost |

## Reproducing the Eval Numbers

| Script | What it measures | Output |
|---|---|---|
| `eval/run_relevance_eval.py` | RAG vs. no-retrieval relevance (embedding cosine + token F1) on `rag-mini-wikipedia` QA pairs | `eval/results/relevance_eval.json` |
| `eval/run_emotion_eval.py` | LoRA-fine-tuned vs. un-adapted classifier accuracy on `dair-ai/emotion` test split | `eval/results/emotion_eval.json` |
| `eval/run_safety_eval.py` | Unsafe-output rate before/after the bias filter, on a curated 40-prompt safety set | `eval/results/safety_eval.json` |

## Project Structure

```
rag-emotion-chatbot/
├── src/            # the running system: ingestion, RAG, emotion, safety, API
├── eval/           # the evidence: scripts that produce real measured numbers
├── infra/          # Docker/compose (runnable) + AWS IaC/docs (illustrative)
├── scripts/        # Windows setup helpers
├── data/           # datasets and the persisted Chroma store (gitignored)
└── tests/          # unit tests (chunking, RAG pipeline, API), mocked external calls
```

## Tech Stack Rationale

- **`infra/entrypoint.sh`** — the container checks whether the Chroma collection already exists on the mounted volume and runs `python -m src.ingest` automatically on first start if not. (This was added after discovering the API container crashed on startup the first time it was actually run — the original `Dockerfile` copied the knowledge-base source files but never ran ingestion, so the vector store was empty. Fixed and verified — see `docker compose up --build` in Quick Start.)
- **No LangChain** — the retrieval/prompt-building logic is hand-written (~80 lines) so it can be explained line-by-line in an interview, rather than pointed at framework internals.
- **Ollama** — runs `llama3.2:3b` locally with no API key/cost, matches the "Llama models" claim for the generation step.
- **ChromaDB** — zero-setup embedded vector store, a widely recognized name.
- **LoRA/PEFT on DistilBERT** — the most credible, actually-runnable stand-in for "fine-tuned Llama" on laptop hardware; genuinely trains in minutes.

## Future Work (what a production version would add)

- Real Llama fine-tuning via GPU rental (Unsloth/HF + QLoRA) instead of the DistilBERT stand-in.
- Managed vector DB (OpenSearch/Pinecone) and Bedrock-hosted inference at scale.
- Actual AWS provisioning behind CI/CD, with cost monitoring.
