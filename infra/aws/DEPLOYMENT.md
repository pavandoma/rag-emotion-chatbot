# AWS Deployment Mapping (Design Only — Nothing Provisioned)

This document maps the local architecture (FastAPI + Ollama + ChromaDB, see
`infra/docker-compose.yml`) onto AWS services. **No resources in this repo
are actually provisioned on a live AWS account** — `infra/aws/template.yaml`
validates with `sam validate` but was never deployed, to avoid incurring
cost on a personal account. This is stated explicitly so the scope is
honestly represented in interviews.

## Architecture mapping

| Component | Local (this repo) | AWS option A | AWS option B |
|---|---|---|---|
| API layer | FastAPI + Uvicorn in Docker | **ECS Fargate** behind an Application Load Balancer | Lambda + API Gateway |
| LLM inference | Ollama container (`llama3.2:3b`) | Self-hosted Ollama on an **EC2 GPU instance** (e.g. `g5.xlarge`) | **Amazon Bedrock** hosted Llama models (no self-hosting) |
| Vector DB | ChromaDB, local volume | ChromaDB on an EBS-backed EC2/ECS volume (small scale) | Managed: OpenSearch with vector engine, or Pinecone (production scale) |
| CI/CD | none (local dev) | GitHub Actions → ECR push → ECS service update | — |

### Why ECS/EC2 over Lambda here

Lambda + API Gateway is attractive for bursty, low-traffic APIs, but this
service loads several models into memory at startup (sentence-transformer,
LoRA emotion classifier, toxic-bert) — that's a multi-second cold start on
every Lambda cold invocation, which makes ECS Fargate (always-warm container)
or EC2 the more realistic choice despite the higher idle cost.

### Why EC2 GPU or Bedrock for inference

Ollama serving `llama3.2:3b` runs adequately on CPU for a demo, but a
production system handling concurrent requests benefits from a GPU instance.
Amazon Bedrock is the lower-ops alternative: no instance management, but
loses the "self-hosted Llama" story this project demonstrates locally.

## Illustrative artifacts in this repo

- `infra/aws/template.yaml` — AWS SAM template defining an ECS Fargate
  service, task definition, and ALB. Validate with:
  ```
  sam validate -t infra/aws/template.yaml
  ```
  Do **not** run `sam deploy` against a real account unless you intend to
  provision and pay for these resources.
- `infra/aws/ecs-task-def.json` — example ECS task definition (CPU/memory
  sizing, container definitions for `api` and `ollama`, environment
  variables) for reference/discussion.

## Sketch CI/CD flow (described, not implemented)

1. GitHub Actions on push to `main`: build Docker image, run tests.
2. Push image to Amazon ECR.
3. Update the ECS service to the new image (`aws ecs update-service --force-new-deployment`).
4. ALB health check (`/health`) gates the rollout.
