````markdown
# Fine-Tuning Service

Copyright (C) 2024-2025 Intel Corporation
SPDX-License-Identifier: Apache-2.0

This is a reference/blueprint fine-tuning solution for **Intel® AI for Enterprise Inference (Agentic Toolkit)** that deploys a complete LLM fine-tuning stack alongside the existing inference cluster.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [What Gets Deployed](#what-gets-deployed)
3. [Directory Structure](#directory-structure)
4. [Prerequisites](#prerequisites)
5. [End-to-End Deployment Guide](#end-to-end-deployment-guide)
   - [Step 1: Deploy the Nvidia Fine-Tuning Engine (GPU machine)](#step-1-deploy-the-nvidia-fine-tuning-engine-gpu-machine)
   - [Step 2: Enable the plugin in the Agentic Toolkit](#step-2-enable-the-plugin-in-the-agentic-toolkit)
   - [Step 3: Configure the Fine-Tuning Service](#step-3-configure-the-fine-tuning-service)
   - [Step 4: Generate Vault Secrets](#step-4-generate-vault-secrets)
   - [Step 5: Deploy the Stack](#step-5-deploy-the-stack)
   - [Step 6: Verify the Deployment](#step-6-verify-the-deployment)
6. [Using the Fine-Tuning Service](#using-the-fine-tuning-service)
   - [Access URLs](#access-urls)
   - [Typical Workflow](#typical-workflow)
   - [Importing Data from Langfuse](#importing-data-from-langfuse)
7. [Configuration Reference](#configuration-reference)
8. [Manual Deployment](#manual-deployment-without-inference-stack-deploysh)
9. [Operations](#operations)
10. [Troubleshooting](#troubleshooting)
11. [Uninstalling](#uninstalling)

---

## Architecture Overview

The fine-tuning service is a **plugin** for the Enterprise Inference / Agentic Toolkit cluster. It adds three cluster-side services (API, Data Prep, UI) plus their own datastores, and communicates over an authenticated HTTPS channel with a **separate Nvidia GPU machine** that runs the actual training workload (Unsloth).

```
 ┌──────────────────────────────────────────────┐        ┌─────────────────────────────┐
 │   Enterprise Inference Cluster (K8s)         │        │  Nvidia GPU Machine         │
 │                                              │        │                             │
 │   ┌─────────┐  ┌──────────┐  ┌────────────┐  │  OAuth2│  ┌───────────────────────┐  │
 │   │   UI    │─▶│ FT API   │─▶│ Dataprep   │  │◀──────▶│  │ Fine-Tuning Engine    │  │
 │   └─────────┘  └──────────┘  └────────────┘  │  HTTPS │  │ (Unsloth + Keycloak)  │  │
 │        │           │              │          │        │  └───────────────────────┘  │
 │   Postgres/Redis  MinIO       Celery workers │        └─────────────────────────────┘
 └──────────────────────────────────────────────┘
```

## What Gets Deployed

| Component | Namespace | Purpose |
|-----------|-----------|--------|
| Fine-Tuning API | `finetuning-api` | OpenAI-compatible API for managing fine-tuning jobs |
| Data Preparation Service | `dataprep` | Document processing, Q&A dataset generation |
| Celery Workers | `dataprep` | Async processing (Docling, LlamaIndex) |
| Fine-Tuning UI | `finetuning-ui` | Web interface |
| PostgreSQL (x2) | `dataprep`, `finetuning-api` | Persistent storage per service |
| Redis (x2) | `dataprep`, `finetuning-api` | Caching and task queuing |
| MinIO | `dataprep` | Shared object storage for training files |

> **Note — Nvidia GPU Training Engine (Unsloth):** The actual GPU fine-tuning workload runs on a **separate Nvidia GPU machine**, not on the Enterprise Inference cluster. To deploy the Nvidia/Unsloth fine-tuning engine on that machine, follow the instructions in [src/finetuning-engine/README.md](src/finetuning-engine/README.md). Once it is running, set its URL and Keycloak credentials in `blueprints/finetuning_service/finetune-config.cfg` before deploying this service.

## Directory Structure

```
blueprints/finetuning_service/
├── README.md                          # This file
├── finetune-config.cfg                # User-facing configuration (DO NOT COMMIT SECRETS)
├── playbooks/
│   ├── deploy-all.yml                 # Main orchestration playbook
│   ├── deploy-finetuning-api.yml      # Fine-Tuning API
│   ├── deploy-dataprep.yml            # Data Preparation Service
│   ├── deploy-ui.yml                  # Fine-Tuning UI
│   └── build-images.yml               # Container image builds
├── vars/
│   └── finetune-plugin-vars.yml       # Internal deployment variables
├── scripts/
│   └── setup-keycloak-finetuning.sh   # Keycloak realm/client setup
└── src/
    ├── api/                           # Fine-Tuning API source (FastAPI)
    ├── dataprep/                      # Data Preparation source (FastAPI)
    ├── ui/                            # Fine-Tuning UI source (Next.js)
    └── finetuning-engine/             # Nvidia/Unsloth GPU training backend
```

---

## Prerequisites

Before deploying, ensure you have:

- A running **Enterprise Inference / Agentic Toolkit** cluster (Kubernetes-based, deployed via `core/inference-stack-deploy.sh`).
- Cluster admin `kubectl` access from the deployment host.
- **Ansible ≥ 2.14** and Python 3.10+ on the deployment host.
- A **separate Nvidia GPU machine** with:
  - Docker + NVIDIA Container Toolkit
  - A reachable HTTPS endpoint from the cluster
  - Its own Keycloak (deployed together with the finetuning engine)
- Network connectivity from the cluster egress to the GPU machine on the configured port (default `8443`).
- (Optional) A Langfuse instance if you plan to import traces as training data.

---

## End-to-End Deployment Guide

### Step 1: Deploy the Nvidia Fine-Tuning Engine (GPU machine)

The training backend must be running **before** deploying the cluster-side plugin.

On the GPU machine:

```bash
git clone https://github.com/manishintel/enterprise-agent-toolkit.git
cd enterprise-agent-toolkit/blueprints/finetuning_service/src/finetuning-engine
# Follow the README in this directory to bring up Unsloth + Keycloak
cat README.md
```

Once the engine is up, collect the following values — you will need them in Step 3:

- **Backend URL** (e.g. `https://<gpu-host>:8443`)
- **Keycloak token URL** (e.g. `https://<gpu-host>/realms/finetuning/protocol/openid-connect/token`)
- **Client ID** (default: `finetuning-api`)
- **Client secret** (from the Keycloak admin console on the GPU machine)

Verify the backend is reachable from the cluster:

```bash
curl -k https://<gpu-host>:8443/health
```

### Step 2: Enable the plugin in the Agentic Toolkit

Edit `core/inventory/inference-config.cfg`:

```properties
deploy_finetune_plugin=on
```

### Step 3: Configure the Fine-Tuning Service

Edit `blueprints/finetuning_service/finetune-config.cfg` with the values gathered in Step 1:

```properties
# URL of the Nvidia/Unsloth fine-tuning engine (deployed separately on a GPU machine)
nvidia_finetune_backend_url: https://your-nvidia-gpu-server:8443

# Keycloak token endpoint on the Nvidia machine's Keycloak
nvidia_keycloak_token_url: https://your-keycloak-server/realms/finetuning/protocol/openid-connect/token

# OAuth2 client credentials used by the Fine-Tuning API to authenticate with the Nvidia backend
nvidia_keycloak_client_id: finetuning-api
nvidia_keycloak_client_secret: <client-secret-from-nvidia-keycloak>

# Set to false only in development with self-signed certificates
nvidia_keycloak_verify_ssl: true
```

> ⚠️ **Do not commit `finetune-config.cfg` or any file containing the client secret to git.** The provided `.gitignore` should already exclude `*.cfg`; verify before pushing.

### Step 4: Generate Vault Secrets

The service uses Ansible Vault to store cluster-side secrets (database passwords, JWT signing keys, MinIO credentials, etc.).

```bash
cd core/scripts
./generate-vault-secrets.sh
```

This creates/updates `core/inventory/metadata/vault.yml` (encrypted).

### Step 5: Deploy the Stack

From the repository root:

```bash
cd core
./inference-stack-deploy.sh
```

Choose one of:

- **Option 1 — Fresh Install:** for a brand-new cluster.
- **Option 3 — Update Cluster:** to add the finetuning plugin to an existing deployment.

The script will:

1. Build container images (API, Dataprep, UI) if enabled.
2. Configure Keycloak realms/clients on the cluster (`finetuning-backend`, `finetuning-ui`).
3. Deploy PostgreSQL, Redis, MinIO in the `dataprep` and `finetuning-api` namespaces.
4. Deploy the API, Dataprep, and UI workloads.
5. Configure the ingress under `/enterprise-ai/*`.

### Step 6: Verify the Deployment

```bash
kubectl get pods -n dataprep
kubectl get pods -n finetuning-api
kubectl get pods -n finetuning-ui
```

All pods should be `Running` / `Ready`. Then check the ingress:

```bash
kubectl get ingress -A | grep enterprise-ai
```

---

## Using the Fine-Tuning Service

### Access URLs

After a successful deployment:

- **UI:** `https://<cluster-url>/enterprise-ai/ui`
- **Fine-Tuning API docs (Swagger):** `https://<cluster-url>/enterprise-ai/api/docs`
- **Data Prep API docs (Swagger):** `https://<cluster-url>/enterprise-ai/dataprep/docs`

Login uses the cluster's Keycloak (same credentials as the rest of the Agentic Toolkit).

### Typical Workflow

1. **Log in** to the UI at `/enterprise-ai/ui`.
2. **Create a dataset** — upload PDFs / text / JSONL, or import Langfuse traces (see next section).
3. **Prepare / generate Q&A** — the Data Prep service uses Docling + LlamaIndex to chunk documents and (optionally) synthesise Q&A pairs via an inference model on the cluster.
4. **Review and export** the dataset to JSONL format.
5. **Create a fine-tuning job** — pick a base model (e.g. `meta-llama/Llama-3.1-8B`), a training dataset, hyperparameters (LoRA rank, epochs, LR, batch size).
6. **Submit** — the API forwards the job to the Nvidia backend over OAuth2.
7. **Monitor** — logs, loss curves, and status stream back to the UI.
8. **Download / deploy** — retrieve the adapter/model artifact from the backend, or push it to an inference endpoint on the cluster.

### Importing Data from Langfuse

The **Import from Langfuse** page reads traces one project at a time. A Langfuse
API key is itself project-scoped, so a project is added by giving the service its
key pair — the project's id and name are then read back from Langfuse.

By default the page uses `langfuse_public_key` / `langfuse_secret_key` from the
vault, which is all a single-project deployment needs. To offer more projects in
the page's **Project** dropdown, add their key pairs to the vault as
`publicKey:secretKey`, comma-separated:

```yaml
# core/inventory/metadata/vault.yml
langfuse_project_keys: "pk-lf-aaa:sk-lf-aaa,pk-lf-bbb:sk-lf-bbb"
```

The default pair is always listed first and is preselected on the page. Malformed
entries are logged and skipped rather than failing the page. Projects belonging to
different organizations are grouped by organization in the dropdown; with a single
organization the list stays flat.

Within the selected project the page can also keep only traces a reviewer scored
in Langfuse:

- **Annotation score** — a score name, listed with how many traces carry it.
  Names come from the project's score configs plus any score recorded ad-hoc.
- **Score value** — a threshold (`≥ 4`) for numeric scores, or the category the
  annotator picked for categorical/boolean ones.
- **Score source** — `ANNOTATION` (a person, in the Langfuse UI), `API` or `EVAL`.
- **Annotation queue** — traces queued for review, optionally narrowed to items
  still `PENDING` or already `COMPLETED`.

Leave them empty to import regardless of review. Scores are looked up without the
time window, since an annotation is written later than the trace it judges; the
window still applies to the traces themselves. Combining filters intersects them,
so `model` + score + queue keeps only traces satisfying all three. Two settings
bound the lookups:

```yaml
langfuse_max_scores_per_scan: 10000       # LANGFUSE_MAX_SCORES_PER_SCAN
langfuse_max_queue_items_per_scan: 10000  # LANGFUSE_MAX_QUEUE_ITEMS_PER_SCAN
```

---

## Configuration Reference

### 1. Plugin toggle (`core/inventory/inference-config.cfg`)

```properties
deploy_finetune_plugin=on
```

### 2. Nvidia backend & Keycloak (`blueprints/finetuning_service/finetune-config.cfg`)

See [Step 3](#step-3-configure-the-fine-tuning-service) above.

### 3. Advanced settings (`blueprints/finetuning_service/vars/finetune-plugin-vars.yml`)

Customisable:
- Resource requests/limits (CPU, Memory)
- Replica counts
- Storage sizes (PVCs for Postgres, MinIO)
- Image repositories and tags
- Base URL paths under the ingress

---

## Manual Deployment (without inference-stack-deploy.sh)

Run from the `core/` directory:

```bash
ansible-playbook -i inventory/hosts.yml \
  ../blueprints/finetuning_service/playbooks/deploy-all.yml \
  --vault-password-file inventory/.vault-passfile
```

Or deploy individual components:

```bash
# Data Preparation Service
ansible-playbook -i inventory/hosts.yml \
  ../blueprints/finetuning_service/playbooks/deploy-dataprep.yml \
  --vault-password-file inventory/.vault-passfile

# Fine-Tuning API
ansible-playbook -i inventory/hosts.yml \
  ../blueprints/finetuning_service/playbooks/deploy-finetuning-api.yml \
  --vault-password-file inventory/.vault-passfile

# Fine-Tuning UI
ansible-playbook -i inventory/hosts.yml \
  ../blueprints/finetuning_service/playbooks/deploy-ui.yml \
  --vault-password-file inventory/.vault-passfile
```

---

## Operations

### Deployment status

```bash
kubectl get pods -n dataprep
kubectl get pods -n finetuning-api
kubectl get pods -n finetuning-ui
```

### Tail logs

```bash
kubectl logs -n finetuning-api deploy/finetuning-api -f
kubectl logs -n dataprep deploy/dataprep -f
kubectl logs -n finetuning-ui deploy/finetuning-ui -f
```

### Updating configuration

```bash
vi core/inventory/inference-config.cfg
# or
vi blueprints/finetuning_service/finetune-config.cfg

cd core && ./inference-stack-deploy.sh    # choose option 3 (Update)
```

---

## Troubleshooting

### Pod not starting

```bash
kubectl describe pod -n <namespace> <pod-name>
kubectl logs -n <namespace> <pod-name>
```

### Keycloak authentication issues

Re-run the Keycloak setup script:

```bash
bash blueprints/finetuning_service/scripts/setup-keycloak-finetuning.sh
```

Verify both clients exist in the Keycloak admin console: `finetuning-backend` (confidential) and `finetuning-ui` (public).

### Nvidia backend connectivity

From inside the API pod:

```bash
kubectl exec -n finetuning-api deploy/finetuning-api -- \
  curl -k -v $NVIDIA_FINETUNE_BACKEND_URL/health
```

Common issues: wrong client secret, expired token URL, self-signed cert without `nvidia_keycloak_verify_ssl: false` in dev.

### Database connection issues

```bash
kubectl get pods -n dataprep | grep postgres
kubectl get pods -n finetuning-api | grep postgres
```

---

## Uninstalling

```bash
kubectl delete namespace dataprep
kubectl delete namespace finetuning-api
kubectl delete namespace finetuning-ui
```

Or disable in config and redeploy:

```properties
deploy_finetune_plugin=off
```
````