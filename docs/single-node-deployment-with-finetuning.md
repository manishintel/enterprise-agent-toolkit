# Single-Node Deployment with the Fine-Tuning Service

Complete from-scratch runbook for deploying this fork
(`manishintel/enterprise-agent-toolkit`) on one Ubuntu node, behind a corporate
proxy, with the **fine-tuning service blueprint** enabled alongside the base stack.

Everything here was verified on a working deployment — the endpoint responses,
timings, namespace names and error messages are observed, not inferred. Where this
fork's own docs are wrong, the correct behaviour is documented and the discrepancy
called out.

**Read first if you are following the other docs:** `single-node-deployment.md` does
not cover fine-tuning, `prerequisites.md` overstates the manual setup (see
[What you do not need to do](#what-you-do-not-need-to-do)), and
`blueprints/finetuning_service/README.md` is written against a different repo layout
(see [Corrections to the blueprint README](#corrections-to-the-blueprint-readme)).

## Table of Contents

- [What you get](#what-you-get)
- [Architecture and request path](#architecture-and-request-path)
- [Prerequisites](#prerequisites)
- [Step 0 — Clone](#step-0--clone)
- [Step 1 — Fine-tuning engine on the GPU machine](#step-1--fine-tuning-engine-on-the-gpu-machine)
- [Step 2 — DNS and TLS](#step-2--dns-and-tls)
- [Step 3 — Configure the base stack](#step-3--configure-the-base-stack)
- [Step 4 — Configure the fine-tuning service](#step-4--configure-the-fine-tuning-service)
- [Step 5 — Proxy configuration](#step-5--proxy-configuration)
- [Step 6 — Deploy](#step-6--deploy)
- [Step 7 — Verify](#step-7--verify)
- [Using the fine-tuning service](#using-the-fine-tuning-service)
- [Re-runs, resume and teardown](#re-runs-resume-and-teardown)
- [Known issues in this fork](#known-issues-in-this-fork)
- [Corrections to the blueprint README](#corrections-to-the-blueprint-readme)
- [Security notes](#security-notes)
- [Troubleshooting](#troubleshooting)

---

## What you get

A complete deployment with `deploy_finetune_plugin=on` produces these namespaces:

| Namespace | Contents |
|---|---|
| `kube-system` | K8s control plane, Calico, CoreDNS, in-cluster registry, dashboard, NRI balloons |
| `local-path-storage` | `local-path` PVC provisioner |
| `ingress-nginx` | Ingress controller, hostPort 80/443 |
| `genai-gateway` | LiteLLM + Postgres + Redis; Langfuse (`-trace-*`: web, worker, ClickHouse, ZooKeeper×3, S3) |
| `observability` | kube-prometheus-stack, Grafana, Loki, Alertmanager, OTel |
| `default` | vLLM model serving (`vllm-qwen-2-5-coder-14b-cpu`) |
| `redis` | Redis Stack (shared agent memory) |
| `pgvector` | PostgreSQL 16 + pgvector |
| `agent-sandbox-system` | Sandbox controller, router, `python-pool` warm pods |
| **`auth-apisix`** | APISIX gateway + ingress controller + `ApisixRoute` CRDs |
| **`dataprep`** | Data Prep backend, Celery worker, MinIO, Postgres, Redis |
| **`finetuning-api`** | Fine-Tuning API + Postgres |
| **`finetuning-ui`** | Fine-Tuning web UI |

The four bold namespaces are the fine-tuning blueprint. Roughly 60 pods total.

## Architecture and request path

The fine-tuning **training workload does not run on this cluster.** The cluster hosts
the API, Data Prep and UI; training runs on a separate GPU machine reached over
OAuth2. That machine must be up before you deploy.

```
 ┌──────────────────────────────────────────────────────────┐    ┌──────────────────────┐
 │  Single Ubuntu node — Kubernetes, CPU only               │    │  GPU machine         │
 │                                                          │    │                      │
 │  ingress-nginx  (hostPort 80/443)                        │    │  Unsloth fine-tuning │
 │    ├── api.example.com/            → LiteLLM             │    │  engine  :8000       │
 │    ├── api.example.com/v1          → vLLM via LiteLLM    │    │                      │
 │    ├── api.example.com/observability → Grafana           │    │  Keycloak     :8080  │
 │    ├── trace-api.example.com       → Langfuse            │    │                      │
 │    └── api.example.com/enterprise-ai ──┐                 │    └──────────┬───────────┘
 │                                        ▼                 │               │ OAuth2
 │                       auth-apisix-gateway:80             │               │ client
 │                       (APISIX, ApisixRoute matching)     │               │ credentials
 │                          ├── /enterprise-ai/ui      → finetuning-ui      │
 │                          ├── /enterprise-ai/api     → finetuning-api ────┘
 │                          ├── /enterprise-ai/v1/fine_tuning → finetuning-api
 │                          ├── /enterprise-ai/v1/models      → finetuning-api
 │                          └── /enterprise-ai/v1/files, /dataprep → data-prep-backend
 │                                                          │
 │  MinIO (datasets) · Celery (async prep) · Postgres ×3 · Redis ×2
 └──────────────────────────────────────────────────────────┘
```

**APISIX sits behind ingress-nginx, not beside it.** Only ingress-nginx binds
hostPort 80/443. A single nginx `Ingress` object in `auth-apisix` forwards one prefix
to the APISIX gateway, which then does `ApisixRoute` matching:

```
kubectl get ingress finetuning-service-apisix-ingress -n auth-apisix \
  -o jsonpath='{range .spec.rules[*].http.paths[*]}{.path}{" -> "}{.backend.service.name}{"\n"}{end}'
# /enterprise-ai -> auth-apisix-gateway
```

So there is no port conflict, and `/enterprise-ai/*` is the only prefix APISIX ever
sees. `auth-apisix-gateway` is a `NodePort` service, but you do not need the node
port — traffic arrives through ingress-nginx on 443.

## Prerequisites

### Hardware

| Resource | Minimum | Notes |
|---|---|---|
| CPU | 48 cores | Fine-tuning adds 4 image builds, 3 Postgres, 2 Redis, MinIO, APISIX+etcd |
| RAM | 32 GB | 64 GB+ strongly recommended with fine-tuning; vLLM on CPU is the main consumer |
| Disk | 150 GB | Fine-tuning PVCs alone request ~72 GB (MinIO 50Gi + 2×10Gi + 2Gi) |
| OS | Ubuntu 22.04 / 24.04 | x86_64 |

The reference deployment for this guide used 256 threads / 251 GB RAM / 1.7 TB free,
which is comfortable. At the 48-core minimum expect the vLLM pod to be slow rather
than to fail.

### Access

- Non-root user with `sudo`. **Do not run the deploy script as root** — it refuses.
- Hugging Face token with read access to gated models.
- Internet egress, proxy is fine, to: `github.com`, `registry-1.docker.io`,
  `huggingface.co`, `registry.k8s.io`, `charts.apiseven.com`.
- Network path from this node to the GPU machine's engine and Keycloak ports.

Verify egress before starting — a proxy that blocks one of these fails the deploy
midway:

```bash
for u in https://github.com https://registry-1.docker.io/v2/ https://huggingface.co \
         https://registry.k8s.io https://charts.apiseven.com; do
  printf "%-40s %s\n" "$u" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$u")"
done
# 200 / 401 / 200 / 307 / 301 are all fine — 401 on Docker Hub is its auth challenge
```

### Tooling

Only `git`, `curl`, `python3` ≥ 3.10 and `envsubst` (`gettext-base`) need to exist
up front. `envsubst` is worth checking explicitly because the BuildKit deploy scripts
use it and its absence surfaces as a confusing template failure:

```bash
command -v git curl python3 envsubst
```

`helm`, `kubectl`, `ansible`, `sshpass`, `jq`, `unzip`, `conntrack`, `socat` and
`ebtables` are installed for you — by `install_prereqs` in the deploy script, or by
the `inference-tools` Ansible role, which fetches Helm via
`https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3`.

### Cluster addons the blueprint depends on

Both are already enabled in `core/inventory/metadata/addons.yml`. **Do not disable
them** — fine-tuning breaks without either:

- `registry_enabled: true` — the blueprint builds four images with BuildKit in-cluster
  and pushes them to `registry.kube-system.svc.cluster.local:5000`.
- `local_path_provisioner_enabled: true` — every fine-tuning PVC uses storage class
  `local-path`.

### What you do *not* need to do

`prerequisites.md` walks through generating SSH keys and hand-editing
`core/inventory/hosts.yaml`. **Skip both for single node** — the deploy script does it:

- `setup_ssh()` generates `~/.ssh/id_ed25519` if absent, appends it to
  `~/.ssh/authorized_keys`, and keyscans localhost.
- `write_hosts_yaml()` overwrites `core/inventory/hosts.yaml` with a one-host
  inventory using `ansible_connection: local` — no SSH transport is used at all. It
  leaves the file alone only if it already contains more than one `ansible_host:`.
- `generate_certs()` creates a self-signed cert if you have not supplied one.
- `write_config()` creates `agentic-config.cfg` from defaults if it does not exist.
- `core/scripts/generate-vault-secrets.sh` runs automatically and generates every
  password, including the four fine-tuning ones.

## Step 0 — Clone

```bash
git clone https://github.com/manishintel/enterprise-agent-toolkit.git
cd enterprise-agent-toolkit
```

All paths below are relative to this directory.

## Step 1 — Fine-tuning engine on the GPU machine

Bring up the Unsloth engine and its Keycloak on the GPU machine **first**, then
collect four values:

| Value | Example |
|---|---|
| Backend URL | `http://gpu-host:8000` (no trailing slash) |
| Keycloak token URL | `http://gpu-host:8080/realms/finetuning/protocol/openid-connect/token` |
| Client ID | `finetuning-backend` |
| Client secret | from that Keycloak's admin console |

> `src/finetuning-engine/` is **not present in this fork**, so the engine cannot be
> deployed from this repo. Obtain it separately. The blueprint README references it
> and a `scripts/setup-keycloak-finetuning.sh`; neither exists here.

After Step 4 fills these into `finetune-config.cfg`, validate with the bundled
checker — it unsets the proxy, requests a token, and makes an authenticated call:

```bash
bash blueprints/finetuning_service/check-finetune-engine.sh
```

```
API health:  OK (200)
Token:       OK
Auth call:   OK (200)
STATUS: Fine-tuning engine is reachable and authenticating successfully.
```

**Do not deploy until all three lines pass.** The fine-tuning plugin runs before
vLLM, Redis, pgvector and agent-sandbox and `exit 1`s on failure — see
[ordering risk](#ordering-risk).

## Step 2 — DNS and TLS

Pick a `cluster_url` (this guide uses `api.example.com`) and map it plus the `trace-`
subdomain. Use the **node's own IP**, not `127.0.0.1` — ingress-nginx uses `hostPort`,
which binds every interface, so the node IP works identically and also makes the
stack reachable from other machines:

```bash
hostname -I                                        # get the node IP
echo "<NODE_IP>  api.example.com  trace-api.example.com" | sudo tee -a /etc/hosts
getent hosts api.example.com trace-api.example.com # both must resolve
```

The fine-tuning service needs **no additional host entries** — it is path-based under
`https://api.example.com/enterprise-ai/`, not a subdomain.

> If you already have an entry for `cluster_url`, `generate_certs()` leaves
> `/etc/hosts` alone — its append is `grep`-guarded, so it will not replace your node
> IP with `127.0.0.1`.

**Certificates.** If `~/certs/cert.pem` and `~/certs/key.pem` are absent, the deploy
script generates a self-signed pair with SANs for `api.example.com`,
`trace-api.example.com` and `*.api.example.com`. To supply your own, place them at
those paths or point `cert_file`/`key_file` at them in Step 3.

Note that `run_deployment` passes `--cert-file "${CERT_DIR}/cert.pem"` where
`CERT_DIR="${HOME}/certs"`, so `$HOME/certs` is what actually gets used. Keeping
`cert_file` in the config aligned with it avoids confusion.

> Use **absolute** paths for `cert_file`/`key_file`. Several playbooks read them with
> `lookup('file', cert_file)`, and Ansible's file lookup does not expand `~`.

## Step 3 — Configure the base stack

Edit `core/inventory/agentic-config.cfg`:

```ini
cluster_url=api.example.com
cert_file=/home/<user>/certs/cert.pem
key_file=/home/<user>/certs/key.pem
hugging_face_token=hf_xxxxxxxxxxxxxxxxxxxx
models=cpu-qwen2-5-coder-14b
deploy_kubernetes_fresh=on
deploy_ingress_controller=on
deploy_genai_gateway=on
deploy_observability=on
deploy_llm_models=on
deploy_agenticai_plugin=off
deploy_redis=on
deploy_pgvector=on
deploy_kuberay=off
deploy_agent_sandbox=on
deploy_finetune_plugin=on
http_proxy=http://your.proxy:912
https_proxy=http://your.proxy:912
no_proxy=<see Step 5>
```

### The fine-tuning toggle is `deploy_finetune_plugin`

**The single easiest thing to get wrong.** The key is exactly
`deploy_finetune_plugin=on`, read at
`core/lib/cluster/deployment/fresh-install.sh:144` and
`core/lib/system/setup-env.sh:184`.

`read_config_file()` turns *every* key it finds into a shell variable without
validating names, so a misspelling — `deploy_finetuning`, `deploy_finetune` — is
accepted in silence, fine-tuning is skipped, and the only trace is
`Skipping Fine-Tuning Plugin deployment...` in `deploy.log`. There is no warning and
no non-zero exit.

### Model selection

Set `models=` to one value (or a comma-separated list). CPU options:

| Value | Model |
|---|---|
| `cpu-qwen3-coder-30b` | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| `cpu-qwen2-5-coder-14b` | Qwen/Qwen2.5-Coder-14B-Instruct *(default)* |
| `cpu-bge-base-en` | BAAI/bge-base-en-v1.5 (embedding) |
| `cpu-bge-reranker-base` | BAAI/bge-reranker-base (reranking) |
| `cpu-qwen3-30b-a3b` | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| `cpu-gemma4-26b-a4b` | google/gemma-4-26B-A4B-it |
| `cpu-llama-8b` | meta-llama/Llama-3.1-8B-Instruct |
| `cpu-llama-8b-specdec` | Llama-3.1-8B with speculative decoding |
| `cpu-qwen3-coder-30b-specdec` | Qwen3-Coder-30B with speculative decoding |

### Other config behaviour worth knowing

- `on`/`off` are translated to `yes`/`no` internally. Write `on`/`off`.
- `write_config()` **skips entirely** if the file already exists — it logs
  `agentic-config.cfg already exists — skipping config write`. Your values are never
  clobbered. (The comment above the function claims it adds missing keys; it does not.)
- The deploy script **rewrites this file** on re-runs. See
  [Re-runs, resume and teardown](#re-runs-resume-and-teardown).
- **This file is tracked by git.** It will hold your HF token. See
  [Security notes](#security-notes).

### Vault secrets — automatic

No manual step. `setup_initial_env` calls `core/scripts/generate-vault-secrets.sh`,
which writes `core/inventory/metadata/vault.yml` (gitignored, mode `0640`) with every
password, including the four the blueprint needs:
`finetune_api_postgres_password`, `finetune_api_redis_password`,
`dataprep_postgres_password`, `dataprep_redis_password`.

> **Check this only if `vault.yml` already exists** — e.g. from a run predating the
> blueprint. The `mandatory_keys` list at `core/lib/system/setup-env.sh:125` does
> **not** include the four fine-tuning keys, so an existing vault is treated as valid
> without them and is never regenerated; `deploy-all.yml` then fails its password
> validation.
> ```bash
> for k in finetune_api_postgres_password finetune_api_redis_password \
>          dataprep_postgres_password dataprep_redis_password; do
>   grep -q "^${k}:" core/inventory/metadata/vault.yml && echo "OK   $k" || echo "MISS $k"
> done
> ```
> If any is missing, append it by hand as `<key>: "<random>"`. Regenerating the whole
> file rotates every other secret and breaks already-deployed components.

## Step 4 — Configure the fine-tuning service

Create `blueprints/finetuning_service/finetune-config.cfg`. **This file does not exist
in a fresh clone**, and the deploy hard-fails without it —
`blueprints/finetuning_service/playbooks/deploy-all.yml:30` loads it via
`include_vars` in `pre_tasks`, before any other task runs.

Despite the `.cfg` extension it is parsed as **YAML**:

```yaml
---
nvidia_finetune_backend_url: "http://<gpu-host>:8000"
nvidia_keycloak_token_url: "http://<gpu-host>:8080/realms/finetuning/protocol/openid-connect/token"
nvidia_keycloak_client_id: "finetuning-backend"
nvidia_keycloak_client_secret: "<client-secret>"
nvidia_keycloak_verify_ssl: "false"
```

- The first four are **mandatory**; `deploy-all.yml` fails the run if any is empty.
- **Quote every value.** A bare `key:` parses as YAML `null`, and the playbook's
  `finetune_config[key] | default('') | length == 0` check then raises a Jinja
  `NoneType has no len()` error instead of its intended message.
- `nvidia_keycloak_verify_ssl: "false"` for self-signed certs on the GPU machine.
  Irrelevant for plain `http://`, but harmless.
- Gitignored via `.gitignore:26`. Confirm:
  `git check-ignore -v blueprints/finetuning_service/finetune-config.cfg`

Now run the Step 1 checker.

## Step 5 — Proxy configuration

Set all three fields in `agentic-config.cfg`; do not rely on shell environment
variables. `read_config_file()` writes them into `core/inventory/metadata/all.yml`
both at top level and under `env_proxy:`.

> Blank proxy fields are **not** neutral. They are `sed`-ed into `all.yml`
> unconditionally, so leaving them empty *erases* any proxy config already there.

`no_proxy` is auto-augmented at deploy time by `_build_k8s_no_proxy()` with the
loopback set, apiserver ClusterIP, service and pod CIDRs, `.svc` and
`.svc.cluster.local`, link-local, the node IP, a fixed list of namespace short-forms
(`.default,.genai-gateway,.redis,.ingress-nginx,.agent-sandbox,.flowise`), and
`cluster_url`. You must add three things it misses:

1. **Your internal domains**, including the GPU machine's — the Fine-Tuning API pod
   must reach the engine directly.
2. **Your cluster domain with a leading dot.** Only `cluster_url` itself is
   auto-added, so `trace-api.example.com` would otherwise go to the corporate proxy,
   which cannot route to a private IP.
3. **The fine-tuning namespace short-forms**, absent from the fixed list.

A working value:

```ini
no_proxy=localhost,127.0.0.1,<NODE_IP>,<your.internal.domain>,.<your.internal.domain>,.example.com,example.com,.kube-system,.dataprep,.finetuning-api,.finetuning-ui,.auth-apisix,.pgvector,.local-path-storage,.observability
```

Verify the result bypasses correctly:

```bash
no_proxy="$(grep '^no_proxy=' core/inventory/agentic-config.cfg | cut -d= -f2-)" python3 - <<'EOF'
from urllib.request import proxy_bypass_environment as pb
for h in ["api.example.com","trace-api.example.com","<gpu-host>",
          "registry.kube-system.svc.cluster.local","github.com"]:
    print(f"{h:44} {'DIRECT' if pb(h) else 'via proxy'}")
EOF
# everything but github.com must read DIRECT
```

How the proxy propagates:

| Consumer | Gets proxy? |
|---|---|
| BuildKit image-build jobs | Yes — via `buildkit_proxy_args` from `env_proxy`; required for PyPI/npm |
| containerd | Yes — systemd drop-in written by `setup_kernel_and_containerd` |
| Ansible tasks | Yes — playbook-level `environment: env_proxy` |
| Fine-Tuning API / Dataprep / UI pods | **No** — no proxy env is injected. Correct for an internal engine; a public backend would need it added to the chart |

If apt ends up proxied and you need to undo it, remove the proxy lines from
`/etc/apt/apt.conf` or `/etc/apt/apt.conf.d/`.

## Step 6 — Deploy

```bash
chmod +x deploy-agentic-stack.sh
./deploy-agentic-stack.sh
```

One `yes/no` confirmation prompt, then unattended. Everything is teed to `deploy.log`
in the repo root.

Useful flags (`--help` for the full list):

| Flag | Effect |
|---|---|
| *(none)* | Deploy per `agentic-config.cfg` — what you want |
| `--menu` | Interactive menu: **1** fresh install · **2** reset cluster · **3** update cluster |
| `--docker` | Docker Compose path instead of Kubernetes — unrelated to this guide |

### Timing

Observed on the reference machine, about **65 minutes** end to end:

| Phase | Time |
|---|---|
| Prereqs, kubespray clone, Kubernetes bootstrap | ~18 min |
| ingress-nginx, GenAI Gateway, Langfuse | ~5 min |
| Observability (Prometheus/Grafana/Loki) | ~3 min |
| **Fine-tuning: 4 BuildKit image builds** | ~6.5 min (celery worker alone ~4 min) |
| **Fine-tuning: APISIX** | ~2 min |
| **Fine-tuning: dataprep, API, UI** | ~2 min |
| vLLM, Redis, pgvector, agent-sandbox | ~2 min + model weight pull |
| vLLM pulling model weights | a further 10–15 min in the background |

### Ordering risk

`fresh_installation` runs components in this order:

```
kubernetes → cluster-config → ingress-nginx → genai-gateway → observability
  → agenticai plugin → FINE-TUNING → istio → llm models → redis → kuberay
  → pgvector → agent-sandbox
```

Fine-tuning is 7th and calls `exit 1` on failure — **before** vLLM, Redis, pgvector
and agent-sandbox. A fine-tuning failure therefore leaves you with no model serving
at all. This is why the Step 1 check matters. Re-running after a fix is safe and
resumes cleanly.

To bring the base stack up first and add fine-tuning as a second pass, set
`deploy_finetune_plugin=off`, deploy, then flip it to `on` and re-run.

## Step 7 — Verify

```bash
kubectl get pods -A                       # ~60 pods; all Running or Completed
kubectl get pods -n dataprep -n finetuning-api -n finetuning-ui -n auth-apisix
kubectl get pvc -A | grep -E 'dataprep|finetun'
kubectl get apisixroute -A
helm list -A
```

`Completed` BuildKit pods (`buildkit-*`) are finished build jobs, not failures.

### Endpoint check

Every one of these was verified returning the code shown:

```bash
for u in / /ui /v1/models /observability/login \
         /enterprise-ai/ui /enterprise-ai/api/docs /enterprise-ai/api/health \
         /enterprise-ai/dataprep/docs /enterprise-ai/dataprep/health \
         /enterprise-ai/v1/fine_tuning/jobs /enterprise-ai/v1/files; do
  printf "%-42s %s\n" "$u" \
    "$(curl -sk -o /dev/null -w '%{http_code}' --noproxy '*' https://api.example.com$u)"
done
curl -sk -o /dev/null -w 'langfuse %{http_code}\n' --noproxy '*' https://trace-api.example.com/
```

| Path | Expected | Notes |
|---|---|---|
| `/` | 200 | LiteLLM |
| `/ui` | 307 | redirect into the LiteLLM dashboard |
| `/v1/models` | **401** | correct — needs the master key |
| `/observability/login` | 200 | Grafana |
| `/enterprise-ai/ui` | 200 | Fine-Tuning UI |
| `/enterprise-ai/api/docs` | 200 | Fine-Tuning API Swagger |
| `/enterprise-ai/api/health` | 200 | unauthenticated health route |
| `/enterprise-ai/dataprep/docs` | 200 | Data Prep Swagger |
| `/enterprise-ai/dataprep/health` | 200 | |
| `/enterprise-ai/v1/fine_tuning/jobs` | 200 | **proves the OAuth2 hop to the GPU engine works** |
| `/enterprise-ai/v1/files` | 200 | Data Prep file API |
| `trace-api.example.com/` | 200 | Langfuse |

`--noproxy '*'` matters when a corporate proxy is set — without it curl tries to
reach your private IP through the proxy.

### Model API

```bash
export LITELLM_MASTER_KEY=$(kubectl get deploy -n genai-gateway genai-gateway-deployment \
  -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="LITELLM_MASTER_KEY")].value}')

curl -sk --noproxy '*' https://api.example.com/v1/models \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}"
# {"data":[{"id":"Qwen/Qwen2.5-Coder-14B-Instruct",...}]}

curl -sk --noproxy '*' https://api.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -d '{"model":"Qwen/Qwen2.5-Coder-14B-Instruct",
       "messages":[{"role":"user","content":"Write a Python hello world"}],
       "max_tokens":100}'
```

`print_summary` prints model names with an `openai/` prefix. Use the bare id returned
by `/v1/models`; both are accepted by LiteLLM.

### Fine-tuning → engine link

The decisive check that the cluster and GPU machine are talking:

```bash
curl -sk --noproxy '*' https://api.example.com/enterprise-ai/v1/fine_tuning/jobs
# {"object":"list","data":[],"has_more":false}      <- empty list = authenticated OK

curl -sk --noproxy '*' https://api.example.com/enterprise-ai/v1/models
# the base models the engine offers, e.g. meta-llama/Llama-3.2-3B-Instruct
```

An empty `data` array is success — it means the token was accepted and there are no
jobs yet. A 5xx here points at the client secret or token URL.

### Access URLs

| Service | URL |
|---|---|
| Fine-Tuning UI | `https://api.example.com/enterprise-ai/ui` |
| Fine-Tuning API docs | `https://api.example.com/enterprise-ai/api/docs` |
| Data Prep API docs | `https://api.example.com/enterprise-ai/dataprep/docs` |
| LiteLLM dashboard | `https://api.example.com/ui` |
| Model API | `https://api.example.com/v1` |
| Grafana | `https://api.example.com/observability/login` |
| Langfuse | `https://trace-api.example.com` |

In-cluster endpoints:

```
http://genai-gateway-service.genai-gateway.svc.cluster.local:4000
http://vllm-qwen-2-5-coder-14b-cpu-service.default/v1
redis://default:<pw>@redis-stack-server.redis.svc.cluster.local:6379
postgresql://agentuser:<pw>@pgvector.pgvector.svc.cluster.local:5432/agentdb
http://sandbox-router-svc.agent-sandbox-system.svc.cluster.local:8080
registry.kube-system.svc.cluster.local:5000
```

## Using the fine-tuning service

Open `https://api.example.com/enterprise-ai/ui`.

**Login is username-only — there is no password.** The UI uses NextAuth's
`CredentialsProvider` with a single `username` field; any string matching
`^[a-z0-9._-]{1,64}$` is accepted, and the bearer token handed to the API is simply
`base64(username)`. It is a tenancy label, not authentication. See
[Security notes](#security-notes).

The blueprint README's claim that login goes through the cluster Keycloak does not
match this fork — `apisix.oidc.enabled` is `false` in
`src/api/helm-charts/finetuning-api/values.yaml`, so the API routes carry a CORS
plugin only.

Typical flow, all reachable from the UI or directly via the documented APIs:

1. **Upload a dataset** — `POST /enterprise-ai/v1/files`, stored in MinIO.
2. **Prepare it** — `/enterprise-ai/v1/dataprep`, processed asynchronously by the
   Celery worker.
3. **Start a job** — `POST /enterprise-ai/v1/fine_tuning/jobs`; the API forwards it to
   the GPU engine over OAuth2 and polls status.
4. **Deploy the result** — the UI's *Deploy Model* button renders the packaged vLLM
   chart from the `vllm-chart` ConfigMap in `default`, with an init container that
   pulls the adapter from MinIO. Registration with the GenAI Gateway is the last
   step: the chart's `-litellm-register-r<revision>` Job polls the model's own
   `/v1/models` endpoint through its Service and only adds the model to LiteLLM
   once it answers, so the gateway never lists a model that is still loading.
   The UI reports that step as *Registering with Gateway*.

Both Swagger UIs are live; use them as the API reference.

The *Deploy Model* button depends on a ConfigMap published during deployment:

```bash
kubectl get cm vllm-chart -n default
# must exist — created by the "Publish vllm Chart to the Inference Namespace" task
```

If it is missing, that task failed — see the `pipefail` entry in
[Troubleshooting](#troubleshooting).

## Re-runs, resume and teardown

**Re-running `./deploy-agentic-stack.sh` resumes; it does not start over.**
`deploy_kubernetes_fresh=on` does not force a rebuild.

`_auto_skip_deployed_components()` at `deploy-agentic-stack.sh:677` queries the live
cluster before deploying and **rewrites `agentic-config.cfg` with `sed`**, flipping
already-deployed components to `off`. Your `on` values are inputs to that check, not
imperatives:

| Key | Auto-skip check |
|---|---|
| `deploy_kubernetes_fresh` | a node is `Ready` and cluster nodes ≥ `ansible_host:` count in `hosts.yaml` |
| `deploy_ingress_controller` | `ingress-nginx` namespace exists |
| `deploy_genai_gateway` | `genai-gateway` namespace exists |
| `deploy_observability` | `observability` namespace exists |
| `deploy_llm_models` | the *specific* model's helm release exists |
| `deploy_redis` | `redis` namespace exists |
| `deploy_agent_sandbox` | `agent-sandbox` namespace exists — **never true**, see [Known issues](#known-issues-in-this-fork) |
| `deploy_pgvector` | **no check** — always re-runs |
| `deploy_finetune_plugin` | **no check** — always re-runs in full |

Consequences:

- Expect those keys to read `off` in your config file afterwards. That is the resume
  mechanism, not corruption. To force a genuine fresh install, set them back to `on`
  yourself.
- Fine-tuning restarts from the top of `deploy-all.yml` every time. Everything is
  idempotent (`helm upgrade --install`), but the four BuildKit builds have no
  existence guard and rebuild — budget ~6.5 min per re-run.
- `prepare_repo` keeps the kubespray clone if it matches the pinned
  `kubespray_version` in `core/inventory/metadata/agentic-metadata.cfg`, else
  re-clones.
- `generate_certs` reuses existing certs and leaves `/etc/hosts` alone.

**Teardown.** `./deploy-agentic-stack.sh --menu` option **2** (`reset_cluster`) is the
only path that destroys the cluster; see `decommission-and-redeploy.md`. To drop just
the fine-tuning stack:

```bash
kubectl delete namespace dataprep finetuning-api finetuning-ui auth-apisix
```

## Known issues in this fork

**`set -euo pipefail` under `/bin/sh`.** The *Publish vllm Chart to the Inference
Namespace* task in `blueprints/finetuning_service/playbooks/deploy-finetuning-api.yml`
uses `set -euo pipefail`, but Ansible's `shell` module defaults to `/bin/sh` — `dash`
on Ubuntu, which has no `pipefail`. The task dies on its first line with
`/bin/sh: 1: set: Illegal option -o pipefail`, taking the whole deploy with it. Fix
by adding the repo's standard block to that task:

```yaml
  args:
    executable: /bin/bash
```

This is applied in this fork. It is the only such task in the blueprint — the other
26 `shell:` tasks avoid bashisms — but it is the pattern to reach for if you add one.

**`agent-sandbox` namespace name mismatch.** The chart deploys into
`agent-sandbox-system`, but `_auto_skip_deployed_components()` and `print_summary`
both check for `agent-sandbox`. Effects: agent-sandbox is re-deployed on every run
(harmless, idempotent), the summary always reports it as `not deployed`, and the
router URL the summary prints (`...agent-sandbox.svc.cluster.local`) is wrong — the
real one is `sandbox-router-svc.agent-sandbox-system.svc.cluster.local:8080`.

**`print_summary` misnames the pgvector toggle.** When pgvector is absent it advises
`enable deploy_agenticai_plugin=on`. The correct key is `deploy_pgvector=on`.

**`print_summary` never prints the ingress address.** It computes `_ingress_addr`,
including a correct hostPort fallback to the node IP, then never echoes it. Harmless
dead code — but do not go looking for an ingress line in the summary.

**Missing from this fork.** `blueprints/finetuning_service/src/finetuning-engine/`
and `scripts/setup-keycloak-finetuning.sh` are both referenced by the blueprint README
and absent.

## Corrections to the blueprint README

`blueprints/finetuning_service/README.md` targets upstream Enterprise Inference and
is wrong for this fork in these places:

| README says | This fork |
|---|---|
| `deploy_finetune_plugin=on` goes in `core/inventory/inference-config.cfg` | Right key, wrong file — it goes in `core/inventory/agentic-config.cfg`; `inference-config.cfg` does not exist |
| Deploy via `cd core && ./inference-stack-deploy.sh` | `./deploy-agentic-stack.sh` at repo root; `inference-stack-deploy.sh` does not exist |
| Menu options 1 / 3 (Fresh / Update) | `./deploy-agentic-stack.sh --menu`, options 1 / 2 / 3 |
| Run `core/scripts/generate-vault-secrets.sh` manually | Automatic in `setup_initial_env` |
| Engine at `src/finetuning-engine/` | Absent |
| `scripts/setup-keycloak-finetuning.sh` for Keycloak repair | Absent |
| Login uses cluster Keycloak | Local NextAuth, username-only, no password |
| Backend on HTTPS `:8443` | Whatever the engine exposes; plain `http://host:8000` works |

## Security notes

This stack is built for an internal, trusted network. Before exposing it more widely,
understand these four things.

**The Fine-Tuning UI and API are unauthenticated.** Login takes a username and no
password; the bearer token is `base64(username)`. `apisix.oidc.enabled: false` means
the APISIX routes apply CORS only — no `openid-connect` plugin. Anyone who can reach
`https://<cluster_url>/enterprise-ai/` can list, create and delete fine-tuning jobs
and read every uploaded dataset. The OIDC plumbing exists in the chart and can be
switched on, but it is off by default.

**`core/inventory/agentic-config.cfg` is tracked by git and not gitignored**, and it
holds your Hugging Face token. Of the three credential-bearing files, only it is
exposed:

| File | Holds | Git status |
|---|---|---|
| `blueprints/finetuning_service/finetune-config.cfg` | Keycloak client secret | Gitignored (`.gitignore:26`) ✅ |
| `core/inventory/metadata/vault.yml` | All generated passwords | Gitignored (`.gitignore:8`) ✅ |
| `core/inventory/agentic-config.cfg` | **Hugging Face token** | **Tracked** ⚠️ |

Keep it out of the index before any commit:

```bash
git update-index --skip-worktree core/inventory/agentic-config.cfg
git status --short core/inventory/     # agentic-config.cfg must not appear
```

**`core/inventory/.vault-passfile` is committed upstream** — the Ansible Vault
password is identical in every clone of this repo.

**`vault.yml` is plain YAML, not vault-encrypted.** Its `0640` mode is the only thing
protecting it; the vault password is irrelevant. Treat it as a plaintext secret file.

## Troubleshooting

**Fine-tuning silently skipped.** `grep -n "Fine-Tuning" deploy.log`. If it says
`Skipping Fine-Tuning Plugin deployment...`, the toggle key is misspelled — it must
be exactly `deploy_finetune_plugin=on`.

**`Could not find or access '.../finetune-config.cfg'`.** Step 4 was not done.

**`NoneType has no len()` in the validation loop.** An unquoted empty key in
`finetune-config.cfg`. Quote all values.

**`ERROR: <name> is not set!`** A blank mandatory value in `finetune-config.cfg`.

**`<name> password not set! Run: core/scripts/generate-vault-secrets.sh`.** The
fine-tuning keys are missing from a pre-existing `vault.yml` — see the note in Step 3.

**`/bin/sh: 1: set: Illegal option -o pipefail`** in *Publish vllm Chart to the
Inference Namespace*. Add `args: {executable: /bin/bash}` to that task — see
[Known issues](#known-issues-in-this-fork).

**Job fails with `backend_error: Failed to download file-...` or `Upload failed`.**
The engine is handed only file IDs, so it uses this cluster's FILES API for both
legs: it pulls the training file from
`https://<cluster_url>/enterprise-ai/v1/files/<id>/content` and POSTs the trained
model back to `.../v1/files` (that POST is the only thing that mints a
`file-<uuid>`, and the vllm init container then reads it straight out of MinIO at
`<user_id>/<file_id>`). Neither leg is about training. Three distinct causes:

- **The engine is pointed at a stale ingress address.** This is the trap: there
  is no DNS record for `cluster_url` and no LoadBalancer VIP
  (`kubectl -n ingress-nginx get svc` shows `EXTERNAL-IP <pending>`), so the
  entry point is just the node's own address and it *moves on every rebuild*
  — `10.165.117.73` → `.210` (Aug 27) → `.174` (Sep 2). The engine resolves the
  name itself from its `FILES_API_RESOLVE` setting, which has to be hand-updated
  each time. **Hand the engine host the new address whenever the cluster is
  rebuilt, not just the new cert.**

  A stale address and a stale cert raise the *identical*
  `CERTIFICATE_VERIFY_FAILED: self-signed certificate`, because the old host is
  usually still up and answers for an unknown SNI with ingress-nginx's built-in
  cert. Tell them apart by the subject, not the error:

  ```bash
  openssl s_client -connect <ip>:443 -servername <cluster_url> </dev/null 2>/dev/null \
    | openssl x509 -noout -subject -dates -fingerprint -sha256
  ```

  `CN = Kubernetes Ingress Controller Fake Certificate` (and a 146-byte
  `<center>nginx</center>` 404 on any path) means **wrong address** — that host
  has no Ingress for this hostname. `CN = <cluster_url>` with an unexpected
  fingerprint means wrong cert.
- **The engine does not trust the cert.** `cert_file` is self-signed, so every
  regeneration breaks it. Install it on the engine host by whatever mechanism its
  client actually reads — for the Unsloth engine that is its own
  `FILES_API_TLS_VERIFY` pem path (`certs/dataprep-ingress.pem`), **not**
  `REQUESTS_CA_BUNDLE` and not the system CA store: it passes `verify=` to
  `requests` explicitly and forces `trust_env=False` for `FILES_API_RESOLVE`
  hosts so the site proxy cannot intercept this leg, which makes both env-based
  mechanisms inert. Confirm the knob before recommending one.
- **`401`** — with `oidc_enabled: true` the files route requires a browser
  session, which the engine has no way to obtain (the plugin does no bearer
  validation: it has neither `introspection_endpoint` nor `public_key`). Set
  `finetune_engine_client_ips` in `finetune-config.cfg`; it opens two
  higher-priority routes, `GET .../content` and `POST /v1/files`, matched on the
  `X-Real-IP` that ingress-nginx stamps from the real peer address. To cover a
  whole node pool, change that expr's `op` to `RegexMatch` with e.g.
  `^10\.14\.219\.` in
  `src/dataprep/helmcharts/data-prep-backend/templates/apisixroute.yaml` — `In`
  compares exact strings and does not understand CIDRs.

Identity on those routes travels in `Authorization: Bearer <base64-username>`
(rule 3 of `core/handlers/auth_handler.get_current_user_id`), which the
fine-tuning API forwards to the engine as `ft-api-key`. A **404** rather than 401
therefore means the route matched but the identity did not resolve to the file's
owner — that is the expected answer to an unauthenticated probe, and a useful
signal that the exemption is live.

**Both halves of this are worth removing permanently**, since a rebuild
currently requires a manual cert copy *and* a manual address edit on a machine
this repo does not manage: give `cluster_url` a real DNS A record (which retires
`FILES_API_RESOLVE` entirely) and issue the ingress cert from a stable internal
CA so the engine can pin the **CA** rather than a leaf that is reissued on every
rebuild.

**Engine unreachable from a pod.**

```bash
kubectl exec -n finetuning-api deploy/finetuning-service -- sh -c \
  'echo "$NVIDIA_API_URL"; curl -sk -o /dev/null -w "%{http_code}\n" --max-time 15 $NVIDIA_API_URL/docs'
# prints the engine URL then 200
kubectl logs -n finetuning-api deploy/finetuning-service --tail=50
```

Note the env var is `NVIDIA_API_URL`, not the config key name. The mapping from
`finetune-config.cfg` into the pod is:

| Config key | Pod env var |
|---|---|
| `nvidia_finetune_backend_url` | `NVIDIA_API_URL` |
| `nvidia_keycloak_token_url` | `NVIDIA_KEYCLOAK_TOKEN_URL` |
| `nvidia_keycloak_client_id` | `NVIDIA_KEYCLOAK_CLIENT_ID` |
| `nvidia_keycloak_client_secret` | `NVIDIA_KEYCLOAK_CLIENT_SECRET` |
| `nvidia_keycloak_verify_ssl` | `NVIDIA_KEYCLOAK_VERIFY_SSL` |

There is also an `ENABLE_NVIDIA` flag on the deployment. Usual failure causes are a
wrong client secret or token URL. These pods get **no** proxy env, so a failure is
more likely DNS or the engine being down than proxy interference.

**Image builds fail.** The BuildKit jobs need both the proxy and the in-cluster
registry:

```bash
kubectl get jobs -A | grep buildkit
kubectl logs -n dataprep job/buildkit-data-prep-backend
kubectl logs -n finetuning-api job/buildkit-finetuning-service
kubectl get pods -n kube-system | grep registry
```

**Pods `Pending` on PVCs.**

```bash
kubectl get sc                          # expect: local-path
kubectl get pods -n local-path-storage
kubectl describe pvc -n dataprep
```

**`/enterprise-ai/*` returns 404.** Check the chain in order — nginx Ingress, then
APISIX, then the routes:

```bash
kubectl get ingress -n auth-apisix
kubectl get pods -n auth-apisix
kubectl get apisixroute -A
kubectl logs -n auth-apisix deploy/auth-apisix-ingress-controller --tail=50
```

**Everything returns a proxy error from the node itself.** Add `--noproxy '*'` to
curl, or confirm your `no_proxy` covers `cluster_url` *and* the `trace-` subdomain.

**vLLM stuck `ContainerCreating` or `0/1 Running`.** It is pulling model weights —
10–15 min is normal. `kubectl logs -n default deploy/vllm-qwen-2-5-coder-14b-cpu -f`.
A 401/403 there means a bad or unauthorized `hugging_face_token`.
