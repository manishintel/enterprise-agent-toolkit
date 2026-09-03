#!/bin/bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -e
#sudo apt-get install gettext-base
# Get the absolute path of the ui directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export BUILD_CONTEXT
export REGISTRY_URL="${REGISTRY_URL:-registry.kube-system.svc.cluster.local:5000}"
NS="${NAMESPACE:-finetuning-ui}"

echo "Building with context: $BUILD_CONTEXT"
echo "Registry: $REGISTRY_URL"

export NEXT_PUBLIC_AUTH_URL="${NEXT_PUBLIC_AUTH_URL:-}"
export NEXT_PUBLIC_BASE_PATH="${NEXT_PUBLIC_BASE_PATH:-/enterprise-ai/ui}"
export NEXT_PUBLIC_FILES_BASE_URL="${NEXT_PUBLIC_FILES_BASE_URL:-}"
export NEXT_PUBLIC_DATAPREP_BASE_URL="${NEXT_PUBLIC_DATAPREP_BASE_URL:-}"
export NEXT_PUBLIC_FINETUNING_API_URL="${NEXT_PUBLIC_FINETUNING_API_URL:-}"
export NEXT_PUBLIC_DEPLOYMENT_API_URL="${NEXT_PUBLIC_DEPLOYMENT_API_URL:-}"
export NEXT_TELEMETRY_DISABLED="${NEXT_TELEMETRY_DISABLED:-1}"

# Delete the old job and wait for the name to free up. Deletion is asynchronous,
# so re-applying immediately races it: the apply is rejected because the object is
# being deleted, and everything after then reports on the *old* job -- a build
# that never ran looks exactly like a build that failed.
if kubectl get job buildkit-frontend -n "$NS" >/dev/null 2>&1; then
  echo "Deleting old buildkit job..."
  kubectl delete job buildkit-frontend -n "$NS" --wait=true
  for _ in $(seq 1 30); do
    kubectl get job buildkit-frontend -n "$NS" >/dev/null 2>&1 || break
    sleep 1
  done
fi

# Apply the job with substituted values using envsubst
echo "Creating BuildKit build job..."
# Pass proxy settings into the build if set in the environment
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export no_proxy="${no_proxy:-}"
envsubst < "$SCRIPT_DIR/buildkit-job.yaml" | kubectl apply -f -

echo "BuildKit job created successfully"
echo "Monitor with: kubectl logs -f job/buildkit-frontend -n ${NAMESPACE:-finetuning-ui}"

# Wait for the pod to be *running* before attaching. Attaching while it is still
# ContainerCreating fails with a BadRequest, and the old loop treated the pod
# merely existing as its cue -- so the build output, type errors included, was
# never shown and a failure had to be diagnosed from the pod log afterwards.
echo "Waiting for BuildKit pod to start..."
POD_NAME=""
for _ in $(seq 1 60); do
  POD_NAME=$(kubectl get pods -n "$NS" -l job-name=buildkit-frontend -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  if [ -n "$POD_NAME" ]; then
    PHASE=$(kubectl get pod "$POD_NAME" -n "$NS" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [ "$PHASE" != "Pending" ] && [ -n "$PHASE" ]; then
      echo "Pod started: $POD_NAME ($PHASE)"
      kubectl logs -f "$POD_NAME" -n "$NS" || true
      break
    fi
  fi
  sleep 5
done

# Poll for either outcome rather than waiting on one of them. `kubectl wait
# --for=condition=complete` sits out its whole timeout when the job has already
# failed, which turned a build that failed in under a minute into a ten-minute
# wait before the fallback ran.
echo "Waiting for job to complete..."
JOB_STATUS=""
JOB_FAILED=""
for _ in $(seq 1 120); do
  JOB_STATUS=$(kubectl get job buildkit-frontend -n "$NS" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
  JOB_FAILED=$(kubectl get job buildkit-frontend -n "$NS" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
  if [ "$JOB_STATUS" = "True" ] || [ "$JOB_FAILED" = "True" ]; then
    break
  fi
  sleep 5
done

if [ "$JOB_STATUS" = "True" ]; then
  echo "✓ UI image build completed successfully!"
elif [ "$JOB_FAILED" = "True" ]; then
  echo "✗ UI image build failed!"
  exit 1
else
  echo "Warning: Could not determine job completion status"
fi