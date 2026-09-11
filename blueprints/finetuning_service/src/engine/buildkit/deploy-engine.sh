#!/bin/bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e

# Get the absolute path of the finetuning-engine root directory
# buildkit is 1 level deep from root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT="$(cd "$SCRIPT_DIR/.." && pwd)"
export BUILD_CONTEXT
export REGISTRY_URL="${REGISTRY_URL:-registry.kube-system.svc.cluster.local:5000}"
export IMAGE_TAG="${IMAGE_TAG:-latest}"
export NAMESPACE="${NAMESPACE:-finetuning}"

echo "Building with context: $BUILD_CONTEXT"
echo "Registry: $REGISTRY_URL"
echo "Image Tag: $IMAGE_TAG"
echo "Namespace: $NAMESPACE"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
else
    echo "Namespace $NAMESPACE already exists"
fi

# Delete old job if exists
if kubectl get job buildkit-finetuning-engine -n "$NAMESPACE" >/dev/null 2>&1; then
  echo "Deleting old buildkit job..."
  kubectl delete job buildkit-finetuning-engine -n "$NAMESPACE" --wait=true
  # Deletion is asynchronous. Re-applying while it is in flight is rejected, and
  # everything after then reports on the *old* job -- a build that never started
  # looks exactly like one that failed.
  for _ in $(seq 1 30); do
    kubectl get job buildkit-finetuning-engine -n "$NAMESPACE" >/dev/null 2>&1 || break
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

echo ""
echo "BuildKit job created successfully"
echo "Monitor with: kubectl logs -f job/buildkit-finetuning-engine -n $NAMESPACE"
echo ""

# Wait for job to start
echo "Waiting for pod to start..."
sleep 3

# Show logs
POD_NAME=""
for _ in {1..60}; do
  POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l job-name=buildkit-finetuning-engine -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  if [ -n "$POD_NAME" ]; then
    # Attaching while it is still ContainerCreating fails with a BadRequest and
    # the build output -- including whatever made it fail -- is never shown.
    PHASE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [ -n "$PHASE" ] && [ "$PHASE" != "Pending" ]; then
      break
    fi
  fi
  sleep 2
done

if [ -n "$POD_NAME" ]; then
  echo "Following build logs..."
  echo "=========================================="
  kubectl logs -f "$POD_NAME" -n "$NAMESPACE" || true
  echo "=========================================="
  
  # Wait for job to complete and check status
  echo ""
  echo "Waiting for job to complete..."
  # Poll rather than wait on one condition: `--for=condition=complete` sits out its
  # entire timeout when the job has already failed, turning a build that failed in
  # under a minute into a ten-minute wait.
  for _ in $(seq 1 120); do
    _done=$(kubectl get job buildkit-finetuning-engine -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
    _failed=$(kubectl get job buildkit-finetuning-engine -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    if [ "$_done" = "True" ] || [ "$_failed" = "True" ]; then
      break
    fi
    sleep 5
  done
  
  # Check final status
  JOB_STATUS=$(kubectl get job buildkit-finetuning-engine -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
  JOB_FAILED=$(kubectl get job buildkit-finetuning-engine -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
  
  if [ "$JOB_STATUS" == "True" ]; then
    echo ""
    echo "✓ Build completed successfully!"
    echo "Image: $REGISTRY_URL/finetuning-engine:$IMAGE_TAG"
    echo ""
    echo "Note: Layer caching is enabled. Subsequent builds will be faster."
    exit 0
  elif [ "$JOB_FAILED" == "True" ]; then
    echo ""
    echo "✗ Build failed!"
    exit 1
  else
    echo ""
    echo "✗ Build status unknown"
    exit 1
  fi
else
  echo "Warning: Could not find pod for buildkit job"
  exit 1
fi
