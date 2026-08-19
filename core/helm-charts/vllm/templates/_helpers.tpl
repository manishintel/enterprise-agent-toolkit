{{/*
Expand the name of the chart.
*/}}
{{- define "vllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "vllm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "vllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Convert chart name to a string suitable as metric prefix
*/}}
{{- define "vllm.metricPrefix" -}}
{{- include "vllm.fullname" . | replace "-" "_" | regexFind "[a-zA-Z_:][a-zA-Z0-9_:]*" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "vllm.labels" -}}
helm.sh/chart: {{ include "vllm.chart" . }}
{{ include "vllm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "vllm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vllm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Directory a fine-tuned model is published to on the model volume.
The fetch/extract init containers place a servable model here and vLLM is
pointed at it, so LLM_MODEL_ID does not have to be supplied for fine-tuned
deployments.
*/}}
{{- define "vllm.finetunedModelPath" -}}
{{- printf "%s/%s" (.Values.finetune.extractPath | trimSuffix "/") .Values.finetune.fileId -}}
{{- end }}

{{/*
Scratch directory used to download and unpack the fine-tuning archive.
Lives on the model volume so the rename into place stays on one filesystem.
*/}}
{{- define "vllm.finetuneStagePath" -}}
{{- printf "%s/.staging" (.Values.finetune.extractPath | trimSuffix "/") -}}
{{- end }}

{{/*
Value for vLLM's --model: a local directory for fine-tuned models, otherwise
the configured HuggingFace model id.
*/}}
{{- define "vllm.modelPath" -}}
{{- if .Values.finetune.enabled -}}
{{- include "vllm.finetunedModelPath" . -}}
{{- else -}}
{{- .Values.LLM_MODEL_ID -}}
{{- end -}}
{{- end }}

{{/*
Value for vLLM's --served-model-name, i.e. the name clients use over the
OpenAI API. Never a filesystem path: fine-tuned deployments fall back to the
result file id when SERVED_MODEL_NAME is not set.
*/}}
{{- define "vllm.servedModelName" -}}
{{- if .Values.SERVED_MODEL_NAME -}}
{{- .Values.SERVED_MODEL_NAME -}}
{{- else if .Values.finetune.enabled -}}
{{- .Values.finetune.fileId -}}
{{- else -}}
{{- .Values.LLM_MODEL_ID -}}
{{- end -}}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "vllm.serviceAccountName" -}}
{{- if .Values.global.sharedSAName }}
{{- .Values.global.sharedSAName }}
{{- else if .Values.serviceAccount.create }}
{{- default (include "vllm.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}