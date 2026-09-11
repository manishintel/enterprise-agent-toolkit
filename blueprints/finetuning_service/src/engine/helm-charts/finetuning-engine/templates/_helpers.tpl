{{/*
Expand the name of the chart.
*/}}
{{- define "finetuning-engine.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "finetuning-engine.fullname" -}}
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
{{- define "finetuning-engine.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "finetuning-engine.labels" -}}
helm.sh/chart: {{ include "finetuning-engine.chart" . }}
{{ include "finetuning-engine.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "finetuning-engine.selectorLabels" -}}
app.kubernetes.io/name: {{ include "finetuning-engine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Name of the ServiceAccount the engine runs as.
*/}}
{{- define "finetuning-engine.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "finetuning-engine.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Claim holding the training log store. An operator-supplied existingClaim wins, so
a redeployment can be pointed at a volume that already has history on it — a
chart-created PVC is deleted with the release and takes the logs with it.
*/}}
{{- define "finetuning-engine.logStoreClaimName" -}}
{{- default (printf "%s-logs" (include "finetuning-engine.fullname" .)) .Values.logStore.persistence.existingClaim }}
{{- end }}

{{/*
Whether the log store is backed by a volume at all. Both flags have to be on:
enabled without persistence would write into the container filesystem, where the
logs die with the pod and the whole point is lost.
*/}}
{{- define "finetuning-engine.logStoreMounted" -}}
{{- if and .Values.logStore.enabled .Values.logStore.persistence.enabled -}}
true
{{- end -}}
{{- end }}

{{/*
Path the GPU cluster's kubeconfig is mounted at, or "" when no Secret is
configured. Empty tells the engine to fall back to in-cluster credentials, which
is only correct when the GPU nodes are in *this* cluster.
*/}}
{{- define "finetuning-engine.trainKubeconfigPath" -}}
{{- if .Values.trainCluster.kubeconfigSecret.name -}}
{{ printf "%s/%s" (trimSuffix "/" .Values.trainCluster.kubeconfigMountPath) .Values.trainCluster.kubeconfigSecret.key }}
{{- end -}}
{{- end }}
