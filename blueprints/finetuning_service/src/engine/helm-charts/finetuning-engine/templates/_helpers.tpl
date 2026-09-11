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
Path the GPU cluster's kubeconfig is mounted at, or "" when no Secret is
configured. Empty tells the engine to fall back to in-cluster credentials, which
is only correct when the GPU nodes are in *this* cluster.
*/}}
{{- define "finetuning-engine.trainKubeconfigPath" -}}
{{- if .Values.trainCluster.kubeconfigSecret.name -}}
{{ printf "%s/%s" (trimSuffix "/" .Values.trainCluster.kubeconfigMountPath) .Values.trainCluster.kubeconfigSecret.key }}
{{- end -}}
{{- end }}
