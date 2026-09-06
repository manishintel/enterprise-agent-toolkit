'use client';

/**
 * Deployment state for one fine-tuned model: where it is in coming up, how to
 * call it once it is serving, and the Helm command for anyone who would rather
 * run it themselves.
 *
 * Lifted off the fine-tuning job page. Training and serving are different
 * questions asked at different times -- "did my job converge" and "is my model
 * answering requests" -- and putting both on one page meant neither had a
 * readable shape. Logs are deliberately *not* here; they are a tab of their own,
 * because a 14-row log box in the middle of a status summary pushes everything
 * that matters below the fold.
 */

import React, { useState } from 'react';
import { Alert, Button, Progress, Space, Tag, Typography, Input } from 'antd';
import {
  CheckCircleFilled,
  ClockCircleOutlined,
  CloseCircleFilled,
  CodeOutlined,
  CopyOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import type {
  DeploymentCapacity,
  DeploymentPhase,
  DeploymentStepStatus,
  FineTuningJob,
  ModelDeploymentStatus,
} from '@features/finetuning/types';
import { getFineTunedModelName } from '@features/finetuning/utils';

const { Text } = Typography;

export const DEPLOYMENT_PHASE_TEXT: Record<DeploymentPhase, string> = {
  not_deployed: 'Not Deployed',
  installing: 'Installing',
  downloading: 'Downloading Model',
  extracting: 'Unpacking Model',
  loading: 'Loading Model',
  registering: 'Registering with Gateway',
  ready: 'Serving',
  failed: 'Failed',
  uninstalling: 'Removing',
  unavailable: 'Unavailable',
};

export const DEPLOYMENT_PHASE_COLOR: Record<DeploymentPhase, string> = {
  not_deployed: 'default',
  installing: 'processing',
  downloading: 'processing',
  extracting: 'processing',
  loading: 'processing',
  registering: 'processing',
  ready: 'success',
  failed: 'error',
  uninstalling: 'warning',
  unavailable: 'default',
};

const DEPLOYMENT_STEP_ICON: Record<DeploymentStepStatus, React.ReactNode> = {
  pending: <ClockCircleOutlined style={{ color: '#bfbfbf' }} />,
  active: <LoadingOutlined style={{ color: '#1890ff' }} />,
  done: <CheckCircleFilled style={{ color: '#52c41a' }} />,
  error: <CloseCircleFilled style={{ color: '#ff4d4f' }} />,
};

export interface DeploymentPanelProps {
  job: FineTuningJob;
  deployment?: ModelDeploymentStatus;
  capacity?: DeploymentCapacity;
  resultFileId?: string;
  /** Told to the capacity query, which only runs while the Helm block is open. */
  onHelmVisibilityChange?: (visible: boolean) => void;
}

export default function DeploymentPanel({
  job,
  deployment,
  capacity,
  resultFileId,
  onHelmVisibilityChange,
}: DeploymentPanelProps) {
  const [showHelmCommand, setShowHelmCommand] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const phase = (deployment?.phase || 'not_deployed') as DeploymentPhase;

  // Release names must be RFC 1123 labels, so derive one from the job id. The API
  // derives the same name, so prefer whatever it reports.
  const releaseName =
    deployment?.release_name ||
    `ft-${job.id}`
      .toLowerCase()
      .replace(/[^a-z0-9-]/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 53);

  const servedModelName = deployment?.served_model_name || getFineTunedModelName(job);

  // cpu/memory are shown because the chart emits resource requests and limits only
  // when they are set: run this without them and the model is scheduled onto
  // whatever node the scheduler picks with nothing reserved for it.
  const suggestedCpu = capacity?.recommended?.cpu;
  const suggestedMemory = capacity?.recommended?.memory;
  const helmCommand = `helm install ${releaseName} vllm/ \\
  -f vllm/xeon-values.yaml \\
  --set finetune.enabled=true \\
  --set finetune.fileId=${resultFileId} \\
  --set SERVED_MODEL_NAME=${servedModelName} \\
  --set litellmRegister.enabled=true \\
  --set pvc.enabled=true \\
  --set cpu=${suggestedCpu ?? '<cores>'} \\
  --set memory=${suggestedMemory ?? '<size>Gi'} \\
  --set tensor_parallel_size=1 \\
  --set pipeline_parallel_size=1`;

  const toggleHelm = () => {
    const next = !showHelmCommand;
    setShowHelmCommand(next);
    onHelmVisibilityChange?.(next);
  };

  return (
    <Space orientation="vertical" style={{ width: '100%' }} size="middle">
      <div>
        <Space size="middle" style={{ marginBottom: 8 }}>
          <Tag
            color={DEPLOYMENT_PHASE_COLOR[phase] || 'default'}
            style={{ fontSize: '14px', padding: '4px 8px' }}
          >
            {DEPLOYMENT_PHASE_TEXT[phase] || phase}
          </Tag>
          <Text type="secondary">{deployment?.message}</Text>
        </Space>
        {phase !== 'not_deployed' && phase !== 'unavailable' && (
          <Progress
            percent={deployment?.progress ?? 0}
            status={phase === 'failed' ? 'exception' : phase === 'ready' ? 'success' : 'active'}
            showInfo
          />
        )}
      </div>

      {deployment?.error && (
        <Alert title="Deployment Failed" description={deployment.error} type="error" showIcon />
      )}

      {phase === 'unavailable' && (
        <Alert
          title="One-Click Deployment Not Available"
          description="This environment is not set up to deploy models from the UI. Use the Helm command below instead."
          type="warning"
          showIcon
        />
      )}

      {phase === 'not_deployed' && (
        <Alert
          title="Ready to Deploy"
          description={`Deploying starts vLLM on your fine-tuned model and, once it answers requests, registers it with the GenAI Gateway as ${servedModelName}. The first start takes several minutes while the model is downloaded from object storage and loaded; registration is the last step, so the gateway only ever lists a model that is ready to use.`}
          type="info"
          showIcon
        />
      )}

      {!!deployment?.steps?.length && (
        <div>
          {deployment.steps.map((step) => (
            <div
              key={step.key}
              style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '4px 0' }}
            >
              <span style={{ lineHeight: '22px' }}>
                {DEPLOYMENT_STEP_ICON[step.status] || DEPLOYMENT_STEP_ICON.pending}
              </span>
              <div>
                <Text
                  type={step.status === 'pending' ? 'secondary' : undefined}
                  strong={step.status === 'active'}
                >
                  {step.title}
                </Text>
                {step.detail && (
                  <div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {step.detail}
                    </Text>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {phase === 'ready' && (
        <Alert
          title="Model Is Serving"
          description={
            <Space orientation="vertical" size="small">
              <Text>
                Call it as <Text code copyable>{servedModelName}</Text>
                {deployment?.gateway_registered
                  ? ' through the GenAI Gateway.'
                  : '. Gateway registration could not be confirmed.'}
              </Text>
              {deployment?.service_url && (
                <Text type="secondary">
                  In-cluster endpoint: <Text code copyable>{deployment.service_url}</Text>
                </Text>
              )}
            </Space>
          }
          type="success"
          showIcon
        />
      )}

      <div>
        <Button
          type="link"
          size="small"
          icon={<CodeOutlined />}
          style={{ paddingLeft: 0 }}
          onClick={toggleHelm}
        >
          {showHelmCommand ? 'Hide Helm command' : 'Deploy manually with Helm instead'}
        </Button>

        {showHelmCommand && (
          <Space orientation="vertical" style={{ width: '100%', marginTop: 12 }} size="middle">
            <Alert
              title="Deploy with Helm"
              description="The Deploy button runs exactly this. Run it yourself from the core/helm-charts directory of the deployment repo on a control-plane node. The chart pulls your fine-tuned model out of object storage and starts vLLM on it."
              type="info"
              showIcon
            />

            <div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8,
                }}
              >
                <Text strong>Helm Install Command:</Text>
                <Button
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={async () => {
                    await navigator.clipboard.writeText(helmCommand);
                    setIsCopied(true);
                    setTimeout(() => setIsCopied(false), 2000);
                  }}
                >
                  {isCopied ? 'Copied!' : 'Copy'}
                </Button>
              </div>
              <Input.TextArea
                value={helmCommand}
                readOnly
                autoSize={{ minRows: 8, maxRows: 12 }}
                style={{ fontFamily: 'monospace', fontSize: '13px', backgroundColor: '#f5f5f5' }}
              />
            </div>

            <Alert
              title="After deployment"
              description={
                <Space orientation="vertical" size="small">
                  <Text>
                    Model file ID: <Text code copyable>{resultFileId}</Text>
                  </Text>
                  <Text>
                    Once the pod is ready, the model is served as{' '}
                    <Text code copyable>{servedModelName}</Text>.
                  </Text>
                  <Text type="secondary">
                    First start is slow: the model is downloaded from object storage and unpacked
                    onto a persistent volume. Follow progress with{' '}
                    <Text code>kubectl logs -f deploy/{releaseName}-vllm -c fetch-finetuned-model</Text>
                    .
                  </Text>
                  <Text type="secondary">Customize the release name and other parameters as needed.</Text>
                </Space>
              }
              type="success"
              showIcon
            />
          </Space>
        )}
      </div>
    </Space>
  );
}
