'use client';

import React, { useState } from 'react';
import {
  Card,
  Typography,
  Button,
  Space,
  Tag,
  Progress,
  Alert,
  Spin,
  App,
  Descriptions,
  Timeline,
  Row,
  Col,
  Statistic,
  Input,
} from 'antd';
import {
  ArrowLeftOutlined,
  StopOutlined,
  ReloadOutlined,
  ClockCircleOutlined,
  FileTextOutlined,
  RobotOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CopyOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  LoadingOutlined,
  CodeOutlined,
} from '@ant-design/icons';
import { useRouter, useParams } from 'next/navigation';
import {
  useFineTuningJob,
  useJobEvents,
  useCancelFineTuningJob,
  useModelDeployment,
  useDeployModel,
  useUndeployModel,
  isDeploymentInProgress,
} from '@features/finetuning';
import type { DeploymentPhase, DeploymentStepStatus } from '@features/finetuning/types';
import { FileNameDisplay } from '@/app/files/components';
import {
  getFineTuningStatusColor,
  formatCreatedAt,
  formatDuration,
  canCancelFineTuningJob,
  getFineTuningStatusText,
  getFineTuningProgress,
  getFineTunedModelName,
  getJobQueueSeconds,
  getJobTrainingSeconds,
  getJobTotalSeconds,
} from '@features/finetuning/utils';

const { Title, Text } = Typography;

const DEPLOYMENT_PHASE_TEXT: Record<DeploymentPhase, string> = {
  not_deployed: 'Not Deployed',
  installing: 'Installing',
  downloading: 'Downloading Model',
  extracting: 'Unpacking Model',
  loading: 'Loading Model',
  ready: 'Serving',
  failed: 'Failed',
  uninstalling: 'Removing',
  unavailable: 'Unavailable',
};

const DEPLOYMENT_PHASE_COLOR: Record<DeploymentPhase, string> = {
  not_deployed: 'default',
  installing: 'processing',
  downloading: 'processing',
  extracting: 'processing',
  loading: 'processing',
  ready: 'success',
  failed: 'error',
  uninstalling: 'warning',
  unavailable: 'default',
};

// Event payloads are engine metrics, not something a reader should have to parse
// out of a JSON dump, so each known key gets a label and a formatter. Unknown
// keys fall through to their raw value rather than being hidden.
const EVENT_DATA_LABELS: Record<string, string> = {
  progress_percent: 'Progress',
  current_step: 'Step',
  total_steps: 'Total steps',
  training_loss: 'Loss',
  current_phase: 'Phase',
  elapsed_seconds: 'Engine time',
  queued_seconds: 'Queued for',
  output_file_id: 'Result file',
  model: 'Base model',
};

const formatEventDataValue = (key: string, value: unknown): string => {
  if (key === 'elapsed_seconds' || key === 'queued_seconds') {
    return formatDuration(Number(value));
  }
  if (key === 'progress_percent') {
    return `${Math.round(Number(value))}%`;
  }
  if (key === 'training_loss') {
    return Number(value).toFixed(4);
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
};

const DEPLOYMENT_STEP_ICON: Record<DeploymentStepStatus, React.ReactNode> = {
  pending: <ClockCircleOutlined style={{ color: '#bfbfbf' }} />,
  active: <LoadingOutlined style={{ color: '#1890ff' }} />,
  done: <CheckCircleFilled style={{ color: '#52c41a' }} />,
  error: <CloseCircleFilled style={{ color: '#ff4d4f' }} />,
};

const FineTuningJobDetailPage = () => {
  const router = useRouter();
  const params = useParams();
  const { modal } = App.useApp();
  const [isCopied, setIsCopied] = useState(false);
  // The Deploy button is the normal way in; the Helm command stays available for
  // anyone who wants to run it themselves.
  const [showHelmCommand, setShowHelmCommand] = useState(false);

  const jobId = params.id as string;

  const {
    data: jobData,
    isLoading: loading,
    error,
    refetch: refetchJob
  } = useFineTuningJob(jobId);

  const {
    data: jobEventsData,
    isLoading: eventsLoading,
    refetch: refetchEvents
  } = useJobEvents(jobId, { limit: 100 });

  const cancelJobMutation = useCancelFineTuningJob({
    onSuccess: () => {
      refetchJob();
    },
  });

  // Get the result file ID
  const resultFileId = jobData?.result_files?.[0];

  const jobEvents = jobEventsData?.data || [];

  const canBeDeployed = jobData?.status === 'succeeded' && !!resultFileId;

  const {
    data: deployment,
    isLoading: deploymentLoading,
    isFetching: deploymentFetching,
    error: deploymentError,
    refetch: refetchDeployment,
  } = useModelDeployment(jobId, { enabled: canBeDeployed });

  const deployModelMutation = useDeployModel();
  const undeployModelMutation = useUndeployModel();

  const handleDeploy = () => {
    if (!jobData) return;

    modal.confirm({
      title: 'Deploy Fine-Tuned Model',
      content:
        'This starts the model on the cluster and registers it with the GenAI Gateway. ' +
        'The first start takes several minutes while the model is downloaded and loaded.',
      okText: 'Deploy',
      cancelText: 'Cancel',
      onOk: () => {
        deployModelMutation.mutate(jobData.id);
      },
    });
  };

  const handleUndeploy = () => {
    if (!jobData) return;

    modal.confirm({
      title: 'Remove Deployment',
      content:
        'This stops the model and removes it from the GenAI Gateway. The fine-tuned model ' +
        'files are kept, so it can be deployed again later.',
      okText: 'Remove',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: () => {
        undeployModelMutation.mutate(jobData.id);
      },
    });
  };

  const handleCancelJob = async () => {
    if (!jobData) return;

    modal.confirm({
      title: 'Cancel Fine-Tuning Job',
      content: 'Are you sure you want to cancel this fine-tuning job? This action cannot be undone.',
      okText: 'Yes, Cancel',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          cancelJobMutation.mutate(jobData.id);
        } catch {
          // Error handled by mutation
        }
      },
    });
  };

  const handleRefresh = () => {
    refetchJob();
    refetchEvents();
    if (canBeDeployed) {
      refetchDeployment();
    }
  };

  const handleBack = () => {
    router.push('/finetuning');
  };

  const getEventIcon = (level: string) => {
    switch (level) {
      case 'warning': return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'error': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'debug': return <CheckCircleOutlined style={{ color: '#bfbfbf' }} />;
      default: return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const renderJobStatus = () => {
    if (!jobData) return null;

    // While training, the engine reports real progress; the status-derived
    // number is only a stand-in for the phases it does not measure.
    const progress =
      jobData.status === 'running' && jobData.progress_percent != null
        ? Math.round(jobData.progress_percent)
        : getFineTuningProgress(jobData.status);
    const statusText = getFineTuningStatusText(jobData.status);
    const statusColor = getFineTuningStatusColor(jobData.status);

    return (
      <Card title="Job Status" size="small">
        <Space orientation="vertical" style={{ width: '100%' }}>
          <div>
            <Tag color={statusColor} style={{ fontSize: '14px', padding: '4px 8px' }}>
              {statusText}
            </Tag>
          </div>
          <Progress
            percent={progress}
            status={jobData.status === 'failed' ? 'exception' : undefined}
            showInfo
          />
          {jobData.error && (
            <Alert
              title="Job Error"
              description={`${jobData.error.code}: ${jobData.error.message}`}
              type="error"
              showIcon
            />
          )}
        </Space>
      </Card>
    );
  };

  const renderJobMetrics = () => {
    if (!jobData) return null;

    const queueSeconds = getJobQueueSeconds(jobData);
    const trainingSeconds = getJobTrainingSeconds(jobData);
    const steps = jobData.total_steps
      ? `${jobData.current_step ?? 0} / ${jobData.total_steps}`
      : jobData.current_step
        ? `${jobData.current_step}`
        : '—';

    // The training engine reports no token counts, so a permanent "Trained
    // Tokens 0" said nothing about the job. These are the numbers it does
    // report, with training time kept apart from queue wait.
    const metrics: Array<{ title: string; value: string; hint?: string }> = [
      {
        title: 'Training Time',
        value: trainingSeconds === null ? '—' : formatDuration(trainingSeconds),
        hint: 'On a worker, queue wait excluded',
      },
      {
        title: 'Queue Wait',
        value: queueSeconds === null ? '—' : formatDuration(queueSeconds),
        hint: 'Submitted until training started',
      },
      { title: 'Steps', value: steps },
      {
        title: 'Training Loss',
        value: jobData.training_loss != null ? jobData.training_loss.toFixed(4) : '—',
      },
    ];

    if (jobData.trained_tokens) {
      metrics.push({
        title: 'Trained Tokens',
        value: jobData.trained_tokens.toLocaleString(),
      });
    }

    return (
      <Card title="Job Metrics" size="small">
        <Row gutter={[16, 16]}>
          {metrics.map((metric) => (
            <Col key={metric.title} xs={12} md={6}>
              <Statistic title={metric.title} value={metric.value} />
              {metric.hint && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {metric.hint}
                </Text>
              )}
            </Col>
          ))}
        </Row>
        {jobData.current_phase && (
          <div style={{ marginTop: 12 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {jobData.current_phase}
            </Text>
          </div>
        )}
      </Card>
    );
  };

  const renderTimeline = () => {
    if (!jobData) return null;

    const queueSeconds = getJobQueueSeconds(jobData);
    const trainingSeconds = getJobTrainingSeconds(jobData);
    const totalSeconds = getJobTotalSeconds(jobData);
    // Total wall-clock on its own reads as a seven-hour fine-tune when almost
    // all of it was spent waiting for a worker, so the split is spelled out.
    const breakdown = [
      queueSeconds ? `${formatDuration(queueSeconds)} queued` : null,
      trainingSeconds ? `${formatDuration(trainingSeconds)} training` : null,
    ].filter(Boolean);

    return (
      <Card title={<><ClockCircleOutlined /> Timeline</>} size="small">
        <Descriptions column={1} size="small">
          <Descriptions.Item label="Created">
            {formatCreatedAt(jobData.created_at)}
          </Descriptions.Item>
          <Descriptions.Item label="Training Started">
            {jobData.started_at ? (
              formatCreatedAt(jobData.started_at)
            ) : (
              <Text type="secondary">
                {queueSeconds !== null
                  ? `Waiting for a worker (${formatDuration(queueSeconds)} so far)`
                  : 'Not started'}
              </Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="Finished">
            {jobData.finished_at ? (
              formatCreatedAt(jobData.finished_at)
            ) : (
              <Text type="secondary">Not finished</Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="Total Elapsed">
            {totalSeconds === null ? (
              <Text type="secondary">—</Text>
            ) : (
              <>
                {formatDuration(totalSeconds)}
                {breakdown.length > 0 && (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {` (${breakdown.join(' + ')})`}
                  </Text>
                )}
              </>
            )}
          </Descriptions.Item>
        </Descriptions>
      </Card>
    );
  };

  const renderDeploymentStatus = () => {
    if (!jobData || !canBeDeployed) return null;

    // Release names must be RFC 1123 labels, so derive one from the job id. The
    // API derives the same name, so prefer whatever it reports.
    const releaseName =
      deployment?.release_name ||
      `ft-${jobData.id}`
        .toLowerCase()
        .replace(/[^a-z0-9-]/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '')
        .slice(0, 53);

    // The chart resolves the archive in MinIO from finetune.fileId, unpacks it
    // onto the model volume and points vLLM at it, so LLM_MODEL_ID is not set
    // here. SERVED_MODEL_NAME is the name to call the model by over the API and
    // the name it appears under in the GenAI Gateway.
    const servedModelName = deployment?.served_model_name || getFineTunedModelName(jobData);
    const helmCommand = `helm install ${releaseName} vllm/ \\
  -f vllm/xeon-values.yaml \\
  --set finetune.enabled=true \\
  --set finetune.fileId=${resultFileId} \\
  --set SERVED_MODEL_NAME=${servedModelName} \\
  --set litellmRegister.enabled=true \\
  --set pvc.enabled=true \\
  --set tensor_parallel_size=1 \\
  --set pipeline_parallel_size=1`;

    const phase = (deployment?.phase || 'not_deployed') as DeploymentPhase;
    const mutating = deployModelMutation.isPending || undeployModelMutation.isPending;

    return (
      <Card
        title={<><CloudUploadOutlined /> Model Deployment</>}
        size="small"
        extra={
          <Space>
            {deployment?.can_deploy && (
              <Button
                type="primary"
                icon={<CloudUploadOutlined />}
                onClick={handleDeploy}
                loading={deployModelMutation.isPending}
                disabled={mutating}
              >
                {phase === 'failed' ? 'Retry Deployment' : 'Deploy Model'}
              </Button>
            )}
            {deployment?.can_undeploy && (
              <Button
                danger
                icon={<DeleteOutlined />}
                onClick={handleUndeploy}
                loading={undeployModelMutation.isPending}
                disabled={mutating}
              >
                Remove Deployment
              </Button>
            )}
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => refetchDeployment()}
              loading={deploymentFetching}
            >
              Refresh
            </Button>
          </Space>
        }
      >
        <Space orientation="vertical" style={{ width: '100%' }} size="middle">
          {deploymentLoading ? (
            <div style={{ textAlign: 'center', padding: '20px' }}>
              <Spin />
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">Checking deployment status...</Text>
              </div>
            </div>
          ) : deploymentError ? (
            <Alert
              title="Could Not Read Deployment Status"
              description={deploymentError.message}
              type="warning"
              showIcon
            />
          ) : (
            <>
              <div>
                <Space size="middle" style={{ marginBottom: 8 }}>
                  <Tag color={DEPLOYMENT_PHASE_COLOR[phase] || 'default'} style={{ fontSize: '14px', padding: '4px 8px' }}>
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
                <Alert
                  title="Deployment Failed"
                  description={deployment.error}
                  type="error"
                  showIcon
                />
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
                  description={`Deploying starts vLLM on your fine-tuned model and registers it with the GenAI Gateway as ${servedModelName}. The first start takes several minutes while the model is downloaded from object storage and loaded.`}
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
                            <Text type="secondary" style={{ fontSize: '12px' }}>{step.detail}</Text>
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

              {!!deployment?.logs?.length && (
                <div>
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>Latest Output</Text>
                    {deployment.log_source && (
                      <Text type="secondary" style={{ fontSize: '12px', marginLeft: 8 }}>
                        {deployment.log_source}
                      </Text>
                    )}
                  </div>
                  <Input.TextArea
                    value={deployment.logs.join('\n')}
                    readOnly
                    autoSize={{ minRows: 4, maxRows: 14 }}
                    style={{
                      fontFamily: 'monospace',
                      fontSize: '12px',
                      backgroundColor: '#f5f5f5',
                    }}
                  />
                </div>
              )}
            </>
          )}

          <div>
            <Button
              type="link"
              size="small"
              icon={<CodeOutlined />}
              style={{ paddingLeft: 0 }}
              onClick={() => setShowHelmCommand(!showHelmCommand)}
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
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
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
                    style={{
                      fontFamily: 'monospace',
                      fontSize: '13px',
                      backgroundColor: '#f5f5f5'
                    }}
                  />
                </div>

                <Alert
                  title="After deployment"
                  description={
                    <Space orientation="vertical" size="small">
                      <Text>Model file ID: <Text code copyable>{resultFileId}</Text></Text>
                      <Text>Once the pod is ready, the model is served as <Text code copyable>{servedModelName}</Text>.</Text>
                      <Text type="secondary">
                        First start is slow: the model is downloaded from object storage and unpacked onto a
                        persistent volume. Follow progress with{' '}
                        <Text code>kubectl logs -f deploy/{releaseName}-vllm -c fetch-finetuned-model</Text>.
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
      </Card>
    );
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>Loading fine-tuning job details...</Text>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={handleBack}
          style={{ marginBottom: 16 }}
        >
          Back to Fine-Tuning Jobs
        </Button>
        <Alert
          title="Error Loading Job Details"
          description={error?.message || 'Failed to load job details'}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={handleRefresh}>
              Retry
            </Button>
          }
        />
      </div>
    );
  }

  if (!jobData) {
    return (
      <div>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={handleBack}
          style={{ marginBottom: 16 }}
        >
          Back to Fine-Tuning Jobs
        </Button>
        <Alert
          title="Job Not Found"
          description="The requested fine-tuning job could not be found."
          type="warning"
          showIcon
        />
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={handleBack}
          style={{ marginBottom: 16 }}
        >
          Back to Fine-Tuning Jobs
        </Button>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <Title level={2} style={{ marginBottom: 8 }}>
              Fine-Tuning Job Details
            </Title>
            <Text code style={{ fontSize: '16px' }}>{jobData.id}</Text>
          </div>

          <Space>
            {canCancelFineTuningJob(jobData) && (
              <Button
                icon={<StopOutlined />}
                danger
                onClick={handleCancelJob}
              >
                Cancel Job
              </Button>
            )}
            <Button
              icon={<ReloadOutlined />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Space>
        </div>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          {renderJobStatus()}
        </Col>
        <Col xs={24} lg={12}>
          {renderJobMetrics()}
        </Col>
      </Row>

      {/* Deployment Status Card */}
      {jobData?.status === 'succeeded' && resultFileId && (
        <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
          <Col xs={24}>
            {renderDeploymentStatus()}
          </Col>
        </Row>
      )}

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title={<><RobotOutlined /> Model Information</>} size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="Base Model">
                <Text strong>{jobData.model}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Fine-Tuned Model">
                {jobData.status === 'succeeded' ? (
                  <Text code copyable>{getFineTunedModelName(jobData)}</Text>
                ) : (
                  <Text type="secondary">Not yet available</Text>
                )}
              </Descriptions.Item>
              {jobData.fine_tuned_model && (
                <Descriptions.Item label="Model File ID">
                  <Text code>{jobData.fine_tuned_model}</Text>
                </Descriptions.Item>
              )}
              <Descriptions.Item label="Organization">
                <Text>{jobData.organization_id}</Text>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title={<><FileTextOutlined /> Training Data</>} size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="Training File">
                <FileNameDisplay fileId={jobData.training_file} />
              </Descriptions.Item>
              <Descriptions.Item label="Validation File">
                {jobData.validation_file ? (
                  <FileNameDisplay fileId={jobData.validation_file} />
                ) : (
                  <Text type="secondary">None specified</Text>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="Result Files">
                {jobData.result_files && jobData.result_files.length > 0 ? (
                  <div>
                    {jobData.result_files.map((file, index) => (
                      <div key={index}>
                        <FileNameDisplay fileId={file} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <Text type="secondary">No result files yet</Text>
                )}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title={<><SettingOutlined /> Hyperparameters</>} size="small">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="Epochs">
                {jobData.hyperparameters.n_epochs || 'Default (3)'}
              </Descriptions.Item>
              <Descriptions.Item label="Batch Size">
                {jobData.hyperparameters.batch_size || 'Default (4)'}
              </Descriptions.Item>
              <Descriptions.Item label="Learning Rate Multiplier">
                {jobData.hyperparameters.learning_rate_multiplier || 'Default (1.0)'}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          {renderTimeline()}
        </Col>
      </Row>

      {/* Job Events Timeline */}
      <Card
        title="Job Events"
        style={{ marginTop: 16 }}
        extra={
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => refetchEvents()}
            loading={eventsLoading}
          >
            Refresh Events
          </Button>
        }
      >
        {eventsLoading ? (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Spin />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">Loading events...</Text>
            </div>
          </div>
        ) : jobEvents.length > 0 ? (
          <Timeline
            items={jobEvents.map((event, index) => {
              // Only the metrics that were actually reported, one readable tag
              // each instead of a JSON blob.
              const details = Object.entries(event.data || {}).filter(
                ([, value]) => value !== null && value !== undefined && value !== ''
              );

              return {
                key: event.id || index,
                icon: getEventIcon(event.level),
                content: (
                  <div>
                    <Text strong style={{ color: event.level === 'error' ? '#ff4d4f' : undefined }}>
                      {event.message}
                    </Text>
                    <div>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {formatCreatedAt(event.created_at)}
                      </Text>
                    </div>
                    {details.length > 0 && (
                      <Space wrap size={[4, 4]} style={{ marginTop: 6 }}>
                        {details.map(([key, value]) => (
                          <Tag key={key} color={event.level === 'error' ? 'error' : undefined}>
                            {EVENT_DATA_LABELS[key] || key}: {formatEventDataValue(key, value)}
                          </Tag>
                        ))}
                      </Space>
                    )}
                  </div>
                ),
              };
            })}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Text type="secondary">
              {jobData.status === 'queued' || jobData.status === 'validating_files'
                ? 'No events yet — the job is still waiting to start.'
                : 'No events reported for this job.'}
            </Text>
          </div>
        )}
      </Card>
    </div>
  );
};

export default FineTuningJobDetailPage;