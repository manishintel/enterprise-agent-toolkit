'use client';

/**
 * One fine-tuning job: how the training run went.
 *
 * Restructured from a single flat column of eight cards. That layout showed a
 * running job and a month-old finished one identically, and gave the same weight
 * to the progress bar someone is watching and the hyperparameters they set once
 * and never looked at again. Three changes fix most of it.
 *
 * **The status strip is always visible.** Status, progress and elapsed time sit
 * above the tabs, because on a running job that is the only thing the visitor came
 * for and it should not move when they change tabs.
 *
 * **Everything else is grouped by when you need it.** Overview is what you read
 * while the job runs; Configuration is what you check when reproducing or
 * debugging it; Events is the log you open when something looks wrong.
 *
 * **Deployment left this page.** Serving a model is a separate job from training
 * one -- it happens later, is repeated, and is undone independently -- so it has
 * its own section, and this page hands off to it with a single banner. What used
 * to be ~300 lines of deployment card in the middle of the page is now one line.
 */

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
  Tabs,
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
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CloudServerOutlined,
  DashboardOutlined,
  ProfileOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { useRouter, useParams } from 'next/navigation';
import {
  useFineTuningJob,
  useJobEvents,
  useCancelFineTuningJob,
  useModelDeployment,
} from '@features/finetuning';
import { FileNameDisplay } from '@/app/files/components';
import { DEPLOYMENT_PHASE_TEXT } from '../components';
import type { DeploymentPhase } from '@features/finetuning/types';
import {
  getFineTuningStatusColor,
  formatCreatedAt,
  formatDuration,
  canCancelFineTuningJob,
  getFineTuningStatusText,
  resolveJobProgress,
  getFineTunedModelName,
  getJobQueueSeconds,
  getJobTrainingSeconds,
  getJobTotalSeconds,
} from '@features/finetuning/utils';

const { Title, Text } = Typography;

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
  if (key === 'current_phase') {
    // The API carries the engine's raw token (e.g. preparing_environment). Rows
    // written before it was normalised hold the engine's upper case, so lower it
    // here rather than trusting the stored casing.
    const spaced = String(value).toLowerCase().replace(/_/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
};

const FineTuningJobDetailPage = () => {
  const router = useRouter();
  const params = useParams();
  const { modal } = App.useApp();
  const [activeTab, setActiveTab] = useState('overview');

  const jobId = params.id as string;

  const {
    data: jobData,
    isLoading: loading,
    error,
    refetch: refetchJob,
  } = useFineTuningJob(jobId);

  const {
    data: jobEventsData,
    isLoading: eventsLoading,
    refetch: refetchEvents,
  } = useJobEvents(jobId, { limit: 100 });

  const cancelJobMutation = useCancelFineTuningJob({
    onSuccess: () => refetchJob(),
  });

  const resultFileId = jobData?.result_files?.[0];
  const jobEvents = jobEventsData?.data || [];
  const canBeDeployed = jobData?.status === 'succeeded' && !!resultFileId;

  // Read only to label the hand-off banner ("Serving" vs "Not deployed"). The
  // deployment itself is managed on its own page.
  const { data: deployment } = useModelDeployment(jobId, { enabled: canBeDeployed });

  const handleCancelJob = async () => {
    if (!jobData) return;

    modal.confirm({
      title: 'Cancel Fine-Tuning Job',
      content:
        'Are you sure you want to cancel this fine-tuning job? This action cannot be undone.',
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
  };

  const handleBack = () => router.push('/finetuning');

  const getEventIcon = (level: string) => {
    switch (level) {
      case 'warning':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'debug':
        return <CheckCircleOutlined style={{ color: '#bfbfbf' }} />;
      default:
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    }
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
        <Button icon={<ArrowLeftOutlined />} onClick={handleBack} style={{ marginBottom: 16 }}>
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
        <Button icon={<ArrowLeftOutlined />} onClick={handleBack} style={{ marginBottom: 16 }}>
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

  // Percentage, phase label and whether the engine is actually measuring anything
  // all come from one place, so the list and this page agree.
  const progress = resolveJobProgress(jobData);
  const statusText = getFineTuningStatusText(jobData.status);
  const statusColor = getFineTuningStatusColor(jobData.status);
  const baseModelShort = jobData.model.split('/').pop() || jobData.model;

  // Once there is a model, its name is the useful title. Before that there is no
  // model yet, so name the run by what it is training.
  const heading =
    jobData.status === 'succeeded' ? getFineTunedModelName(jobData) : `Fine-tuning ${baseModelShort}`;

  const queueSeconds = getJobQueueSeconds(jobData);
  const trainingSeconds = getJobTrainingSeconds(jobData);
  const totalSeconds = getJobTotalSeconds(jobData);

  const steps = jobData.total_steps
    ? `${jobData.current_step ?? 0} / ${jobData.total_steps}`
    : jobData.current_step
      ? `${jobData.current_step}`
      : '—';

  // The training engine reports no token counts, so a permanent "Trained Tokens 0"
  // said nothing about the job. These are the numbers it does report, with
  // training time kept apart from queue wait.
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

  // Total wall-clock on its own reads as a seven-hour fine-tune when almost all of
  // it was spent waiting for a worker, so the split is spelled out.
  const breakdown = [
    queueSeconds ? `${formatDuration(queueSeconds)} queued` : null,
    trainingSeconds ? `${formatDuration(trainingSeconds)} training` : null,
  ].filter(Boolean);

  const deploymentPhase = (deployment?.phase ?? 'not_deployed') as DeploymentPhase;

  const overviewTab = (
    <Space orientation="vertical" style={{ width: '100%' }} size={16}>
      {jobData.error && (
        <Alert
          title="Job Error"
          description={`${jobData.error.code}: ${jobData.error.message}`}
          type="error"
          showIcon
        />
      )}

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

      <Card
        title={
          <>
            <ClockCircleOutlined /> Timeline
          </>
        }
        size="small"
      >
        <Descriptions column={{ xs: 1, md: 2 }} size="small">
          <Descriptions.Item label="Created">{formatCreatedAt(jobData.created_at)}</Descriptions.Item>
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
    </Space>
  );

  const configurationTab = (
    <Row gutter={[16, 16]}>
      <Col xs={24} lg={12}>
        <Card
          title={
            <>
              <RobotOutlined /> Model Information
            </>
          }
          size="small"
        >
          <Descriptions column={1} size="small">
            <Descriptions.Item label="Base Model">
              <Text strong>{jobData.model}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Fine-Tuned Model">
              {jobData.status === 'succeeded' ? (
                <Text code copyable>
                  {getFineTunedModelName(jobData)}
                </Text>
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
        <Card
          title={
            <>
              <FileTextOutlined /> Training Data
            </>
          }
          size="small"
        >
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

      <Col xs={24} lg={12}>
        <Card
          title={
            <>
              <SettingOutlined /> Hyperparameters
            </>
          }
          size="small"
        >
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
        <Card title="Identifiers" size="small">
          <Descriptions column={1} size="small">
            <Descriptions.Item label="Job ID">
              <Text code copyable>
                {jobData.id}
              </Text>
            </Descriptions.Item>
          </Descriptions>
        </Card>
      </Col>
    </Row>
  );

  const eventsTab = (
    <Card
      size="small"
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
            // Only the metrics that were actually reported, one readable tag each
            // instead of a JSON blob.
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
  );

  return (
    <div>
      <Button icon={<ArrowLeftOutlined />} onClick={handleBack} style={{ marginBottom: 16 }}>
        Back to Fine-Tuning Jobs
      </Button>

      {/* Header: the model being produced is the subject, not the job id. */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: 16,
        }}
      >
        <div>
          <Space align="center" wrap>
            <Title level={2} style={{ margin: 0 }}>
              {heading}
            </Title>
            <Tag color={statusColor}>{statusText}</Tag>
          </Space>
          <div style={{ marginTop: 4 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Base model {baseModelShort} · created {formatCreatedAt(jobData.created_at)}
            </Text>
          </div>
        </div>

        <Space>
          {canCancelFineTuningJob(jobData) && (
            <Button icon={<StopOutlined />} danger onClick={handleCancelJob}>
              Cancel Job
            </Button>
          )}
          <Button icon={<ReloadOutlined />} onClick={handleRefresh}>
            Refresh
          </Button>
        </Space>
      </div>

      {/* Status strip. Above the tabs and outside them: on a running job this is
          the only thing most visitors came for, so it should not move or vanish
          when they switch tabs. */}
      <Card size="small" style={{ marginTop: 16 }}>
        <Progress
          percent={progress.percent}
          status={
            jobData.status === 'failed'
              ? 'exception'
              : // Animate whenever the number is only inferred from the phase: it
                // will not move again until the phase changes, and a still bar
                // there reads as a stalled job.
                progress.active && !progress.measured
                ? 'active'
                : undefined
          }
          showInfo
        />
        <Text type="secondary" style={{ fontSize: 12 }}>
          {progress.label}
          {!progress.measured && progress.active && ' (estimated)'}
          {jobData.elapsed_seconds != null && ` · ${formatDuration(jobData.elapsed_seconds)} elapsed`}
        </Text>
      </Card>

      {/* Hand-off to serving. Replaces ~300 lines of deployment card that used to
          sit in the middle of this page. */}
      {canBeDeployed && (
        <Alert
          style={{ marginTop: 16 }}
          type={deploymentPhase === 'ready' ? 'success' : 'info'}
          showIcon
          icon={<CloudServerOutlined />}
          title={
            deploymentPhase === 'ready'
              ? 'This model is serving'
              : deploymentPhase === 'not_deployed'
                ? 'This model is ready to deploy'
                : `Deployment: ${DEPLOYMENT_PHASE_TEXT[deploymentPhase] ?? deploymentPhase}`
          }
          description={
            deploymentPhase === 'ready'
              ? 'Manage serving and semantic routing from the deployment page.'
              : 'Deploy it to start vLLM on it and register it with the GenAI Gateway.'
          }
          action={
            <Button
              size="small"
              type={deploymentPhase === 'not_deployed' ? 'primary' : 'default'}
              onClick={() => router.push(`/deployments/${jobData.id}`)}
            >
              {deploymentPhase === 'not_deployed' ? 'Deploy' : 'Open deployment'}
            </Button>
          }
        />
      )}

      <Tabs
        style={{ marginTop: 8 }}
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'overview',
            label: (
              <span>
                <DashboardOutlined /> Overview
              </span>
            ),
            children: overviewTab,
          },
          {
            key: 'configuration',
            label: (
              <span>
                <ProfileOutlined /> Configuration
              </span>
            ),
            children: configurationTab,
          },
          {
            key: 'events',
            label: (
              <span>
                <HistoryOutlined /> Events
                {jobEvents.length > 0 && (
                  <Text type="secondary" style={{ fontSize: 11, marginLeft: 4 }}>
                    ({jobEvents.length})
                  </Text>
                )}
              </span>
            ),
            children: eventsTab,
          },
        ]}
      />
    </div>
  );
};

export default FineTuningJobDetailPage;
