'use client';

/**
 * Deployments: the fine-tuned models this cluster can serve, and what each one is
 * doing right now.
 *
 * Entered from the sidebar as a peer of Fine-Tuning, because serving a model is a
 * separate job from training one and happens on a different schedule. A user who
 * wants to know "is my model up, and is traffic reaching it" should not have to
 * remember which fine-tuning job produced it.
 *
 * The list is derived client-side from succeeded jobs -- see
 * `useDeploymentsOverview` for why, and what keeps it cheap.
 */

import React from 'react';
import { Card, Typography, Table, Button, Space, Tag, Tooltip, Progress, Empty, Alert } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  ApiOutlined,
  CloudServerOutlined,
  EyeOutlined,
  ReloadOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useDeploymentsOverview, isDeploymentInProgress } from '@features/finetuning/hooks';
import type { DeployableModel } from '@features/finetuning/hooks';
import { QueryLoading, QueryErrorDisplay } from '@/app/components';
import { formatCreatedAt } from '@features/finetuning/utils';
import { DEPLOYMENT_PHASE_COLOR, DEPLOYMENT_PHASE_TEXT } from '../finetuning/components';
import type { DeploymentPhase } from '@features/finetuning/types';

const { Title, Text } = Typography;

const DeploymentsPage = () => {
  const router = useRouter();
  const { models, isLoading, isFetching, isError, error, routesUnavailable, refetch } =
    useDeploymentsOverview();

  const openDeployment = (jobId: string) => router.push(`/deployments/${jobId}`);

  const columns: ColumnsType<DeployableModel> = [
    {
      title: 'Model',
      key: 'model',
      width: 260,
      render: (_v, row) => (
        <Button
          type="link"
          style={{ padding: 0, textAlign: 'left', height: 'auto' }}
          onClick={() => openDeployment(row.job.id)}
        >
          <Space orientation="vertical" size={0} style={{ alignItems: 'flex-start' }}>
            <Text strong>{row.modelName}</Text>
            <Text type="secondary" style={{ fontSize: 11 }}>
              from {row.job.model.split('/').pop() || row.job.model}
            </Text>
          </Space>
        </Button>
      ),
    },
    {
      title: 'Status',
      key: 'phase',
      width: 200,
      render: (_v, row) => {
        // Until the per-row read lands there is genuinely nothing known, so say so
        // rather than showing "Not Deployed", which is a different claim.
        if (row.deploymentLoading && !row.deployment) {
          return <Text type="secondary">Checking…</Text>;
        }
        const phase = (row.deployment?.phase ?? 'not_deployed') as DeploymentPhase;
        return (
          <Space orientation="vertical" size={4} style={{ width: '100%' }}>
            <Tag color={DEPLOYMENT_PHASE_COLOR[phase] || 'default'}>
              {DEPLOYMENT_PHASE_TEXT[phase] || phase}
            </Tag>
            {isDeploymentInProgress(phase) && (
              <Progress percent={row.deployment?.progress ?? 0} size="small" status="active" />
            )}
          </Space>
        );
      },
    },
    {
      title: (
        <Tooltip title="Whether the shared semantic router sends matching queries to this model. Routing is opt-in: only requests addressed to the router are affected.">
          <span>
            Routing <ApiOutlined />
          </span>
        </Tooltip>
      ),
      key: 'routed',
      width: 150,
      render: (_v, row) => {
        if (routesUnavailable) {
          return (
            <Tooltip title="The gateway could not be read, so routing state is unknown.">
              <Text type="secondary">Unknown</Text>
            </Tooltip>
          );
        }
        if (!row.route) {
          return <Text type="secondary">Not routed</Text>;
        }
        return (
          <Tooltip title={`${row.route.utterances.length} utterances`}>
            <Tag color="blue">Routed ({row.route.utterances.length})</Tag>
          </Tooltip>
        );
      },
    },
    {
      title: 'Served As',
      key: 'served',
      width: 220,
      ellipsis: true,
      render: (_v, row) =>
        row.deployment?.phase === 'ready' ? (
          <Text code copyable style={{ fontSize: 12 }}>
            {row.deployment.served_model_name}
          </Text>
        ) : (
          <Text type="secondary" style={{ fontSize: 12 }}>
            —
          </Text>
        ),
    },
    {
      title: 'Trained',
      key: 'created',
      width: 160,
      render: (_v, row) => (
        <Text style={{ fontSize: 12 }}>{formatCreatedAt(row.job.created_at)}</Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 110,
      render: (_v, row) => (
        <Space size="small">
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            title="Open deployment"
            onClick={() => openDeployment(row.job.id)}
          />
          <Button
            type="text"
            size="small"
            icon={<ExperimentOutlined />}
            title="View the fine-tuning job that produced this"
            onClick={() => router.push(`/finetuning/${row.job.id}`)}
          />
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Title level={2} style={{ marginBottom: 8 }}>
        Deployments
      </Title>
      <Text type="secondary">
        Fine-tuned models ready to serve. Deploy one to start vLLM on it and register it with the
        GenAI Gateway, then set up routing to send matching queries its way.
      </Text>

      {isError && (
        <div style={{ marginTop: 16 }}>
          <QueryErrorDisplay error={error} onRetry={refetch} showRetry />
        </div>
      )}

      {routesUnavailable && (
        <Alert
          style={{ marginTop: 16 }}
          type="info"
          showIcon
          title="Routing state unavailable"
          description="The GenAI Gateway could not be read, so the Routing column is showing as unknown. Deployment status is unaffected."
        />
      )}

      <Card style={{ marginTop: 16 }}>
        <div style={{ marginBottom: 16 }}>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={refetch} loading={isFetching}>
              Refresh
            </Button>
          </Space>
        </div>

        <QueryLoading
          isLoading={isLoading}
          isFetching={isFetching}
          loadingType="skeleton"
          skeletonType="table"
          skeletonRows={5}
        >
          {models.length === 0 ? (
            <Empty
              image={<CloudServerOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />}
              description={
                <Space orientation="vertical" size={4}>
                  <Text>No fine-tuned models yet</Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    A model appears here once a fine-tuning job succeeds and produces a result file.
                  </Text>
                </Space>
              }
            >
              <Button type="primary" onClick={() => router.push('/finetuning')}>
                Go to Fine-Tuning
              </Button>
            </Empty>
          ) : (
            <Table
              columns={columns}
              dataSource={models.map((model) => ({ ...model, key: model.job.id }))}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} models`,
              }}
              scroll={{ x: 1100 }}
            />
          )}
        </QueryLoading>
      </Card>
    </div>
  );
};

export default DeploymentsPage;
