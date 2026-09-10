'use client';

/**
 * One fine-tuned model's serving state, in three tabs.
 *
 * **Overview** -- phase, the step list while it comes up, how to call it once it is
 * serving, and the Helm escape hatch.
 *
 * **Routing** -- semantic routing setup. This lives here rather than in its own
 * sidebar section for two reasons. Every routing API is scoped to a single job
 * (`/jobs/{id}/semantic-route`), so a standalone section would have no backing
 * resource to list. And a route names the model it points at, so the model has to
 * be serving before a route can exist -- as a tab that dependency is a disabled
 * tab with a reason, where a separate page would let a user arrive somewhere they
 * cannot act.
 *
 * **Logs** -- pod output, read when this tab is open and when Refresh is pressed.
 * Its own tab because a tall log box wedged into a status summary pushes everything
 * that matters below the fold, and its own request because status is polled while a
 * deployment comes up: tailing a container on every poll spends a cluster
 * round-trip on output nobody is reading, and for a healthy model that output is
 * almost entirely liveness probes.
 */

import React, { useEffect, useState } from 'react';
import {
  Alert,
  App,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Empty,
  Input,
  Select,
  Space,
  Spin,
  Tabs,
  Tag,
  Typography,
} from 'antd';
import {
  ApiOutlined,
  ArrowLeftOutlined,
  CloudUploadOutlined,
  CodeOutlined,
  DeleteOutlined,
  ExperimentOutlined,
  InfoCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useParams, useRouter } from 'next/navigation';
import { useQueryClient } from '@tanstack/react-query';
import {
  useFineTuningJob,
  useModelDeployment,
  useDeploymentCapacity,
  useDeployModel,
  useUndeployModel,
  useSemanticRoute,
  useExtractUtterances,
  useApplySemanticRoute,
  useRemoveSemanticRoute,
  useTestSemanticRoute,
  useRouteReadiness,
  useDeploymentLogs,
} from '@features/finetuning';
import { queryKeys } from '@core/query/queryClient';
import type { DeploymentPhase, DeployModelRequest } from '@features/finetuning/types';
import { getFineTunedModelName } from '@features/finetuning/utils';
import {
  DeployModelForm,
  DeploymentPanel,
  SemanticRoutingPanel,
  DEPLOYMENT_PHASE_COLOR,
  DEPLOYMENT_PHASE_TEXT,
} from '../../finetuning/components';
import type { RouteProgress } from '../../finetuning/components';

const { Title, Text } = Typography;

const DeploymentDetailPage = () => {
  const router = useRouter();
  const params = useParams();
  const { modal } = App.useApp();
  const jobId = params.id as string;

  const [activeTab, setActiveTab] = useState('overview');
  const [helmOpen, setHelmOpen] = useState(false);
  // Which router the Routing tab is looking at. Undefined means the installation
  // default, which is what the API returns for an unnamed request.
  const [routerName, setRouterName] = useState<string | undefined>(undefined);
  // Set the moment a route change is accepted, and cleared by the user. It drives
  // the readiness poll below, which is the only way to know a restart has landed.
  const [routeProgress, setRouteProgress] = useState<RouteProgress | null>(null);
  const [logTail, setLogTail] = useState(200);
  const [hideProbes, setHideProbes] = useState(true);

  const queryClient = useQueryClient();

  const { data: job, isLoading: jobLoading, error: jobError } = useFineTuningJob(jobId);

  const resultFileId = job?.result_files?.[0];
  const canBeDeployed = job?.status === 'succeeded' && !!resultFileId;

  const {
    data: deployment,
    isLoading: deploymentLoading,
    isFetching: deploymentFetching,
    error: deploymentError,
    refetch: refetchDeployment,
  } = useModelDeployment(jobId, { enabled: canBeDeployed });

  const deployModelMutation = useDeployModel();
  const undeployModelMutation = useUndeployModel();

  const phase = (deployment?.phase || 'not_deployed') as DeploymentPhase;
  const isServing = phase === 'ready';

  // The deploy form lives on the page rather than in a dialog, so it is on screen
  // exactly when deploying is possible -- there is nothing else to do here in that
  // state, and its sizing verdict updates as capacity is polled.
  const showDeployForm = !!deployment?.can_deploy;

  // Cluster-wide read, so only while something on screen needs it. Both consumers
  // are on the Overview tab, so reading logs or routing costs nothing.
  const { data: capacity, isLoading: capacityLoading } = useDeploymentCapacity(
    jobId,
    canBeDeployed && activeTab === 'overview' && (showDeployForm || helmOpen)
  );

  // The gateway is only read once the user is actually on the Routing tab.
  const routingActive = activeTab === 'routing' && isServing;
  const { data: routeStatus, isLoading: routeLoading } = useSemanticRoute(
    jobId,
    routingActive,
    routerName
  );
  const extractUtterances = useExtractUtterances();
  const applyRoute = useApplySemanticRoute();
  const removeRoute = useRemoveSemanticRoute();
  const testRoute = useTestSemanticRoute();

  // Deliberately not tied to the tab being open: a restart takes about a minute,
  // and a user who clicks Apply and goes to look at the logs should still come back
  // to a finished progress card rather than a frozen one.
  const { data: readiness } = useRouteReadiness(
    jobId,
    !!routeProgress,
    routeProgress?.routerName,
    routeProgress?.expectRoute ?? true
  );

  // The applied status was reported from what was written, because reading it back
  // through a restarting gateway races the pod. Once the route is confirmed live,
  // the real state is readable, so replace the optimistic copy with it.
  useEffect(() => {
    if (readiness?.ready) {
      queryClient.invalidateQueries({
        queryKey: queryKeys.fineTuning.jobs.semanticRoute(jobId, routerName),
      });
    }
  }, [readiness?.ready, jobId, routerName, queryClient]);

  // Logs are read when the tab is open and not polled: a serving model's output is
  // almost all health checks, so refreshing it on a timer spends a cluster
  // round-trip to redraw the same lines.
  const {
    data: deploymentLogs,
    isFetching: logsFetching,
    error: logsError,
    refetch: refetchLogs,
  } = useDeploymentLogs(jobId, canBeDeployed && activeTab === 'logs', {
    tail: logTail,
    hideProbes,
  });

  const handleApplyRoute = (utterances: string[], threshold: number, router: string) => {
    applyRoute.mutate(
      { jobId, body: { utterances, score_threshold: threshold, router_name: router } },
      {
        onSuccess: (data) => {
          // Follow the router that was actually written, which is the one just
          // created when the user named a new one.
          setRouterName(data.router_name);
          setRouteProgress({
            startedAt: Date.now(),
            expectRoute: true,
            routerName: data.router_name,
          });
        },
      }
    );
  };

  const handleRemoveRoute = () => {
    modal.confirm({
      title: 'Stop routing here',
      content:
        'Questions that used to reach this model will go to the router\u2019s fallback model instead. ' +
        'The model keeps serving, and other models routed through this router are unaffected. ' +
        'Applying this restarts the gateway.',
      okText: 'Stop routing',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: () =>
        removeRoute.mutate(
          { jobId, routerName: routeStatus?.router_name },
          {
            onSuccess: (data) =>
              setRouteProgress({
                startedAt: Date.now(),
                expectRoute: false,
                routerName: data.router_name,
              }),
          }
        ),
    });
  };

  const mutating = deployModelMutation.isPending || undeployModelMutation.isPending;

  // No dialog to close on success: once the deployment is accepted `can_deploy`
  // goes false and the form gives way to the progress steps above it.
  const handleDeploy = (overrides: DeployModelRequest) => {
    if (!job) return;
    deployModelMutation.mutate({ jobId: job.id, overrides });
  };

  const handleUndeploy = () => {
    if (!job) return;
    modal.confirm({
      title: 'Remove Deployment',
      content:
        'This stops the model and removes it from the GenAI Gateway. The fine-tuned model ' +
        'files are kept, so it can be deployed again later.',
      okText: 'Remove',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: () => undeployModelMutation.mutate(job.id),
    });
  };

  if (jobLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>Loading deployment…</Text>
        </div>
      </div>
    );
  }

  if (jobError || !job) {
    return (
      <div>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => router.push('/deployments')}
          style={{ marginBottom: 16 }}
        >
          Back to Deployments
        </Button>
        <Alert
          title="Model Not Found"
          description={jobError?.message || 'This fine-tuned model could not be loaded.'}
          type="error"
          showIcon
        />
      </div>
    );
  }

  // A job that never produced a model has nothing to serve, and the deployment
  // APIs would fail on it, so say that plainly instead of rendering empty tabs.
  if (!canBeDeployed) {
    return (
      <div>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => router.push('/deployments')}
          style={{ marginBottom: 16 }}
        >
          Back to Deployments
        </Button>
        <Alert
          title="Nothing to deploy yet"
          description={
            job.status === 'succeeded'
              ? 'This job succeeded but produced no result file, so there is no model to serve.'
              : `This fine-tuning job is ${job.status.replace(/_/g, ' ')}. A model can be deployed once it succeeds.`
          }
          type="info"
          showIcon
          action={
            <Button size="small" onClick={() => router.push(`/finetuning/${job.id}`)}>
              View job
            </Button>
          }
        />
      </div>
    );
  }

  const servedModelName = deployment?.served_model_name || getFineTunedModelName(job);

  const overviewTab = (
    <Space orientation="vertical" style={{ width: '100%' }} size="middle">
      <Card size="small">
        {deploymentLoading ? (
          <div style={{ textAlign: 'center', padding: 20 }}>
            <Spin />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">Checking deployment status…</Text>
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
          <DeploymentPanel
            job={job}
            deployment={deployment}
            capacity={capacity}
            resultFileId={resultFileId}
            onHelmVisibilityChange={setHelmOpen}
          />
        )}
      </Card>

      {showDeployForm && (
        <Card
          size="small"
          title={phase === 'failed' ? 'Deploy again' : 'Deployment settings'}
          extra={
            <Text type="secondary" style={{ fontSize: 12 }}>
              sized for this model — adjust before deploying
            </Text>
          }
        >
          <DeployModelForm
            modelId={job.model}
            capacity={capacity}
            loading={capacityLoading}
            submitting={deployModelMutation.isPending}
            submitLabel={phase === 'failed' ? 'Retry Deployment' : 'Deploy Model'}
            onDeploy={handleDeploy}
          />
        </Card>
      )}
    </Space>
  );

  const routingTab = !isServing ? (
    <Card size="small">
      <Empty
        image={<ApiOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />}
        description={
          <Space orientation="vertical" size={4}>
            <Text>Routing needs a serving model</Text>
            <Text type="secondary" style={{ fontSize: 12 }}>
              A route names the model it points at, so this model has to be deployed and answering
              requests before traffic can be sent to it. Deploy it from the Overview tab first.
            </Text>
          </Space>
        }
      />
    </Card>
  ) : (
    // No Card wrapper: the panel lays itself out as four numbered cards, and nesting
    // those inside another one buries the step structure in borders.
    <SemanticRoutingPanel
      status={routeStatus}
      statusLoading={routeLoading}
      routerName={routerName}
      onRouterChange={(name) => {
        setRouterName(name);
        // A finished progress card belongs to the router it was applied to, so it
        // goes when the view moves to a different one.
        setRouteProgress(null);
      }}
      extraction={extractUtterances.data}
      extracting={extractUtterances.isPending}
      testResult={testRoute.data}
      testing={testRoute.isPending}
      applying={applyRoute.isPending}
      removing={removeRoute.isPending}
      progress={routeProgress}
      readiness={readiness}
      onDismissProgress={() => setRouteProgress(null)}
      onExtract={(options) => extractUtterances.mutate({ jobId, options })}
      onTest={(query, utterances, score_threshold) =>
        testRoute.mutate({ jobId, query, utterances, score_threshold, router_name: routerName })
      }
      onApply={handleApplyRoute}
      onRemove={handleRemoveRoute}
    />
  );

  // Fetched when this tab is open and when Refresh is pressed, never on a timer.
  // A healthy model logs little but liveness probes, so a self-refreshing view was
  // a cluster round-trip every few seconds to redraw `GET /health 200 OK`.
  const logsTab = (
    <Card
      size="small"
      title={
        <Space wrap size={12}>
          <Checkbox checked={hideProbes} onChange={(e) => setHideProbes(e.target.checked)}>
            <Text style={{ fontSize: 13 }}>Hide health checks</Text>
          </Checkbox>
          <Space size={4}>
            <Text type="secondary" style={{ fontSize: 13 }}>
              Lines
            </Text>
            <Select
              size="small"
              value={logTail}
              onChange={setLogTail}
              style={{ width: 90 }}
              options={[100, 200, 500, 1000].map((n) => ({ value: n, label: String(n) }))}
            />
          </Space>
        </Space>
      }
      extra={
        <Button
          size="small"
          icon={<ReloadOutlined />}
          onClick={() => refetchLogs()}
          loading={logsFetching}
        >
          Refresh
        </Button>
      }
    >
      {logsError ? (
        <Alert
          title="Could Not Read Logs"
          description={logsError.message}
          type="warning"
          showIcon
        />
      ) : logsFetching && !deploymentLogs ? (
        <div style={{ textAlign: 'center', padding: 32 }}>
          <Spin />
          <div style={{ marginTop: 8 }}>
            <Text type="secondary">Reading the container log…</Text>
          </div>
        </div>
      ) : deploymentLogs?.logs?.length ? (
        <Space orientation="vertical" style={{ width: '100%' }} size="small">
          <Space wrap size={12}>
            {deploymentLogs.log_source && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                <CodeOutlined /> {deploymentLogs.log_source}
              </Text>
            )}
            {deploymentLogs.hidden_lines > 0 && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {deploymentLogs.hidden_lines} health-check lines hidden
              </Text>
            )}
          </Space>
          <Input.TextArea
            value={deploymentLogs.logs.join('\n')}
            readOnly
            autoSize={{ minRows: 18, maxRows: 40 }}
            style={{ fontFamily: 'monospace', fontSize: 12, backgroundColor: '#f5f5f5' }}
          />
        </Space>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            deploymentLogs?.message ||
            (phase === 'not_deployed'
              ? 'No logs yet — this model has not been deployed.'
              : 'No output reported for this deployment.')
          }
        />
      )}
    </Card>
  );

  return (
    <div>
      <Button
        icon={<ArrowLeftOutlined />}
        onClick={() => router.push('/deployments')}
        style={{ marginBottom: 16 }}
      >
        Back to Deployments
      </Button>

      {/* The model's name is the title. The job id is provenance, not identity, so
          it moves to a link rather than being the heading. */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: 16,
          marginBottom: 16,
        }}
      >
        <div>
          <Space align="center" wrap>
            <Title level={2} style={{ margin: 0 }}>
              {servedModelName}
            </Title>
            <Tag color={DEPLOYMENT_PHASE_COLOR[phase] || 'default'}>
              {DEPLOYMENT_PHASE_TEXT[phase] || phase}
            </Tag>
          </Space>
          <div style={{ marginTop: 4 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Fine-tuned from {job.model} ·{' '}
              <Button
                type="link"
                size="small"
                style={{ padding: 0, fontSize: 12 }}
                icon={<ExperimentOutlined />}
                onClick={() => router.push(`/finetuning/${job.id}`)}
              >
                view training job
              </Button>
            </Text>
          </div>
        </div>

        {/* No Deploy button here: the action belongs to the form on the Overview
            tab, and two of them would be two different-looking ways to do one
            thing. The other tabs both say where to go while nothing is serving. */}
        <Space wrap>
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
            icon={<ReloadOutlined />}
            onClick={() => refetchDeployment()}
            loading={deploymentFetching}
          >
            Refresh
          </Button>
        </Space>
      </div>

      {isServing && (
        <Descriptions
          size="small"
          bordered
          column={{ xs: 1, sm: 2, lg: 3 }}
          style={{ marginBottom: 16 }}
        >
          <Descriptions.Item label="Call it as">
            <Text code copyable>
              {servedModelName}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="Gateway">
            {deployment?.gateway_registered ? (
              <Tag color="success">Registered</Tag>
            ) : (
              <Tag color="warning">Not confirmed</Tag>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="Routing">
            {routeStatus?.this_route ? (
              <Tag color="blue">{routeStatus.this_route.utterances.length} utterances</Tag>
            ) : (
              <Text type="secondary">Not configured</Text>
            )}
          </Descriptions.Item>
        </Descriptions>
      )}

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'overview',
            label: (
              <span>
                <CloudUploadOutlined /> Overview
              </span>
            ),
            children: overviewTab,
          },
          {
            key: 'routing',
            label: (
              <span>
                <ApiOutlined /> Routing
                {!isServing && (
                  <Text type="secondary" style={{ fontSize: 11, marginLeft: 6 }}>
                    <InfoCircleOutlined />
                  </Text>
                )}
              </span>
            ),
            children: routingTab,
          },
          {
            key: 'logs',
            label: (
              <span>
                <CodeOutlined /> Logs
              </span>
            ),
            children: logsTab,
          },
        ]}
      />
    </div>
  );
};

export default DeploymentDetailPage;
