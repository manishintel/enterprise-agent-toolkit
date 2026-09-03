'use client';

/**
 * Deploy dialog: what is free, what this model will ask for, and whether it fits.
 *
 * Two things drive the design.
 *
 * The numbers are **reservations, not measurements**. Kubernetes admits a pod by
 * comparing its requests against a node's allocatable, and this cluster has no
 * metrics-server, so "free" here means unreserved, and a node can be busy while
 * showing room. The panel says so rather than implying a utilisation graph.
 *
 * The chart sets requests and limits to the same value, so the number chosen is
 * both the guarantee and the ceiling: too low is an OOM kill mid-load, too high
 * will not schedule. That trade-off is stated in the dialog, because it is not
 * something a caller can infer from a pair of input boxes.
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Descriptions,
  Divider,
  Form,
  InputNumber,
  Modal,
  Progress,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import { InfoCircleOutlined, WarningOutlined } from '@ant-design/icons';
import type {
  DeploymentCapacity,
  DeployModelRequest,
  ServingLimit,
} from '@features/finetuning/types';

const { Text, Paragraph } = Typography;

const GIB = 1024 ** 3;

export interface DeployModelDialogProps {
  open: boolean;
  modelId: string;
  capacity?: DeploymentCapacity;
  loading: boolean;
  submitting: boolean;
  onCancel: () => void;
  onDeploy: (overrides: DeployModelRequest) => void;
}

const formatCores = (millis: number): string => {
  const cores = millis / 1000;
  return Number.isInteger(cores) ? `${cores}` : cores.toFixed(1);
};

const formatGib = (bytes: number): string => {
  const gib = bytes / GIB;
  return Number.isInteger(gib) ? `${gib}` : gib.toFixed(1);
};

/**
 * One resource, as a bar of "already reserved" + "this deployment" against the
 * node's allocatable, so the request is shown in the space it has to fit into.
 */
function ResourceBar({
  label,
  unit,
  allocatable,
  committed,
  requested,
}: {
  label: string;
  unit: string;
  allocatable: number;
  committed: number;
  requested: number;
}) {
  if (!allocatable) return null;

  const committedPct = Math.min(100, (committed / allocatable) * 100);
  const requestedPct = Math.min(100 - committedPct, (requested / allocatable) * 100);
  const free = Math.max(0, allocatable - committed);
  const overflows = requested > free;
  const fmt = unit === 'cores' ? formatCores : formatGib;

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <Text strong>{label}</Text>
        <Text type="secondary">
          {fmt(free)} of {fmt(allocatable)} {unit} unreserved
        </Text>
      </div>
      <Progress
        percent={committedPct + requestedPct}
        success={{ percent: committedPct }}
        status={overflows ? 'exception' : 'normal'}
        showInfo={false}
        size="small"
      />
      <Text type="secondary" style={{ fontSize: 11 }}>
        in use by other workloads {fmt(committed)} · this model {fmt(requested)}
        {overflows && (
          <Text type="danger" style={{ fontSize: 11 }}>
            {' '}
            · exceeds what is free
          </Text>
        )}
      </Text>
    </div>
  );
}

export default function DeployModelDialog({
  open,
  modelId,
  capacity,
  loading,
  submitting,
  onCancel,
  onDeploy,
}: DeployModelDialogProps) {
  const [cpuCores, setCpuCores] = useState<number | null>(null);
  const [memoryGib, setMemoryGib] = useState<number | null>(null);
  const [maxModelLen, setMaxModelLen] = useState<number | null>(null);
  const [maxNumSeqs, setMaxNumSeqs] = useState<number | null>(null);
  const [maxBatchedTokens, setMaxBatchedTokens] = useState<number | null>(null);
  const [dtype, setDtype] = useState<string | undefined>(undefined);
  const [kvCacheGib, setKvCacheGib] = useState<number | null>(null);
  const [temperature, setTemperature] = useState<number | null>(null);
  const [topP, setTopP] = useState<number | null>(null);
  const [override, setOverride] = useState(false);
  const [touched, setTouched] = useState(false);

  const recommended = capacity?.recommended;
  const defaults = capacity?.serving_defaults;
  const limits = capacity?.serving_limits ?? {};

  /**
   * Label showing the chart's own default and the range, so an empty field is
   * self-explanatory rather than "Chart default when empty".
   */
  const hint = (field: string, fallback?: string): string => {
    const limit: ServingLimit | undefined = limits[field];
    const value = (defaults as Record<string, unknown> | undefined)?.[field];
    const shown = value === null || value === undefined ? fallback ?? 'model default' : String(value);
    const range = limit ? `, allowed ${limit.min}-${limit.max}` : '';
    return `default ${shown}${range}`;
  };

  // Seed the form from the recommendation once it arrives, but never overwrite
  // what the user has typed -- capacity is polled while this is open.
  //
  // Memory tracks the KV cache field: vLLM reserves VLLM_CPU_KVCACHE_SPACE
  // whatever the model's size, so halving the cache genuinely frees that much of
  // the request, and leaving the two out of step is what gets a pod OOM-killed.
  useEffect(() => {
    if (!recommended || touched) return;
    const kvDefault = defaults?.kv_cache_space_gib ?? null;
    const kvDelta = kvCacheGib !== null && kvDefault !== null ? kvCacheGib - kvDefault : 0;
    setCpuCores(recommended.cpu_millis / 1000);
    setMemoryGib(Math.max(1, Math.round(recommended.memory_bytes / GIB) + kvDelta));
  }, [recommended, touched, kvCacheGib, defaults]);

  useEffect(() => {
    if (!open) {
      setTouched(false);
      setOverride(false);
      setMaxModelLen(null);
      setMaxNumSeqs(null);
      setMaxBatchedTokens(null);
      setDtype(undefined);
      setKvCacheGib(null);
      setTemperature(null);
      setTopP(null);
    }
  }, [open]);

  const node = useMemo(() => {
    const usable = (capacity?.nodes ?? []).filter((n) => n.schedulable);
    if (!usable.length) return undefined;
    return usable.reduce((a, b) => (b.free.cpu_millis > a.free.cpu_millis ? b : a));
  }, [capacity]);

  const requestedMillis = (cpuCores ?? 0) * 1000;
  const requestedBytes = (memoryGib ?? 0) * GIB;

  // Recomputed locally as the user types; the server re-checks on deploy, since
  // this endpoint is reachable without the UI.
  const fits = useMemo(() => {
    if (!capacity?.available || !node) return undefined;
    return (
      node.free.cpu_millis >= requestedMillis &&
      node.free.memory_bytes >= requestedBytes &&
      (node.free.pods ?? 1) >= 1
    );
  }, [capacity, node, requestedMillis, requestedBytes]);

  const atDeploymentLimit =
    !!capacity && capacity.deployments_max > 0 && capacity.deployments_used >= capacity.deployments_max;

  const blocked = fits === false || atDeploymentLimit;
  const canDeploy =
    !submitting &&
    !!cpuCores &&
    !!memoryGib &&
    !atDeploymentLimit &&
    (fits !== false || (override && !!capacity?.override_allowed));

  const submit = () => {
    const overrides: DeployModelRequest = {};
    if (cpuCores) overrides.cpu = `${cpuCores}`;
    if (memoryGib) overrides.memory = `${memoryGib}Gi`;
    if (maxModelLen) overrides.max_model_len = maxModelLen;
    if (maxNumSeqs) overrides.max_num_seqs = maxNumSeqs;
    if (maxBatchedTokens) overrides.max_num_batched_tokens = maxBatchedTokens;
    if (dtype) overrides.dtype = dtype;
    if (kvCacheGib) overrides.kv_cache_space_gib = kvCacheGib;
    if (temperature !== null) overrides.temperature = temperature;
    if (topP !== null) overrides.top_p = topP;
    if (fits === false && override) overrides.force = true;
    onDeploy(overrides);
  };

  return (
    <Modal
      open={open}
      title="Deploy Fine-Tuned Model"
      onCancel={onCancel}
      width={680}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>,
        <Button key="deploy" type="primary" loading={submitting} disabled={!canDeploy} onClick={submit}>
          Deploy
        </Button>,
      ]}
    >
      {loading && !capacity ? (
        <div style={{ textAlign: 'center', padding: 32 }}>
          <Spin tip="Checking what the cluster has free..." />
        </div>
      ) : (
        <Space orientation="vertical" style={{ width: '100%' }} size="middle">
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            This starts the model on the cluster and registers it with the GenAI Gateway. The first
            start takes several minutes while the model is downloaded and loaded.
          </Paragraph>

          {capacity && !capacity.available && (
            <Alert
              type="warning"
              showIcon
              icon={<WarningOutlined />}
              title="Cluster capacity is unknown"
              description={
                <>
                  {capacity.message} Deploying is still possible, but nothing here can tell you in
                  advance whether the model will fit.
                </>
              }
            />
          )}

          {capacity?.available && node && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <Text strong>{node.name}</Text>
                <Tooltip
                  title={
                    'Kubernetes places a pod by comparing its request against what is unreserved on ' +
                    'a node, so these are reservations rather than live usage. This cluster has no ' +
                    'metrics-server, so actual utilisation is not available.'
                  }
                >
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    reserved vs allocatable <InfoCircleOutlined />
                  </Text>
                </Tooltip>
              </div>
              <ResourceBar
                label="CPU"
                unit="cores"
                allocatable={node.allocatable.cpu_millis}
                committed={node.committed.cpu_millis}
                requested={requestedMillis}
              />
              <ResourceBar
                label="Memory"
                unit="GiB"
                allocatable={node.allocatable.memory_bytes}
                committed={node.committed.memory_bytes}
                requested={requestedBytes}
              />
              {(capacity.nodes ?? []).length > 1 && (
                <Text type="secondary" style={{ fontSize: 11 }}>
                  Roomiest of {capacity.nodes.length} nodes. A model runs on one node, so it has to
                  fit here rather than in the cluster total.
                </Text>
              )}
            </div>
          )}

          {atDeploymentLimit && (
            <Alert
              type="error"
              showIcon
              title={`Already serving ${capacity?.deployments_used} of ${capacity?.deployments_max} models`}
              description="Remove a deployed model before deploying another one."
            />
          )}

          {!atDeploymentLimit && fits === false && (
            <Alert
              type="error"
              showIcon
              title="Not enough room on any node"
              description={
                <Space orientation="vertical" size={4}>
                  <span>
                    {capacity?.shortfall ??
                      'The request is larger than what is unreserved on the roomiest node.'}
                  </span>
                  <span>Lower the request, or remove a deployed model to free space.</span>
                  {capacity?.override_allowed && (
                    <label style={{ display: 'block', marginTop: 4 }}>
                      <input
                        type="checkbox"
                        checked={override}
                        onChange={(event) => setOverride(event.target.checked)}
                        style={{ marginRight: 6 }}
                      />
                      Deploy anyway. Kubernetes will leave the pod Pending until room appears.
                    </label>
                  )}
                </Space>
              }
            />
          )}

          {!blocked && fits === true && (
            <Alert type="success" showIcon title="Fits on this node" />
          )}

          <Form layout="vertical" size="small">
            <Space size="middle" style={{ width: '100%' }}>
              <Form.Item
                label="CPU (cores)"
                extra="Request and limit are set to this same value."
                style={{ marginBottom: 8 }}
              >
                <InputNumber
                  min={1}
                  step={1}
                  value={cpuCores}
                  onChange={(value) => {
                    setTouched(true);
                    setCpuCores(value);
                  }}
                  style={{ width: 150 }}
                />
              </Form.Item>
              <Form.Item
                label="Memory (GiB)"
                extra="Weights + KV cache + overhead. Too low is an OOM kill."
                style={{ marginBottom: 8 }}
              >
                <InputNumber
                  min={1}
                  step={4}
                  value={memoryGib}
                  onChange={(value) => {
                    setTouched(true);
                    setMemoryGib(value);
                  }}
                  style={{ width: 150 }}
                />
              </Form.Item>
              <Form.Item
                label="KV cache (GiB)"
                extra={hint('kv_cache_space_gib')}
                style={{ marginBottom: 8 }}
              >
                <Tooltip title="VLLM_CPU_KVCACHE_SPACE. Reserved up front whatever the model's size, so on CPU it is usually the largest part of the memory request. Lowering it lowers the memory needed, at the cost of how much context and concurrency fit.">
                  <InputNumber
                    min={limits.kv_cache_space_gib?.min ?? 1}
                    max={limits.kv_cache_space_gib?.max ?? 512}
                    step={4}
                    value={kvCacheGib}
                    onChange={setKvCacheGib}
                    placeholder={String(defaults?.kv_cache_space_gib ?? '')}
                    style={{ width: 150 }}
                  />
                </Tooltip>
              </Form.Item>
            </Space>

            <Divider style={{ margin: '8px 0' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Serving limits — leave empty to keep the chart&apos;s value
              </Text>
            </Divider>

            <Space size="middle" wrap style={{ width: '100%' }}>
              <Form.Item
                label="Max context length"
                extra={hint('max_model_len', "the model's own maximum")}
                style={{ marginBottom: 8 }}
              >
                <InputNumber
                  min={limits.max_model_len?.min ?? 256}
                  max={limits.max_model_len?.max}
                  step={1024}
                  value={maxModelLen}
                  onChange={setMaxModelLen}
                  placeholder={defaults?.max_model_len ? String(defaults.max_model_len) : 'model max'}
                  style={{ width: 150 }}
                />
              </Form.Item>
              <Form.Item
                label="Max concurrent sequences"
                extra={hint('max_num_seqs')}
                style={{ marginBottom: 8 }}
              >
                <InputNumber
                  min={limits.max_num_seqs?.min ?? 1}
                  max={limits.max_num_seqs?.max}
                  step={16}
                  value={maxNumSeqs}
                  onChange={setMaxNumSeqs}
                  placeholder={String(defaults?.max_num_seqs ?? '')}
                  style={{ width: 150 }}
                />
              </Form.Item>
              <Form.Item
                label="Max batched tokens"
                extra={hint('max_num_batched_tokens')}
                style={{ marginBottom: 8 }}
              >
                <InputNumber
                  min={limits.max_num_batched_tokens?.min ?? 256}
                  max={limits.max_num_batched_tokens?.max}
                  step={256}
                  value={maxBatchedTokens}
                  onChange={setMaxBatchedTokens}
                  placeholder={String(defaults?.max_num_batched_tokens ?? '')}
                  style={{ width: 150 }}
                />
              </Form.Item>
              <Form.Item
                label="Weight precision"
                extra={hint('dtype')}
                style={{ marginBottom: 8 }}
              >
                <Select
                  allowClear
                  value={dtype}
                  onChange={setDtype}
                  placeholder={defaults?.dtype ?? 'auto'}
                  style={{ width: 150 }}
                  options={(capacity?.dtype_choices ?? []).map((choice) => ({
                    value: choice,
                    label: choice,
                  }))}
                />
              </Form.Item>
            </Space>

            <Divider style={{ margin: '8px 0' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Sampling defaults
              </Text>
            </Divider>

            <Space size="middle" style={{ width: '100%' }}>
              <Form.Item label="Temperature" extra={hint('temperature', 'the model\'s own')} style={{ marginBottom: 0 }}>
                <InputNumber
                  min={limits.temperature?.min ?? 0}
                  max={limits.temperature?.max ?? 2}
                  step={0.1}
                  value={temperature}
                  onChange={setTemperature}
                  placeholder="model default"
                  style={{ width: 150 }}
                />
              </Form.Item>
              <Form.Item label="Top-p" extra={hint('top_p', 'the model\'s own')} style={{ marginBottom: 0 }}>
                <InputNumber
                  min={limits.top_p?.min ?? 0}
                  max={limits.top_p?.max ?? 1}
                  step={0.05}
                  value={topP}
                  onChange={setTopP}
                  placeholder="model default"
                  style={{ width: 150 }}
                />
              </Form.Item>
            </Space>
            <Text type="secondary" style={{ fontSize: 11 }}>
              A default only — vLLM has no server-side temperature, so a request that sends its own
              wins. Enforce it at the gateway if it has to hold.
            </Text>
          </Form>

          <Alert
            type="info"
            showIcon
            title="Sizing this model"
            description={
              <Space orientation="vertical" size={2}>
                {(recommended?.notes ?? []).map((note) => (
                  <span key={note}>{note}</span>
                ))}
                <span>
                  The KV cache is reserved up front, so it is part of the memory request whether or
                  not the model is busy. Lowering it lowers the memory needed; raising context length
                  or concurrency without room in the cache is what makes requests queue.
                </span>
                {defaults?.source === 'fallback' && (
                  <span>
                    The packaged chart could not be read, so the defaults shown are this
                    installation&apos;s documented values rather than the chart&apos;s own.
                  </span>
                )}
              </Space>
            }
          />

          {capacity && (
            <Descriptions size="small" column={2}>
              <Descriptions.Item label="Base model">
                <Text code style={{ fontSize: 11 }}>
                  {modelId}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Models served">
                <Tag color={atDeploymentLimit ? 'red' : 'default'}>
                  {capacity.deployments_used} / {capacity.deployments_max}
                </Tag>
              </Descriptions.Item>
            </Descriptions>
          )}
        </Space>
      )}
    </Modal>
  );
}
