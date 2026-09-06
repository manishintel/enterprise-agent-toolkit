'use client';

/**
 * Deploy form: what is free, what this model will ask for, and whether it fits.
 *
 * Inline on the deployment page rather than in a modal. Deploying is the whole
 * purpose of that page while a model is not yet serving, so there is nothing for a
 * dialog to interrupt -- and the sizing feedback here (the bars, the fits/does not
 * fit verdict) updates as capacity is polled, which a modal hid behind a click.
 *
 * Two things drive the design.
 *
 * The numbers are **reservations, not measurements**. Kubernetes admits a pod by
 * comparing its requests against a node's allocatable, and this cluster has no
 * metrics-server, so "free" here means unreserved, and a node can be busy while
 * showing room. The form says so rather than implying a utilisation graph.
 *
 * The chart sets requests and limits to the same value, so the number chosen is
 * both the guarantee and the ceiling: too low is an OOM kill mid-load, too high
 * will not schedule. That trade-off is stated here, because it is not something a
 * caller can infer from a pair of input boxes.
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Divider,
  Form,
  InputNumber,
  Progress,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import {
  CloudUploadOutlined,
  InfoCircleOutlined,
  UndoOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import type {
  DeploymentCapacity,
  DeployModelRequest,
  ServingLimit,
} from '@features/finetuning/types';

const { Text } = Typography;

const GIB = 1024 ** 3;

export interface DeployModelFormProps {
  modelId: string;
  capacity?: DeploymentCapacity;
  loading: boolean;
  submitting: boolean;
  /** "Deploy Model" normally, "Retry Deployment" after a failure. */
  submitLabel?: string;
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

export default function DeployModelForm({
  modelId,
  capacity,
  loading,
  submitting,
  submitLabel = 'Deploy Model',
  onDeploy,
}: DeployModelFormProps) {
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
   * CPU and memory bounds come from the API, which enforces the same numbers on
   * the request. Keeping no local fallback is deliberate: a guessed floor here
   * would be a second rule that disagrees with the one that actually decides, and
   * the disagreement only shows up as a rejected deploy.
   */
  const bounds = capacity?.request_limits;
  const cpuMin = bounds ? bounds.cpu_min_millis / 1000 : undefined;
  const cpuMax = bounds ? Math.floor(bounds.cpu_max_millis / 1000) : undefined;
  const memoryMin = bounds ? Math.ceil(bounds.memory_min_bytes / GIB) : undefined;
  const memoryMax = bounds ? Math.floor(bounds.memory_max_bytes / GIB) : undefined;

  // Below the floor the pod cannot load the weights; above the ceiling it cannot
  // be scheduled at all. Flagged rather than silently clamped, so a value the user
  // typed on purpose is not quietly changed under them.
  const cpuBelowMin = cpuMin !== undefined && cpuCores !== null && cpuCores < cpuMin;
  const memoryBelowMin = memoryMin !== undefined && memoryGib !== null && memoryGib < memoryMin;
  const cpuAboveMax = cpuMax !== undefined && cpuCores !== null && cpuCores > cpuMax;
  const memoryAboveMax = memoryMax !== undefined && memoryGib !== null && memoryGib > memoryMax;
  const outOfRange = cpuBelowMin || memoryBelowMin || cpuAboveMax || memoryAboveMax;

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
  // what the user has typed -- capacity is polled while this is on screen.
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

  // `touched` deliberately tracks only CPU and memory, because those are what the
  // effect above re-seeds. Reset has to react to any edit, so it gets its own test.
  const hasEdits =
    touched ||
    override ||
    kvCacheGib !== null ||
    maxModelLen !== null ||
    maxNumSeqs !== null ||
    maxBatchedTokens !== null ||
    dtype !== undefined ||
    temperature !== null ||
    topP !== null;

  // Nothing here is submitted until Deploy is pressed, so "reset" is just dropping
  // the edits and letting the effect above re-seed from the recommendation.
  const resetToRecommended = () => {
    setTouched(false);
    setOverride(false);
    setMaxModelLen(null);
    setMaxNumSeqs(null);
    setMaxBatchedTokens(null);
    setDtype(undefined);
    setKvCacheGib(null);
    setTemperature(null);
    setTopP(null);
  };

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
  // `outOfRange` is not overridable by force: force exists for a capacity snapshot
  // that may be stale, whereas a request under the model's own floor is wrong now
  // and will still be wrong when the pod starts.
  const canDeploy =
    !submitting &&
    !!cpuCores &&
    !!memoryGib &&
    !atDeploymentLimit &&
    !outOfRange &&
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

  if (loading && !capacity) {
    return (
      <div style={{ textAlign: 'center', padding: 32 }}>
        <Spin tip="Checking what the cluster has free..." />
      </div>
    );
  }

  return (
    <Space orientation="vertical" style={{ width: '100%' }} size="middle">
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
              Roomiest of {capacity.nodes.length} nodes. A model runs on one node, so it has to fit
              here rather than in the cluster total.
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

      {!blocked && fits === true && <Alert type="success" showIcon title="Fits on this node" />}

      {/* Stated in full rather than only as a range on the inputs. A bare "minimum
          48Gi" reads as an arbitrary rule; the arithmetic shows which term to
          change -- and that the KV cache, not the model, is usually the big one. */}
      {bounds && (
        <Descriptions
          size="small"
          bordered
          column={1}
          title={
            <Text style={{ fontSize: 13 }}>
              Minimum this model needs
              {bounds.basis === 'installation-default' && (
                <Text type="secondary" style={{ fontSize: 12, fontWeight: 'normal' }}>
                  {' '}
                  — size could not be read from the model name, so these are installation defaults
                </Text>
              )}
            </Text>
          }
        >
          <Descriptions.Item label="CPU">
            <Text style={{ fontSize: 12 }}>{bounds.cpu_formula}</Text>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>
                A practical floor, not a hard one: fewer cores serve more slowly rather than
                failing.
              </Text>
            </div>
          </Descriptions.Item>
          <Descriptions.Item label="Memory">
            <Text style={{ fontSize: 12 }}>{bounds.memory_formula}</Text>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>
                A hard requirement: below this the pod is OOM-killed while loading. The KV cache is
                usually the largest term — lower it and this minimum drops with it.
              </Text>
            </div>
          </Descriptions.Item>
          <Descriptions.Item label="Maximum">
            <Text style={{ fontSize: 12 }}>
              {cpuMax} cores / {memoryMax} GiB
            </Text>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>
                What {bounds.ceiling_node || 'the roomiest node'} has allocatable. A pod runs on one
                node, so this is a node&apos;s limit rather than the cluster total.
              </Text>
            </div>
          </Descriptions.Item>
        </Descriptions>
      )}

      {bounds?.exceeds_hardware && (
        <Alert
          type="error"
          showIcon
          title="This model needs more than any single node has"
          description={`Its minimum is ${bounds.cpu_min} cores and ${bounds.memory_min}, which no node here can provide. Lower the KV cache to reduce the minimum, or serve it on a larger node.`}
        />
      )}

      {outOfRange && (
        <Alert
          type="error"
          showIcon
          title="Outside the range this model can run in"
          description={
            <Space orientation="vertical" size={2}>
              {cpuBelowMin && <span>CPU is below the {cpuMin}-core minimum.</span>}
              {memoryBelowMin && (
                <span>
                  Memory is below the {memoryMin} GiB minimum — the weights and the KV cache
                  reservation would not fit, and the pod would be OOM-killed while loading.
                </span>
              )}
              {cpuAboveMax && <span>CPU is above the {cpuMax} cores the node has.</span>}
              {memoryAboveMax && <span>Memory is above the {memoryMax} GiB the node has.</span>}
            </Space>
          }
        />
      )}

      <Form layout="vertical" size="small">
        <Space size="middle" wrap style={{ width: '100%' }}>
          <Form.Item
            label="CPU (cores)"
            required
            validateStatus={cpuBelowMin || cpuAboveMax ? 'error' : undefined}
            extra={
              bounds ? (
                <Tooltip title={`Minimum: ${bounds.cpu_formula}. Below this the model still runs, but too slowly to be useful. Maximum is what ${bounds.ceiling_node || 'the roomiest node'} has, since a pod runs on one node.`}>
                  <span>
                    {cpuMin}–{cpuMax} cores <InfoCircleOutlined />
                  </span>
                </Tooltip>
              ) : (
                'Request and limit are set to this same value.'
              )
            }
            style={{ marginBottom: 8 }}
          >
            <InputNumber
              min={cpuMin}
              max={cpuMax}
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
            required
            validateStatus={memoryBelowMin || memoryAboveMax ? 'error' : undefined}
            extra={
              bounds ? (
                <Tooltip title={`Minimum: ${bounds.memory_formula}. This is a hard requirement — below it the weights and the KV cache reservation do not fit and the pod is OOM-killed while loading. Lowering the KV cache field lowers this minimum.`}>
                  <span>
                    {memoryMin}–{memoryMax} GiB <InfoCircleOutlined />
                  </span>
                </Tooltip>
              ) : (
                'Weights + KV cache + overhead. Too low is an OOM kill.'
              )
            }
            style={{ marginBottom: 8 }}
          >
            <InputNumber
              min={memoryMin}
              max={memoryMax}
              step={4}
              value={memoryGib}
              onChange={(value) => {
                setTouched(true);
                setMemoryGib(value);
              }}
              style={{ width: 150 }}
            />
          </Form.Item>
          {/* Capped at what a node holds, not the config's 512Gi: this is a
              reservation inside the container, so a value above the node's memory
              makes the memory floor unsatisfiable and the form unfillable. */}
          <Form.Item
            label="KV cache (GiB)"
            extra={
              <Tooltip title="Changing this moves the memory minimum with it, because vLLM reserves the cache before it loads anything.">
                <span>
                  {hint('kv_cache_space_gib')} <InfoCircleOutlined />
                </span>
              </Tooltip>
            }
            style={{ marginBottom: 8 }}
          >
            <Tooltip title="VLLM_CPU_KVCACHE_SPACE. Reserved up front whatever the model's size, so on CPU it is usually the largest part of the memory request. Lowering it lowers the memory needed, at the cost of how much context and concurrency fit.">
              <InputNumber
                min={limits.kv_cache_space_gib?.min ?? 1}
                max={Math.min(limits.kv_cache_space_gib?.max ?? 512, memoryMax ?? 512)}
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
          <Form.Item label="Weight precision" extra={hint('dtype')} style={{ marginBottom: 8 }}>
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

        <Space size="middle" wrap style={{ width: '100%' }}>
          <Form.Item
            label="Temperature"
            extra={hint('temperature', "the model's own")}
            style={{ marginBottom: 0 }}
          >
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
          <Form.Item
            label="Top-p"
            extra={hint('top_p', "the model's own")}
            style={{ marginBottom: 0 }}
          >
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
          A default only — vLLM has no server-side temperature, so a request that sends its own wins.
          Enforce it at the gateway if it has to hold.
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
              The KV cache is reserved up front, so it is part of the memory request whether or not
              the model is busy. Lowering it lowers the memory needed; raising context length or
              concurrency without room in the cache is what makes requests queue.
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
        <Descriptions size="small" column={{ xs: 1, sm: 2 }}>
          <Descriptions.Item label="Base model">
            <Text code style={{ fontSize: 11 }}>
              {modelId}
            </Text>
          </Descriptions.Item>
          {/* With no count cap configured (the default) a "3 / 3" reads as a limit
              that does not exist, and the number counts every vLLM instance in the
              namespace -- base and embedding models included -- not just
              fine-tuned ones. So say what it is instead of implying a budget. */}
          <Descriptions.Item label="vLLM instances running">
            <Tooltip
              title={
                capacity.deployments_max > 0
                  ? `This installation caps concurrent deployments at ${capacity.deployments_max}.`
                  : 'Every vLLM model on the cluster, including base and embedding models. There is no cap on the count — whether another one fits is decided by the CPU and memory above.'
              }
            >
              <Tag color={atDeploymentLimit ? 'red' : 'default'}>
                {capacity.deployments_used}
                {capacity.deployments_max > 0 ? ` / ${capacity.deployments_max}` : ''}
              </Tag>
            </Tooltip>
          </Descriptions.Item>
        </Descriptions>
      )}

      {/* Inline, so the action needs its own resting place -- a dialog footer used
          to supply one. Reset replaces Cancel: there is no dialog to dismiss, but
          there are edits worth being able to drop. */}
      <Card size="small" style={{ background: '#fafafa' }}>
        <Space wrap>
          <Button
            type="primary"
            icon={<CloudUploadOutlined />}
            loading={submitting}
            disabled={!canDeploy}
            onClick={submit}
          >
            {submitLabel}
          </Button>
          <Button
            icon={<UndoOutlined />}
            onClick={resetToRecommended}
            disabled={submitting || !hasEdits}
          >
            Reset to recommended
          </Button>
        </Space>
      </Card>
    </Space>
  );
}
