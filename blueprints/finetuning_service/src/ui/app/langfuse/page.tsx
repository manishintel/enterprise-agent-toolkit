'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  DatePicker,
  Divider,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd';
import {
  CloudDownloadOutlined,
  DatabaseOutlined,
  EyeOutlined,
  FileTextOutlined,
  FilterOutlined,
  ImportOutlined,
} from '@ant-design/icons';
import { notify } from '@notification';
import { useRouter } from 'next/navigation';
import dayjs, { Dayjs } from 'dayjs';
import type { SelectProps } from 'antd';
import {
  importLangfuse,
  listLangfuseAnnotations,
  listLangfuseFields,
  listLangfuseModels,
  listLangfuseProjects,
  previewLangfuse,
} from '@features/dataprep/api/client';
import type {
  LangfuseAnnotationQueue,
  LangfuseImportFormat,
  LangfuseImportRequest,
  LangfuseModelOption,
  LangfuseProject,
  LangfuseScoreOption,
  LangfuseScoreSource,
} from '@features/dataprep/types';

const { Title, Text, Paragraph } = Typography;
const { RangePicker } = DatePicker;

interface FormValues {
  project_id?: string;
  range?: [Dayjs | null, Dayjs | null];
  order_by?: string;
  max_traces?: number;
  model?: string;
  score_name?: string;
  score_source?: LangfuseScoreSource;
  score_operator?: string;
  score_value?: number;
  score_string_value?: string;
  annotation_queue_id?: string;
  annotation_queue_status?: string;
  format: LangfuseImportFormat;
  fields?: string[];
  filename?: string;
}

/** Score fields are cleared together — the condition only means something next to its score. */
const SCORE_CONDITION_FIELDS = ['score_operator', 'score_value', 'score_string_value'] as const;

const SCORE_SOURCE_LABELS: Record<LangfuseScoreSource, string> = {
  ANNOTATION: 'ANNOTATION — entered by a person',
  API: 'API — written programmatically',
  EVAL: 'EVAL — evaluator output',
};

interface Telemetry {
  scanned: number;
  skipped: number;
  returned: number;
  lastAction: 'preview' | 'import' | null;
}

const EMPTY_TELEMETRY: Telemetry = {
  scanned: 0,
  skipped: 0,
  returned: 0,
  lastAction: null,
};

const LangfusePage: React.FC = () => {
  const router = useRouter();
  const [form] = Form.useForm<FormValues>();
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry>(EMPTY_TELEMETRY);
  const [previewing, setPreviewing] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableFields, setAvailableFields] = useState<string[]>([]);
  const [fieldsLoading, setFieldsLoading] = useState(false);
  const [models, setModels] = useState<LangfuseModelOption[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsWarning, setModelsWarning] = useState<string | null>(null);
  const [projects, setProjects] = useState<LangfuseProject[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [scoreOptions, setScoreOptions] = useState<LangfuseScoreOption[]>([]);
  const [queues, setQueues] = useState<LangfuseAnnotationQueue[]>([]);
  const [annotationsLoading, setAnnotationsLoading] = useState(false);
  const [annotationsWarning, setAnnotationsWarning] = useState<string | null>(null);

  // Watched rather than read on demand: the hint line and the button state have
  // to follow the selection, and form values alone don't re-render.
  const selectedProjectId = Form.useWatch('project_id', form);
  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId),
    [projects, selectedProjectId]
  );
  const selectedProjectLabel = useMemo(() => {
    if (!selectedProject) return null;
    const { name, organization } = selectedProject;
    return organization && organization !== name ? `${name} (org ${organization})` : name;
  }, [selectedProject]);
  // With projects listed, an import needs one chosen; without them the server
  // falls back to its default project, so the buttons stay usable.
  const projectMissing = projects.length > 0 && !selectedProjectId;

  const projectLabel = useCallback(
    (project: LangfuseProject) =>
      project.name === project.id ? project.name : `${project.name} (${project.id})`,
    []
  );

  /**
   * Grouped by organisation once there is more than one: across orgs the same
   * project name can legitimately appear twice, and the org is the only thing
   * that tells them apart. With a single org the grouping is noise, so it's flat.
   */
  const projectOptions = useMemo<SelectProps['options']>(() => {
    const organizations = new Set(projects.map((p) => p.organization || ''));
    if (organizations.size <= 1) {
      return projects.map((p) => ({ value: p.id, label: projectLabel(p) }));
    }
    const grouped = new Map<string, { value: string; label: string }[]>();
    projects.forEach((p) => {
      const key = p.organization || 'No organization';
      const bucket = grouped.get(key) ?? [];
      bucket.push({ value: p.id, label: projectLabel(p) });
      grouped.set(key, bucket);
    });
    return Array.from(grouped, ([organization, options]) => ({
      label: organization,
      title: organization,
      options,
    }));
  }, [projects, projectLabel]);

  // The score decides what a "condition" even looks like: a threshold for numeric
  // scores, a category for the rest.
  const selectedScoreName = Form.useWatch('score_name', form);
  const selectedScore = useMemo(
    () => scoreOptions.find((s) => s.name === selectedScoreName),
    [scoreOptions, selectedScoreName]
  );
  const scoreIsNumeric =
    selectedScore?.data_type === 'NUMERIC' || selectedScore?.data_type == null;
  const scoreCategories = selectedScore?.categories ?? [];
  // A queue status narrows a queue, so it stays disabled until one is picked.
  const selectedQueueId = Form.useWatch('annotation_queue_id', form);

  const buildRequest = useCallback((values: FormValues): LangfuseImportRequest => {
    const [from, to] = values.range ?? [];
    return {
      project_id: values.project_id || undefined,
      from_timestamp: from ? from.toISOString() : undefined,
      to_timestamp: to ? to.toISOString() : undefined,
      order_by: values.order_by || undefined,
      model: values.model || undefined,
      score_name: values.score_name || undefined,
      score_source: values.score_source || undefined,
      score_operator: values.score_value != null ? values.score_operator || '>=' : undefined,
      score_value: values.score_value ?? undefined,
      score_string_value: values.score_string_value?.trim() || undefined,
      annotation_queue_id: values.annotation_queue_id || undefined,
      // A status on its own filters nothing server-side; don't record it either.
      annotation_queue_status: values.annotation_queue_id
        ? values.annotation_queue_status || undefined
        : undefined,
      format: values.format,
      fields: values.fields?.length ? values.fields : undefined,
      filename: values.filename?.trim() || undefined,
      max_traces: values.max_traces,
      preview_limit: 10,
    };
  }, []);

  const loadFields = useCallback(
    async (projectId?: string) => {
      setFieldsLoading(true);
      try {
        const resp = await listLangfuseFields(projectId ?? form.getFieldValue('project_id'));
        setAvailableFields(resp.fields);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to load fields';
        setError(msg);
      } finally {
        setFieldsLoading(false);
      }
    },
    [form]
  );

  useEffect(() => {
    const cur = form.getFieldValue('format');
    if (cur === 'custom' && availableFields.length === 0) {
      loadFields();
    }
  }, [form, availableFields.length, loadFields]);

  /**
   * Models are scoped to the selected project and window, so picking one can
   * never yield an empty import. Reloads whenever either changes.
   */
  const loadModels = useCallback(
    async (range?: [Dayjs | null, Dayjs | null], projectId?: string) => {
      const [from, to] = range ?? [];
      setModelsLoading(true);
      try {
        const resp = await listLangfuseModels({
          project_id: projectId ?? form.getFieldValue('project_id'),
          from_timestamp: from ? from.toISOString() : undefined,
          to_timestamp: to ? to.toISOString() : undefined,
        });
        setModels(resp.models);
        setModelsWarning(resp.deployed_filter_applied ? null : resp.warning ?? null);

        // Clear a selection that has no traces in the new window.
        const selected: string | undefined = form.getFieldValue('model');
        if (selected && !resp.models.some((m) => m.id === selected)) {
          form.setFieldValue('model', undefined);
        }
      } catch (e: unknown) {
        setModels([]);
        setModelsWarning(e instanceof Error ? e.message : 'Failed to load models');
      } finally {
        setModelsLoading(false);
      }
    },
    [form]
  );

  /**
   * Annotations (Langfuse scores) and annotation queues are per project, and both
   * are listed with the number of traces they'd keep so an empty filter is
   * visible before it wastes an import.
   */
  const loadAnnotations = useCallback(
    async (projectId?: string) => {
      setAnnotationsLoading(true);
      try {
        const resp = await listLangfuseAnnotations({
          project_id: projectId ?? form.getFieldValue('project_id'),
        });
        setScoreOptions(resp.scores);
        setQueues(resp.queues);
        setAnnotationsWarning(null);

        // Drop a selection this project doesn't have.
        const score: string | undefined = form.getFieldValue('score_name');
        if (score && !resp.scores.some((s) => s.name === score)) {
          form.setFieldValue('score_name', undefined);
          SCORE_CONDITION_FIELDS.forEach((field) => form.setFieldValue(field, undefined));
        }
        const queue: string | undefined = form.getFieldValue('annotation_queue_id');
        if (queue && !resp.queues.some((q) => q.id === queue)) {
          form.setFieldValue('annotation_queue_id', undefined);
          form.setFieldValue('annotation_queue_status', undefined);
        }
      } catch (e: unknown) {
        setScoreOptions([]);
        setQueues([]);
        setAnnotationsWarning(
          e instanceof Error ? e.message : 'Failed to load annotations'
        );
      } finally {
        setAnnotationsLoading(false);
      }
    },
    [form]
  );

  /**
   * The project comes first: traces, models and fields all live inside one, so
   * nothing else can be loaded until we know which one to ask about. The default
   * project is preselected so a single-project deployment needs no extra click.
   */
  const loadProjects = useCallback(async () => {
    setProjectsLoading(true);
    try {
      const resp = await listLangfuseProjects();
      setProjects(resp.projects);
      setProjectsError(null);

      const current: string | undefined = form.getFieldValue('project_id');
      const stillValid = current && resp.projects.some((p) => p.id === current);
      const next = stillValid
        ? current
        : resp.default_project_id ?? resp.projects[0]?.id;
      if (next !== current) {
        form.setFieldValue('project_id', next);
      }
      return next;
    } catch (e: unknown) {
      setProjects([]);
      // Without a project list the page still works against the server's default
      // project, so this is a warning on the field rather than a dead end.
      setProjectsError(e instanceof Error ? e.message : 'Failed to load projects');
      return undefined;
    } finally {
      setProjectsLoading(false);
    }
  }, [form]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const projectId = await loadProjects();
      if (!cancelled) {
        loadModels(form.getFieldValue('range'), projectId);
        loadAnnotations(projectId);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [form, loadProjects, loadModels, loadAnnotations]);

  /** Everything on the page belongs to one project, so switching resets it all. */
  const handleProjectChange = useCallback(
    (projectId?: string) => {
      form.setFieldValue('model', undefined);
      form.setFieldValue('fields', undefined);
      form.setFieldValue('score_name', undefined);
      form.setFieldValue('score_source', undefined);
      form.setFieldValue('annotation_queue_id', undefined);
      form.setFieldValue('annotation_queue_status', undefined);
      SCORE_CONDITION_FIELDS.forEach((field) => form.setFieldValue(field, undefined));
      setAvailableFields([]);
      setModels([]);
      setScoreOptions([]);
      setQueues([]);
      setPreviewRows([]);
      setTelemetry(EMPTY_TELEMETRY);
      setError(null);
      loadModels(form.getFieldValue('range'), projectId);
      loadAnnotations(projectId);
      if (form.getFieldValue('format') === 'custom') {
        loadFields(projectId);
      }
    },
    [form, loadAnnotations, loadFields, loadModels]
  );

  const handlePreview = async () => {
    setError(null);
    setPreviewing(true);
    try {
      const values = await form.validateFields();
      const resp = await previewLangfuse(buildRequest(values));
      setPreviewRows(resp.records);
      setTelemetry({
        scanned: resp.scanned ?? resp.returned,
        skipped: resp.skipped ?? 0,
        returned: resp.returned,
        lastAction: 'preview',
      });
      if (resp.returned === 0) {
        const scanned = resp.scanned ?? 0;
        if (scanned === 0) {
          notify.info({
            message: selectedProjectLabel
              ? `No traces in ${selectedProjectLabel} for the selected time range.`
              : 'No traces in the selected time range.',
          });
        } else {
          notify.warning({
            message: `Scanned ${scanned}, none convertible.`,
            description:
              'Traces exist but have no assistant output. Try format="raw" or widen the range.',
          });
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Preview failed';
      setError(msg);
    } finally {
      setPreviewing(false);
    }
  };

  const handleImport = async () => {
    setError(null);
    setImporting(true);
    try {
      const values = await form.validateFields();
      const resp = await importLangfuse(buildRequest(values));
      setTelemetry({
        scanned: resp.scanned ?? resp.n_records,
        skipped: resp.skipped ?? 0,
        returned: resp.n_records,
        lastAction: 'import',
      });
      notify.success({
        message: 'Import complete',
        description:
          `${resp.filename} — ${resp.n_records} records saved` +
          (resp.project_id ? ` from project ${resp.project_id}` : '') +
          (resp.scanned != null ? ` (scanned ${resp.scanned}, skipped ${resp.skipped ?? 0}).` : '.'),
      });
      setPreviewRows([]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Import failed';
      setError(msg);
    } finally {
      setImporting(false);
    }
  };

  const handleReset = () => {
    form.resetFields();
    setPreviewRows([]);
    setTelemetry(EMPTY_TELEMETRY);
    setError(null);
    setAvailableFields([]);
    // resetFields clears the project too, so pick the default again before
    // reloading anything that depends on it.
    loadProjects().then((projectId) => {
      loadModels(form.getFieldValue('range'), projectId);
      loadAnnotations(projectId);
    });
  };

  const previewCols = useMemo(() => {
    const keys = new Set<string>();
    previewRows.forEach((r) => Object.keys(r ?? {}).forEach((k) => keys.add(k)));
    return Array.from(keys).map((k) => ({
      title: k,
      dataIndex: k,
      key: k,
      // Collapsed to a single line so one record stays one row. The full value
      // lives in the expanded row, so nothing is actually hidden from view.
      ellipsis: true,
      render: (v: unknown) =>
        typeof v === 'string' ? v : <code>{JSON.stringify(v)}</code>,
    }));
  }, [previewRows]);

  return (
    <div style={{ padding: 24, minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space align="center">
              <CloudDownloadOutlined style={{ fontSize: 24 }} />
              <Title level={2} style={{ margin: 0 }}>Import from Langfuse</Title>
            </Space>
            <Paragraph type="secondary" style={{ marginTop: 4, marginBottom: 0 }}>
              Filter Langfuse traces, convert them into a training dataset, and save the JSONL to MinIO.
            </Paragraph>
          </Col>
        </Row>

        {error && (
          <Alert
            type="error"
            message={error}
            closable
            onClose={() => setError(null)}
          />
        )}

        <Card
          title={
            <Space>
              <FilterOutlined /> Filters
            </Space>
          }
          extra={
            <Button size="small" onClick={handleReset} disabled={previewing || importing}>
              Reset
            </Button>
          }
        >
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              format: 'openai_chat',
              range: [dayjs().subtract(7, 'day'), dayjs()],
              max_traces: 1000,
              order_by: 'timestamp.desc',
            }}
            onValuesChange={(changed: Partial<FormValues>) => {
              if ('project_id' in changed) {
                handleProjectChange(changed.project_id);
                return;
              }
              if ('range' in changed) {
                loadModels(changed.range);
              }
            }}
          >
            <Row gutter={16} align="top">
              <Col span={8}>
                <Form.Item
                  label="Project"
                  name="project_id"
                  tooltip="Langfuse keeps traces per project. Everything below is filtered within the project selected here."
                  rules={
                    projects.length > 0
                      ? [{ required: true, message: 'Select a project.' }]
                      : undefined
                  }
                  extra={
                    projectsError ? (
                      <Text type="warning" style={{ fontSize: 12 }}>
                        {projectsError} — using the default project.
                      </Text>
                    ) : undefined
                  }
                >
                  <Select
                    showSearch
                    optionFilterProp="label"
                    loading={projectsLoading}
                    disabled={projectsLoading || projects.length === 0}
                    placeholder={
                      projectsLoading
                        ? 'Loading projects…'
                        : projects.length === 0
                          ? 'Default project'
                          : 'Select a project'
                    }
                    options={projectOptions}
                  />
                </Form.Item>
              </Col>
              <Col span={16}>
                <Form.Item label=" " colon={false}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {selectedProjectLabel
                      ? `Importing traces recorded in ${selectedProjectLabel}. Models and fields below are read from this project only.`
                      : 'Pick the project whose traces you want to turn into a dataset.'}
                  </Text>
                </Form.Item>
              </Col>
            </Row>

            <Divider style={{ margin: '4px 0 16px' }} />

            <Row gutter={16}>
              <Col span={10}>
                <Form.Item
                  label="Timestamp range"
                  name="range"
                  rules={[{ required: true, message: 'Pick a date range.' }]}
                >
                  <RangePicker showTime style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item
                  label="Model"
                  name="model"
                  tooltip="Deployed models with at least one trace in the selected project and range. Leave empty to include every model."
                  extra={
                    modelsWarning ? (
                      <Text type="warning" style={{ fontSize: 12 }}>
                        {modelsWarning}
                      </Text>
                    ) : undefined
                  }
                >
                  <Select
                    allowClear
                    showSearch
                    optionFilterProp="label"
                    loading={modelsLoading}
                    disabled={!modelsLoading && models.length === 0}
                    placeholder={
                      modelsLoading
                        ? 'Loading models…'
                        : models.length === 0
                          ? 'No deployed models with traces'
                          : 'All models'
                    }
                    options={models.map((m) => ({
                      value: m.id,
                      label: `${m.id}  (${m.trace_count} trace${m.trace_count === 1 ? '' : 's'})`,
                    }))}
                  />
                </Form.Item>
              </Col>
              <Col span={4}>
                <Form.Item label="Order by" name="order_by">
                  <Select
                    options={[
                      { value: 'timestamp.desc', label: 'newest first' },
                      { value: 'timestamp.asc', label: 'oldest first' },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={4}>
                <Form.Item label="Max traces" name="max_traces">
                  <InputNumber min={1} max={100000} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Divider style={{ margin: '4px 0 12px' }} />

            <div style={{ marginBottom: 8 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                <b>Annotations</b> — keep only traces a reviewer scored in Langfuse.
                Leave empty to import regardless of review. Counts are traces
                carrying that score in this project.
              </Text>
            </div>

            <Row gutter={16}>
              <Col span={7}>
                <Form.Item
                  label="Annotation score"
                  name="score_name"
                  tooltip="Scores defined for this project plus any recorded ad-hoc. Selecting one keeps only traces that carry it."
                  extra={
                    annotationsWarning ? (
                      <Text type="warning" style={{ fontSize: 12 }}>
                        {annotationsWarning}
                      </Text>
                    ) : selectedScore?.description ? (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {selectedScore.description}
                      </Text>
                    ) : undefined
                  }
                >
                  <Select
                    allowClear
                    showSearch
                    optionFilterProp="label"
                    loading={annotationsLoading}
                    disabled={!annotationsLoading && scoreOptions.length === 0}
                    placeholder={
                      annotationsLoading
                        ? 'Loading annotations…'
                        : scoreOptions.length === 0
                          ? 'No annotations in this project'
                          : 'Any annotation'
                    }
                    // The condition belongs to the score that was selected; keeping
                    // it across a change would silently filter on the wrong scale.
                    onChange={() =>
                      SCORE_CONDITION_FIELDS.forEach((field) =>
                        form.setFieldValue(field, undefined)
                      )
                    }
                    options={scoreOptions.map((s) => ({
                      value: s.name,
                      label:
                        `${s.name}  (${s.trace_count} trace${s.trace_count === 1 ? '' : 's'})` +
                        (s.configured ? '' : ' · ad-hoc'),
                    }))}
                  />
                </Form.Item>
              </Col>
              <Col span={7}>
                {scoreIsNumeric ? (
                  <Form.Item
                    label="Score value"
                    tooltip="Numeric threshold, e.g. ≥ 4. Leave the value empty to accept any score."
                  >
                    <Space.Compact style={{ width: '100%' }}>
                      <Form.Item name="score_operator" noStyle>
                        <Select
                          style={{ width: '38%' }}
                          disabled={!selectedScoreName}
                          placeholder="≥"
                          options={[
                            { value: '>=', label: '≥' },
                            { value: '>', label: '>' },
                            { value: '=', label: '=' },
                            { value: '!=', label: '≠' },
                            { value: '<', label: '<' },
                            { value: '<=', label: '≤' },
                          ]}
                        />
                      </Form.Item>
                      <Form.Item name="score_value" noStyle>
                        <InputNumber
                          style={{ width: '62%' }}
                          disabled={!selectedScoreName}
                          min={selectedScore?.min_value ?? undefined}
                          max={selectedScore?.max_value ?? undefined}
                          placeholder={
                            selectedScore?.min_value != null || selectedScore?.max_value != null
                              ? `${selectedScore?.min_value ?? '−∞'} … ${selectedScore?.max_value ?? '∞'}`
                              : 'Any value'
                          }
                        />
                      </Form.Item>
                    </Space.Compact>
                  </Form.Item>
                ) : (
                  <Form.Item
                    label="Score value"
                    name="score_string_value"
                    tooltip="Category or label the annotator picked. Leave empty to accept any."
                  >
                    {scoreCategories.length > 0 ? (
                      <Select
                        allowClear
                        showSearch
                        optionFilterProp="label"
                        placeholder="Any value"
                        options={scoreCategories.map((c) => ({
                          value: c.label,
                          label: c.value != null ? `${c.label}  (${c.value})` : c.label,
                        }))}
                      />
                    ) : (
                      <Input allowClear placeholder="Any value" />
                    )}
                  </Form.Item>
                )}
              </Col>
              <Col span={5}>
                <Form.Item
                  label="Score source"
                  name="score_source"
                  tooltip="ANNOTATION is a human verdict entered in Langfuse; API and EVAL are written by code."
                >
                  <Select
                    allowClear
                    placeholder="Any source"
                    options={(
                      Object.keys(SCORE_SOURCE_LABELS) as LangfuseScoreSource[]
                    ).map((source) => ({
                      value: source,
                      label: SCORE_SOURCE_LABELS[source],
                    }))}
                  />
                </Form.Item>
              </Col>
              <Col span={5}>
                <Form.Item
                  label="Annotation queue"
                  tooltip="Import only traces queued for review. Status narrows that to items still pending or already completed."
                >
                  <Space.Compact style={{ width: '100%' }}>
                    <Form.Item name="annotation_queue_id" noStyle>
                      <Select
                        allowClear
                        showSearch
                        optionFilterProp="label"
                        style={{ width: '56%' }}
                        loading={annotationsLoading}
                        disabled={!annotationsLoading && queues.length === 0}
                        placeholder={queues.length === 0 ? 'No queues' : 'Any queue'}
                        onChange={(value) => {
                          if (!value) form.setFieldValue('annotation_queue_status', undefined);
                        }}
                        options={queues.map((q) => ({ value: q.id, label: q.name }))}
                      />
                    </Form.Item>
                    <Form.Item name="annotation_queue_status" noStyle>
                      <Select
                        allowClear
                        style={{ width: '44%' }}
                        disabled={!selectedQueueId}
                        placeholder="Any state"
                        options={[
                          { value: 'PENDING', label: 'pending' },
                          { value: 'COMPLETED', label: 'done' },
                        ]}
                      />
                    </Form.Item>
                  </Space.Compact>
                </Form.Item>
              </Col>
            </Row>

            <Divider style={{ margin: '4px 0 16px' }} />

            <Row gutter={16} align="bottom">
              <Col span={8}>
                <Form.Item label="Format" name="format" rules={[{ required: true }]}>
                  <Select
                    onChange={(v) => {
                      if (v === 'custom' && availableFields.length === 0) {
                        loadFields();
                      }
                    }}
                    options={[
                      { value: 'openai_chat', label: 'OpenAI chat  { messages: [ … ] }' },
                      { value: 'raw', label: 'Raw (full trace object)' },
                      { value: 'custom', label: 'Custom fields' },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={10}>
                <Form.Item
                  noStyle
                  shouldUpdate={(prev, next) => prev.format !== next.format}
                >
                  {({ getFieldValue }) =>
                    getFieldValue('format') === 'custom' ? (
                      <Form.Item
                        label="Fields to keep"
                        name="fields"
                        rules={[{ required: true, message: 'Pick at least one field.' }]}
                      >
                        <Select
                          mode="multiple"
                          allowClear
                          placeholder={fieldsLoading ? 'Loading fields…' : 'Select fields'}
                          loading={fieldsLoading}
                          options={availableFields.map((f) => ({ label: f, value: f }))}
                          maxTagCount="responsive"
                        />
                      </Form.Item>
                    ) : (
                      <Form.Item label="Fields" tooltip="Only used for format=custom">
                        <Input disabled placeholder="—" />
                      </Form.Item>
                    )
                  }
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item label="Filename" name="filename">
                  <Input placeholder="auto-generated" />
                </Form.Item>
              </Col>
            </Row>

            <Row justify="end">
              <Space>
                <Button
                  icon={<EyeOutlined />}
                  onClick={handlePreview}
                  loading={previewing}
                  disabled={importing || projectMissing}
                >
                  Preview
                </Button>
                <Button
                  type="primary"
                  icon={<ImportOutlined />}
                  onClick={handleImport}
                  loading={importing}
                  disabled={previewing || projectMissing}
                >
                  Download
                </Button>
              </Space>
            </Row>
          </Form>
        </Card>

        <Card
          title={
            <Space>
              <DatabaseOutlined /> Telemetry
            </Space>
          }
        >
          <Row gutter={16}>
            <Col span={8}>
              <Statistic
                title="Traces scanned"
                value={telemetry.scanned}
                prefix={<DatabaseOutlined />}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Convertible records"
                value={telemetry.returned}
                valueStyle={{ color: telemetry.returned > 0 ? '#3f8600' : undefined }}
                prefix={<FileTextOutlined />}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Skipped (no assistant output)"
                value={telemetry.skipped}
                valueStyle={{ color: telemetry.skipped > 0 ? '#cf1322' : undefined }}
              />
            </Col>
          </Row>
          {telemetry.lastAction === 'import' && telemetry.returned > 0 && (
            <>
              <Divider style={{ margin: '16px 0 12px' }} />
              <Space>
                <Tag color="green">Saved to MinIO</Tag>
                <Text type="secondary">
                  See the new file under{' '}
                  <a onClick={() => router.push('/files')}>Files Management</a>.
                </Text>
              </Space>
            </>
          )}
        </Card>

        <Card
          title={
            <Space>
              <EyeOutlined /> Preview
              {previewRows.length > 0 && <Tag>{previewRows.length} records</Tag>}
            </Space>
          }
        >
          {previewRows.length === 0 ? (
            <Text type="secondary">
              Click <b>Preview</b> above to see how the first records will look after conversion.
            </Text>
          ) : (
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Rows are collapsed to one line. Expand a row to read the complete
                record exactly as it will be written to the JSONL file.
              </Text>
              <Table
                size="small"
                rowKey={(_, i) => String(i)}
                columns={previewCols}
                dataSource={previewRows}
                pagination={false}
                scroll={{ x: true, y: 400 }}
                expandable={{
                  expandedRowRender: (record) => (
                    <pre
                      style={{
                        margin: 0,
                        maxHeight: 360,
                        overflow: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        fontSize: 12,
                      }}
                    >
                      {JSON.stringify(record, null, 2)}
                    </pre>
                  ),
                }}
              />
            </Space>
          )}
        </Card>
      </Space>
    </div>
  );
};

export default LangfusePage;
