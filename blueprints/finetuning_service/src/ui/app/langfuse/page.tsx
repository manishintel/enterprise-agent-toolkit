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
  ExperimentOutlined,
  EyeOutlined,
  FileTextOutlined,
  FilterOutlined,
  FolderOutlined,
  ImportOutlined,
  ReloadOutlined,
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
  LangfuseSystemMode,
  LangfuseImportRequest,
  LangfuseModelOption,
  LangfuseOrganization,
  LangfuseProject,
  LangfuseScoreOption,
  LangfuseScoreSource,
} from '@features/dataprep/types';

const { Title, Text, Paragraph } = Typography;
const { RangePicker } = DatePicker;

/** Sentinel for "don't narrow by organisation" — a Select cannot hold undefined. */
const ALL_ORGS = '__all__';

interface FormValues {
  /** Narrows the project list only; never sent to the server. */
  organization_id?: string;
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
  system_mode: LangfuseSystemMode;
  system_text?: string;
  filename?: string;
}

/**
 * Offered as the replacement system prompt. It has to be the instruction the
 * fine-tuned model will actually be served with, because that is the whole point
 * of replacing: the training examples should look like inference.
 */
const DEFAULT_SYSTEM_TEXT =
  'You are a helpful assistant. Answer the question clearly and concisely.';

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
  /** The max_traces that was in force, so a truncation can name its own limit. */
  cap: number | null;
  /**
   * The scan stopped on the cap rather than running out of traces, so there are
   * probably more. Worth saying while Max traces can still be raised, since
   * generating with the same value would write the same truncated dataset.
   */
  capped: boolean;
  lastAction: 'preview' | 'generate' | null;
}

const EMPTY_TELEMETRY: Telemetry = {
  scanned: 0,
  skipped: 0,
  returned: 0,
  cap: null,
  capped: false,
  lastAction: null,
};

/**
 * The dataset the last run produced. Generating it is only half the job — the
 * file sits in object storage until a fine-tuning job is submitted against it,
 * and nothing on this page does that — so the result is kept in state to offer
 * the way there.
 */
interface GeneratedDataset {
  fileId: string;
  filename: string;
  records: number;
}

const LangfusePage: React.FC = () => {
  const router = useRouter();
  const [form] = Form.useForm<FormValues>();
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry>(EMPTY_TELEMETRY);
  const [previewing, setPreviewing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generated, setGenerated] = useState<GeneratedDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableFields, setAvailableFields] = useState<string[]>([]);
  const [fieldsLoading, setFieldsLoading] = useState(false);
  const [models, setModels] = useState<LangfuseModelOption[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsWarning, setModelsWarning] = useState<string | null>(null);
  const [projects, setProjects] = useState<LangfuseProject[]>([]);
  const [organizations, setOrganizations] = useState<LangfuseOrganization[]>([]);
  const [canProvision, setCanProvision] = useState(false);
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
    (project: LangfuseProject) => {
      const base =
        project.name === project.id ? project.name : `${project.name} (${project.id})`;
      // Worth saying out loud: the choice works, but the first import in that
      // project has a key creation behind it, and if provisioning is off it
      // cannot work at all.
      if (project.has_credentials === false) {
        return canProvision ? `${base} — key on first use` : `${base} — no API key`;
      }
      return base;
    },
    [canProvision]
  );

  const orgName = useCallback(
    (organization?: string | null, organizationId?: string | null) =>
      organization || organizationId || 'Unnamed organisation',
    []
  );

  const selectedOrgId = Form.useWatch('organization_id', form);

  /**
   * The organisation filter. Offered whenever Langfuse reports more than one:
   * with a single org it is a control with one choice, which is just noise.
   */
  const organizationOptions = useMemo<SelectProps['options']>(() => {
    if (organizations.length <= 1) return [];
    return [
      { value: ALL_ORGS, label: `All organisations (${projects.length})` },
      ...organizations.map((o) => ({
        value: o.id ?? '',
        label: `${orgName(o.name, o.id)} (${o.project_count})`,
      })),
    ];
  }, [organizations, projects.length, orgName]);

  const hasOrgFilter = (organizationOptions?.length ?? 0) > 0;

  const visibleProjects = useMemo(
    () =>
      !selectedOrgId || selectedOrgId === ALL_ORGS
        ? projects
        : projects.filter((p) => (p.organization_id ?? '') === selectedOrgId),
    [projects, selectedOrgId]
  );

  /**
   * Grouped by organisation once there is more than one and none is selected:
   * across orgs the same project name can legitimately appear twice, and the org
   * is the only thing that tells them apart. Inside one org the grouping repeats
   * the filter above, so it's flat.
   */
  const projectOptions = useMemo<SelectProps['options']>(() => {
    const toOption = (p: LangfuseProject) => ({
      value: p.id,
      label: projectLabel(p),
      // Listed but unusable: better than hiding it, which reads as "Langfuse
      // lost my project" rather than "this one needs a key".
      disabled: p.has_credentials === false && !canProvision,
    });
    const orgs = new Set(visibleProjects.map((p) => p.organization_id ?? ''));
    if (orgs.size <= 1) {
      return visibleProjects.map(toOption);
    }
    const grouped = new Map<string, ReturnType<typeof toOption>[]>();
    visibleProjects.forEach((p) => {
      const key = orgName(p.organization, p.organization_id);
      const bucket = grouped.get(key) ?? [];
      bucket.push(toOption(p));
      grouped.set(key, bucket);
    });
    return Array.from(grouped, ([organization, options]) => ({
      label: organization,
      title: organization,
      options,
    }));
  }, [visibleProjects, projectLabel, canProvision, orgName]);

  /** The line under the project field, including why a project may not be readable. */
  const projectHint = useMemo(() => {
    if (!selectedProject || !selectedProjectLabel) {
      return 'Pick the project whose traces you want to turn into a dataset.';
    }
    if (selectedProject.has_credentials === false) {
      return canProvision
        ? `${selectedProjectLabel} was found through its organisation and has no API key here yet. One is created for it on first use, then models and fields below will fill in.`
        : `${selectedProjectLabel} was found through its organisation, but reading its traces needs a project API key that is not configured. Add it as LANGFUSE_PROJECT_KEYS, or enable automatic key creation.`;
    }
    return `Importing traces recorded in ${selectedProjectLabel}. Models and fields below are read from this project only.`;
  }, [selectedProject, selectedProjectLabel, canProvision]);

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
      // Only openai_chat has messages to rewrite, and the server rejects
      // replace without text, so a half-filled form must not send the mode.
      system_mode:
        values.format === 'openai_chat' &&
        (values.system_mode !== 'replace' || !!values.system_text?.trim())
          ? values.system_mode
          : undefined,
      system_text:
        values.format === 'openai_chat' && values.system_mode === 'replace'
          ? values.system_text?.trim() || undefined
          : undefined,
      filename: values.filename?.trim() || undefined,
      // The one bound on both preview and generate, so a preview shows the
      // dataset that generating would write rather than a sample of it.
      max_traces: values.max_traces,
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
  const loadProjects = useCallback(
    async (refresh = false) => {
      setProjectsLoading(true);
      try {
        const resp = await listLangfuseProjects(refresh);
        setProjects(resp.projects);
        setOrganizations(resp.organizations ?? []);
        setCanProvision(resp.can_provision ?? false);
        setProjectsError(null);

        const current: string | undefined = form.getFieldValue('project_id');
        const stillValid = current && resp.projects.some((p) => p.id === current);
        const next = stillValid
          ? current
          : resp.default_project_id ?? resp.projects[0]?.id;
        if (next !== current) {
          form.setFieldValue('project_id', next);
        }
        // The organisation follows the project rather than the other way round:
        // the project is what everything else is scoped to, and starting on "all"
        // would hide which org the preselected one belongs to.
        const selected = resp.projects.find((p) => p.id === next);
        form.setFieldValue('organization_id', selected?.organization_id ?? ALL_ORGS);
        return next;
      } catch (e: unknown) {
        setProjects([]);
        setOrganizations([]);
        // Without a project list the page still works against the server's default
        // project, so this is a warning on the field rather than a dead end.
        setProjectsError(e instanceof Error ? e.message : 'Failed to load projects');
        return undefined;
      } finally {
        setProjectsLoading(false);
      }
    },
    [form]
  );

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
      setGenerated(null);
      setError(null);
      loadModels(form.getFieldValue('range'), projectId);
      loadAnnotations(projectId);
      if (form.getFieldValue('format') === 'custom') {
        loadFields(projectId);
      }
    },
    [form, loadAnnotations, loadFields, loadModels]
  );

  /**
   * Narrowing to an organisation moves the project too when the current one is
   * outside it, so the two controls can never disagree about what is being read.
   */
  const handleOrganizationChange = useCallback(
    (organizationId?: string) => {
      const current: string | undefined = form.getFieldValue('project_id');
      const inScope =
        !organizationId || organizationId === ALL_ORGS
          ? projects
          : projects.filter((p) => (p.organization_id ?? '') === organizationId);
      if (current && inScope.some((p) => p.id === current)) return;
      const next = inScope.find((p) => p.has_credentials !== false) ?? inScope[0];
      form.setFieldValue('project_id', next?.id);
      handleProjectChange(next?.id);
    },
    [form, projects, handleProjectChange]
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
        cap: resp.cap ?? values.max_traces ?? null,
        capped: !!resp.capped,
        lastAction: 'preview',
      });
      if (resp.capped) {
        notify.warning({
          message: `Stopped at Max traces (${resp.cap ?? values.max_traces}).`,
          description:
            'There are probably more traces than this. Raise Max traces and preview ' +
            'again to see them all — generating now would write this same subset.',
        });
      }
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

  const handleGenerateDataset = async () => {
    setError(null);
    setGenerating(true);
    try {
      const values = await form.validateFields();
      const resp = await importLangfuse(buildRequest(values));
      setTelemetry({
        scanned: resp.scanned ?? resp.n_records,
        skipped: resp.skipped ?? 0,
        returned: resp.n_records,
        cap: resp.cap ?? values.max_traces ?? null,
        capped: !!resp.capped,
        lastAction: 'generate',
      });
      setGenerated({
        fileId: resp.file_id,
        filename: resp.filename,
        records: resp.n_records,
      });
      notify.success({
        message: 'Dataset generated',
        description:
          `${resp.filename} — ${resp.n_records} records saved` +
          (resp.project_id ? ` from project ${resp.project_id}` : '') +
          (resp.scanned != null ? ` (scanned ${resp.scanned}, skipped ${resp.skipped ?? 0}).` : '.'),
      });
      if (resp.capped) {
        notify.warning({
          message: `Max traces (${resp.cap ?? values.max_traces}) was reached.`,
          description:
            'The dataset stops there and there are probably more traces. Raise ' +
            'Max traces and generate again to include them.',
        });
      }
      setPreviewRows([]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Generating the dataset failed';
      setError(msg);
    } finally {
      setGenerating(false);
    }
  };

  /**
   * Hand the new dataset straight to the fine-tuning form rather than only
   * naming it: the file id is the one thing the user would otherwise have to
   * copy across by hand.
   */
  const handleContinueToFineTuning = () => {
    if (!generated) return;
    router.push(`/finetuning/new?training_file=${encodeURIComponent(generated.fileId)}`);
  };

  const handleReset = () => {
    form.resetFields();
    setPreviewRows([]);
    setTelemetry(EMPTY_TELEMETRY);
    setGenerated(null);
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
      <Space orientation="vertical" size="large" style={{ width: '100%' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space align="center">
              <CloudDownloadOutlined style={{ fontSize: 24 }} />
              <Title level={2} style={{ margin: 0 }}>Import from Langfuse</Title>
            </Space>
            <Paragraph type="secondary" style={{ marginTop: 4, marginBottom: 0 }}>
              Filter Langfuse traces and generate a training dataset from them, saved as
              JSONL in object storage. Fine-tuning on the dataset is the next step, on the
              Fine-Tuning page.
            </Paragraph>
          </Col>
        </Row>

        {error && (
          <Alert
            type="error"
            title={error}
            closable
            onClose={() => setError(null)}
          />
        )}

        {generated && (
          <Alert
            type="success"
            showIcon
            title={`Dataset generated — ${generated.filename}`}
            description={
              <Space orientation="vertical" size="small">
                <Text>
                  {generated.records} record{generated.records === 1 ? '' : 's'} saved as{' '}
                  <Text code copyable>{generated.fileId}</Text>. Generating the dataset does
                  not start any training: continue to Fine-Tuning to submit a job against it.
                </Text>
                <Space wrap>
                  <Button
                    type="primary"
                    icon={<ExperimentOutlined />}
                    onClick={handleContinueToFineTuning}
                  >
                    Continue to Fine-Tuning
                  </Button>
                  <Button icon={<FolderOutlined />} onClick={() => router.push('/files')}>
                    View in Files
                  </Button>
                </Space>
              </Space>
            }
            closable
            onClose={() => setGenerated(null)}
          />
        )}

        <Card
          title={
            <Space>
              <FilterOutlined /> Filters
            </Space>
          }
          extra={
            <Space>
              {/* Projects are cached for a few minutes, so one created in Langfuse
                  a moment ago needs this rather than a wait. */}
              <Button
                size="small"
                icon={<ReloadOutlined />}
                loading={projectsLoading}
                onClick={() => loadProjects(true)}
                disabled={previewing || generating}
              >
                Refresh projects
              </Button>
              <Button size="small" onClick={handleReset} disabled={previewing || generating}>
                Reset
              </Button>
            </Space>
          }
        >
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              organization_id: ALL_ORGS,
              format: 'openai_chat',
              // Keep, so an import behaves as it always has until asked otherwise.
              system_mode: 'keep',
              system_text: DEFAULT_SYSTEM_TEXT,
              range: [dayjs().subtract(7, 'day'), dayjs()],
              max_traces: 1000,
              order_by: 'timestamp.desc',
            }}
            onValuesChange={(changed: Partial<FormValues>) => {
              if ('organization_id' in changed) {
                handleOrganizationChange(changed.organization_id);
                return;
              }
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
              {hasOrgFilter && (
                <Col span={7}>
                  <Form.Item
                    label="Organisation"
                    name="organization_id"
                    tooltip="Narrows the project list below. Projects are unique to an organisation, so this only decides which ones you can choose from."
                  >
                    <Select
                      showSearch
                      optionFilterProp="label"
                      loading={projectsLoading}
                      options={organizationOptions}
                      placeholder="All organisations"
                    />
                  </Form.Item>
                </Col>
              )}
              <Col span={hasOrgFilter ? 9 : 8}>
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
                          : visibleProjects.length === 0
                            ? 'No projects in this organisation'
                            : 'Select a project'
                    }
                    options={projectOptions}
                  />
                </Form.Item>
              </Col>
              <Col span={hasOrgFilter ? 8 : 16}>
                <Form.Item label=" " colon={false}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {projectHint}
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
                <Form.Item
                  label="Max traces"
                  name="max_traces"
                  tooltip={
                    'The only limit on either button. Preview shows every matching ' +
                    'trace up to this many, and Generate writes the same set — so ' +
                    'raise it if a run reports that it stopped here.'
                  }
                >
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

            {/* The system turn is recorded exactly as it was sent, which is wrong
                for any request that carried context the served model will not
                have. Only openai_chat has messages, so the row hides otherwise. */}
            <Form.Item
              noStyle
              shouldUpdate={(prev, next) =>
                prev.format !== next.format || prev.system_mode !== next.system_mode
              }
            >
              {({ getFieldValue }) =>
                getFieldValue('format') !== 'openai_chat' ? null : (
                  <Row gutter={16} align="bottom">
                    <Col span={8}>
                      <Form.Item
                        label="System prompt"
                        name="system_mode"
                        tooltip="Traces record the system prompt that was sent. Replace it when those requests carried per-request context — a retrieved document, a policy excerpt — that the fine-tuned model will not have at inference, or it learns to copy the answer out of its context instead of remembering it."
                      >
                        <Select
                          options={[
                            { value: 'keep', label: 'Keep as traced' },
                            { value: 'drop', label: 'Drop it' },
                            { value: 'replace', label: 'Replace with…' },
                          ]}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={16}>
                      {getFieldValue('system_mode') === 'replace' ? (
                        <Form.Item
                          label="Replacement system prompt"
                          name="system_text"
                          rules={[
                            {
                              required: true,
                              whitespace: true,
                              message: 'Give the system prompt to substitute, or choose Drop.',
                            },
                          ]}
                        >
                          <Input.TextArea
                            rows={2}
                            maxLength={4000}
                            showCount
                            placeholder={DEFAULT_SYSTEM_TEXT}
                          />
                        </Form.Item>
                      ) : (
                        <Form.Item label=" " colon={false}>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {getFieldValue('system_mode') === 'drop'
                              ? 'Every system turn is removed, leaving question and answer only.'
                              : 'The system prompt from each traced request is kept in the dataset as-is.'}
                          </Text>
                        </Form.Item>
                      )}
                    </Col>
                  </Row>
                )
              }
            </Form.Item>

            <Row justify="end">
              <Space>
                <Button
                  icon={<EyeOutlined />}
                  onClick={handlePreview}
                  loading={previewing}
                  disabled={generating || projectMissing}
                >
                  Preview
                </Button>
                <Button
                  type="primary"
                  icon={<ImportOutlined />}
                  onClick={handleGenerateDataset}
                  loading={generating}
                  disabled={previewing || projectMissing}
                >
                  {generating ? 'Generating Dataset…' : 'Generate Dataset'}
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
          {telemetry.capped && (
            <>
              <Divider style={{ margin: '16px 0 12px' }} />
              <Space>
                <Tag color="orange">Truncated at Max traces</Tag>
                <Text type="secondary">
                  The scan stopped on the {telemetry.cap} trace limit, so there are
                  likely more to import. Raise <b>Max traces</b> and run it again.
                </Text>
              </Space>
            </>
          )}
          {telemetry.lastAction === 'generate' && telemetry.returned > 0 && (
            <>
              <Divider style={{ margin: '16px 0 12px' }} />
              <Space>
                <Tag color="green">Dataset generated</Tag>
                <Text type="secondary">
                  Saved to object storage — see it under{' '}
                  <a onClick={() => router.push('/files')}>Files Management</a>, or{' '}
                  <a onClick={handleContinueToFineTuning}>fine-tune on it</a>.
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
              Click <b>Preview</b> above to see every record the filters select, up
              to <b>Max traces</b>, exactly as they will be converted.
            </Text>
          ) : (
            <Space orientation="vertical" size="small" style={{ width: '100%' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Every record the dataset would contain, not a sample. Rows are
                collapsed to one line — expand one to read the complete record
                exactly as it will be written to the JSONL file.
              </Text>
              <Table
                size="small"
                rowKey={(_, i) => String(i)}
                columns={previewCols}
                dataSource={previewRows}
                // Paged because a preview is now the whole dataset: thousands of
                // expandable rows in one DOM tree is what would make the page crawl.
                pagination={{
                  defaultPageSize: 20,
                  showSizeChanger: true,
                  pageSizeOptions: ['20', '50', '100'],
                  showTotal: (total, [start, end]) =>
                    `${start}-${end} of ${total} records`,
                }}
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
