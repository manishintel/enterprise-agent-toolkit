'use client';

/**
 * Semantic routing setup: pick a router, collect example questions, set how close a
 * match has to be, apply, and watch the change go live.
 *
 * This is the body of what used to be a modal, lifted out so it can be a tab on
 * the deployment page. The work is a review a user iterates on -- mine, prune,
 * test, adjust, test again -- which a dialog is a poor container for: it steals the
 * page, cannot be linked to, and encourages a "just press Apply" reading of what is
 * really a calibration task. It is laid out as four numbered cards for the same
 * reason: the order matters, and a flat stack of controls hid that.
 *
 * Six things this has to make clear, because none of them is guessable from the
 * controls alone.
 *
 * **Routing is opt-in by model name.** The router is itself a model; callers have
 * to address it to be routed. Setting this up changes nothing for traffic still
 * addressed to a specific model, so the router's name is shown as the thing to
 * point clients at.
 *
 * **Which router.** More than one can exist, so the router is chosen here rather
 * than assumed. Adding to an existing one puts this model in competition with the
 * routes already in it; a new one keeps unrelated sets apart. Both are one click.
 *
 * **The working set is the user's, not the extractor's.** Mining adds to the list
 * and never overwrites it: anything typed by hand survives a second extraction, and
 * what gets applied is whatever is *selected*, so pruning is a checkbox rather than a
 * delete. Losing hand-written examples to a stray click on Extract is the one thing
 * this screen must not do.
 *
 * **Similarity is shown as a percentage**, because 0.52 on an unlabelled 0-1 scale
 * told users nothing. The number is the gateway's cosine similarity times 100 --
 * a relabelling, not a different measure -- so it is honest arithmetic, but it is
 * *not* a percentage of words in common: unrelated text lands around 35-45%, not 0%.
 * The scale is annotated with that floor so the middle of the slider is not read as
 * "half a match".
 *
 * **The score is a mean, not a best match.** The router averages the similarity of
 * the nearest few examples, which reads below the closest one. Both are shown, with
 * the mean as the number the threshold applies to, or a user calibrates against the
 * wrong figure.
 *
 * **Applying takes a minute, visibly.** An auto-router is cached in the gateway
 * process, so the change only lands once the gateway has restarted. Rather than
 * warning about that up front and then going silent, the wait is shown as it
 * happens -- saved, restarting, live -- and confirmed when the route is actually
 * serving.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Empty,
  Input,
  InputNumber,
  Progress,
  Radio,
  Select,
  Slider,
  Space,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import {
  CheckCircleFilled,
  ClockCircleOutlined,
  CloseOutlined,
  DownOutlined,
  EditOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
  PlusOutlined,
  SearchOutlined,
  SettingOutlined,
  StopOutlined,
  ThunderboltOutlined,
  UpOutlined,
} from '@ant-design/icons';
import type {
  ExtractUtterancesResponse,
  GatewayReadiness,
  SemanticRouteStatus,
  SemanticRouteTestResponse,
} from '@features/finetuning/types';

const { Text } = Typography;

/** Sentinel for the "create a new router" entry in the picker. */
const NEW_ROUTER = '__new__';

/**
 * Similarity is held as a whole percent in this component and divided by 100 on the
 * way out. The API and the gateway both speak 0-1; converting at the boundary keeps
 * one representation on screen and avoids 0.51999999 in a label.
 */
const PCT_MIN = 20;
const PCT_MAX = 90;
const DEFAULT_PCT = 50;

/**
 * Roughly where unrelated text lands with a sentence encoder. Not a constant of
 * nature -- it moves with the embedding model -- but close enough to stop the scale
 * being read as "0% means nothing in common", which is the one reading that makes a
 * percentage misleading here.
 */
const NOISE_FLOOR_PCT = 40;

/** A change being rolled out, so the wait can be shown instead of guessed at. */
export interface RouteProgress {
  startedAt: number;
  /** False for a removal, where the change has landed once the route is gone. */
  expectRoute: boolean;
  routerName: string;
}

export interface SemanticRoutingPanelProps {
  status?: SemanticRouteStatus;
  statusLoading: boolean;
  /** Router being viewed. Undefined means the installation default. */
  routerName?: string;
  onRouterChange: (routerName?: string) => void;
  extraction?: ExtractUtterancesResponse;
  extracting: boolean;
  testResult?: SemanticRouteTestResponse;
  testing: boolean;
  applying: boolean;
  removing: boolean;
  progress?: RouteProgress | null;
  readiness?: GatewayReadiness;
  onDismissProgress: () => void;
  onExtract: (options: { limit: number; first_turn_only: boolean; redact_pii: boolean }) => void;
  /** Threshold is passed as the 0-1 fraction the API expects, not the percent shown. */
  onTest: (query: string, utterances: string[], threshold: number) => void;
  onApply: (utterances: string[], threshold: number, router: string) => void;
  onRemove: () => void;
}

/** Where an entry in the working set came from, which is worth showing per row. */
type UtteranceSource = 'applied' | 'dataset' | 'manual';

interface WorkingUtterance {
  /** The normalised text: unique by construction, so it doubles as the row key. */
  key: string;
  text: string;
  source: UtteranceSource;
  /** How many near-duplicate phrasings in the dataset this one stands for. */
  represents?: number;
}

/**
 * How a row's origin is marked.
 *
 * A dot and a word rather than a coloured chip: the table tints selected rows blue,
 * and chips in blue and green on top of that tint made the busiest column the least
 * readable one. A 6px dot carries the same information at a fraction of the ink.
 */
const SOURCE_TAG: Record<UtteranceSource, { dot: string; label: string; hint: string }> = {
  applied: { dot: '#52c41a', label: 'Live', hint: 'Already applied to this router' },
  dataset: { dot: '#8c8c8c', label: 'From data', hint: 'Found in the training dataset' },
  manual: { dot: '#722ed1', label: 'Yours', hint: 'You added or edited this' },
};

/** Human-readable labels for the funnel counters. */
const DROP_LABELS: Record<string, string> = {
  too_short: 'too short',
  too_long: 'too long',
  pleasantry: 'greetings only',
  empty: 'empty',
  duplicate: 'exact duplicates',
  near_duplicate: 'near duplicates',
};

/**
 * The form two utterances are compared in.
 *
 * Case and punctuation should not decide whether something is a duplicate: mining
 * twice, or pasting in a question that is already listed, would otherwise add a
 * second copy that quietly takes a slot in the applied set.
 */
const norm = (text: string): string =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

/** What a given threshold means in practice, in one line, at the value chosen. */
function thresholdAdvice(pct: number): { tone: 'secondary' | 'warning'; text: string } {
  if (pct <= NOISE_FLOOR_PCT) {
    return {
      tone: 'warning',
      text:
        `Very loose. Unrelated questions typically score ${NOISE_FLOOR_PCT - 5}-${NOISE_FLOOR_PCT + 5}%, ` +
        'so at this setting some traffic that has nothing to do with this model will be routed to it.',
    };
  }
  if (pct <= 55) {
    return {
      tone: 'secondary',
      text: 'Loose. Catches loosely related questions too — good when missing a real question costs more than answering an odd one.',
    };
  }
  if (pct <= 70) {
    return {
      tone: 'secondary',
      text: 'Strict. Only clearly on-topic questions are routed; borderline ones go to the fallback model.',
    };
  }
  return {
    tone: 'warning',
    text:
      'Very strict. Because the score is an average over the nearest few examples, real questions ' +
      'rarely reach this high — expect almost everything to fall through to the fallback model.',
  };
}

/** A numbered marker, so the four cards read as an order rather than a pile. */
function StepBadge({ n }: { n: number }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 22,
        height: 22,
        borderRadius: '50%',
        background: '#1677ff',
        color: '#fff',
        fontSize: 12,
        fontWeight: 600,
        flex: '0 0 auto',
      }}
    >
      {n}
    </span>
  );
}

export default function SemanticRoutingPanel({
  status,
  statusLoading,
  onRouterChange,
  extraction,
  extracting,
  testResult,
  testing,
  applying,
  removing,
  progress,
  readiness,
  onDismissProgress,
  onExtract,
  onTest,
  onApply,
  onRemove,
}: SemanticRoutingPanelProps) {
  const [limit, setLimit] = useState<number>(30);
  const [firstTurnOnly, setFirstTurnOnly] = useState(true);
  const [redact, setRedact] = useState(true);
  const [showOptions, setShowOptions] = useState(false);
  const [thresholdPct, setThresholdPct] = useState<number>(DEFAULT_PCT);
  const [items, setItems] = useState<WorkingUtterance[]>([]);
  const [selected, setSelected] = useState<React.Key[]>([]);
  const [added, setAdded] = useState('');
  const [filter, setFilter] = useState('');
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [query, setQuery] = useState('');
  const [mergeNote, setMergeNote] = useState<string | null>(null);
  const [newRouterName, setNewRouterName] = useState('');
  const [creatingRouter, setCreatingRouter] = useState(false);

  const applied = status?.this_route;
  const routers = status?.available_routers ?? [];
  const currentRouter = routers.find((r) => r.name === status?.router_name);

  // Everything below is scoped to one router, so the name has to resolve before
  // anything can be applied: either the one being viewed, or the one being named.
  const effectiveRouter = creatingRouter ? newRouterName.trim() : status?.router_name || '';
  const nameError =
    creatingRouter && newRouterName.trim() && !/^[A-Za-z0-9][A-Za-z0-9._-]{1,119}$/.test(newRouterName.trim())
      ? 'Letters, digits, dot, dash and underscore only, starting with a letter or digit.'
      : routers.some((r) => r.name === newRouterName.trim()) && creatingRouter
        ? 'A router with that name already exists — pick it from the list instead.'
        : null;

  // Start from whatever is already applied, so opening this on a configured route
  // is an edit rather than a blank slate. Keyed on the utterances themselves: the
  // status object is replaced on every poll, and re-running this on each one would
  // undo the user's edits under them.
  // Serialised rather than joined on a separator, so an utterance that happens to
  // contain the separator cannot come back out as two.
  const appliedSignature = JSON.stringify(applied?.utterances ?? []);
  useEffect(() => {
    const texts = JSON.parse(appliedSignature) as string[];
    if (!texts.length) return;
    const seen = new Map<string, WorkingUtterance>();
    texts.forEach((text) => {
      const key = norm(text);
      if (key && !seen.has(key)) seen.set(key, { key, text, source: 'applied' });
    });
    setItems(Array.from(seen.values()));
    setSelected(Array.from(seen.keys()));
  }, [appliedSignature]);

  useEffect(() => {
    if (applied?.score_threshold != null) {
      setThresholdPct(Math.round(applied.score_threshold * 100));
    }
  }, [applied?.score_threshold]);

  // Mining *adds*. Replacing the list here was the single worst thing this screen
  // did: a second Extract silently deleted every hand-written example, with no
  // undo. Duplicates are skipped on normalised text and reported, so pressing it
  // twice is a safe no-op rather than a surprise.
  const lastExtraction = useRef<ExtractUtterancesResponse | undefined>(undefined);
  useEffect(() => {
    if (!extraction || extraction === lastExtraction.current) return;
    lastExtraction.current = extraction;

    const existing = new Set(items.map((i) => i.key));
    const additions: WorkingUtterance[] = [];
    let duplicates = 0;
    extraction.utterances.forEach((u) => {
      const key = norm(u.text);
      if (!key) return;
      if (existing.has(key)) {
        duplicates += 1;
        return;
      }
      existing.add(key);
      additions.push({ key, text: u.text, source: 'dataset', represents: u.represents });
    });

    if (additions.length) {
      setItems([...items, ...additions]);
      setSelected([...selected, ...additions.map((a) => a.key)]);
    }
    setMergeNote(
      additions.length
        ? `Added ${additions.length} new example${additions.length === 1 ? '' : 's'}` +
            (duplicates ? `, skipped ${duplicates} already in your list.` : '.') +
            ` Your list now has ${items.length + additions.length}.`
        : `Nothing new — all ${duplicates} examples found are already in your list.`
    );
  }, [extraction, items, selected]);

  const report = extraction?.report;
  const dropped = useMemo(
    () => Object.entries(report?.dropped ?? {}).filter(([, count]) => count > 0),
    [report]
  );
  const redacted = useMemo(
    () => Object.entries(report?.redacted ?? {}).filter(([, count]) => count > 0),
    [report]
  );

  const selectedTexts = useMemo(() => {
    const keys = new Set(selected.map(String));
    return items.filter((i) => keys.has(i.key)).map((i) => i.text);
  }, [items, selected]);

  // Filtering is display-only: selection and deletion work on keys, so a filtered
  // view cannot silently narrow what gets applied.
  const visibleItems = useMemo(() => {
    const needle = norm(filter);
    if (!needle) return items;
    return items.filter((i) => i.key.includes(needle));
  }, [items, filter]);

  // Whether the ticked set differs from what the router is actually serving. Worth
  // stating: the list is seeded from the live route, so an unchanged screen and a
  // screen with pending edits otherwise look identical.
  const dirty = useMemo(() => {
    if (!applied) return selectedTexts.length > 0;
    const live = [...applied.utterances].map(norm).sort();
    const next = [...selectedTexts].map(norm).sort();
    const sameThreshold =
      applied.score_threshold == null ||
      Math.round(applied.score_threshold * 100) === thresholdPct;
    return !sameThreshold || live.length !== next.length || live.some((t, i) => t !== next[i]);
  }, [applied, selectedTexts, thresholdPct]);

  const blocked = !!status && (!status.available || !!status.message);
  const canApply =
    !applying && !progress && selectedTexts.length > 0 && !blocked && !!effectiveRouter && !nameError;

  const removeSelected = () => {
    const keys = new Set(selected.map(String));
    setItems((current) => current.filter((i) => !keys.has(i.key)));
    setSelected([]);
  };

  const removeOne = (key: string) => {
    setItems((current) => current.filter((i) => i.key !== key));
    setSelected((current) => current.filter((k) => String(k) !== key));
  };

  const startEdit = (row: WorkingUtterance) => {
    setEditingKey(row.key);
    setEditText(row.text);
  };

  /**
   * Save an edited example.
   *
   * The row key is the normalised text, so editing changes it -- selection has to be
   * carried across, or a ticked row would come back unticked and quietly drop out of
   * the applied set. Editing one example into the wording of another merges them
   * rather than creating a duplicate.
   */
  const commitEdit = (row: WorkingUtterance) => {
    const text = editText.trim();
    setEditingKey(null);
    if (!text || text === row.text) return;
    const key = norm(text);
    if (!key) return;
    if (key !== row.key && items.some((i) => i.key === key)) {
      setItems(items.filter((i) => i.key !== row.key));
      setSelected(selected.filter((k) => String(k) !== row.key));
      setMergeNote('That wording is already in the list, so the two were merged.');
      return;
    }
    setItems(items.map((i) => (i.key === row.key ? { ...i, key, text, source: 'manual' } : i)));
    setSelected(selected.map((k) => (String(k) === row.key ? key : k)));
  };

  // A paste of several lines is several examples, which is how anyone with a list
  // in a file or a spreadsheet will arrive here.
  const addTyped = () => {
    const lines = added
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    if (!lines.length) return;
    const existing = new Set(items.map((i) => i.key));
    const additions: WorkingUtterance[] = [];
    lines.forEach((text) => {
      const key = norm(text);
      if (!key || existing.has(key)) return;
      existing.add(key);
      additions.push({ key, text, source: 'manual' });
    });
    if (additions.length) {
      setItems([...items, ...additions]);
      setSelected([...selected, ...additions.map((a) => a.key)]);
    }
    setAdded('');
    setMergeNote(
      additions.length
        ? `Added ${additions.length} example${additions.length === 1 ? '' : 's'}.`
        : 'Those are already in your list.'
    );
  };

  const advice = thresholdAdvice(thresholdPct);

  if (statusLoading && !status) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="Reading the gateway..." />
      </div>
    );
  }

  return (
    <Space orientation="vertical" style={{ width: '100%' }} size={16}>
      {status && !status.available && (
        <Alert
          type="warning"
          showIcon
          title="Semantic routing is unavailable"
          description={status.message}
        />
      )}
      {status?.available && status.message && (
        <Alert type="warning" showIcon title="Not ready yet" description={status.message} />
      )}

      {/* The live thing on the screen goes first, not buried under the form. */}
      {progress && (
        <RouteProgressCard progress={progress} readiness={readiness} onDismiss={onDismissProgress} />
      )}

      {/* --- 1 · Router ---------------------------------------------------- */}
      <Card
        size="small"
        title={
          <Space size={8}>
            <StepBadge n={1} />
            <span>Router</span>
          </Space>
        }
        extra={
          currentRouter ? (
            <Space size={4}>
              {currentRouter.is_default && <Tag color="blue">default</Tag>}
              <Tag>
                {currentRouter.routes} route{currentRouter.routes === 1 ? '' : 's'}
              </Tag>
            </Space>
          ) : creatingRouter ? (
            <Tag color="orange">new</Tag>
          ) : null
        }
      >
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          <Space wrap align="start" size={12}>
            <Select
              value={creatingRouter ? NEW_ROUTER : status?.router_name}
              style={{ minWidth: 320 }}
              onChange={(value) => {
                if (value === NEW_ROUTER) {
                  setCreatingRouter(true);
                  setNewRouterName('');
                } else {
                  setCreatingRouter(false);
                  onRouterChange(value);
                }
              }}
              options={[
                ...routers.map((r) => ({
                  value: r.name,
                  label: (
                    <Space size={4}>
                      <span>{r.name}</span>
                      {r.is_default && <Tag color="blue">default</Tag>}
                      <Tag>
                        {r.routes} route{r.routes === 1 ? '' : 's'}
                      </Tag>
                      {r.has_this_model && <Tag color="green">this model</Tag>}
                    </Space>
                  ),
                })),
                // Offered whether or not any router exists yet: with none, this is
                // the only way forward, and the name still ought to be the user's.
                {
                  value: NEW_ROUTER,
                  label: (
                    <Space size={4}>
                      <PlusOutlined />
                      <span>Create a new router…</span>
                    </Space>
                  ),
                },
              ]}
            />
            {creatingRouter && (
              <div>
                <Input
                  placeholder="Name it, e.g. support-router"
                  value={newRouterName}
                  status={nameError ? 'error' : undefined}
                  style={{ width: 280 }}
                  onChange={(e) => setNewRouterName(e.target.value)}
                  onBlur={() => {
                    const name = newRouterName.trim();
                    if (name && !nameError) onRouterChange(name);
                  }}
                />
                <div>
                  <Text type={nameError ? 'danger' : 'secondary'} style={{ fontSize: 11 }}>
                    {nameError ?? 'Clients call this name to be routed. It is created when you apply.'}
                  </Text>
                </div>
              </div>
            )}
          </Space>

          {!routers.length && !creatingRouter && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              No router exists yet. Applying creates <Text code>{status?.router_name}</Text>, or pick
              “Create a new router…” to name your own.
            </Text>
          )}

          {status?.available && !creatingRouter && (
            <Descriptions size="small" column={1} bordered>
              <Descriptions.Item label="Clients call">
                <Text code copyable>
                  {status.router_name}
                </Text>{' '}
                <Text type="secondary" style={{ fontSize: 11 }}>
                  — only requests addressed to this name are routed; traffic sent to a specific model
                  is unaffected.
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Matches go to">
                <Text code>{status.this_model}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Everything else">
                <Text code>
                  {status.default_model ??
                    status.available_chat_models.find((m) => m !== status.this_model) ??
                    '—'}
                </Text>
              </Descriptions.Item>
              {status.routes.filter((r) => !r.is_this_job).length > 0 && (
                <Descriptions.Item label="Sharing this router">
                  <Space size={4} wrap>
                    {status.routes
                      .filter((r) => !r.is_this_job)
                      .map((r) => (
                        <Tag key={r.model}>
                          {r.model} · {r.utterances.length}
                        </Tag>
                      ))}
                  </Space>
                </Descriptions.Item>
              )}
            </Descriptions>
          )}
        </Space>
      </Card>

      {/* --- 2 · Example questions ----------------------------------------- */}
      <Card
        size="small"
        title={
          <Space size={8}>
            <StepBadge n={2} />
            <span>Example questions</span>
          </Space>
        }
        extra={
          <Tag color={selected.length ? 'blue' : 'orange'}>
            {selected.length} of {items.length} selected
          </Tag>
        }
      >
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            A route is defined by examples of the questions that should reach this model. The dataset
            it was trained on is the best place to find them, and you can add your own. Only the{' '}
            <strong>selected</strong> rows are applied.
          </Text>

          {/* Toolbar: acquire on the left, act on the list on the right. */}
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 8,
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Space wrap size={8}>
              <Tooltip title="Reads this model's training dataset and offers the questions users actually asked">
                <Button
                  type="primary"
                  icon={<SearchOutlined />}
                  loading={extracting}
                  onClick={() =>
                    onExtract({ limit, first_turn_only: firstTurnOnly, redact_pii: redact })
                  }
                >
                  {items.length ? 'Find more' : 'Find examples'}
                </Button>
              </Tooltip>
              <Button
                size="small"
                type="text"
                icon={<SettingOutlined />}
                onClick={() => setShowOptions((v) => !v)}
              >
                Options {showOptions ? <UpOutlined /> : <DownOutlined />}
              </Button>
            </Space>
            <Space wrap size={8}>
              <Input
                allowClear
                size="small"
                prefix={<SearchOutlined style={{ color: '#bfbfbf' }} />}
                placeholder="Filter list"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                style={{ width: 200 }}
              />
              <Button
                size="small"
                danger
                disabled={!selected.length}
                icon={<CloseOutlined />}
                onClick={removeSelected}
              >
                Delete selected
              </Button>
            </Space>
          </div>

          {/* Plain language, one idea per line, with the default spelled out. The
              previous version was three bare labels ("Opening turn only", "Redact
              identifiers") that only made sense if you already knew the pipeline. */}
          {showOptions && (
            <Card
              size="small"
              style={{ background: '#fff', borderStyle: 'dashed', borderColor: '#d9d9d9' }}
            >
              <Space orientation="vertical" size={12} style={{ width: '100%' }}>
                <div>
                  <Space align="center" wrap>
                    <Text>How many examples to look for</Text>
                    <InputNumber
                      min={1}
                      max={200}
                      value={limit}
                      onChange={(v) => setLimit(v ?? 30)}
                      style={{ width: 90 }}
                    />
                  </Space>
                  <div>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      10–50 is usually right. They are picked to be as different from each other as
                      possible, so a wider spread beats a longer list of near-identical questions.
                    </Text>
                  </div>
                </div>

                <div>
                  <Text>Which messages to look at</Text>
                  <div style={{ marginTop: 4 }}>
                    <Radio.Group
                      value={firstTurnOnly ? 'first' : 'all'}
                      onChange={(e) => setFirstTurnOnly(e.target.value === 'first')}
                    >
                      <Space orientation="vertical" size={2}>
                        <Radio value="first">
                          The first question of each conversation{' '}
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            (recommended — this is what a router has to match on)
                          </Text>
                        </Radio>
                        <Radio value="all">
                          Every message the user sent{' '}
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            (more examples, but follow-ups like “what about the second one?” make no
                            sense on their own)
                          </Text>
                        </Radio>
                      </Space>
                    </Radio.Group>
                  </div>
                </div>

                <div>
                  <Checkbox checked={redact} onChange={(e) => setRedact(e.target.checked)}>
                    Mask personal details{' '}
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      (recommended)
                    </Text>
                  </Checkbox>
                  <div>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      Replaces emails, phone numbers, long reference numbers and amounts with
                      placeholders such as <Text code>[EMAIL]</Text>. These examples are stored as
                      gateway configuration, readable by anyone with gateway admin access, so real
                      details from your data should not end up there.
                    </Text>
                  </div>
                </div>
              </Space>
            </Card>
          )}

          {mergeNote && (
            <Alert
              type="success"
              showIcon
              closable
              onClose={() => setMergeNote(null)}
              description={mergeNote}
            />
          )}

          {report && (
            <div
              style={{
                border: '1px solid #f0f0f0',
                borderLeft: '3px solid #d9d9d9',
                borderRadius: 6,
                padding: '8px 12px',
                background: '#fff',
              }}
            >
              <Space orientation="vertical" size={2} style={{ fontSize: 12, color: '#595959' }}>
                  <span>
                    {report.rows} conversations → {report.user_turns} user turns → {report.unique}{' '}
                    distinct → <strong>{report.selected} offered</strong>
                    {report.selection_basis === 'embeddings'
                      ? ' (spread out by meaning)'
                      : ' (spread out by word overlap)'}
                  </span>
                  {dropped.length > 0 && (
                    <span>
                      Skipped:{' '}
                      {dropped
                        .map(([key, count]) => `${count} ${DROP_LABELS[key] ?? key}`)
                        .join(', ')}
                    </span>
                  )}
                  {redacted.length > 0 && (
                    <span>
                      Masked: {redacted.map(([kind, count]) => `${count}× ${kind}`).join(', ')}
                    </span>
                  )}
                  {report.warnings.map((w) => (
                    <Text type="warning" key={w} style={{ fontSize: 12 }}>
                      {w}
                    </Text>
                  ))}
              </Space>
            </div>
          )}

          {/* Boxed and white, so the list reads as one object on the page rather
              than as rows bleeding into the surrounding text and controls. */}
          <div
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: 8,
              overflow: 'hidden',
              background: '#fff',
            }}
          >
            <Table<WorkingUtterance>
              size="small"
              rowKey="key"
              dataSource={visibleItems}
              scroll={{ y: 336 }}
              pagination={
                visibleItems.length > 25
                  ? {
                      pageSize: 25,
                      size: 'small',
                      showSizeChanger: false,
                      showTotal: (total) => `${total} example${total === 1 ? '' : 's'}`,
                    }
                  : false
              }
              locale={{
                emptyText: (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description={
                      items.length
                        ? 'Nothing matches that filter.'
                        : 'No examples yet — find some in the training data, or add your own below.'
                    }
                  />
                ),
              }}
              rowSelection={{
                selectedRowKeys: selected,
                onChange: setSelected,
                // Ticking is how the applied set is chosen, so the header checkbox
                // needs to reach across pages, not just the visible rows.
                selections: [Table.SELECTION_ALL, Table.SELECTION_INVERT, Table.SELECTION_NONE],
              }}
              columns={[
                {
                  title: 'Question',
                  dataIndex: 'text',
                  sorter: (a, b) => a.text.localeCompare(b.text),
                  render: (text: string, row) =>
                    editingKey === row.key ? (
                      <Input
                        size="small"
                        autoFocus
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        onPressEnter={() => commitEdit(row)}
                        onBlur={() => commitEdit(row)}
                      />
                    ) : (
                      <span
                        onDoubleClick={() => startEdit(row)}
                        style={{ cursor: 'text', display: 'block' }}
                      >
                        {text}
                      </span>
                    ),
                },
                {
                  title: 'Source',
                  dataIndex: 'source',
                  width: 170,
                  filters: [
                    { text: 'From data', value: 'dataset' },
                    { text: 'Added by you', value: 'manual' },
                    { text: 'Already live', value: 'applied' },
                  ],
                  onFilter: (value, row) => row.source === value,
                  render: (source: UtteranceSource, row) => (
                    <Space size={8}>
                      <Tooltip title={SOURCE_TAG[source].hint}>
                        <span style={{ whiteSpace: 'nowrap' }}>
                          <span
                            style={{
                              display: 'inline-block',
                              width: 6,
                              height: 6,
                              borderRadius: '50%',
                              marginRight: 6,
                              verticalAlign: 'middle',
                              background: SOURCE_TAG[source].dot,
                            }}
                          />
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {SOURCE_TAG[source].label}
                          </Text>
                        </span>
                      </Tooltip>
                      {!!row.represents && row.represents > 1 && (
                        <Tooltip
                          title={`Stands in for ${row.represents} similarly worded questions in the dataset`}
                        >
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            ×{row.represents}
                          </Text>
                        </Tooltip>
                      )}
                    </Space>
                  ),
                },
                {
                  title: '',
                  key: 'actions',
                  width: 76,
                  align: 'right',
                  render: (_v, row) => (
                    <Space size={0}>
                      <Tooltip title="Edit wording">
                        <Button
                          type="text"
                          size="small"
                          icon={<EditOutlined />}
                          onClick={() => startEdit(row)}
                        />
                      </Tooltip>
                      <Tooltip title="Remove from list">
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<CloseOutlined />}
                          onClick={() => removeOne(row.key)}
                        />
                      </Tooltip>
                    </Space>
                  ),
                },
              ]}
            />
          </div>

          <Space.Compact style={{ width: '100%' }}>
            <Input.TextArea
              placeholder="Add your own — one question per line"
              value={added}
              autoSize={{ minRows: 1, maxRows: 5 }}
              onChange={(e) => setAdded(e.target.value)}
            />
            <Button icon={<PlusOutlined />} onClick={addTyped} disabled={!added.trim()}>
              Add
            </Button>
          </Space.Compact>
        </Space>
      </Card>

      {/* --- 3 · Similarity threshold -------------------------------------- */}
      <Card
        size="small"
        title={
          <Space size={8}>
            <StepBadge n={3} />
            <span>How similar a question has to be</span>
          </Space>
        }
        extra={
          <Text strong style={{ fontSize: 16 }}>
            {thresholdPct}%
          </Text>
        }
      >
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            A question is routed here when it is at least this similar to your examples. Higher is
            stricter.
          </Text>

          <Space wrap size={8}>
            <InputNumber
              min={0}
              max={100}
              value={thresholdPct}
              onChange={(v) => setThresholdPct(v ?? DEFAULT_PCT)}
              formatter={(v) => `${v}%`}
              parser={(v) => Number((v ?? '').toString().replace('%', '')) || 0}
              style={{ width: 90 }}
            />
            <Button
              size="small"
              type={thresholdPct === 45 ? 'primary' : 'default'}
              onClick={() => setThresholdPct(45)}
            >
              Loose · 45%
            </Button>
            <Button
              size="small"
              type={thresholdPct === 50 ? 'primary' : 'default'}
              onClick={() => setThresholdPct(50)}
            >
              Balanced · 50%
            </Button>
            <Button
              size="small"
              type={thresholdPct === 60 ? 'primary' : 'default'}
              onClick={() => setThresholdPct(60)}
            >
              Strict · 60%
            </Button>
          </Space>

          <div style={{ padding: '0 4px' }}>
            <Slider
              min={PCT_MIN}
              max={PCT_MAX}
              step={1}
              value={Math.min(PCT_MAX, Math.max(PCT_MIN, thresholdPct))}
              onChange={setThresholdPct}
              tooltip={{ formatter: (v) => `${v}%` }}
              marks={{
                [PCT_MIN]: `${PCT_MIN}%`,
                [NOISE_FLOOR_PCT]: { style: { color: '#faad14' }, label: 'noise' },
                50: '50%',
                60: '60%',
                [PCT_MAX]: `${PCT_MAX}%`,
              }}
            />
          </div>

          <Text type={advice.tone} style={{ fontSize: 12 }}>
            {advice.text}
          </Text>

          <Alert
            type="info"
            showIcon
            icon={<InfoCircleOutlined />}
            description={
              <Text style={{ fontSize: 12 }}>
                The percentage is the encoder&apos;s similarity score, not a share of words in
                common: two unrelated questions still score around {NOISE_FLOOR_PCT}%, so
                treat 0% as “nothing to compare”, not “opposite”. A route is scored by the{' '}
                <strong>average</strong> of its nearest few examples, which reads below the closest
                single one. Because the exact numbers depend on your data and the encoder, test one
                question that should reach this model and one that should not, then set the
                threshold between them.
              </Text>
            }
          />

          <Space.Compact style={{ width: '100%' }}>
            <Input
              placeholder="Try a real question your users would ask"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onPressEnter={() =>
                query.trim() && onTest(query.trim(), selectedTexts, thresholdPct / 100)
              }
            />
            <Button
              icon={<ThunderboltOutlined />}
              loading={testing}
              disabled={!query.trim() || selectedTexts.length === 0}
              onClick={() => onTest(query.trim(), selectedTexts, thresholdPct / 100)}
            >
              Test
            </Button>
          </Space.Compact>
          {selectedTexts.length === 0 && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              Select at least one example above to test against.
            </Text>
          )}

          {testResult && <TestResult result={testResult} />}
        </Space>
      </Card>

      {/* --- 4 · Apply ------------------------------------------------------ */}
      <Card
        size="small"
        title={
          <Space size={8}>
            <StepBadge n={4} />
            <span>Apply</span>
          </Space>
        }
        extra={
          applied ? (
            dirty ? (
              <Tag color="orange">unsaved changes</Tag>
            ) : (
              <Tag color="green">matches what is live</Tag>
            )
          ) : null
        }
      >
        <Space orientation="vertical" size={8} style={{ width: '100%' }}>
          <Space wrap>
            <Button
              type="primary"
              loading={applying}
              disabled={!canApply}
              onClick={() => onApply(selectedTexts, thresholdPct / 100, effectiveRouter)}
            >
              {applied ? 'Update route' : creatingRouter ? 'Create router and apply' : 'Apply route'}
            </Button>
            {applied && (
              <Button
                danger
                icon={<StopOutlined />}
                loading={removing}
                disabled={!!progress}
                onClick={onRemove}
              >
                Stop routing here
              </Button>
            )}
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>
            <InfoCircleOutlined />{' '}
            {selectedTexts.length === 0
              ? 'Select at least one example question to apply a route.'
              : `Applies ${selectedTexts.length} example${selectedTexts.length === 1 ? '' : 's'} at ` +
                `${thresholdPct}% to ${effectiveRouter || 'the router'}, then restarts the gateway so it ` +
                'picks the change up. That takes about a minute and progress is shown here; requests ' +
                'in flight during the restart may fail.'}
          </Text>
        </Space>
      </Card>
    </Space>
  );
}

/**
 * A test result, as a bar against the line it has to clear.
 *
 * Two numbers and a verdict were easy to misread -- 0.512 against 0.50 does not look
 * like a pass at a glance. The bar puts the score and the threshold on the same
 * scale, so "just over" and "nowhere near" look different.
 */
function TestResult({ result }: { result: SemanticRouteTestResponse }) {
  const scorePct = Math.round(result.score * 100);
  const thresholdPct = Math.round(result.threshold * 100);
  const closestPct =
    result.scores[0]?.closest_score != null
      ? Math.round(result.scores[0].closest_score * 100)
      : null;

  return (
    <Card
      size="small"
      style={{ background: '#fff', borderColor: result.matched ? '#b7eb8f' : '#ffe58f' }}
    >
      <Space orientation="vertical" size={8} style={{ width: '100%' }}>
        <Space wrap size={8} align="center">
          <Tag color={result.matched ? 'green' : 'orange'}>
            {result.matched ? 'Would be routed' : 'Would not be routed'}
          </Tag>
          <Text style={{ fontSize: 12 }}>
            {result.matched ? (
              <>
                to <Text code>{result.matched_model}</Text>
              </>
            ) : (
              <>
                — falls back to <Text code>{result.matched_model ?? 'the default model'}</Text>
              </>
            )}
          </Text>
        </Space>

        <div>
          <div
            style={{
              position: 'relative',
              height: 10,
              background: '#f0f0f0',
              borderRadius: 5,
              overflow: 'visible',
            }}
          >
            <div
              style={{
                width: `${Math.min(100, Math.max(0, scorePct))}%`,
                height: 10,
                borderRadius: 5,
                background: result.matched ? '#52c41a' : '#faad14',
              }}
            />
            <Tooltip title={`Threshold ${thresholdPct}%`}>
              <div
                style={{
                  position: 'absolute',
                  left: `${Math.min(100, Math.max(0, thresholdPct))}%`,
                  top: -4,
                  width: 2,
                  height: 18,
                  background: '#434343',
                }}
              />
            </Tooltip>
          </div>
          <div style={{ marginTop: 6 }}>
            <Text style={{ fontSize: 12 }}>
              Similarity <strong>{scorePct}%</strong> against a threshold of {thresholdPct}%
              {closestPct != null && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {' '}
                  · closest single example {closestPct}%
                </Text>
              )}
            </Text>
          </div>
        </div>

        {result.closest_utterance && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            Closest example: “{result.closest_utterance}”
          </Text>
        )}
      </Space>
    </Card>
  );
}

/**
 * The wait, shown as it happens.
 *
 * Before this, applying was a button that returned and then nothing: the gateway
 * was restarting for the best part of a minute with no way to tell whether the
 * change had landed, failed, or was still going. Three states are enough to make
 * the wait legible -- saved, restarting, live -- and the last one is read back from
 * the gateway rather than assumed from a timer.
 */
function RouteProgressCard({
  progress,
  readiness,
  onDismiss,
}: {
  progress: RouteProgress;
  readiness?: GatewayReadiness;
  onDismiss: () => void;
}) {
  const [elapsed, setElapsed] = useState(() =>
    Math.max(0, Math.round((Date.now() - progress.startedAt) / 1000))
  );

  useEffect(() => {
    const tick = setInterval(
      () => setElapsed(Math.max(0, Math.round((Date.now() - progress.startedAt) / 1000))),
      1000
    );
    return () => clearInterval(tick);
  }, [progress.startedAt]);

  const ready = !!readiness?.ready;
  const gatewayBack = ready || !!readiness?.gateway_responding;

  const steps: { title: string; done: boolean; active: boolean; detail?: string }[] = [
    {
      title: progress.expectRoute
        ? `Route saved to ${progress.routerName}`
        : `Route removed from ${progress.routerName}`,
      done: true,
      active: false,
    },
    {
      title: 'Restarting the gateway',
      detail: 'A router is cached in the gateway process, so it has to restart to see the change.',
      done: gatewayBack,
      active: !gatewayBack,
    },
    {
      title: progress.expectRoute ? 'Router loaded and route live' : 'Old route cleared',
      done: ready,
      active: gatewayBack && !ready,
    },
  ];

  const percent = ready ? 100 : gatewayBack ? 66 : 33;
  // Past this the rollout is no longer "about a minute", and saying so beats a
  // spinner that looks identical at ten seconds and at five minutes.
  const slow = !ready && elapsed > 120;

  return (
    <Card
      size="small"
      style={{ borderColor: ready ? '#b7eb8f' : '#91caff' }}
      title={
        <Space>
          {ready ? (
            <CheckCircleFilled style={{ color: '#52c41a' }} />
          ) : (
            <LoadingOutlined style={{ color: '#1890ff' }} />
          )}
          <span>
            {ready
              ? progress.expectRoute
                ? 'Route is live'
                : 'Routing stopped'
              : 'Applying your change'}
          </span>
          <Text type="secondary" style={{ fontSize: 12, fontWeight: 'normal' }}>
            {elapsed}s
          </Text>
        </Space>
      }
      extra={
        ready ? (
          <Button size="small" type="link" onClick={onDismiss}>
            Dismiss
          </Button>
        ) : null
      }
    >
      <Space orientation="vertical" size={8} style={{ width: '100%' }}>
        <Progress percent={percent} status={ready ? 'success' : 'active'} showInfo={false} />
        <div>
          {steps.map((step) => (
            <div
              key={step.title}
              style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '2px 0' }}
            >
              <span style={{ lineHeight: '22px' }}>
                {step.done ? (
                  <CheckCircleFilled style={{ color: '#52c41a' }} />
                ) : step.active ? (
                  <LoadingOutlined style={{ color: '#1890ff' }} />
                ) : (
                  <ClockCircleOutlined style={{ color: '#bfbfbf' }} />
                )}
              </span>
              <div>
                <Text type={step.done || step.active ? undefined : 'secondary'} strong={step.active}>
                  {step.title}
                </Text>
                {step.active && step.detail && (
                  <div>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {step.detail}
                    </Text>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {readiness?.replicas_desired != null && !ready && (
          <Text type="secondary" style={{ fontSize: 11 }}>
            Gateway pods ready: {readiness.replicas_ready ?? 0} of {readiness.replicas_desired}
          </Text>
        )}

        {ready && progress.expectRoute && (
          <Text style={{ fontSize: 12 }}>
            Call it as{' '}
            <Text code copyable>
              {progress.routerName}
            </Text>{' '}
            — matching questions now reach this model.
          </Text>
        )}

        {slow && (
          <Alert
            type="warning"
            showIcon
            title="This is taking longer than usual"
            description={
              readiness?.message ||
              'The gateway has not come back yet. It is still being watched; if it does not ' +
                'recover, check the gateway deployment in the cluster.'
            }
          />
        )}
      </Space>
    </Card>
  );
}
