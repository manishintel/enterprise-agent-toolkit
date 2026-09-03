'use client';

/**
 * Semantic routing setup: mine utterances from the training dataset, review them,
 * calibrate a threshold, apply.
 *
 * Four things this has to make clear, because none of them is guessable from the
 * controls alone.
 *
 * **Routing is opt-in by model name.** The router is itself a model; callers have
 * to address it to be routed. Setting this up changes nothing for traffic still
 * addressed to a specific model, so the router's name is shown as the thing to
 * point clients at.
 *
 * **The score is a mean, not a best match.** The router averages the similarity of
 * the nearest few utterances, which reads well below the closest one (0.62 where
 * the best utterance scores 0.86). Both are shown, with the mean as the number the
 * threshold applies to, or a user calibrates against the wrong figure.
 *
 * **Utterances become gateway configuration.** They are mined from production
 * traces, so redaction is on by default and turning it off is spelled out rather
 * than being a bare toggle.
 *
 * **Applying restarts the gateway.** An auto-router is cached in the gateway
 * process, so a change only takes effect on restart. That is stated before the
 * button is pressed, not discovered afterwards.
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Descriptions,
  Divider,
  Empty,
  Form,
  Input,
  InputNumber,
  Modal,
  Slider,
  Space,
  Spin,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import { DeleteOutlined, InfoCircleOutlined, ThunderboltOutlined } from '@ant-design/icons';
import type {
  ExtractUtterancesResponse,
  SemanticRouteStatus,
  SemanticRouteTestResponse,
} from '@features/finetuning/types';

const { Text, Paragraph } = Typography;

export interface SemanticRoutingDialogProps {
  open: boolean;
  jobId: string;
  status?: SemanticRouteStatus;
  statusLoading: boolean;
  extraction?: ExtractUtterancesResponse;
  extracting: boolean;
  testResult?: SemanticRouteTestResponse;
  testing: boolean;
  applying: boolean;
  removing: boolean;
  onExtract: (options: { limit: number; first_turn_only: boolean; redact_pii: boolean }) => void;
  onTest: (query: string, utterances: string[], threshold: number) => void;
  onApply: (utterances: string[], threshold: number) => void;
  onRemove: () => void;
  onCancel: () => void;
}

/** Human-readable labels for the funnel counters. */
const DROP_LABELS: Record<string, string> = {
  too_short: 'too short',
  too_long: 'too long',
  pleasantry: 'greetings only',
  empty: 'empty',
  duplicate: 'exact duplicates',
  near_duplicate: 'near duplicates',
};

export default function SemanticRoutingDialog({
  open,
  jobId,
  status,
  statusLoading,
  extraction,
  extracting,
  testResult,
  testing,
  applying,
  removing,
  onExtract,
  onTest,
  onApply,
  onRemove,
  onCancel,
}: SemanticRoutingDialogProps) {
  const [limit, setLimit] = useState<number>(30);
  const [firstTurnOnly, setFirstTurnOnly] = useState(true);
  const [redact, setRedact] = useState(true);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [utterances, setUtterances] = useState<string[]>([]);
  const [added, setAdded] = useState('');
  const [query, setQuery] = useState('');

  const applied = status?.this_route;

  // Start from whatever is already applied, so opening this on a configured route
  // is an edit rather than a blank slate.
  useEffect(() => {
    if (!open) return;
    if (applied?.utterances?.length) {
      setUtterances(applied.utterances);
      if (applied.score_threshold != null) setThreshold(applied.score_threshold);
    }
  }, [open, applied]);

  // Extraction results replace the working set; the user can then edit it.
  useEffect(() => {
    if (extraction?.utterances) {
      setUtterances(extraction.utterances.map((u) => u.text));
    }
  }, [extraction]);

  useEffect(() => {
    if (!open) {
      setQuery('');
      setAdded('');
    }
  }, [open]);

  const report = extraction?.report;
  const dropped = useMemo(
    () => Object.entries(report?.dropped ?? {}).filter(([, count]) => count > 0),
    [report]
  );
  const redacted = useMemo(
    () => Object.entries(report?.redacted ?? {}).filter(([, count]) => count > 0),
    [report]
  );

  const blocked = !!status && (!status.available || !!status.message);
  const canApply = !applying && utterances.length > 0 && !blocked;

  const removeAt = (index: number) =>
    setUtterances((current) => current.filter((_, i) => i !== index));

  const addUtterance = () => {
    const text = added.trim();
    if (!text) return;
    setUtterances((current) => (current.includes(text) ? current : [...current, text]));
    setAdded('');
  };

  return (
    <Modal
      open={open}
      title="Semantic routing"
      onCancel={onCancel}
      width={760}
      footer={[
        applied ? (
          <Button key="remove" danger icon={<DeleteOutlined />} loading={removing} onClick={onRemove}>
            Stop routing here
          </Button>
        ) : null,
        <Button key="cancel" onClick={onCancel} disabled={applying}>
          Close
        </Button>,
        <Button key="apply" type="primary" loading={applying} disabled={!canApply} onClick={() => onApply(utterances, threshold)}>
          {applied ? 'Update route' : 'Apply route'}
        </Button>,
      ]}
    >
      {statusLoading && !status ? (
        <div style={{ textAlign: 'center', padding: 32 }}>
          <Spin tip="Reading the gateway..." />
        </div>
      ) : (
        <Space orientation="vertical" style={{ width: '100%' }} size="middle">
          {status && !status.available && (
            <Alert type="warning" showIcon title="Semantic routing is unavailable" description={status.message} />
          )}
          {status?.available && status.message && (
            <Alert type="warning" showIcon title="Not ready yet" description={status.message} />
          )}

          {status?.available && (
            <Descriptions size="small" column={1} bordered>
              <Descriptions.Item label="Clients call">
                <Text code>{status.router_name}</Text>{' '}
                <Text type="secondary" style={{ fontSize: 11 }}>
                  — routing only happens for requests addressed to this name; traffic sent to a
                  specific model is unaffected.
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="Routes here">
                <Text code>{status.this_model}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Everything else">
                <Text code>{status.default_model ?? status.available_chat_models.find((m) => m !== status.this_model) ?? '—'}</Text>
              </Descriptions.Item>
              {status.routes.filter((r) => !r.is_this_job).length > 0 && (
                <Descriptions.Item label="Other routes">
                  {status.routes
                    .filter((r) => !r.is_this_job)
                    .map((r) => (
                      <Tag key={r.model}>
                        {r.model} ({r.utterances.length})
                      </Tag>
                    ))}
                </Descriptions.Item>
              )}
            </Descriptions>
          )}

          <Divider style={{ margin: '4px 0' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              1 — Mine utterances from the training dataset
            </Text>
          </Divider>

          <Space wrap align="end">
            <Form.Item label="How many" style={{ marginBottom: 0 }} extra="Aim for 10-50">
              <InputNumber min={1} max={200} value={limit} onChange={(v) => setLimit(v ?? 30)} style={{ width: 110 }} />
            </Form.Item>
            <Form.Item
              label="Opening turn only"
              style={{ marginBottom: 0 }}
              extra="Later turns are follow-ups"
            >
              <Switch checked={firstTurnOnly} onChange={setFirstTurnOnly} />
            </Form.Item>
            <Form.Item
              label={
                <Tooltip title="Replaces card numbers, IBANs, account numbers, emails, phone numbers and amounts with placeholders. These utterances are stored as gateway configuration and are readable by anyone with gateway admin access, so leaving this on keeps trace content out of it.">
                  <span>
                    Redact identifiers <InfoCircleOutlined />
                  </span>
                </Tooltip>
              }
              style={{ marginBottom: 0 }}
            >
              <Switch checked={redact} onChange={setRedact} />
            </Form.Item>
            <Button
              type="primary"
              ghost
              loading={extracting}
              onClick={() => onExtract({ limit, first_turn_only: firstTurnOnly, redact_pii: redact })}
            >
              Extract
            </Button>
          </Space>

          {report && (
            <Alert
              type="info"
              showIcon={false}
              description={
                <Space orientation="vertical" size={2} style={{ fontSize: 12 }}>
                  <span>
                    {report.rows} conversations → {report.user_turns} user turns → {report.unique}{' '}
                    distinct → <strong>{report.selected} selected</strong>
                    {report.selection_basis === 'embeddings'
                      ? ' (spread out by meaning)'
                      : ' (spread out by word overlap)'}
                  </span>
                  {dropped.length > 0 && (
                    <span>
                      Dropped:{' '}
                      {dropped.map(([key, count]) => `${count} ${DROP_LABELS[key] ?? key}`).join(', ')}
                    </span>
                  )}
                  {redacted.length > 0 && (
                    <span>
                      Redacted: {redacted.map(([kind, count]) => `${count}× ${kind}`).join(', ')}
                    </span>
                  )}
                  {report.warnings.map((w) => (
                    <Text type="warning" key={w} style={{ fontSize: 12 }}>
                      {w}
                    </Text>
                  ))}
                </Space>
              }
            />
          )}

          <Divider style={{ margin: '4px 0' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              2 — Review ({utterances.length})
            </Text>
          </Divider>

          {utterances.length === 0 ? (
            <Empty
              description="No utterances yet — extract them from the dataset, or add your own below."
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            <Table
              size="small"
              pagination={utterances.length > 8 ? { pageSize: 8, size: 'small' } : false}
              rowKey={(_, index) => String(index)}
              dataSource={utterances.map((text, index) => ({ text, index }))}
              columns={[
                { title: 'Utterance', dataIndex: 'text', ellipsis: true },
                {
                  title: '',
                  width: 44,
                  render: (_v, row: { index: number }) => (
                    <Button
                      type="text"
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() => removeAt(row.index)}
                    />
                  ),
                },
              ]}
            />
          )}

          <Space.Compact style={{ width: '100%' }}>
            <Input
              placeholder="Add an utterance by hand"
              value={added}
              onChange={(e) => setAdded(e.target.value)}
              onPressEnter={addUtterance}
            />
            <Button onClick={addUtterance}>Add</Button>
          </Space.Compact>

          <Divider style={{ margin: '4px 0' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              3 — Calibrate the threshold
            </Text>
          </Divider>

          <div>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={setThreshold}
              marks={{ 0.4: '0.4', 0.5: '0.5', 0.6: '0.6', 0.8: '0.8' }}
            />
            <Paragraph type="secondary" style={{ fontSize: 11, marginBottom: 8 }}>
              The router scores a route by the <strong>mean</strong> similarity of its nearest few
              utterances — not its best match — so the number is lower than it looks. On a banking
              set, in-domain queries scored 0.57-0.62 and unrelated ones 0.40-0.44. Test a couple of
              real queries rather than guessing.
            </Paragraph>
            <Space.Compact style={{ width: '100%' }}>
              <Input
                placeholder="Try a query, e.g. how do I report a lost card?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onPressEnter={() => query.trim() && onTest(query.trim(), utterances, threshold)}
              />
              <Button
                icon={<ThunderboltOutlined />}
                loading={testing}
                disabled={!query.trim() || utterances.length === 0}
                onClick={() => onTest(query.trim(), utterances, threshold)}
              >
                Test
              </Button>
            </Space.Compact>

            {testResult && (
              <Alert
                style={{ marginTop: 8 }}
                type={testResult.matched ? 'success' : 'info'}
                showIcon
                title={
                  testResult.matched
                    ? `Would route to ${testResult.matched_model}`
                    : `Would fall back to ${testResult.matched_model ?? 'the default model'}`
                }
                description={
                  <Space orientation="vertical" size={0} style={{ fontSize: 12 }}>
                    <span>
                      Score <strong>{testResult.score.toFixed(3)}</strong> against a threshold of{' '}
                      {testResult.threshold.toFixed(2)}
                      {testResult.scores[0]?.closest_score != null && (
                        <>
                          {' '}
                          (closest single utterance {testResult.scores[0].closest_score.toFixed(3)})
                        </>
                      )}
                    </span>
                    {testResult.closest_utterance && (
                      <span>Closest: “{testResult.closest_utterance}”</span>
                    )}
                  </Space>
                }
              />
            )}
          </div>

          {status?.restart_required_on_apply && (
            <Alert
              type="warning"
              showIcon
              title="Applying restarts the gateway"
              description={
                <>
                  The gateway caches a router in memory, so a change only takes effect once it
                  restarts. Requests in flight during the restart may fail; it is usually back within
                  a minute. Job {jobId} is unaffected.
                </>
              }
            />
          )}
        </Space>
      )}
    </Modal>
  );
}
