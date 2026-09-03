import type { FineTuningJob } from '../types';

/**
 * What to draw for a job's progress.
 *
 * `measured` says whether `percent` came from the engine counting training steps
 * or is only inferred from which phase the job is in. The bar is animated while
 * unmeasured so it reads as "working" rather than "stuck", because an inferred
 * number does not move between phase changes.
 */
export interface JobProgress {
  percent: number;
  label: string;
  measured: boolean;
  active: boolean;
}

/**
 * Engine phase tokens, as carried in `current_phase`. The API keeps the token
 * rather than a display string precisely so the mapping to a label and to a
 * position on the bar lives here.
 *
 * The engine's pipeline is
 * `downloading_data → preparing_environment → training → merging →
 * uploading_model`, and `current_phase` is null once the job is terminal, in
 * which case the API falls back to the coarse status. `status` itself stays
 * coarse by design (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED), so the phase is
 * the only source of granularity — the two must not be conflated.
 *
 * The percentages are deliberately coarse: they are a phase indicator, not a
 * measurement. Training occupies 40-90 so that a real step count (see
 * `stepPercent`) lands in the same band and the bar never jumps backwards when
 * the engine starts reporting steps mid-run.
 */
const PHASES: Record<string, { percent: number; label: string }> = {
  pending: { percent: 20, label: 'Queued' },
  queued: { percent: 20, label: 'Queued' },
  initializing: { percent: 25, label: 'Initializing' },
  downloading_data: { percent: 30, label: 'Downloading training data' },
  preparing_environment: { percent: 40, label: 'Preparing environment' },
  running: { percent: 60, label: 'Training' },
  training: { percent: 60, label: 'Training' },
  merging: { percent: 90, label: 'Merging adapter' },
  uploading_model: { percent: 95, label: 'Uploading model' },
  succeeded: { percent: 100, label: 'Completed' },
  completed: { percent: 100, label: 'Completed' },
};

/**
 * Phases whose position may be read from the step counters. Only `training` has
 * steps behind it: they stop moving when it ends but keep their final value, so
 * honouring them during `merging` or `uploading_model` would pin the bar at the
 * end of training and label the upload "Training — step 20/20".
 */
const STEP_COUNTED_PHASES = new Set(['training', 'running']);

const TRAINING_FLOOR = 40;
const TRAINING_CEILING = 90;

/** Turn an unrecognised engine token into something presentable. */
function prettifyPhase(phase: string): string {
  const spaced = phase.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function phaseTokenOf(job: FineTuningJob): string | null {
  return job.current_phase?.toLowerCase() || null;
}

function phaseOf(job: FineTuningJob): { percent: number; label: string } | null {
  const token = phaseTokenOf(job);
  if (!token) return null;
  return PHASES[token] ?? { percent: PHASES.running.percent, label: prettifyPhase(token) };
}

/** Step-based percentage, mapped into the training band. */
function stepPercent(current: number, total: number): number {
  const ratio = Math.min(1, Math.max(0, current / total));
  return Math.round(TRAINING_FLOOR + (TRAINING_CEILING - TRAINING_FLOOR) * ratio);
}

export function resolveJobProgress(job: FineTuningJob | undefined): JobProgress {
  if (!job) return { percent: 0, label: 'Unknown', measured: false, active: false };

  const phaseToken = phaseTokenOf(job);
  const phase = phaseOf(job);
  const step = job.current_step ?? 0;
  const total = job.total_steps ?? 0;
  const hasSteps = total > 0 && step > 0;

  const stepLabel = () => {
    const parts = [`step ${step}/${total}`];
    if (job.num_train_epochs != null) parts.push(`epoch ${job.num_train_epochs.toFixed(2)}`);
    if (job.training_loss != null) parts.push(`loss ${job.training_loss.toFixed(4)}`);
    return parts.join(' · ');
  };

  // Where the job got to, from the best evidence available and independent of
  // status, so one that failed part-way stops there rather than at either end of
  // the bar. The step counters survive the phase they belong to, which is what
  // lets a failure during merging be told apart from one during training.
  const stoppedAt = hasSteps ? stepPercent(step, total) : (phase?.percent ?? 100);

  switch (job.status) {
    case 'succeeded':
      return { percent: 100, label: 'Completed', measured: true, active: false };
    case 'failed':
      return {
        percent: stoppedAt,
        label: hasSteps ? `Failed — ${stepLabel()}` : 'Failed',
        measured: false,
        active: false,
      };
    case 'cancelled':
      return {
        percent: stoppedAt,
        label: hasSteps ? `Cancelled — ${stepLabel()}` : 'Cancelled',
        measured: false,
        active: false,
      };
    case 'validating_files':
      return { percent: 10, label: 'Validating files', measured: false, active: true };
    case 'queued':
      return { percent: 20, label: 'Queued', measured: false, active: true };
    default:
      break;
  }

  // Running. Key off the phase, and consult the step counters only inside the
  // phase that has them — outside it they are a stale leftover from training.
  const stepsApply = phaseToken === null || STEP_COUNTED_PHASES.has(phaseToken);

  if (stepsApply && hasSteps) {
    return {
      percent: stepPercent(step, total),
      label: `Training — ${stepLabel()}`,
      measured: true,
      active: true,
    };
  }

  if (stepsApply && job.progress_percent != null && job.progress_percent > 0) {
    // The engine's own percentage. Zero means "not measured yet" rather than "no
    // progress" — treating it as a reading is what made the bar sit at 0% for a
    // whole run.
    return {
      percent: Math.round(job.progress_percent),
      label: phase?.label ?? 'Training',
      measured: true,
      active: true,
    };
  }

  const fallback = phase ?? { percent: PHASES.running.percent, label: 'Training' };
  return { percent: fallback.percent, label: fallback.label, measured: false, active: true };
}

/** Percentage only, for callers that just need a number. */
export function calculateJobProgress(job: FineTuningJob | undefined): number {
  return resolveJobProgress(job).percent;
}
