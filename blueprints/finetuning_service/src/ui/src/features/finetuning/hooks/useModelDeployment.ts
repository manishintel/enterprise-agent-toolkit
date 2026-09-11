import {
  useMutation,
  useQuery,
  useQueryClient,
  UseMutationOptions,
  UseQueryOptions,
} from '@tanstack/react-query';
import { fineTuningApi, FineTuningApiError } from '../api/client';
import type {
  DeploymentCapacity,
  DeploymentLogs,
  DeploymentPhase,
  DeployModelRequest,
  ModelDeploymentStatus,
} from '../types';
import { queryKeys, handleQueryError } from '@core/query/queryClient';

// Phases where something is still happening in the cluster, so the status is
// worth polling. Everything else is a resting state.
const ACTIVE_PHASES: DeploymentPhase[] = [
  'installing',
  'downloading',
  'extracting',
  'loading',
  'registering',
  'uninstalling',
];

export const isDeploymentInProgress = (phase?: DeploymentPhase): boolean =>
  !!phase && ACTIVE_PHASES.includes(phase);

export function useModelDeployment(
  jobId: string,
  options?: Omit<UseQueryOptions<ModelDeploymentStatus, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.deployment(jobId),
    queryFn: () => fineTuningApi.getModelDeployment(jobId),
    enabled: !!jobId,
    staleTime: 2 * 1000,
    // Bringing a model up takes minutes (download, unpack, load weights), so
    // follow it while it is in flight and stop once it settles.
    refetchInterval: (query) => (isDeploymentInProgress(query.state.data?.phase) ? 5 * 1000 : false),
    ...options,
  });
}

export interface DeploymentLogOptions {
  tail?: number;
  hideProbes?: boolean;
}

/**
 * A tail of the deployment's container output, fetched when asked for.
 *
 * Deliberately not polled. A serving model's log is almost entirely liveness
 * probes, so a self-refreshing log view spends a cluster round-trip every few
 * seconds to redraw the same health checks; reading a log is a thing someone does,
 * so it happens when they open the view or press Refresh.
 */
export function useDeploymentLogs(
  jobId: string,
  enabled: boolean,
  logOptions?: DeploymentLogOptions,
  options?: Omit<UseQueryOptions<DeploymentLogs, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.deploymentLogs(jobId, logOptions),
    queryFn: () => fineTuningApi.getDeploymentLogs(jobId, logOptions),
    enabled: !!jobId && enabled,
    // Long enough that switching tabs back and forth does not refetch, short
    // enough that Refresh always goes to the cluster.
    staleTime: 5 * 1000,
    refetchInterval: false,
    refetchOnWindowFocus: false,
    ...options,
  });
}

/**
 * Room to serve this model, plus the sizing to start from.
 *
 * Only fetched while the deploy dialog is open: it reads cluster-wide node and
 * pod state, so there is no reason to poll it from a page nobody is deploying
 * from. It is polled while open because another deployment or a fine-tuning job
 * can take the room in the meantime.
 */
export function useDeploymentCapacity(
  jobId: string,
  enabled: boolean,
  options?: Omit<UseQueryOptions<DeploymentCapacity, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.deploymentCapacity(jobId),
    queryFn: () => fineTuningApi.getDeploymentCapacity(jobId),
    enabled: !!jobId && enabled,
    staleTime: 5 * 1000,
    refetchInterval: enabled ? 15 * 1000 : false,
    ...options,
  });
}

export interface DeployModelVariables {
  jobId: string;
  overrides?: DeployModelRequest;
}

export function useDeployModel(
  options?: UseMutationOptions<ModelDeploymentStatus, FineTuningApiError, DeployModelVariables>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ jobId, overrides }: DeployModelVariables) =>
      fineTuningApi.deployModel(jobId, overrides),
    onSuccess: (data) => {
      // The response is the status right after the deployment was started, so
      // seeding it starts the polling above without waiting for a refetch.
      queryClient.setQueryData(queryKeys.fineTuning.jobs.deployment(data.job_id), data);
      // What is free changed the moment this was admitted.
      queryClient.invalidateQueries({
        queryKey: queryKeys.fineTuning.jobs.deploymentCapacity(data.job_id),
      });
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

export function useUndeployModel(
  options?: UseMutationOptions<ModelDeploymentStatus, FineTuningApiError, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => fineTuningApi.undeployModel(jobId),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.fineTuning.jobs.deployment(data.job_id), data);
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}
