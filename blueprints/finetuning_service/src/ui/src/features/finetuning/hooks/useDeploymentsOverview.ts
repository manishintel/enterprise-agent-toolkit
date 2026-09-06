import { useMemo } from 'react';
import { useQueries } from '@tanstack/react-query';
import { fineTuningApi, type FineTuningApiError } from '../api/client';
import type {
  FineTuningJob,
  ModelDeploymentStatus,
  SemanticRouteEntry,
} from '../types';
import { queryKeys } from '@core/query/queryClient';
import { getFineTunedModelName } from '../utils';
import { useFineTuningJobsList } from './useFineTuningQueries';
import { useSemanticRoute } from './useSemanticRoute';
import { isDeploymentInProgress } from './useModelDeployment';

/**
 * The Deployments list, assembled client-side.
 *
 * There is no "list deployments" endpoint -- every deployment and route API is
 * scoped to a single job (`/jobs/{id}/deployment`). So this derives the list from
 * the jobs that produced a model, and reads each one's state individually.
 *
 * Two things keep that from being expensive.
 *
 * Only jobs that actually produced a model are asked about: a job has to have
 * succeeded *and* have a result file before there is anything to deploy, so
 * queued, running and failed jobs cost nothing here.
 *
 * The route table is fetched once, not per row. A job's semantic-route response
 * carries every route registered on the shared router, with `is_this_job`
 * marking its own -- so one call answers "which of these models is routed" for
 * the whole list. Asking each row separately would be N calls for the same
 * payload.
 */

export interface DeployableModel {
  job: FineTuningJob;
  /** Name the model is (or would be) served under. */
  modelName: string;
  deployment?: ModelDeploymentStatus;
  deploymentLoading: boolean;
  /** This model's entry on the shared router, if traffic is routed to it. */
  route?: SemanticRouteEntry;
}

export interface DeploymentsOverview {
  models: DeployableModel[];
  isLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  /** Narrower than `unknown` so it can be handed straight to QueryErrorDisplay. */
  error: FineTuningApiError | null;
  /** True when the gateway could not be read, so route state is unknown. */
  routesUnavailable: boolean;
  refetch: () => void;
}

export function useDeploymentsOverview(): DeploymentsOverview {
  const {
    data: jobsResponse,
    isLoading: jobsLoading,
    isFetching: jobsFetching,
    isError,
    error,
    refetch: refetchJobs,
  } = useFineTuningJobsList({ limit: 100 });

  // A model exists only once the job succeeded and left a result file behind.
  const deployableJobs = useMemo(
    () =>
      (jobsResponse?.data ?? []).filter(
        (job) => job.status === 'succeeded' && !!job.result_files?.[0]
      ),
    [jobsResponse]
  );

  const deploymentQueries = useQueries({
    queries: deployableJobs.map((job) => ({
      queryKey: queryKeys.fineTuning.jobs.deployment(job.id),
      queryFn: () => fineTuningApi.getModelDeployment(job.id),
      staleTime: 10 * 1000,
      // Same rule as the detail page: follow a deployment while it is coming up
      // or going away, then stop. A settled list makes no requests.
      refetchInterval: (query: { state: { data?: ModelDeploymentStatus } }) =>
        isDeploymentInProgress(query.state.data?.phase) ? 5 * 1000 : (false as const),
      // A single unreadable deployment should leave a gap in one row, not blank
      // the page, so failures are not retried into a spinner.
      retry: false,
    })),
  });

  // One call for the whole route table -- see the note above.
  const routeProbeJobId = deployableJobs[0]?.id ?? '';
  const { data: routeStatus, refetch: refetchRoutes } = useSemanticRoute(
    routeProbeJobId,
    !!routeProbeJobId
  );

  const models = useMemo<DeployableModel[]>(
    () =>
      deployableJobs.map((job, index) => {
        const query = deploymentQueries[index];
        const deployment = query?.data;
        const modelName = deployment?.served_model_name || getFineTunedModelName(job);

        // Routes name a model, not a job, so match on the served name. Fall back
        // to the job's own flag when the gateway reports one.
        const route = routeStatus?.routes?.find(
          (entry) => entry.model === modelName || entry.model === deployment?.served_model_name
        );

        return {
          job,
          modelName,
          deployment,
          deploymentLoading: !!query?.isLoading,
          route,
        };
      }),
    [deployableJobs, deploymentQueries, routeStatus]
  );

  return {
    models,
    isLoading: jobsLoading,
    isFetching: jobsFetching || deploymentQueries.some((query) => query.isFetching),
    isError,
    error,
    routesUnavailable: !!routeProbeJobId && routeStatus?.available === false,
    refetch: () => {
      refetchJobs();
      refetchRoutes();
      deploymentQueries.forEach((query) => query.refetch());
    },
  };
}
