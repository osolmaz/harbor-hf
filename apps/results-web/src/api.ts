export interface RunSummary {
  run_id: string;
  publication_id: string;
  campaign_id: string;
  evaluation_id: string;
  publication_role: "final" | "component" | "diagnostic";
  component_kind: "base" | "correction" | null;
  source_publication_ids: string[];
  benchmark: string;
  benchmark_revision: string;
  model_repo: string;
  model_revision: string;
  agent_name: string;
  agent_revision: string;
  provider: string;
  region: string;
  hardware: string;
  accelerator_count: number;
  result_kind: string;
  outcome: string;
  quality: "clean" | "degraded";
  score: number;
  passed_trials: number;
  planned_trial_count: number;
  scored_trial_count: number;
  agent_failed_count: number;
  benchmark_failed_count: number;
  infrastructure_exhausted_count: number;
  unsupported_count: number;
  execution_count: number;
  failed_executions: number;
  duration_seconds: number;
  completed_at: string;
}

export interface RunsResponse {
  items: RunSummary[];
  total: number;
  next_cursor: string | null;
  facets: {
    benchmarks: string[];
    models: string[];
    hardware: string[];
    agents: string[];
  };
}

export type RunSortField =
  | "score"
  | "benchmark"
  | "model_repo"
  | "agent_name"
  | "hardware"
  | "passed_trials"
  | "duration_seconds"
  | "completed_at";

export type SortOrder = "asc" | "desc";

export interface RunDetail {
  summary: RunSummary;
  sources: RunSummary[];
  configuration: Record<string, unknown>;
  trials: Array<Record<string, unknown> & {
    trial_id: string;
    task_name: string;
    outcome:
      | "scored"
      | "agent_failed"
      | "benchmark_failed"
      | "infrastructure_exhausted"
      | "unsupported";
    score: number | null;
    execution_count: number;
  }>;
  executions: Array<Record<string, unknown>>;
  metrics: Array<Record<string, unknown>>;
  artifacts: Array<Record<string, unknown>>;
  provenance: Record<string, unknown>;
}

export interface Comparison {
  compatible: boolean;
  left: RunSummary;
  right: RunSummary;
  score_delta: number;
  tasks: Array<{
    task_name: string;
    logical_attempt: number;
    left_score: number | null;
    right_score: number | null;
    delta: number | null;
  }>;
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(
      body.error?.message ?? body.detail ?? `Request failed with ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
}
