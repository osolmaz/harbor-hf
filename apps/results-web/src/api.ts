export interface RunSummary {
  run_id: string;
  publication_id: string;
  campaign_id: string;
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
  score: number;
  passed_trials: number;
  trial_count: number;
  execution_count: number;
  infrastructure_failures: number;
  duration_seconds: number;
  completed_at: string;
}

export interface RunsResponse {
  items: RunSummary[];
  total: number;
  facets: {
    benchmarks: string[];
    models: string[];
    hardware: string[];
    agents: string[];
  };
}

export interface RunDetail {
  summary: RunSummary;
  configuration: Record<string, unknown>;
  trials: Array<Record<string, unknown> & {
    trial_id: string;
    task_name: string;
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
