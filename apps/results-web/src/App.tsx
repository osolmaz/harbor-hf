import {
  ArrowLeftRight,
  Check,
  ChevronRight,
  Database,
  ExternalLink,
  GitCompareArrows,
  ListFilter,
  Moon,
  Search,
  Server,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Link,
  NavLink,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import { Comparison, getJson, RunDetail, RunsResponse, RunSummary } from "./api";

function useData<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    setData(null);
    setError("");
    getJson<T>(path)
      .then((value) => active && setData(value))
      .catch((reason: Error) => active && setError(reason.message));
    return () => {
      active = false;
    };
  }, [path]);
  return { data, error };
}

function App() {
  const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link className="brand" to="/">
          <span className="brand-mark">H</span>
          <span>Harbor Results</span>
        </Link>
        <nav>
          <NavLink to="/">Runs</NavLink>
          <NavLink to="/campaigns">Campaigns</NavLink>
        </nav>
        <div className="topbar-actions">
          <a
            className="icon-button"
            href="https://github.com/osolmaz/harbor-hf"
            target="_blank"
            rel="noreferrer"
            title="Source repository"
          >
            <ExternalLink size={16} />
          </a>
          <button
            className="icon-button"
            type="button"
            onClick={() => setDark((value) => !value)}
            title={dark ? "Use light theme" : "Use dark theme"}
          >
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<RunsPage />} />
          <Route path="/campaigns" element={<CampaignsPage />} />
          <Route path="/campaigns/:campaignId" element={<CampaignPage />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/runs/:runId/compare/:otherId" element={<ComparePage />} />
          <Route path="/trials/:trialId" element={<EntityPage kind="trial" />} />
          <Route path="/executions/:executionId" element={<EntityPage kind="execution" />} />
          <Route path="*" element={<EmptyState message="Page not found" />} />
        </Routes>
      </main>
    </div>
  );
}

function RunsPage() {
  const { data, error } = useData<RunsResponse>("/api/v1/runs");
  const [search, setSearch] = useState("");
  const [benchmark, setBenchmark] = useState("");
  const [model, setModel] = useState("");
  const [hardware, setHardware] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const navigate = useNavigate();
  const runs = useMemo(() => {
    if (!data) return [];
    const needle = search.toLowerCase();
    return data.items.filter(
      (run) =>
        (!needle ||
          `${run.run_id} ${run.benchmark} ${run.model_repo} ${run.agent_name}`
            .toLowerCase()
            .includes(needle)) &&
        (!benchmark || run.benchmark === benchmark) &&
        (!model || run.model_repo === model) &&
        (!hardware || run.hardware === hardware),
    );
  }, [data, search, benchmark, model, hardware]);
  const toggle = (runId: string) => {
    setSelected((current) =>
      current.includes(runId)
        ? current.filter((value) => value !== runId)
        : [...current.slice(-1), runId],
    );
  };
  if (error) return <EmptyState message={error} />;
  if (!data) return <Loading />;
  return (
    <>
      <PageHeader
        eyebrow="Published evaluations"
        title="Benchmark runs"
        meta={`${data.total} immutable runs`}
      />
      <section className="summary-strip">
        <SummaryStat label="Runs" value={String(data.total)} icon={<Database size={16} />} />
        <SummaryStat
          label="Benchmarks"
          value={String(data.facets.benchmarks.length)}
          icon={<ListFilter size={16} />}
        />
        <SummaryStat
          label="Hardware profiles"
          value={String(data.facets.hardware.length)}
          icon={<Server size={16} />}
        />
        <SummaryStat
          label="Best score"
          value={formatPercent(Math.max(...data.items.map((run) => run.score), 0))}
          icon={<Check size={16} />}
        />
      </section>
      <section className="toolbar" aria-label="Run filters">
        <label className="search-field">
          <Search size={15} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search runs"
          />
        </label>
        <FilterSelect label="Benchmark" value={benchmark} values={data.facets.benchmarks} onChange={setBenchmark} />
        <FilterSelect label="Model" value={model} values={data.facets.models} onChange={setModel} />
        <FilterSelect label="Hardware" value={hardware} values={data.facets.hardware} onChange={setHardware} />
        {(search || benchmark || model || hardware) && (
          <button className="icon-button" type="button" title="Clear filters" onClick={() => {
            setSearch(""); setBenchmark(""); setModel(""); setHardware("");
          }}><X size={15} /></button>
        )}
        <button
          className="command-button"
          type="button"
          disabled={selected.length !== 2}
          onClick={() => navigate(`/runs/${selected[0]}/compare/${selected[1]}`)}
        >
          <GitCompareArrows size={15} /> Compare {selected.length}/2
        </button>
      </section>
      <RunsTable runs={runs} selected={selected} onToggle={toggle} />
    </>
  );
}

function RunsTable({ runs, selected, onToggle }: {
  runs: RunSummary[];
  selected: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <section className="table-wrap">
      <table>
        <thead><tr><th className="select-cell">Compare</th><th>Score</th><th>Benchmark</th><th>Model</th><th>Agent</th><th>Hardware</th><th>Trials</th><th>Duration</th><th>Completed</th><th /></tr></thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id}>
              <td className="select-cell"><input aria-label={`Compare ${run.run_id}`} type="checkbox" checked={selected.includes(run.run_id)} onChange={() => onToggle(run.run_id)} /></td>
              <td><Score value={run.score} /></td>
              <td><Link className="primary-link" to={`/runs/${run.run_id}`}>{run.benchmark}</Link><SmallText>{shortId(run.run_id)}</SmallText></td>
              <td><span className="model-name">{run.model_repo}</span><SmallText>{shortRevision(run.model_revision)}</SmallText></td>
              <td>{run.agent_name}<SmallText>{run.agent_revision}</SmallText></td>
              <td>{run.hardware}<SmallText>{run.accelerator_count} accelerator{run.accelerator_count === 1 ? "" : "s"}</SmallText></td>
              <td>{run.passed_trials}/{run.trial_count}<SmallText>{run.infrastructure_failures ? `${run.infrastructure_failures} infra errors` : "complete"}</SmallText></td>
              <td>{formatDuration(run.duration_seconds)}</td>
              <td>{formatDate(run.completed_at)}</td>
              <td><Link className="icon-button compact" title="Open run" to={`/runs/${run.run_id}`}><ChevronRight size={15} /></Link></td>
            </tr>
          ))}
        </tbody>
      </table>
      {!runs.length && <EmptyState message="No matching runs" />}
    </section>
  );
}

function RunPage() {
  const { runId = "" } = useParams();
  const { data, error } = useData<RunDetail>(`/api/v1/runs/${encodeURIComponent(runId)}`);
  if (error) return <EmptyState message={error} />;
  if (!data) return <Loading />;
  return (
    <>
      <PageHeader eyebrow={data.summary.benchmark} title={data.summary.model_repo} meta={data.summary.run_id} />
      <section className="summary-strip">
        <SummaryStat label="Score" value={formatPercent(data.summary.score)} />
        <SummaryStat label="Passed" value={`${data.summary.passed_trials}/${data.summary.trial_count}`} />
        <SummaryStat label="Executions" value={String(data.summary.execution_count)} />
        <SummaryStat label="Duration" value={formatDuration(data.summary.duration_seconds)} />
      </section>
      <div className="detail-grid">
        <section className="panel span-two">
          <SectionTitle title="Task results" meta={`${data.trials.length} trials`} />
          <div className="table-wrap flush"><table><thead><tr><th>Task</th><th>Score</th><th>Attempt</th><th>Executions</th><th /></tr></thead><tbody>
            {data.trials.map((trial) => <tr key={trial.trial_id}><td>{trial.task_name}</td><td><Score value={trial.score ?? 0} /></td><td>{String(trial.logical_attempt)}</td><td>{trial.execution_count}</td><td><Link className="icon-button compact" title="Open trial" to={`/trials/${trial.trial_id}`}><ChevronRight size={15} /></Link></td></tr>)}
          </tbody></table></div>
        </section>
        <KeyValuePanel title="Configuration" values={data.configuration} />
        <KeyValuePanel title="Provenance" values={data.provenance} />
        <section className="panel span-two"><SectionTitle title="Artifacts" meta="Public metadata only" /><ArtifactTable artifacts={data.artifacts} /></section>
      </div>
    </>
  );
}

function ComparePage() {
  const { runId = "", otherId = "" } = useParams();
  const { data, error } = useData<Comparison>(`/api/v1/runs/${encodeURIComponent(runId)}/compare/${encodeURIComponent(otherId)}`);
  if (error) return <EmptyState message={error} />;
  if (!data) return <Loading />;
  return (
    <>
      <PageHeader eyebrow="Run comparison" title={`${data.left.model_repo} vs ${data.right.model_repo}`} meta={data.compatible ? data.left.benchmark : "Different benchmarks"} />
      <section className="compare-head">
        <CompareRun run={data.left} label="Baseline" />
        <div className="compare-delta"><ArrowLeftRight size={18} /><strong>{signedPercent(data.score_delta)}</strong><span>score delta</span></div>
        <CompareRun run={data.right} label="Candidate" />
      </section>
      <section className="panel"><SectionTitle title="Task comparison" meta={`${data.tasks.length} tasks`} /><div className="table-wrap flush"><table><thead><tr><th>Task</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead><tbody>
        {data.tasks.map((task) => <tr key={task.task_name}><td>{task.task_name}</td><td>{scoreOrDash(task.left_score)}</td><td>{scoreOrDash(task.right_score)}</td><td className={deltaClass(task.delta)}>{task.delta === null ? "-" : signedPercent(task.delta)}</td></tr>)}
      </tbody></table></div></section>
    </>
  );
}

function CampaignsPage() {
  const { data, error } = useData<{items: Array<{campaign_id: string; run_count: number; benchmark_count: number; model_count: number; completed_at: string; average_score: number}>; total: number}>("/api/v1/campaigns");
  if (error) return <EmptyState message={error} />;
  if (!data) return <Loading />;
  return <><PageHeader eyebrow="Evaluation groups" title="Campaigns" meta={`${data.total} campaigns`} /><section className="table-wrap"><table><thead><tr><th>Campaign</th><th>Average score</th><th>Runs</th><th>Benchmarks</th><th>Models</th><th>Completed</th><th /></tr></thead><tbody>{data.items.map((item) => <tr key={item.campaign_id}><td><Link className="primary-link" to={`/campaigns/${item.campaign_id}`}>{item.campaign_id}</Link></td><td><Score value={item.average_score} /></td><td>{item.run_count}</td><td>{item.benchmark_count}</td><td>{item.model_count}</td><td>{formatDate(item.completed_at)}</td><td><Link className="icon-button compact" title="Open campaign" to={`/campaigns/${item.campaign_id}`}><ChevronRight size={15} /></Link></td></tr>)}</tbody></table></section></>;
}

function CampaignPage() {
  const { campaignId = "" } = useParams();
  const { data, error } = useData<{campaign_id: string; runs: RunSummary[]}>(`/api/v1/campaigns/${encodeURIComponent(campaignId)}`);
  if (error) return <EmptyState message={error} />;
  if (!data) return <Loading />;
  return <><PageHeader eyebrow="Campaign" title={data.campaign_id} meta={`${data.runs.length} runs`} /><RunsTable runs={data.runs} selected={[]} onToggle={() => undefined} /></>;
}

function EntityPage({ kind }: { kind: "trial" | "execution" }) {
  const params = useParams();
  const id = kind === "trial" ? params.trialId : params.executionId;
  const plural = kind === "trial" ? "trials" : "executions";
  const { data, error } = useData<Record<string, unknown>>(`/api/v1/${plural}/${encodeURIComponent(id ?? "")}`);
  if (error) return <EmptyState message={error} />;
  if (!data) return <Loading />;
  const primary = data[kind] as Record<string, unknown>;
  return <><PageHeader eyebrow={kind} title={String(primary[`${kind}_id`])} meta={String(primary.run_id)} /><div className="detail-grid"><KeyValuePanel title="Details" values={primary} /><KeyValuePanel title="Metrics" values={arrayToObject(data.metrics)} /><section className="panel span-two"><SectionTitle title="Artifacts" meta="Public metadata only" /><ArtifactTable artifacts={(data.artifacts as Array<Record<string, unknown>>) ?? []} /></section></div></>;
}

function KeyValuePanel({ title, values }: { title: string; values: Record<string, unknown> }) {
  return <section className="panel"><SectionTitle title={title} /><dl className="key-values">{Object.entries(values).map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{formatValue(value)}</dd></div>)}</dl></section>;
}

function ArtifactTable({ artifacts }: { artifacts: Array<Record<string, unknown>> }) {
  if (!artifacts.length) return <div className="empty-inline">No published artifact metadata</div>;
  return <div className="table-wrap flush"><table><thead><tr><th>Kind</th><th>Path</th><th>Media type</th><th>Size</th><th>Checksum</th></tr></thead><tbody>{artifacts.map((artifact) => <tr key={String(artifact.artifact_id)}><td>{String(artifact.kind)}</td><td>{String(artifact.path)}</td><td>{String(artifact.media_type)}</td><td>{formatBytes(Number(artifact.size_bytes))}</td><td><code>{shortRevision(String(artifact.sha256))}</code></td></tr>)}</tbody></table></div>;
}

function CompareRun({ run, label }: { run: RunSummary; label: string }) {
  return <section className="compare-run"><span className="eyebrow">{label}</span><Link to={`/runs/${run.run_id}`}>{run.model_repo}</Link><Score value={run.score} /><dl><div><dt>Trials</dt><dd>{run.passed_trials}/{run.trial_count}</dd></div><div><dt>Hardware</dt><dd>{run.hardware}</dd></div><div><dt>Agent</dt><dd>{run.agent_name}</dd></div></dl></section>;
}

function PageHeader({ eyebrow, title, meta }: { eyebrow: string; title: string; meta: string }) {
  return <header className="page-header"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1></div><code>{meta}</code></header>;
}

function SectionTitle({ title, meta }: { title: string; meta?: string }) {
  return <header className="section-title"><h2>{title}</h2>{meta && <span>{meta}</span>}</header>;
}

function SummaryStat({ label, value, icon }: { label: string; value: string; icon?: React.ReactNode }) {
  return <div className="summary-stat"><span>{icon}{label}</span><strong>{value}</strong></div>;
}

function FilterSelect({ label, value, values, onChange }: { label: string; value: string; values: string[]; onChange: (value: string) => void }) {
  return <label className="select-field"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="">All</option>{values.map((item) => <option key={item}>{item}</option>)}</select></label>;
}

function Score({ value }: { value: number }) {
  return <div className="score"><strong>{formatPercent(value)}</strong><span><i style={{ width: `${Math.max(0, Math.min(value, 1)) * 100}%` }} /></span></div>;
}

function SmallText({ children }: { children: React.ReactNode }) { return <span className="small-text">{children}</span>; }
function Loading() { return <div className="loading"><span />Loading results</div>; }
function EmptyState({ message }: { message: string }) { return <div className="empty-state">{message}</div>; }

function formatPercent(value: number) { return `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`; }
function signedPercent(value: number) { return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`; }
function scoreOrDash(value: number | null) { return value === null ? "-" : formatPercent(value); }
function formatDuration(seconds: number) { if (seconds < 60) return `${Math.round(seconds)}s`; if (seconds < 3600) return `${Math.round(seconds / 60)}m`; return `${(seconds / 3600).toFixed(1)}h`; }
function formatDate(value: string) { return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value)); }
function formatBytes(value: number) { if (value < 1024) return `${value} B`; if (value < 1048576) return `${(value / 1024).toFixed(1)} KB`; return `${(value / 1048576).toFixed(1)} MB`; }
function shortId(value: string) { return value.length > 18 ? `${value.slice(0, 18)}…` : value; }
function shortRevision(value: string) { return value.length > 12 ? value.slice(0, 12) : value; }
function humanize(value: string) { return value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase()); }
function formatValue(value: unknown) { if (value === null || value === undefined) return "-"; if (typeof value === "object") return JSON.stringify(value); return String(value); }
function arrayToObject(value: unknown) { const rows = Array.isArray(value) ? value : []; return Object.fromEntries(rows.map((row, index) => [`metric_${index + 1}`, row])); }
function deltaClass(value: number | null) { if (value === null || value === 0) return ""; return value > 0 ? "positive" : "negative"; }

export default App;
