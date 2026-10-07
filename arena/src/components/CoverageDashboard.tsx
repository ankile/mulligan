/**
 * Coverage tab — live stage-label coverage & annotation provenance.
 *
 * Rendered entirely from the reactive stageCoverage aggregate, so it reflects
 * every prediction selection and review save immediately.
 *
 * Stage segments use the project stage-ladder ramp (mirrors the frozen
 * STAGE_LADDER_COLORS anchors in mulligan/plotting/colors.py — red = failed early,
 * green = full success) with S# text labels so identity never rides on color
 * alone; coverage segments use a CVD-validated deep-green / warm-orange /
 * neutral trio.
 */
import { useQuery } from "../lib/arenaClient";
import { api } from "../release/api";
import { useSearchParam } from "../lib/useSearchParam";

// Frozen anchors from mulligan/plotting/colors.py STAGE_LADDER_COLORS (S0..S7).
const STAGE_ANCHORS = [
  "#b1342f",
  "#dd6a3e",
  "#eb9a4e",
  "#ecc15e",
  "#bcc862",
  "#86bd66",
  "#4ea76a",
  "#247a4f",
];

// Coverage trio — validated (dataviz six-checks) against the cream surface.
const HUMAN_COLOR = "#196b1f";
const VLM_COLOR = "#eb9a4e";
const UNLABELED_COLOR = "#d9d9d9";

function lerpHex(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const mix = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `#${mix.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** Stage colour for an n-rung ladder, sampling the frozen anchor ramp. */
function stageColor(stage: number, nStages: number): string {
  if (nStages === STAGE_ANCHORS.length) return STAGE_ANCHORS[stage];
  // A degenerate 1-rung ladder would divide by zero (NaN -> undefined anchor
  // -> render crash with no ErrorBoundary above us); pin it to the top anchor.
  const denom = Math.max(nStages - 1, 1);
  const pos = (stage / denom) * (STAGE_ANCHORS.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, STAGE_ANCHORS.length - 1);
  return lerpHex(STAGE_ANCHORS[lo], STAGE_ANCHORS[hi], pos - lo);
}

function textOn(hex: string): string {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b) < 0.3 ? "#ffffff" : "#1a1a1a";
}

function shortRepo(repo: string): string {
  return repo.split("/").pop() ?? repo;
}

function stageReviewUrl(repo: string): string {
  const p = new URLSearchParams({ tab: "explorer", dataset: repo, view: "stage" });
  return `?${p.toString()}`;
}

type Segment = { label: string; value: number; color: string; title?: string };

/** Horizontal stacked bar out of plain divs — 2px surface gaps between fills. */
function StackedBar({ segments, total }: { segments: Segment[]; total: number }) {
  const denom = Math.max(total, 1);
  return (
    <div className="flex h-5 w-full overflow-hidden rounded" style={{ gap: 2 }}>
      {segments
        .filter((s) => s.value > 0)
        .map((s, i) => (
          <div
            key={i}
            title={s.title ?? `${s.label}: ${s.value}`}
            className="flex items-center justify-center rounded-[3px] text-[10px] leading-none"
            style={{
              width: `${(s.value / denom) * 100}%`,
              backgroundColor: s.color,
              color: textOn(s.color),
              minWidth: s.value > 0 ? 2 : 0,
            }}
          >
            {s.value / denom > 0.055 ? s.label : ""}
          </div>
        ))}
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-muted">
      <span className="inline-block h-3 w-3 rounded-sm" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

type TaskCoverage = NonNullable<ReturnType<typeof useTaskCoverage>>;

function useTaskCoverage(task: string, includeAll: boolean) {
  return useQuery(api.stageCoverage.forTask, { task, includeAll });
}

function StageHistBar({
  hist,
  nStages,
  caption,
}: {
  hist: number[];
  nStages: number;
  caption: string;
}) {
  const total = hist.reduce((a, b) => a + b, 0);
  if (total === 0) {
    return (
      <div className="text-xs text-ink-muted">
        {caption}: no labeled episodes
      </div>
    );
  }
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs text-ink-muted">
        <span>{caption}</span>
        <span>n={total}</span>
      </div>
      <StackedBar
        total={total}
        segments={hist.map((v, s) => ({
          label: `S${s}`,
          value: v,
          color: stageColor(s, nStages),
          title: `S${s}: ${v} episodes (${((v / total) * 100).toFixed(1)}%)`,
        }))}
      />
    </div>
  );
}

function TaskSection({ t, readOnly = false }: { t: TaskCoverage; readOnly?: boolean }) {
  const totals = t.repos.reduce(
    (acc, r) => ({
      episodes: acc.episodes + (r.num_episodes ?? 0),
      unregistered: acc.unregistered + (r.num_episodes === null ? 1 : 0),
      prefill: acc.prefill + r.n_prefill,
      committed: acc.committed + r.n_committed,
      uncertain: acc.uncertain + r.n_uncertain,
      flagged: acc.flagged + r.n_flagged,
    }),
    { episodes: 0, unregistered: 0, prefill: 0, committed: 0, uncertain: 0, flagged: 0 }
  );
  return (
    <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8">
      <div className="mb-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="font-display text-xl text-ink">{t.task}</h2>
        <span className="font-mono text-xs text-ink-muted">schema {t.taxonomy_version}</span>
      </div>
      <p className="mb-5 text-sm text-ink-muted">
        {totals.episodes.toLocaleString()} episodes across {t.repos.length} dataset
        {t.repos.length === 1 ? "" : "s"}
        {totals.unregistered > 0 ? ` (${totals.unregistered} unregistered — episode totals incomplete)` : ""} ·{" "}
        {totals.prefill.toLocaleString()} pipeline-labeled ·{" "}
        {totals.committed} human-committed · {totals.uncertain} uncertain ·{" "}
        {totals.flagged} prefills with consistency flags
      </p>

      {/* Per-repo coverage */}
      <div className="mb-2 flex flex-wrap gap-4">
        <LegendSwatch color={HUMAN_COLOR} label="human committed" />
        <LegendSwatch color={VLM_COLOR} label="pipeline only" />
        <LegendSwatch color={UNLABELED_COLOR} label="unlabeled" />
      </div>
      <table className="mb-6 w-full text-sm break-all">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wide text-ink-muted">
            <th className="py-1 pr-3 font-medium">Dataset</th>
            <th className="py-1 pr-3 font-medium text-right">Episodes</th>
            <th className="w-1/2 py-1 font-medium">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {t.repos.map((r) => {
            const labeled = r.n_committed + r.n_vlm_only;
            // A registered episode count SMALLER than the labeled count means
            // stale/broken registration — surface it instead of clamping the
            // bar into a clean-looking 100%.
            const mismatch = r.num_episodes !== null && r.num_episodes < labeled;
            const total = r.num_episodes !== null && !mismatch ? r.num_episodes : labeled;
            const unlabeled = total - labeled;
            return (
              <tr key={r.repo} className="border-t border-warm-200/60">
                <td className="py-1.5 pr-3">
                  <a
                    href={readOnly ? `?tab=explorer&dataset=${encodeURIComponent(r.repo)}` : stageReviewUrl(r.repo)}
                    className="font-mono text-xs text-teal hover:underline"
                    title={`${r.repo} — open stage review`}
                  >
                    {shortRepo(r.repo)}
                  </a>
                  {r.n_uncertain > 0 && (
                    <span className="ml-2 rounded-full bg-gold/15 px-2 py-0.5 text-[10px] text-ink-soft">
                      {r.n_uncertain} uncertain
                    </span>
                  )}
                  {mismatch && (
                    <span
                      className="ml-2 rounded-full bg-coral/15 px-2 py-0.5 text-[10px] text-coral"
                      title={`registered num_episodes (${r.num_episodes}) < labeled episodes (${labeled}) — stale dataset registration`}
                    >
                      count mismatch
                    </span>
                  )}
                </td>
                <td className="py-1.5 pr-3 text-right font-mono text-xs text-ink-soft">
                  {r.num_episodes ?? "?"}
                </td>
                <td className="py-1.5">
                  <StackedBar
                    total={total}
                    segments={[
                      {
                        label: `${r.n_committed}`,
                        value: r.n_committed,
                        color: HUMAN_COLOR,
                        title: `human committed: ${r.n_committed}`,
                      },
                      {
                        label: `${r.n_vlm_only}`,
                        value: r.n_vlm_only,
                        color: VLM_COLOR,
                        title: `pipeline only: ${r.n_vlm_only}`,
                      },
                      {
                        label: `${unlabeled}`,
                        value: unlabeled,
                        color: UNLABELED_COLOR,
                        title: `unlabeled: ${unlabeled}`,
                      },
                    ]}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {!readOnly && <>
      {/* Stage distributions */}
      <div className="mb-6 grid gap-3">
        <StageHistBar
          hist={t.stage_hist_current}
          nStages={t.n_stages}
          caption="Current labels (committed review beats latest prefill) — stage share"
        />
        <StageHistBar
          hist={t.stage_hist_committed}
          nStages={t.n_stages}
          caption="Human-committed subset — stage share (review-targeted, selection-biased)"
        />
        {t.n_unknown_stage > 0 && (
          <div className="text-xs text-coral">
            {t.n_unknown_stage} labeled episode(s) without a readable stage value
          </div>
        )}
      </div>

      {/* Pipelines + reviewers */}
      <div className="grid gap-6 sm:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-muted">
            Pipelines (operative prefills)
          </h3>
          {t.pipelines.length === 0 ? (
            <div className="text-xs text-ink-muted">no prefills pushed</div>
          ) : (
            <table className="w-full text-xs">
              <tbody>
                {t.pipelines.map((p) => (
                  <tr key={`${p.name}@${p.version}|${p.model}`} className="border-t border-warm-200/60">
                    <td className="py-1 pr-2 font-mono">
                      {p.name}@{p.version}
                    </td>
                    <td className="py-1 pr-2 text-ink-muted">{p.model}</td>
                    <td className="py-1 text-right font-mono">{p.n.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-muted">
            Reviewers (latest rows)
          </h3>
          {t.reviewers.length === 0 ? (
            <div className="text-xs text-ink-muted">no reviews yet</div>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-ink-muted">
                  <th className="py-1 pr-2 font-medium">Reviewer</th>
                  <th className="py-1 pr-2 text-right font-medium">Committed</th>
                  <th className="py-1 pr-2 text-right font-medium">Uncertain</th>
                  <th className="py-1 text-right font-medium">Drafts</th>
                </tr>
              </thead>
              <tbody>
                {t.reviewers.map((r) => (
                  <tr key={r.reviewer} className="border-t border-warm-200/60">
                    <td className="py-1 pr-2 font-mono">{r.reviewer}</td>
                    <td className="py-1 pr-2 text-right font-mono">{r.committed}</td>
                    <td className="py-1 pr-2 text-right font-mono">{r.uncertain}</td>
                    <td className="py-1 text-right font-mono">{r.draft}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      </>}
    </div>
  );
}

function TaskSectionLoader({ task, includeAll, readOnly }: { task: string; includeAll: boolean; readOnly: boolean }) {
  const t = useTaskCoverage(task, includeAll);
  if (t === undefined) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
        Loading {task}…
      </div>
    );
  }
  if (t === null || t.repos.length === 0) {
    return (
      <div className="text-xs text-ink-muted">No arena stage data yet for {task}.</div>
    );
  }
  return <TaskSection t={t} readOnly={readOnly} />;
}

export default function CoverageDashboard({readOnly = false}: {readOnly?: boolean}) {
  // Same global mainline/all lens as the other tabs (header toggle owns it).
  const [showParam] = useSearchParam("show", "mainline");
  const showAll = showParam === "all";
  const allTasks = useQuery(api.stageCoverage.tasks);
  if (allTasks === undefined) {
    return (
      <div className="bg-white rounded-2xl border border-warm-200 shadow-sm p-8 text-center text-ink-muted">
        Loading coverage…
      </div>
    );
  }
  const tasks = allTasks.filter((t) => showAll || t.status === "mainline");
  return (
    <div className="grid gap-8" style={{ animation: "fade-up 0.6s ease-out both" }}>
      {readOnly ? <p className="text-sm text-ink-muted">Stage annotation coverage at release export, restricted to the released repositories. Machine stage labels and human stage reviews are separate from evaluation success labels.</p> : <p className="text-sm text-ink-muted">
        Live from the review database — every prefill push and review save shows up here
        immediately. Datasets link into the stage-review surface.
      </p>}
      {tasks.map((t) => (
        <TaskSectionLoader
          key={`${t.task}@${t.taxonomy_version}`}
          task={t.task}
          includeAll={showAll}
          readOnly={readOnly}
        />
      ))}
    </div>
  );
}
