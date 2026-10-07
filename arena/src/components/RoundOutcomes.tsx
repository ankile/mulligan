// Shared surface for "N arms evaluated round by round": per-arm summary
// cards, the paired-tests drawer, and the one-row-per-round outcome grid.
// Used by the single-session detail (arms = the session's policies) and by
// the joined-sessions view (arms = session letter × policy over rounds
// aligned by index), so both inherit the same layout and statistics.

import { Fragment, useMemo, useState, type ReactNode } from "react";
import { armStats, successStepsSummary, type Arm, type ArmRound } from "../lib/armStats";
import { roundNumber } from "../lib/roundNumber";
export type { Arm, ArmResult, ArmRound } from "../lib/armStats";
import { TONE_PILL, meanScore, outcomeLabel, outcomeTone } from "../lib/outcomeScore";
import { formatPValue, pairedRows, type PairedRow } from "../lib/pairedStats";

export function WdlRecord({
  wins,
  draws,
  losses,
  title,
}: {
  wins: number;
  draws: number;
  losses: number;
  title: string;
}) {
  return (
    <span className="flex items-center gap-1.5 font-mono whitespace-nowrap" title={title}>
      <span className="text-teal font-medium">{wins}W</span>
      <span className="text-ink-muted">{draws}D</span>
      <span className="text-coral font-medium">{losses}L</span>
    </span>
  );
}

/** The arm's short label (`3`, or `B3` in the joined view) as a gold chip. */
export function ArmBadge({ label }: { label: string }) {
  return (
    <span className="inline-flex min-w-4 h-4 px-1 rounded bg-gold text-white text-[10px] font-mono font-semibold items-center justify-center shrink-0 leading-none">
      {label}
    </span>
  );
}

// ── Legend ──

/**
 * Policy number → full name, the one place the names are never truncated.
 * One row per policy: in the joined view a model evaluated in sessions A and
 * B shows both its chips (`A3` `B3`) on the same row, so the number alone
 * says "same model".
 */
export function ArmLegend({ arms }: { arms: Arm[] }) {
  const byNumber = new Map<number, { name: string; labels: string[] }>();
  for (const arm of arms) {
    const entry = byNumber.get(arm.policyNumber);
    if (entry) entry.labels.push(arm.label);
    else byNumber.set(arm.policyNumber, { name: arm.name, labels: [arm.label] });
  }
  const rows = [...byNumber].sort(([a], [b]) => a - b);
  return (
    <div className="mb-4 rounded-lg border border-warm-200 bg-warm-50/40 px-3 py-2 grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-1">
      {rows.map(([n, entry]) => (
        <Fragment key={n}>
          <span className="flex items-center gap-1">
            {entry.labels.map((label) => (
              <ArmBadge key={label} label={label} />
            ))}
          </span>
          <span className="font-mono text-xs text-ink break-all">{entry.name}</span>
        </Fragment>
      ))}
    </div>
  );
}

// ── Per-arm summary cards ──

export function PolicyStatCards({
  arms,
  rounds,
  maxMarks,
}: {
  arms: Arm[];
  rounds: ArmRound[];
  maxMarks: number;
}) {
  const stats = armStats(
    arms.map((a) => a.key),
    rounds,
    maxMarks,
  );
  // Sessions submitted before frame counts were recorded have no step data
  // for any arm; hide the row rather than show a column of dashes.
  const hasStepCounts = rounds.some((round) =>
    round.results.some((result) => result.num_frames !== null),
  );
  return (
    <div className="grid gap-3 mb-4 grid-cols-[repeat(auto-fill,minmax(15rem,1fr))]">
      {arms.map((arm) => {
        const s = stats.get(arm.key)!;
        const successRate = s.n > 0 ? (s.successes / s.n) * 100 : 0;
        const score = meanScore(s.graded, maxMarks);
        const unscored = s.graded.filter((g) => g.marks === null).length;
        const steps = successStepsSummary(s.successSteps);
        const unmeasured = s.successes - s.successSteps.length;
        return (
          <div key={arm.key} className="rounded-xl border border-warm-200 bg-warm-50/50 px-4 py-3">
            <div className="flex items-start gap-1.5 font-body font-semibold text-ink text-sm mb-1.5 min-w-0">
              <span className="pt-0.5">
                <ArmBadge label={arm.label} />
              </span>
              <span className="break-all">{arm.name}</span>
            </div>
            {/* One metric per line: label | value | pairwise record. */}
            <div className="grid grid-cols-[auto_1fr_auto] items-baseline gap-x-3 gap-y-1 text-xs">
              <span className="text-[11px] text-ink-muted font-body">success</span>
              <span className="font-mono text-ink whitespace-nowrap">
                {successRate.toFixed(0)}%
                <span className="text-ink-muted">
                  {" "}
                  {s.successes}/{s.n}
                </span>
              </span>
              <WdlRecord
                wins={s.wins}
                draws={s.draws}
                losses={s.losses}
                title="Pairwise record on binary success"
              />
              {score !== null && (
                <>
                  <span
                    className="text-[11px] text-ink-muted font-body"
                    title={
                      `Mean graded score: sub-goal marks reached + success, ` +
                      `max ${maxMarks + 1} per round` +
                      (unscored > 0
                        ? ` (${unscored} round(s) submitted without a mark count)`
                        : "")
                    }
                  >
                    score{unscored > 0 ? "*" : ""}
                  </span>
                  <span className="font-mono text-ink whitespace-nowrap">
                    {score.toFixed(2)}
                    <span className="text-ink-muted"> / {maxMarks + 1}</span>
                  </span>
                  <WdlRecord
                    wins={s.scoreWins}
                    draws={s.scoreDraws}
                    losses={s.scoreLosses}
                    title={`Pairwise record on the graded 0..${maxMarks + 1} score`}
                  />
                </>
              )}
              {hasStepCounts && (
                <>
                  <span
                    className="text-[11px] text-ink-muted font-body"
                    title={
                      `Mean steps to completion over this arm's successful rounds ` +
                      `(± sample SD). Failures run to the step budget and are excluded.` +
                      (unmeasured > 0
                        ? ` (${unmeasured} success(es) submitted without a step count)`
                        : "")
                    }
                  >
                    steps{unmeasured > 0 ? "*" : ""}
                  </span>
                  {steps === null ? (
                    <span className="font-mono text-ink-muted whitespace-nowrap">—</span>
                  ) : (
                    <span className="font-mono text-ink whitespace-nowrap">
                      {steps.mean.toFixed(0)}
                      {steps.sd !== null && (
                        <span className="text-ink-muted"> ± {steps.sd.toFixed(0)}</span>
                      )}
                    </span>
                  )}
                  <span
                    className="font-mono text-ink-muted whitespace-nowrap"
                    title="Successful rounds with a step count"
                  >
                    n={steps?.n ?? 0}
                  </span>
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Paired-tests drawer ──

const WIKI = {
  paired: "https://en.wikipedia.org/wiki/Paired_difference_test",
  bootstrap: "https://en.wikipedia.org/wiki/Bootstrapping_(statistics)",
  sign: "https://en.wikipedia.org/wiki/Sign_test",
  mcnemar: "https://en.wikipedia.org/wiki/McNemar%27s_test",
  permutation: "https://en.wikipedia.org/wiki/Permutation_test",
};

function WikiLink({ href, children }: { href: string; children: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-teal hover:underline whitespace-nowrap"
    >
      {children} ↗
    </a>
  );
}

/**
 * Faint "?" next to the drawer toggle. The definitions card shows on hover
 * and can be pinned by click (touch, or to follow a link comfortably).
 */
function PairedStatsHelp() {
  const [pinned, setPinned] = useState(false);
  const dt = "font-mono text-ink shrink-0 w-14";
  const dd = "text-ink-muted";
  return (
    <span className="relative group inline-flex">
      <button
        onClick={() => setPinned((v) => !v)}
        aria-label="How the paired tests are computed"
        aria-expanded={pinned}
        className={`w-3.5 h-3.5 rounded-full border text-[9px] leading-none font-medium flex items-center justify-center cursor-pointer transition-colors ${
          pinned
            ? "border-ink-muted text-ink-muted"
            : "border-warm-300 text-warm-400 group-hover:border-ink-muted group-hover:text-ink-muted"
        }`}
      >
        ?
      </button>
      <div
        className={`absolute left-0 top-full z-20 pt-1.5 w-[30rem] max-w-[calc(100vw-4rem)] ${
          pinned ? "block" : "hidden group-hover:block"
        }`}
      >
        <div className="rounded-lg border border-warm-200 bg-white shadow-lg px-3 py-2.5 text-[11px] font-body leading-snug">
          <div className="text-ink mb-1.5">
            Every row is a{" "}
            <WikiLink href={WIKI.paired}>paired comparison</WikiLink> of two arms X and
            Y: round k of X is compared with round k of Y (same start state), never X's
            pool against Y's. Arm chips are read off the legend above.
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1">
            <dt className={dt}>n</dt>
            <dd className={dd}>
              Rounds where both arms have a result. Score rows also require a mark
              count on both.
            </dd>
            <dt className={dt}>Δ X−Y</dt>
            <dd className={dd}>
              Mean over paired rounds of X's outcome minus Y's: percentage points for
              success, score points for the graded score.
            </dd>
            <dt className={dt}>95% CI</dt>
            <dd className={dd}>
              Paired <WikiLink href={WIKI.bootstrap}>bootstrap</WikiLink>: resample
              rounds with replacement 20,000 times, recompute Δ each time, take the
              2.5th and 97.5th percentiles.
            </dd>
            <dt className={dt}>W/D/L</dt>
            <dd className={dd}>Rounds where X &gt; Y, X = Y, X &lt; Y.</dd>
            <dt className={dt}>sign p</dt>
            <dd className={dd}>
              Exact two-sided <WikiLink href={WIKI.sign}>sign test</WikiLink> on the
              W and L counts (draws dropped): under H₀ each discordant round is a fair
              coin, p = P(a split at least this lopsided). For binary success this is
              identical to the exact{" "}
              <WikiLink href={WIKI.mcnemar}>McNemar test</WikiLink>, which uses the
              same two discordant cells.
            </dd>
            <dt className={dt}>flip p</dt>
            <dd className={dd}>
              Sign-flip <WikiLink href={WIKI.permutation}>permutation test</WikiLink>{" "}
              of mean Δ = 0: under H₀ each round's difference is equally likely to
              have the opposite sign, so flip signs at random 20,000 times and take
              p = share of flips with |mean| ≥ observed (with +1 correction). Unlike
              the sign test it weighs the size of each difference, so it can disagree
              with sign p when a few rounds swing by 2 points.
            </dd>
          </dl>
          <div className="mt-1.5 text-ink-muted">Bold p: p &lt; 0.05.</div>
        </div>
      </div>
    </span>
  );
}

function ArmChip({ arm }: { arm: Arm }) {
  return (
    <span className="inline-flex" title={arm.name}>
      <ArmBadge label={arm.label} />
    </span>
  );
}

function PairedStatsTable({
  rows,
  arms,
  maxMarks,
}: {
  rows: PairedRow[];
  arms: Map<string, Arm>;
  maxMarks: number;
}) {
  const th = "px-2 py-1 text-[10px] uppercase tracking-wider text-ink-muted font-medium whitespace-nowrap";
  const td = "px-2 py-1 font-mono text-xs text-ink whitespace-nowrap";
  const anyDropped = rows.some((r) => r.droppedUnscored > 0);
  return (
    <div className="mt-2 rounded-lg border border-warm-200 bg-warm-50/40 overflow-x-auto">
      <table className="w-full text-left">
        <thead className="border-b border-warm-200">
          <tr>
            <th className={th} title="Arm chips X vs Y; full names in the legend above">
              X vs Y
            </th>
            <th className={th}>Metric</th>
            <th className={`${th} text-right`}>n</th>
            <th className={`${th} text-right`}>X</th>
            <th className={`${th} text-right`}>Y</th>
            <th className={`${th} text-right`} title="Paired per-round difference, mean(X) - mean(Y)">
              Δ X−Y
            </th>
            <th className={th} title="Paired bootstrap 95% CI on Δ (20,000 resamples of rounds)">
              95% CI
            </th>
            <th className={th} title="Rounds where X beat / tied / lost to Y">
              W/D/L
            </th>
            <th
              className={`${th} text-right`}
              title="Exact two-sided sign test on the discordant rounds (= exact McNemar for binary success)"
            >
              sign p
            </th>
            <th
              className={`${th} text-right`}
              title="Two-sided sign-flip permutation test of mean Δ = 0 (20,000 flips); weighs the size of each round's difference"
            >
              flip p
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const armA = arms.get(row.policyA)!;
            const armB = arms.get(row.policyB)!;
            const isScore = row.metric === "score";
            const fmt = (x: number) => (isScore ? x.toFixed(2) : `${(x * 100).toFixed(0)}%`);
            const fmtDelta = (x: number) =>
              isScore
                ? `${x >= 0 ? "+" : ""}${x.toFixed(2)}`
                : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(0)} pp`;
            const s = row.stats;
            return (
              <tr key={`${row.policyA}:${row.policyB}:${row.metric}`} className="border-b border-warm-100 last:border-b-0">
                <td className={`${td} font-body`}>
                  <span className="flex items-center gap-1.5">
                    <ArmChip arm={armA} />
                    <span className="text-ink-muted">vs</span>
                    <ArmChip arm={armB} />
                  </span>
                </td>
                <td className={`${td} font-body text-ink-muted`}>
                  {isScore ? `score 0–${maxMarks + 1}` : "success"}
                  {row.droppedUnscored > 0 ? "*" : ""}
                </td>
                {s === null ? (
                  <td className={`${td} text-ink-muted`} colSpan={8}>
                    no paired rounds
                  </td>
                ) : (
                  <>
                    <td className={`${td} text-right`}>{s.n}</td>
                    <td className={`${td} text-right`}>{fmt(s.meanA)}</td>
                    <td className={`${td} text-right`}>{fmt(s.meanB)}</td>
                    <td className={`${td} text-right font-medium`}>{fmtDelta(s.delta)}</td>
                    <td className={`${td} text-ink-muted`}>
                      [{fmtDelta(s.ciLo)}, {fmtDelta(s.ciHi)}]
                    </td>
                    <td className={td}>
                      <WdlRecord
                        wins={s.wins}
                        draws={s.draws}
                        losses={s.losses}
                        title={`${armA.name} record vs ${armB.name}`}
                      />
                    </td>
                    <td className={`${td} text-right ${s.signPValue < 0.05 ? "font-semibold" : ""}`}>
                      {formatPValue(s.signPValue)}
                    </td>
                    <td className={`${td} text-right ${s.signFlipPValue < 0.05 ? "font-semibold" : ""}`}>
                      {formatPValue(s.signFlipPValue)}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {anyDropped && (
        <div className="px-2 py-1.5 text-[10px] text-ink-muted font-body border-t border-warm-200">
          * Score rows use only rounds where both arms carry a mark count.
        </div>
      )}
    </div>
  );
}

/**
 * Collapsible "Paired tests" drawer. Pass memoized `arms` / `rounds`: the
 * 20k-resample statistics are recomputed whenever either identity changes.
 */
export function PairedTestsDrawer({
  arms,
  rounds,
  maxMarks,
}: {
  arms: Arm[];
  rounds: ArmRound[];
  maxMarks: number;
}) {
  const [open, setOpen] = useState(false);
  const table = useMemo<PairedRow[] | null>(
    () =>
      open
        ? pairedRows(
            arms.map((a) => a.key),
            rounds,
            maxMarks,
          )
        : null,
    [arms, rounds, maxMarks, open],
  );
  return (
    <div className="mb-4 -mt-1">
      <div className="flex items-center gap-2">
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 text-[11px] text-ink-muted hover:text-ink transition-colors cursor-pointer"
          aria-expanded={open}
        >
          <svg
            width="10"
            height="10"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform ${open ? "rotate-90" : ""}`}
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
          Paired tests
        </button>
        {open && <PairedStatsHelp />}
      </div>
      {open && table && (
        <PairedStatsTable
          rows={table}
          arms={new Map(arms.map((a) => [a.key, a]))}
          maxMarks={maxMarks}
        />
      )}
    </div>
  );
}

// ── Rounds grid ──

/**
 * One grid row per round: Round | one cell per arm (header row carries the
 * arm chip + name, so a pill only holds the outcome and never wraps) | chevron.
 * An arm without a result in a round shows a dash.
 */
export function RoundsGrid({
  arms,
  rounds,
  maxMarks,
  expandedRound,
  expandAll,
  canExpand,
  onToggleRound,
  renderExpanded,
}: {
  arms: Arm[];
  rounds: ArmRound[];
  maxMarks: number;
  expandedRound: number | null;
  expandAll: boolean;
  /** False while video metadata is unavailable: no chevron, rows inert. */
  canExpand: boolean;
  onToggleRound: (index: number | null) => void;
  renderExpanded: (round: ArmRound) => ReactNode;
}) {
  const gridStyle = {
    gridTemplateColumns: `4.5rem repeat(${arms.length}, minmax(0, 1fr)) 0.75rem`,
  };
  return (
    <div className="space-y-2">
      <div className="grid items-center gap-x-2 px-3" style={gridStyle}>
        <span className="text-[10px] uppercase tracking-wider text-ink-muted/70 font-medium">
          Round
        </span>
        {arms.map((arm) => (
          <span
            key={arm.key}
            className="flex items-center gap-1 text-[11px] font-mono text-ink-light min-w-0"
            title={arm.name}
          >
            <ArmBadge label={arm.label} />
            <span className="truncate">{arm.name}</span>
          </span>
        ))}
        <span />
      </div>
      {rounds.map((round) => {
        const isExpanded = canExpand && (expandAll || expandedRound === round.index);
        const byArm = new Map(round.results.map((r) => [r.policy_id, r]));
        return (
          <div key={round.index} className="rounded-lg bg-warm-50/50 overflow-hidden">
            <button
              onClick={() => {
                if (expandAll || !canExpand) return;
                onToggleRound(isExpanded ? null : round.index);
              }}
              className="w-full grid items-center gap-x-2 px-3 py-2 hover:bg-warm-50 transition-colors cursor-pointer text-left"
              style={gridStyle}
            >
              <span className="text-xs font-mono text-ink-muted">
                {round.label ?? `Round ${roundNumber(round.index)}`}
                {round.results.some((r) => r.visit_id) && (
                  <span
                    className="block text-[10px] text-ink-muted/70"
                    title={[...new Set(round.results.map((r) => r.visit_id).filter(Boolean))].join("\n")}
                  >
                    {new Set(round.results.map((r) => r.visit_id)).size === 1
                      ? "1 launch"
                      : `${new Set(round.results.map((r) => r.visit_id)).size} launches`}
                  </span>
                )}
              </span>
              {arms.map((arm) => {
                const result = byArm.get(arm.key);
                if (!result) {
                  return (
                    <span key={arm.key} className="text-[11px] text-ink-muted/50 px-2">
                      —
                    </span>
                  );
                }
                return (
                  <span
                    key={arm.key}
                    className={`block truncate px-2 py-0.5 rounded text-[11px] font-medium text-center ${
                      TONE_PILL[outcomeTone(result.success, result.num_subtask_marks)]
                    }`}
                    title={`${arm.name} — episode ${result.episode_index}${result.visit_id ? ` — launch ${result.visit_id}` : ""}`}
                  >
                    {outcomeLabel(result.success, result.num_subtask_marks, maxMarks)}
                  </span>
                );
              })}
              {canExpand ? (
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className={`text-ink-muted/50 transition-transform duration-200 shrink-0 ${
                    isExpanded ? "rotate-90" : ""
                  }`}
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              ) : (
                <span />
              )}
            </button>
            {isExpanded && <div className="px-3 pb-3">{renderExpanded(round)}</div>}
          </div>
        );
      })}
    </div>
  );
}

export function ExpandAllButton({
  expandAll,
  onToggle,
}: {
  expandAll: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all cursor-pointer ${
        expandAll
          ? "bg-teal text-white border-teal shadow-sm"
          : "bg-white text-ink-muted border-warm-200 hover:border-teal/40 hover:text-ink"
      }`}
    >
      {expandAll ? "Collapse all" : "Expand all rounds"}
    </button>
  );
}
