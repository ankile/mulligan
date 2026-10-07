import { useState } from 'react';
import { fitBradleyTerry, pairOutcomesFromRounds } from '../../convex/bradleyTerry';
import { RoundVideos } from '../components/RoundVideos';
import { exactSignTestPValue, formatPValue } from '../lib/pairedStats';
import type { RoundVideoSpec } from '../lib/roundVideoSpecs';
import { selectedTask } from './types';
import type { Release, Task, Block, Policy, Dataset } from './types';
import '../index.css';
import './overview.css';

const percent = (n: number) => `${(100 * n).toFixed(1)}%`;
const hf = (id: string, revision?: string) => `https://huggingface.co/datasets/${id}${revision ? `/tree/${revision}` : ''}`;

function Results({ task }: {task: Task}) {
  const methods = [...new Set(task.policies.map(p => p.method))];
  const rounds = [...new Set(task.policies.map(p => p.round))];
  const [detail, setDetail] = useState<Policy | null>(null);
  return <>
    <div className="section-heading"><h2>Learning across rounds</h2><span>{task.metric === 'task_progress' ? 'Average task progress' : 'Success rate'}</span></div>
    <p className="muted">{task.domain === 'sim' ? 'Five training seeds per point. Intervals are Student-t 95% confidence intervals across seeds.' : task.metric === 'task_progress' ? 'Progress is the fraction of two clip seats completed. Intervals are one standard error of the per-episode clip score, matching the paper. Full success is reported separately.' : 'Intervals are Wilson one-standard-error intervals, matching the paper. Each cell reports successes / evaluated episodes.'} Select a result for its evidence.</p>
    <div className="table-scroll"><table><thead><tr><th>Method</th>{rounds.map(r => <th key={r}>{r}</th>)}</tr></thead><tbody>{methods.map(method => <tr key={method}><th>{method}</th>{rounds.map(round => {
      const p = task.policies.find(p => p.method === method && p.round === round);
      return <td key={round}>{p ? <button className="result-cell" onClick={() => setDetail(p)} aria-label={`${method} ${round}: ${percent(p.rate)}`}><strong>{percent(p.rate)}</strong><small>{percent(p.lo)}–{percent(p.hi)}</small><small>{p.seeds ? '5 seeds' : `${p.successes} / ${p.episodes} full successes`}</small></button> : <span className="muted">Not evaluated</span>}</td>;
    })}</tr>)}</tbody></table></div>
    {detail && <aside className="evidence"><button className="close" onClick={() => setDetail(null)} aria-label="Close evidence">×</button><h3>{detail.method} · {detail.round}</h3><p>{detail.provenance}</p><code>{detail.modelId}</code>{detail.seeds && <ul>{detail.seeds.map(s => <li key={s.seed}>Seed {s.seed}: {percent(s.rate)} {s.dataUrl && <a href={s.dataUrl}>Per-state outcomes ↗</a>}<br /><code>{s.artifact}</code></li>)}</ul>}</aside>}
  </>;
}

function ComparisonBlock({ task, block }: {task: Task; block: Block}) {
  const [startIndex, setStartIndex] = useState(block.starts[0].index);
  const [camera, setCamera] = useState(block.cameras.includes('observation.images.side_1') ? 'observation.images.side_1' : block.cameras[0]);
  const start = block.starts.find(s => s.index === startIndex)!;
  const policies = task.policies.filter(p => p.blocks.includes(block.id));
  const names = new Map(policies.map(p => [p.id, p.method]));
  const videos: RoundVideoSpec[] = start.results.map((r, i) => {
    const v = r.videos[camera];
    if (!v) throw new Error('Missing selected camera metadata');
    return {policyName: names.get(r.policyId)!, success: r.success, numSubtaskMarks: r.marks,
      maxSubtaskMarks: task.metric === 'task_progress' ? 1 : 0, episodeIndex: r.episode,
      datasetRepo: block.dataset, cameraKey: camera, badge: String(i + 1),
      videoUrl: `https://huggingface.co/datasets/${block.dataset}/resolve/${block.revision}/${v.path}`,
      episode: {episodeIndex: r.episode, numFrames: r.frames, duration: r.frames / block.fps,
        videoFileIndex: v.file, fromTimestamp: v.from, toTimestamp: v.to}};
  });
  const ratings = fitBradleyTerry(pairOutcomesFromRounds(block.starts.map(s => s.results.map(r => ({id:r.policyId, success:r.success})))));
  const pairs = [];
  for (let i = 0; i < policies.length; i++) for (let j = i + 1; j < policies.length; j++) {
    const a = policies[i], b = policies[j]; let wins = 0, losses = 0, draws = 0;
    for (const s of block.starts) {
      const x = s.results.find(r => r.policyId === a.id), y = s.results.find(r => r.policyId === b.id);
      if (!x || !y) continue;
      if (x.success === y.success) draws++; else if (x.success) wins++; else losses++;
    }
    pairs.push({a, b, wins, losses, draws});
  }
  return <>
    <div className="controls"><label>Initial state<select value={startIndex} onChange={e => setStartIndex(Number(e.target.value))}>{block.starts.map(s => <option key={s.index} value={s.index}>State {block.startRange && s.manifestIndex !== undefined ? block.startRange[0] + s.manifestIndex : s.index + 1}</option>)}</select></label><label>Camera<select value={camera} onChange={e => setCamera(e.target.value)}>{block.cameras.map(c => <option key={c} value={c}>{c.replace('observation.images.', '').replaceAll('_', ' ')}</option>)}</select></label><a href={hf(block.dataset, block.revision)}>Open pinned recording ↗</a></div>
    <RoundVideos key={`${block.id}-${startIndex}-${camera}`} videos={videos} />
    <p className="muted">Synchronized recordings from one evaluation session ({block.label ?? block.id}) of <a href={hf(block.roundDataset ?? block.dataset)}>{(block.roundDataset ?? block.dataset).slice('mulligan/'.length)}</a>. Videos include the recorded tail. Binary comparisons below use full success and exact two-sided McNemar tests.</p>
    <div className="table-scroll"><table><thead><tr><th>Method</th><th>Arena rating</th></tr></thead><tbody>{policies.map(p => <tr key={p.id}><td>{p.method}</td><td>{ratings.get(p.id)?.toFixed(0) ?? 'No paired observations'}</td></tr>)}</tbody></table></div>
    <p className="muted">Bradley–Terry ratings are fit over this block's selected full-success comparisons, with draws counting as half a win and the block mean centered at 1500. Ratings from separate blocks are not a cross-round ranking.</p>
    <div className="table-scroll"><table><thead><tr><th>Comparison</th><th>Paired states</th><th>Win / draw / loss</th><th>Exact p</th></tr></thead><tbody>{pairs.map(p => <tr key={p.a.id + p.b.id}><td>{p.a.method} vs {p.b.method}</td><td>{p.wins + p.draws + p.losses}</td><td>{p.wins} / {p.draws} / {p.losses}</td><td>{formatPValue(exactSignTestPValue(p.wins, p.losses))}</td></tr>)}</tbody></table></div>
    <p className="muted">{block.reviewedEpisodes === null ? 'Review coverage varies by source; see the release evidence.' : `${block.reviewedEpisodes} episodes in the original recording have operator reviews; remaining labels retain evaluation-time evidence.`}</p>
  </>;
}

function Compare({ task }: {task: Task}) {
  const [blockId, setBlockId] = useState(task.blocks.at(-1)?.id ?? '');
  if (!task.blocks.length) return <div className="evidence"><h2>Simulation evaluation evidence</h2><p>These results use fixed initial-state grids and five training seeds. The release bundles provide state IDs, actor and critic provenance, and available per-state outcomes. This campaign has no released evaluation videos.</p><a href={hf(`mulligan/${task.datasetTask}-r00-r03-eval`)}>Browse evaluation bundle ↗</a></div>;
  const block = task.blocks.find(b => b.id === blockId)!;
  return <><div className="section-heading"><h2>Compare the same starts</h2></div><label>Evaluation block<select value={blockId} onChange={e => setBlockId(e.target.value)}>{[...task.blocks].sort((a, b) => a.round.localeCompare(b.round) || (a.session ?? '').localeCompare(b.session ?? '')).map((b, i) => <option key={b.id} value={b.id}>{b.round} · {b.label ?? `block ${i + 1}`} · {b.starts.length} starts</option>)}</select></label><ComparisonBlock key={block.id} task={task} block={block} /></>;
}

function Datasets({ task, datasets }: {task: Task; datasets: Dataset[]}) {
  const [query, setQuery] = useState('');
  const [role, setRole] = useState('all');
  const all = datasets.filter(d => d.task === task.datasetTask);
  const rows = all.filter(d => (role === 'all' || d.role === role) && `${d.id} ${d.variant}`.toLowerCase().includes(query.toLowerCase()));
  return <><div className="section-heading"><h2>Data for {task.title}</h2><span>{all.length} repositories</span></div>{task.evaluationDataset && <p><a href={hf(task.evaluationDataset.id, task.evaluationDataset.revision)}>Simulation evaluation outcomes and exact state grids ↗</a></p>}<p className="muted">C indexes collection increments; R indexes evaluated model rounds. Training views share episodes with their raw parent. Prior evaluations can be training input for later models; eligibility is defined by each training recipe.</p><div className="controls"><label>Search<input value={query} onChange={e => setQuery(e.target.value)} placeholder="Dataset or method" /></label><label>Role<select value={role} onChange={e => setRole(e.target.value)}><option value="all">All roles</option>{[...new Set(all.map(d => d.role))].map(r => <option key={r}>{r}</option>)}</select></label></div><div className="dataset-list">{rows.map(d => <article key={d.id}><div><span className="eyebrow">{d.role.replaceAll('-', ' ')} · {d.variant || 'mixed recording'}</span><h3><a href={hf(d.id, d.revision)}>{d.id.slice('mulligan/'.length)} ↗</a></h3><p>{d.episodes.toLocaleString()} episodes · {d.fps} Hz · {d.cameras.length ? `${d.cameras.length} camera streams` : 'state observations'}</p>{d.parent && <small>Derived from <a href={hf(d.parent)}>{d.parent.slice('mulligan/'.length)}</a></small>}</div></article>)}{rows.length === 0 && <p>No datasets match this filter.</p>}</div></>;
}

export default function App({ data }: {data: Release}) {
  const task = selectedTask(data, new URLSearchParams(location.search).get('task'));
  const [tab, setTab] = useState('results');
  return <div className="release-app"><header><a className="wordmark" href="https://mulligan.page">Mulligan<span> / Release overview</span></a><nav><a href="/">Policy Arena</a><a href="https://mulligan.page">Project ↗</a><a href="https://arxiv.org/abs/2610.05882">Paper ↗</a><a href="https://github.com/ankile/mulligan">Code ↗</a><a href="https://huggingface.co/mulligan">Hugging Face ↗</a><a href="/data/catalog.html">Full data catalog ↗</a><a href="/data/release.json">Release data ↓</a></nav></header><main><div className="intro"><p className="eyebrow">Mainline data & results · {data.version}</p><h1>See what the policies learned.</h1><p>Explore the real-world and simulation campaigns behind Mulligan. Compare outcomes, watch the same initial states, and find the data used at each round.</p>{data.state === 'draft' && <p className="draft">Preview build. Publication checks are still in progress.</p>}</div><div className="task-grid">{data.tasks.map(t => <a key={t.id} href={`?task=${t.id}`} aria-current={task.id === t.id ? 'page' : undefined}><span className="eyebrow">{t.domain === 'real' ? 'Real world · D2' : 'Simulation'}</span><h2>{t.title}</h2><span>{new Set(t.policies.map(p => p.round)).size} rounds</span></a>)}</div><div className="tabs" role="tablist" aria-label="Task view">{['results', 'compare', 'datasets'].map(t => <button role="tab" aria-selected={tab === t} key={t} onClick={() => setTab(t)}>{t[0].toUpperCase() + t.slice(1)}</button>)}</div><section>{tab === 'results' ? <Results key={task.id} task={task} /> : tab === 'compare' ? <Compare key={task.id} task={task} /> : <Datasets key={task.id} task={task} datasets={data.datasets} />}</section><details className="release-notes"><summary>Release notes and evidence</summary><p>This release covers the paper's five tasks: {data.tasks.map(t => t.title).join(', ')}. Pilot campaigns and policies used only in ablations are not part of this Arena.</p><p>Thread Nut R5 compares the three policies evaluated together in one session; the critic-guided policy uses its critic after 200k training updates and picks the best of N=32 sampled actions. Route Cable rounds R0–R5 are the campaign's collection rounds 0, 2, 4, 6, 8 and 9. A person reviewed the outcome of all 750 Route Cable episodes.</p><p>Dataset links and video URLs are pinned to fixed dataset revisions. Selected analysis labels and source hashes are included in the downloadable release. Machine stage predictions are separate from success labels.</p><code>Selection SHA-256: {data.selectionSha256}</code></details></main><footer>Mulligan · Performance-Guided Data Collection for Efficient On-Robot Learning</footer></div>;
}

