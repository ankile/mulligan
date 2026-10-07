# hilserl

HiL-SERL in simulation (`app:baselines-hilserl`): the per-session operator-cost table
(`tab:appendix-hilserl-sessions`, `paper/build/tables/hilserl_sessions.tex`, `prepare.py`) and
the eval curves of every session against RLPD (figure `sim_hilserl_sessions.pdf`, `plot.py`).

Inputs pinned by `inputs.json` (`sim/results/hilserl/` of the paper evidence) are the HiL-SERL
session records, one seed per session:

| Input | Content |
|---|---|
| `square_narrow/eval.csv`, `square_narrow/status.json` | Square-Narrow sessions: eval curves and session totals |
| `square_broad/eval.csv`, `square_broad/status.json` | Square-Broad sessions: eval curves and session totals |

The eval CSVs hold the 5-seed RLPD mean and SE (`ref_mean_sr`, `ref_se`), the
operator-free run (`split_nohuman_sr`), and each session's policy-only eval (`<session>_sr`,
50 episodes per 10k-step milestone). Session totals (episodes, human frames, takeovers,
successes, operator hours) come from the actor ledgers; a forked session counts only its
post-fork episodes, and `forked_from_step` records its fork step.

```sh
python -m paper.appendix.hilserl.prepare           # write the table
python -m paper.appendix.hilserl.prepare --check   # require the manuscript's table body
python -m paper.figures --only sim_hilserl_sessions
```
