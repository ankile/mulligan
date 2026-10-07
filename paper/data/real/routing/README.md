# Route Cable (routing_d2) evaluation tables

This folder carries 50 starts, 15 policies, and 750 rollouts, all with an outcome human-reviewed from video, as released in `mulligan/real-routing-d2-r00-r05-eval` (`provenance.json`, `label_rule` and `review_rule`). Under the review rule, an episode that seats only the right-most (second) clip has not seated the first clip: the 31 episodes noted "Second clip only" carry no first-clip mark. Full successes total 128 of 750 (HG-DAgger R4 4/50, R5 9/50; HG-DAgger+Mulligan R4 15/50).

| File | Content |
|---|---|
| `paired_round_outcomes.csv` | per start, the outcome, success, steps and clip marks of all 15 arms |
| `policy_summary.csv` | per arm, success counts and the clip score |
| `routing_d2_headline_sr.csv`, `routing_d2_headline_sr_full_success.csv`, `routing_d2_headline_task_progress.csv` | the headline tables (clip success rate, full success, task progress) |
| `provenance.json` | labeling and review rules, the round mapping, and the sha256 of the files `paper.stats.routing_d2_verify` checks |

| Paper checkpoint | Collection-round checkpoint | Training episodes per arm |
|---|---|---|
| R0 | R0 | 100 |
| R1 | R2 | 200 |
| R2 | R4 | 300 |
| R3 | R6 | 400 |
| R4 | R8 | 500 |
| R5 | R9 | 600 |

The main headline uses clip success rate from `routing_d2_headline_task_progress.csv`: mean 0–2 clip score divided by two, expressed as a percentage. Its whiskers are one standard error across episode scores. The second moment is recovered from total clips and full successes, so the two clips within an episode are not treated as independent trials. The detailed arm plot, endpoint bars, and efficiency plots retain full two-clip success. No early checkpoints are omitted. The combined Ours curve uses DP through R2 and BoN from R3, without selecting the better arm at each checkpoint.

The final-round comparisons are paired bootstrap 95% intervals with 20,000 resamples and seed 20260909, plus exact two-sided McNemar tests; `paper/stats/routing_d2_verify.py` recomputes them with the shared statistics code and checks every headline count against the paired outcomes. The endpoint HiL-IDQL+Mulligan-minus-HG-DAgger contrast is +16.0 pp [+2, +30], exact McNemar p=0.077. Checkpoint-specific tests are exploratory and unadjusted. The round-pooled campaign row of `tab:appendix-campaign-mcnemar` (HG-DAgger vs. HG-DAgger+Mulligan, 33 vs. 51 of 300, +6.0 pp [+1, +11], exact McNemar p=0.0356) is built by `paper.appendix.real_results` from the `real_results` evidence, like the Marker and Nut rows.

Collection and burden panels use the 100-episode increments, which regroup the 50-episode collection sessions (see `paper/appendix/real_results/README.md`). The Cable figures and the Cable rows of the appendix tables read the identical copy in the `real_results` paper evidence (`real/results/routing/headline/`, configured by `paper/appendix/real_results/config.json`).

Check the tables and rebuild the Cable figures (after `uv sync`):

```sh
python -m paper.stats.routing_d2_verify                   # hashes, all 15 arms, endpoint tests
python -m paper.figures --only headline_real_sim real_world_headline_mulligan_vs_baseline_detailed real_world_success_speed real_world_success_throughput
```
