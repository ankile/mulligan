# critic_data

Critic training-data composition (`app:critic-data`): the human-only vs all-data critic table
(`tab:appendix-critic-data`, `paper/build/tables/critic_data_table.tex`, `prepare.py`).

Inputs pinned by `inputs.json` (`sim/critic/data_ablation/` of the paper evidence) are the
ablation outputs:

| Input | Content |
|---|---|
| `paired_seeds.csv`, `summary.json` | per-seed results and the summary of the critic data ablation |

Ten human-only DIVL critic fits (`--dataset.critic_sampling_mode=human_only`) paired by seed
with the ten existing all-data DIVL controls, behind the same frozen R3 actor; final-step N=1
and N=32 success on the 400 fixed Sobol starts. The table recomputes each mean and Student-t
95% interval from the per-seed CSV and asserts it matches the summary.

```sh
python -m paper.appendix.critic_data.prepare           # write the table
python -m paper.appendix.critic_data.prepare --check   # require the manuscript's table body
```
