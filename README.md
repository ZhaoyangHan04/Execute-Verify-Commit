# Execute–Verify–Commit (EVC)

Implementation-only companion: core method, benchmark adapters, and paper settings.
No benchmark data, credentials, experiment outputs, cached responses, or historical
results are included. This is not a complete historical-results reproduction package.

## Where to look

| Path | Contents |
| --- | --- |
| `shadow-verifier/src/shadow_verifier/runtime.py` | Stage once; review; publish that exact candidate or discard it. Majority voting and probability-based early stopping. |
| `shadow-verifier/src/shadow_verifier/reviewers/` | Current rubric, hard/probability output contracts, and reviewer-visible evidence projection. |
| `shadow-verifier/src/shadow_verifier/backends/` | Generative API transport, Jev typed decisions and fixed category feedback. |
| `workflow/{tau_bench,tau2,state_bench}/` | Dataset loading, isolated execution, native observations, and final evaluation. |
| `configs/paper.json` | Models, seeds, role-specific generation limits, task budgets, upstream revisions. |
| `patches/` | Cabin persistence and retail multi-item variant fixes, applied equally to baseline and EVC. |
| `experiments/` | Explicit configuration resolver and task-level launchers. |
| `tests/` | Synthetic offline checks; no benchmark examples. |

## Configure and inspect

Use Python 3.12 or 3.13 in an isolated environment, then `python -m pip install -e .`.

```bash
# No API calls or data downloads: prints the complete resolved configuration.
python -m experiments.run --benchmark tau_bench
python -m experiments.run --benchmark tau2 --producer glm-5.2 --reviewer self --setting b1
python -m experiments.run --benchmark state_bench --setting prob_es --seeds 42
python -m unittest discover -s tests -v
```

Default: Qwen3.8-max producer/reviewer, B1, seeds 42–45, full public context and
tool list, no policy unit tests, no completion check. `baseline` bypasses review;
`b1` uses one hard decision; `b3/b5/b7` use strict majority. `prob_es` samples at
most five probabilities: first `p<0.3` rejects or `p>0.7` approves; otherwise the
mean must exceed 0.7. `hidden_effect` hides only candidate-effect evidence.
These compute/mechanism settings use Qwen self-review, as in the paper.
`jev` uses Jev-1.13.0 approval probability and rejection-category choices; it
returns fixed category feedback, not a generated explanation.

Generative B1 and probability review use temperature 0; multi-vote hard review
uses 0.3. Kimi-K3 uses native sampling without a generation seed. Producers are
capped at 8,192 output tokens and generative reviewers at 512. The outcome judge
stays Qwen3.8-max, independent of reviewer choice. The paper's simulator name is
Qwen3.8-max; the confirmed runtime alias remains `qwen-max` for τ/τ².

## Optional execution with external data

```bash
# Explicitly downloads the public benchmark into ignored external/ and applies its patch.
python -m experiments.prepare tau_bench
python -m pip install -e external/tau-bench

# Set DASHSCOPE_API_KEY in your process environment. Jev also needs TYPESAFE_API_KEY.
# Choose task IDs from your installed official split; four seeds run by default.
python -m experiments.run --benchmark tau_bench --domain retail --tasks 0 --setting b1 --execute
```

Prepare `tau2` or `state_bench` similarly, installing `external/tau2-bench` or
`external/state-bench`. Use a separate environment if upstream dependencies conflict.
For STATE, task IDs must be taken from its official test split. `--workers`
controls concurrent trajectories (default 4); `--output` defaults to ignored
`outputs/`. API errors are surfaced, not silently scored as failures. Credentials
are read only from environment variables. Do not commit generated directories.
