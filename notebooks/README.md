# Analysis notebooks (pilot v2)

These notebooks analyse the multilingual **pilot v2** run. They read aggregated judge outputs under `data/` and stress-test the main claims from [`pilot_v2_eval_exploration.ipynb`](pilot_v2_eval_exploration.ipynb). Judging and aggregation are documented in [`refusalbench/multilingual/README.md`](../refusalbench/multilingual/README.md).

## Shared reference

[`refusalbench.md`](refusalbench.md) is a local copy of the RefusalBench paper (methodology, metrics, taxonomy). The pilot notebooks use the same refusal codes and metric definitions as [`refusalbench.multilingual.aggregate`](../refusalbench/multilingual/aggregate.py), not a separate scoring scheme.

## Data layout

Most pilot artefacts are gitignored (see repo `.gitignore`). The only tracked pilot file in `data/` is:

| Path | Role |
|------|------|
| `data/pilot_v2/refusalbench_translation_shared_prompts.csv` | Translated judge prompt segments and labels (used by `judge_bedrock` and the judge-language notebook) |

Everything below is expected on disk when you run the notebooks but may not be in the repository.

| Path | Role |
|------|------|
| `data/pilot_v2_infer/inference_results/0/{en,pl,ru,zh_cmn,zh_yue}.jsonl` | Full run: model responses for 180 base items × 5 languages (900 records) |
| `data/pilot_v2_infer/inference_results/3/`, `5/`, `8/` | Leakage-filtered subsets of the same prompts (18, 30, 55 items). Same filename pattern per language |
| `data/pilot_v2_eval/` | Canonical evaluation: English judge instructions for all languages |
| `data/pilot_v2_eval_localised/` | Same inference, judges instructed in each target language (`--judge-prompt-language target`) |

### Aggregated CSVs (both eval directories)

Produced by [`aggregate.py`](../refusalbench/multilingual/aggregate.py) from `judgments_raw.jsonl`:

- `consensus.csv`, `judgments_merged.csv`, `metrics_by_language.csv`, `metrics_by_stratum.csv`, `judge_agreement.csv`, `failures.csv`

Judge step also writes `judgments_raw.jsonl` and appends to `failures.csv` before aggregation.

## How to generate inputs (`refusalbench/multilingual`)

There are no `[project.scripts]` entry points in `pyproject.toml`. Run modules from the **repository root** with `uv run python -m ...` (after `uv sync --all-groups` for notebook dependencies).

**Prerequisites:** Model inference JSONL under `data/pilot_v2_infer/inference_results/0/` (not produced by this package). AWS Bedrock credentials and judge model IDs in `.env` (see [`.env.example`](../.env.example): `AWS_BEARER_TOKEN` or key pair, `AWS_REGION`, `REFUSALBENCH_JUDGE_*`).

### 1. Run judges (Bedrock)

```bash
uv run python -m refusalbench.multilingual.judge_bedrock \
  --input-dir data/pilot_v2_infer/inference_results/0 \
  --out-dir data/pilot_v2_eval \
  --prompts-csv data/pilot_v2/refusalbench_translation_shared_prompts.csv
```

Defaults match the above (`--input-dir`, `--out-dir`, `--prompts-csv`, `--judge-prompt-language english`, languages `en pl ru zh_cmn zh_yue`, three judges from env).

**Localised judge prompts** (for [`pilot_v2_judge_prompt_language.ipynb`](pilot_v2_judge_prompt_language.ipynb)):

```bash
uv run python -m refusalbench.multilingual.judge_bedrock \
  --judge-prompt-language target \
  --out-dir data/pilot_v2_eval_localised
```

Use a separate `--out-dir` so canonical and localised runs are not mixed.

### 2. Aggregate to CSVs

```bash
uv run python -m refusalbench.multilingual.aggregate \
  --judgments data/pilot_v2_eval/judgments_raw.jsonl \
  --input-dir data/pilot_v2_infer/inference_results/0/ \
  --out-dir data/pilot_v2_eval/
```

For the localised run:

```bash
uv run python -m refusalbench.multilingual.aggregate \
  --judgments data/pilot_v2_eval_localised/judgments_raw.jsonl \
  --input-dir data/pilot_v2_infer/inference_results/0/ \
  --out-dir data/pilot_v2_eval_localised/
```

| Notebook | Primary data |
|----------|----------------|
| `pilot_v2_eval_exploration.ipynb` | `data/pilot_v2_eval/` (`consensus.csv`, `judgments_merged.csv`, `metrics_by_language.csv`, `metrics_by_stratum.csv`, `judge_agreement.csv`, `failures.csv`) |
| `pilot_v2_leakage_subsets.ipynb` | `data/pilot_v2_eval/consensus.csv` plus `data/pilot_v2_infer/inference_results/{0,3,5,8}/*.jsonl` |
| `pilot_v2_judge_prompt_language.ipynb` | `data/pilot_v2_eval/` and `data/pilot_v2_eval_localised/` (same CSV set), plus `data/pilot_v2/refusalbench_translation_shared_prompts.csv` for prompt checks |

Leakage subset directories `3`, `5`, and `8` are **not** created by `judge_bedrock` or `aggregate`. They are extra inference exports alongside run `0`.

## Notebooks

Read in this order for the narrative arc: exploration, then leakage, then judge language.

### [`pilot_v2_eval_exploration.ipynb`](pilot_v2_eval_exploration.ipynb)

Loads `data/pilot_v2_eval` and interprets what the pilot can and cannot support (180 items × 5 languages, balanced design, metric caveats for pure strata and calibrated scores).

**Figures (matplotlib/seaborn):** inter-judge agreement (κ vs ρ), answerable vs unanswerable counts, confusion-style view on unanswerable items, language intervals vs English, paired disagreement against English on answerable items, perturbation and intensity slices.

**Imports:** `refusalbench.multilingual.aggregate` (`compute_metrics_block`, label constants).

### [`pilot_v2_leakage_subsets.ipynb`](pilot_v2_leakage_subsets.ipynb)

Tests whether English left in translated **prompts** explains the cross-language false-refusal result. Reuses full-run judgments and filters by item id. Compares David's inference subsets `3`, `5`, `8` to run `0`, and builds two per-item leakage scores on all 180 items.

**Figures:** leakage measure agreement, subset metrics vs full run, continuous leakage vs cross-language gap, permutation-style plots.

**Does not re-run** `judge_bedrock` or `aggregate` for subsets.

### [`pilot_v2_judge_prompt_language.ipynb`](pilot_v2_judge_prompt_language.ipynb)

Compares canonical (`data/pilot_v2_eval`) vs localised judge instructions (`data/pilot_v2_eval_localised`). Uses English as a placebo for judge non-determinism. Addresses whether English-only judge prompts could fake the translation effect described in the exploration notebook.

**Figures:** verdict flips vs placebo floor, false refusal under both instruments, Holm-adjusted significance, heatmap of consensus metric shifts.

**Imports:** `aggregate` constants and `judge_bedrock` prompt builders with the shared prompts CSV.

## How to run the notebooks

1. Install dev dependencies: `uv sync --all-groups` (Jupyter, seaborn, plotly are in the `dev` group).
2. Place or generate data paths above (at minimum `data/pilot_v2_eval` for the main notebook).
3. Start Jupyter from the repo root or from `notebooks/`:

   ```bash
   uv run jupyter lab
   ```

Each notebook sets `REPO = Path.cwd()` and, if `data/pilot_v2_eval` is missing, sets `REPO` to the parent directory. Running with cwd = repo root or `notebooks/` both work once `data/pilot_v2_eval` exists in one of those locations.

The package must be importable (`refusalbench.multilingual`). An editable install via `uv sync` satisfies that.
