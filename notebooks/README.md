# Analysis notebooks (pilot v2)

> **Start here for results:** [`FINDINGS.md`](FINDINGS.md) summarises what the three notebooks found and is meant to be readable on its own. Its section 0 defines every statistical term used, so no background is assumed. This README covers how to regenerate the data and run the notebooks.

These notebooks analyse the multilingual **pilot v2** run. They read aggregated judge outputs under `data/` and stress-test the main claims from `pilot_v2_eval_exploration`. Judging and aggregation are documented in [`refusalbench/multilingual/README.md`](../refusalbench/multilingual/README.md).

## Notebook format: marimo `.py` is the source of truth

Each notebook exists twice:

| File | Role |
|------|------|
| `pilot_v2_*.py` | **marimo notebook, the source of truth.** Edit this one. |
| `rendered/pilot_v2_*.ipynb` | Generated artefact for GitHub preview. **Never hand-edit.** Lives in [`rendered/`](rendered/). |

Both the `.ipynb` copies and the seven PNGs in `figures/` are produced by [`export_figures.py`](export_figures.py), which executes each marimo notebook via `marimo export ipynb --include-outputs` and then pulls each figure out of the fresh outputs. Because the images come from the executed notebook rather than being re-plotted, a figure in the report can never disagree with the analysis it came from.

Figures are located by a `# figure: <name>` marker comment in the plotting cell, not by cell index, so cells can be added, removed or reordered freely. Adding a new figure to the report means adding the marker in the notebook and its name to `EXPECTED` in `export_figures.py`.

Regenerate everything (takes a couple of minutes; it runs all three notebooks):

```bash
uv run python notebooks/export_figures.py
```

Edit a notebook interactively:

```bash
uv run marimo edit notebooks/pilot_v2_eval_exploration.py
```

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
| `data/pilot_v2_infer/inference_results/0/{en,pl,ru,zh_cmn,zh_yue}.jsonl` | Full run: model responses for 180 base items × 5 languages (900 records). The evaluated model is **Llama 3.1 70B**, recorded as `llama3.1:70b` in each record's `response_metadata.model` |
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

**Localised judge prompts** (for [`rendered/pilot_v2_judge_prompt_language.ipynb`](rendered/pilot_v2_judge_prompt_language.ipynb)):

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
| `rendered/pilot_v2_eval_exploration.ipynb` | `data/pilot_v2_eval/` (`consensus.csv`, `judgments_merged.csv`, `metrics_by_language.csv`, `metrics_by_stratum.csv`, `judge_agreement.csv`, `failures.csv`) |
| `rendered/pilot_v2_leakage_subsets.ipynb` | `data/pilot_v2_eval/consensus.csv` plus `data/pilot_v2_infer/inference_results/{0,3,5,8}/*.jsonl` |
| `rendered/pilot_v2_judge_prompt_language.ipynb` | `data/pilot_v2_eval/` and `data/pilot_v2_eval_localised/` (same CSV set), plus `data/pilot_v2/refusalbench_translation_shared_prompts.csv` for prompt checks |

Leakage subset directories `3`, `5`, and `8` are **not** created by `judge_bedrock` or `aggregate`. They are extra inference exports alongside run `0`.

## Notebooks

Read in this order for the narrative arc: exploration, then leakage, then judge language.

### `pilot_v2_eval_exploration.py` ([rendered](rendered/pilot_v2_eval_exploration.ipynb))

Loads `data/pilot_v2_eval` and interprets what the pilot can and cannot support (180 items × 5 languages, balanced design, metric caveats for pure strata and calibrated scores).

**Figures (matplotlib/seaborn):** inter-judge agreement (κ vs ρ), answerable vs unanswerable counts, confusion-style view on unanswerable items, language intervals vs English, paired disagreement against English on answerable items, perturbation and intensity slices.

**Imports:** `refusalbench.multilingual.aggregate` (`compute_metrics_block`, label constants).

### `pilot_v2_leakage_subsets.py` ([rendered](rendered/pilot_v2_leakage_subsets.ipynb))

Tests whether English left in translated **prompts** explains the cross-language false-refusal result. Reuses full-run judgments and filters by item id. Compares David's inference subsets `3`, `5`, `8` to run `0`, and builds two per-item leakage scores on all 180 items.

**Figures:** leakage measure agreement, subset metrics vs full run, continuous leakage vs cross-language gap, permutation-style plots.

**Does not re-run** `judge_bedrock` or `aggregate` for subsets.

### `pilot_v2_judge_prompt_language.py` ([rendered](rendered/pilot_v2_judge_prompt_language.ipynb))

Compares canonical (`data/pilot_v2_eval`) vs localised judge instructions (`data/pilot_v2_eval_localised`). Uses English as a placebo for judge non-determinism. Addresses whether English-only judge prompts could fake the translation effect described in the exploration notebook.

**Figures:** verdict flips vs placebo floor, false refusal under both instruments, Holm-adjusted significance, heatmap of consensus metric shifts.

**Imports:** `aggregate` constants and `judge_bedrock` prompt builders with the shared prompts CSV.

## How to run the notebooks

1. Install dev dependencies: `uv sync --all-groups` (marimo, seaborn, plotly are in the `dev` group).
2. Place or generate the data paths above (at minimum `data/pilot_v2_eval` for the main notebook).
3. Open a notebook from the repo root:

   ```bash
   uv run marimo edit notebooks/pilot_v2_eval_exploration.py
   ```

   Or run all three non-interactively and refresh the `.ipynb` copies and figures in one pass:

   ```bash
   uv run python notebooks/export_figures.py
   ```

Each notebook sets `REPO = Path.cwd()` and, if `data/pilot_v2_eval` is missing, sets `REPO` to the parent directory. Running with cwd = repo root or `notebooks/` both work once `data/pilot_v2_eval` exists in one of those locations.

Re-execution is deterministic: the notebooks read fixed CSV/JSONL from `data/`, and the only randomness (the permutation test in the leakage notebook) is seeded.

The package must be importable (`refusalbench.multilingual`). An editable install via `uv sync` satisfies that.
