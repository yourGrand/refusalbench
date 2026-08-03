# Multilingual pilot evaluation

Tools to judge multilingual model outputs with Amazon Bedrock LLMs and to aggregate those judgments into consensus labels and RefusalBench metrics. This package covers the **pilot v2** path only. It does not run inference or build translations.

## Layout

| Module | Role |
|--------|------|
| `judge_bedrock.py` | Async batch judging CLI. Calls Bedrock via LiteLLM, appends to `judgments_raw.jsonl`. |
| `aggregate.py` | Merges judgments with inference metadata, majority vote consensus, metrics, judge agreement. |
| `env.py` | Loads repo `.env`, resolves AWS region and Bedrock auth, reads judge model IDs. |
| `bedrock_retry.py` | Shared LiteLLM completion helper (retries, throttling, adaptive concurrency). Used by the judge, not a CLI. |

There are **no** `[project.scripts]` entry points in `pyproject.toml`. Run the CLIs as modules from the repository root (after `uv sync` or an editable install).

## Prerequisites

- Python 3.10+ (see root `pyproject.toml`).
- Dependencies: `litellm`, `boto3`, `pandas`, `python-dotenv`, `tenacity`, and others listed in `pyproject.toml`.
- AWS credentials that can call the configured Bedrock models (bearer token, access keys, profile, or instance role).
- Judge model IDs in `.env` (copy from `.env.example`).

### Environment variables

| Variable | Purpose |
|----------|---------|
| `AWS_BEARER_TOKEN` or `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key (recommended). |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Alternative auth. |
| `AWS_PROFILE` | Alternative auth. |
| `AWS_REGION` | Region for Bedrock (default fallback in code: `us-east-1`). |
| `REFUSALBENCH_JUDGE_SONNET4_5` | LiteLLM model string for judge `sonnet4_5`. |
| `REFUSALBENCH_JUDGE_MISTRAL_LARGE_3` | LiteLLM model string for judge `mistral-large-3`. |
| `REFUSALBENCH_JUDGE_NEMOTRON_3_SUPER` | LiteLLM model string for judge `nemotron-3-super`. |

Values must be full LiteLLM Bedrock paths (for example `bedrock/converse/<inference-profile-id>`). See comments in `.env.example`.

`env.load_project_env()` loads `<repo>/.env` unless you pass `--env-file`. Existing shell variables win over `.env`.

## Data layout

Paths below are relative to the repository root.

### Shared translations (`data/pilot_v2/`)

- `refusalbench_translation_shared_prompts.csv` (tracked in git): per-language text for system segments and judge prompt segments. Columns include `segment_id`, `english_text`, `PL`, `RU`, `ZH_YUE`, `ZH_CMN`. The judge uses rows such as `sys_query_label`, `sys_context_label`, and (for localised judging) `judge_intro`, `judge_step1`, etc.

### Inference inputs (`data/pilot_v2_infer/`)

Large artefacts are gitignored. Place inference JSONL here before judging.

Default judge input directory:

```text
data/pilot_v2_infer/inference_results/0/
```

One file per language (lowercase code in the filename):

- `en.jsonl`, `pl.jsonl`, `ru.jsonl`, `zh_cmn.jsonl`, `zh_yue.jsonl`

Judging defaults to run `0`. Sibling directories `3`, `5`, and `8` hold leakage-filtered inference subsets (same filename pattern). The leakage notebook filters full-run `consensus.csv` by those item ids and reads inference from runs `0`, `3`, `5`, and `8`. It does not re-judge splits `3`, `5`, or `8` unless you point `--input-dir` there yourself.

**Inference record fields** (from code, not a checked-in schema):

- **Required for judging:** `id` or `example_id`, plus `messages` (chat list) and `response` or `model_output`.
- **Used in prompts and metrics:** `expected_behavior` or `expected_rag_behavior`, `reference_answers`, `perturbation_class`, `intensity`.

Aggregation can fill missing ground truth and query fields from these inference files when they are absent in judgment rows.

### Evaluation outputs (`data/pilot_v2_eval/`)

Default output for the canonical (English-instructed) judge run. A separate directory (for example `data/pilot_v2_eval_localised/`) should be used for target-language judge prompts so runs are not mixed.

## Step 1: Run judges

From the repo root:

```bash
uv sync --all-groups
cp .env.example .env   # then fill in AWS and judge model IDs

uv run python -m refusalbench.multilingual.judge_bedrock \
  --input-dir data/pilot_v2_infer/inference_results/0 \
  --out-dir data/pilot_v2_eval \
  --prompts-csv data/pilot_v2/refusalbench_translation_shared_prompts.csv
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--languages` | `en pl ru zh_cmn zh_yue` | Which `{lang}.jsonl` files to process. |
| `--judges` | all three from `.env` | Subset: `sonnet4_5`, `mistral-large-3`, `nemotron-3-super`. |
| `--limit` | none | Cap records per language file (smoke tests). |
| `--concurrency` | `8` | Concurrent calls per judge. |
| `--max-retry-minutes` | none | Optional wall-clock retry budget per completion. |
| `--judge-prompt-language` | `english` | `english` uses the canonical English judge template. `target` uses translated judge segments from `--prompts-csv` (use a separate `--out-dir`). |
| `--log-file` | `<out-dir>/judge_bedrock.log` | Application logs (progress bar stays on the terminal). |

**Judge outputs** (under `--out-dir`):

| File | Format |
|------|--------|
| `judgments_raw.jsonl` | One JSON object per call. OK rows include `judge`, `language`, `id`, `status`, `classification`, `quality_score`, `explanation`, `judge_prompt_language`, metadata copied from inference, etc. |
| `failures.csv` | Permanent API or parsing failures during judging (`judge`, `language`, `id`, `error`, `recorded_at`). |
| `judge_bedrock.log` | Run log. |

Judging is **resumable**: successful `status: ok` rows in `judgments_raw.jsonl` are skipped. The resume key includes judge prompt language so English and target runs do not skip each other unless they share the same `--out-dir`.

**Target-language judge example** (separate output tree):

```bash
uv run python -m refusalbench.multilingual.judge_bedrock \
  --input-dir data/pilot_v2_infer/inference_results/0 \
  --out-dir data/pilot_v2_eval_localised \
  --judge-prompt-language target \
  --prompts-csv data/pilot_v2/refusalbench_translation_shared_prompts.csv
```

## Step 2: Aggregate

After `judgments_raw.jsonl` exists:

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

| Flag | Default | Meaning |
|------|---------|---------|
| `--judgments` | `data/pilot_v2_eval/judgments_raw.jsonl` | Raw judge JSONL. |
| `--input-dir` | `data/pilot_v2_infer/inference_results/0/` | Inference JSONL tree for metadata merge (`rglob("*.jsonl")`). |
| `--out-dir` | `data/pilot_v2_eval/` | CSV destination. |
| `--judges` | all | Optional filter on judge ids. |

**Aggregation outputs** (under `--out-dir`):

| File | Contents |
|------|----------|
| `judgments_merged.csv` | Per-judge rows with query, ground truth, model output, classification, quality, refusal match. |
| `consensus.csv` | One row per `(language, id)` from majority vote across OK judges (`consensus_classification`, `n_agree`, `no_majority`, etc.). |
| `metrics_by_language.csv` | RefusalBench metrics per judge and for `consensus`, plus `delta_vs_en_*` columns. |
| `metrics_by_stratum.csv` | Same metrics by language, `perturbation_class`, and `intensity`. |
| `judge_agreement.csv` | Pairwise and Fleiss agreement statistics. |
| `failures.csv` | Non-OK rows from the raw JSONL (overwrites the judge-stage `failures.csv` in the same directory). |

Ground truth label for answerable items is `ANSWER_CORRECTLY`. Judge classifications use `answer_attempt` or a `REFUSE_*` code (see `VALID_REFUSAL_CODES` in code). Metric definitions match `aggregate.compute_metrics_block` (used in `notebooks/pilot_v2_eval_exploration.ipynb`).

You can also call `aggregate.aggregate(...)` from Python with the same paths.

## Downstream notebooks

Analysis notebooks live under `notebooks/`. See [`notebooks/README.md`](../../notebooks/README.md) for run order, figures, and notebook-specific data paths.

These notebooks expect aggregated CSVs under `data/pilot_v2_eval/` (and optionally `data/pilot_v2_eval_localised/`):

| Notebook | Reads |
|----------|--------|
| `notebooks/pilot_v2_eval_exploration.ipynb` | `consensus.csv`, `judgments_merged.csv`, `metrics_by_language.csv`, `metrics_by_stratum.csv`, `judge_agreement.csv`, `failures.csv` |
| `notebooks/pilot_v2_judge_prompt_language.ipynb` | Canonical vs localised eval dirs (same CSV set), plus shared prompts CSV |
| `notebooks/pilot_v2_leakage_subsets.ipynb` | Full-run `consensus.csv` plus inference under `data/pilot_v2_infer/inference_results/{0,3,5,8}/` (filters by id, does not re-judge splits) |

Run notebooks from `notebooks/` or the repo root. They resolve `REPO` by looking for `data/pilot_v2_eval`.

## Classifications and consensus

- Three Bedrock judges classify each inference record independently.
- Consensus is a simple majority over OK judge labels. Ties set `no_majority`. Those rows are dropped from consensus metrics.
- Dual evaluation (answer quality vs refusal code match) follows the same rules as the main RefusalBench evaluators (`apply_dual_eval` in `aggregate.py`).

## Tests

Unit tests live under `tests/` (`test_aggregate.py`, `test_judge_bedrock.py`, `test_bedrock_retry.py`). Run:

```bash
uv run pytest
```
