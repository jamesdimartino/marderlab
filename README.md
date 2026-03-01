# Marder Lab Tools

Python-first production pipelines for Marder Lab electrophysiology analysis.

## Scope (current)

- Contracture pipeline
- Nerve-evoked contraction pipeline
- Metadata sync from Google Sheet with local CSV fallback
- Batch rerun (`run-all`) with per-experiment failure isolation
- Sanity checks + run manifests + JSON/HTML reports

## Install

```bash
pip install -e ".[analysis,dev]"
```

## Commands

```bash
marder doctor --config configs/default.yml
marder sync-metadata --config configs/default.yml
marder run --pipeline contracture --config configs/default.yml
marder run --pipeline nerve-evoked --config configs/default.yml
marder run --pipeline hikcontrol --config configs/default.yml
marder run-all --config configs/default.yml
marder genai-chat --agent-config configs/genai.yml --prompt "How does run-all select experiments?"
marder genai-window --agent-config configs/genai.yml
```

Add `--plots` to `run` or `run-all` to save SVG plots.

## GenAI Window

Install app dependency:

```bash
pip install -e ".[analysis,dev,app]"
```

Launch local chat window:

```bash
marder genai-window --agent-config configs/genai.yml --workspace-root .
```

Features:

- Reads project code context directly from your VSCode workspace
- Exposes safe read-only tools (search code, read excerpts, list pipelines/commands)
- Supports multiple model providers via `configs/genai.yml` with fallback
- Saves chat transcripts to `.cache/marderlab/agent_chat/`

Provider env vars:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Ollama: no key required (local server)

## Visual sanity-check subset

Run a small subset with live progress and per-experiment plots:

```bash
marder run \
  --pipeline contracture \
  --config configs/default.yml \
  --pages 997_006,997_008 \
  --max-experiments 2 \
  --plots \
  --live
```

This writes SVG sanity plots to:

```text
{processed_root}/{notebook_page}/plots/
```

It also writes a VSCode-local gallery for that run:

```text
{cache_root}/vscode_sanity/{run_id}/index.html
```

## Output layout

Per experiment:

```text
{processed_root}/{notebook_page}/
  npy/
    {pipeline}_metrics.npy
    {pipeline}_metrics_tidy.csv
  plots/
```

Per run:

- `run_report.json`
- `run_report.html`
- run manifest in cache

## Data safety and rigor

- Raw ABF inputs are treated as read-only.
- Missing crucial metadata fails that experiment.
- Poor signal sets metrics to `0` and adds quality flags.
- Unit conversions and calibration are applied before metrics.
- Every run records config, code/version metadata, inputs, and timestamps.
