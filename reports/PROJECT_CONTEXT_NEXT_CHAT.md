# MarderLab Codex Context (New Chat Handoff)

Last updated: 2026-03-01  
Repo: `C:\Users\james\Desktop\MarderLab\codex`  
Git commit: `abd61fb`

## Objective

Productionize Marder Lab notebook workflows into reusable Python pipelines with:

- preserved scientific functionality
- routine sanity checks
- metadata-driven batch processing
- reproducible outputs/reports
- CLI-first operation (app/GenAI layer on top)

## Non-Negotiable Rules

- Never modify raw ABF files.
- Write only to processed/cache/report outputs.
- If experiment metadata is incomplete: flag and exclude that experiment from the current run.
- Baseline default is 2-second pre-stim window (protocol-specific logic still allowed).
- Poor signal -> metrics zeroed + explicit flags.
- Force data converted from volts to cN before final metrics.
- Keep legacy notebooks untouched (no overwrite).

## Implemented Pipelines

ABF analysis pipelines now available:

- `contracture`
- `nerve_evoked` (also accepts `nerve-evoked`)
- `hikcontrol`
- `control`
- `dualhik`
- `freqrange`
- `gm56acclim`
- `gm56weaklink`
- `muscle`
- `heartbeat`
- `rawheart`

Model/simulation ports:

- `hiksim`
- `modelfiber`
- `musclemodel`
- `untitled-model`

Stimulus generation port:

- `StimulusGen.ipynb` -> `marder stimulus-gen`

## Key Entry Points

- CLI: `src/marderlab_tools/cli.py`
- Orchestration: `src/marderlab_tools/run/orchestrator.py`
- Pipeline modules: `src/marderlab_tools/analysis/`
- Modeling modules: `src/marderlab_tools/modeling/`
- Stimulus modules: `src/marderlab_tools/stimulus/`
- Configs:
  - `configs/default.yml` (pipeline runs)
  - `configs/genai.yml` (GenAI routing/tools)

## Current Commands

Core:

- `marder doctor --config configs/default.yml`
- `marder sync-metadata --config configs/default.yml`
- `marder run --pipeline <name> --config configs/default.yml [--plots] [--pages ...] [--max-experiments N] [--live]`
- `marder run-all --config configs/default.yml [--plots] [--pages ...] [--max-experiments N] [--live]`

GenAI:

- `marder genai-window --agent-config configs/genai.yml --workspace-root .`
- `marder genai-chat --agent-config configs/genai.yml --workspace-root . --prompt "..." --model <optional>`

Modeling/Stimulus:

- `marder simulate --model hiksim --output outputs/hiksim_run.npz`
- `marder simulate --model modelfiber --output outputs/modelfiber_run.npz`
- `marder simulate --model musclemodel --output outputs/musclemodel_run.npz`
- `marder simulate --model untitled-model --output outputs/untitled_run.npz`
- `marder stimulus-gen --output outputs/burst_train.csv`

## GenAI Layer Status

Implemented:

- workspace-aware context service
- tool registry (read-only + safe planning tools)
- model router (mock/OpenAI/Anthropic/Ollama + fallback)
- native tool call handling in agent loop
- deterministic fallback responses (no blank content)
- Streamlit chat window + saved chat logs

Known behavior:

- Anthropic/OpenAI tool payload compatibility patched.
- `api_key_env` must contain env var name, not raw key.

## Outputs

Per experiment:

- `{processed_root}/{notebook_page}/npy/{pipeline}_metrics.npy`
- `{processed_root}/{notebook_page}/npy/{pipeline}_metrics_tidy.csv`
- optional SVG plots in `{processed_root}/{notebook_page}/plots/`

Per run:

- run report JSON + HTML
- manifest in cache
- optional VSCode sanity gallery HTML

## Testing State

Latest full test run passed: `41 passed`.

## Suggested Immediate Next Work

1. Run acceptance batch on real recent datasets and generate parity report by notebook_page.
2. Tighten per-pipeline scientific invariants tests (especially burst windows, weaklink metrics, heartbeat rate logic).
3. Add richer visual sanity checks for each newly ported pipeline.
4. Add explicit pipeline-selection diagnostic report (why each experiment maps to a pipeline or is skipped).
5. Continue hardening GenAI responses with strict structured output + cited tool evidence.

