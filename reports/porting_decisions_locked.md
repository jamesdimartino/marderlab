# Porting Decisions (Locked From User Comments)

Source: comments added in `reports/notebook_pipeline_status_report.md`.

## Priority Queue

1. `dualhik` (interpreted from "hikcontrol")
2. `control`
3. `gm56weaklink`

## Architecture Decisions

- Merge pipelines where they perform similar tasks to keep deployment lightweight.
- Preserve or expand flexibility/optionality from legacy notebooks.
- Build toward an app-ready capability layer (agentic/natural-language-driven orchestration later).
- Never overwrite base notebooks.

## Metadata Mapping (Canonical)

- `FTSpont`: heartbeat-focused analyses (`Heartbeat.ipynb`, `rawheart.ipynb`)
- `FTMuscle`: nerve-evoked analyses
- `FTBath`: bath-evoked contracture + Hi-K pipelines

## Scientific/Behavioral Rules

- Functionality preservation is prioritized over exact legacy implementation details.
- Assume all legacy metrics are required in production outputs.
- Baseline policy:
  - Keep notebook/protocol-specific baseline logic.
  - Standardize baseline extraction window to 2 seconds across notebooks.
- Missing metadata policy:
  - Skip and flag (do not hard-fail entire run).
- Acceptance criterion for a notebook to be "fully ported":
  - Function parity + output parity.

## Statistics Policy

- Add default statistical tests now (not just hooks).
- Include statistical markers for analyses that compare conditions or otherwise require inference.
- Support easy invocation of multiple statistical tests across data types.
- Default test selection should be based on data type; escalate when ambiguous.

## Testing Policy

- Use phase-gate validation rather than per-notebook golden datasets.

## Scope Decisions

- Do not deprecate modeling notebooks (`modelfiber`, `musclemodelrealisticVm`).
- Integrate `StimulusGen.ipynb` outputs now (same package/CLI scope).
- Add dedicated `heartbeat` pipeline command in the next sprint.
- Each newly ported notebook should get its own VSCode visual sanity template.

## Open Clarification

- Run order/dependency graph for "run all":
  - User response indicated uncertainty.
  - Working default until specified: independent pipelines, fixed deterministic order, no inter-pipeline data dependencies unless explicitly declared.
