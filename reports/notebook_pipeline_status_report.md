# Notebook Pipeline Status Report

Generated from all non-checkpoint notebooks under `C:\Users\james\Desktop\MarderLab`.

## Portfolio Snapshot

- Total notebooks: **16**
- Already partially ported: **2**
- Notebooks using ABF data: **10**
- Notebooks using Google Sheet metadata: **8**
- Notebooks with explicit stats tests/markers: **5**
- Readiness bands: **high=7**, **medium=5**, **low=4**

## Per-Notebook Status

| Notebook | Category | Ported Status | Readiness (0-10) | Complexity | ABF | Sheet | Stats | Hardcoded Paths | Notes |
|---|---|---|---:|---|---|---|---|---:|---|
| contracture.ipynb | contracture | partial (contracture pipeline implemented) | 8 | high | yes | yes | no | 16 | has defensive error handling; writes analysis outputs; hardcoded local paths |
| Heartbeat.ipynb | heartbeat | not ported | 9 | high | yes | yes | yes | 24 | has defensive error handling; writes analysis outputs; hardcoded local paths |
| rawheart.ipynb | heartbeat | not ported | 5 | medium | yes | no | no | 3 | writes analysis outputs; hardcoded local paths |
| HIKSIM.ipynb | high-k/control | not ported | 5 | high | no | no | no | 0 | has defensive error handling; has execution entrypoint; naming conventions mixed |
| control.ipynb | high-k/control | not ported | 9 | high | yes | yes | no | 18 | has defensive error handling; has execution entrypoint; writes analysis outputs; hardcoded local paths |
| dualhik.ipynb | high-k/control | not ported | 10 | high | yes | yes | yes | 39 | has defensive error handling; has execution entrypoint; writes analysis outputs; hardcoded local paths |
| Untitled.ipynb | misc | not ported | 3 | low | no | no | no | 0 | - |
| Untitled.ipynb | misc | not ported | 0 | low | no | no | no | 0 | no reusable functions |
| modelfiber.ipynb | modeling | not ported | 4 | medium | no | no | no | 0 | has execution entrypoint |
| Muscle.ipynb | nerve-evoked/muscle | not ported | 6 | medium | yes | no | yes | 2 | hardcoded local paths |
| PairedContractions.ipynb | nerve-evoked/muscle | partial (nerve_evoked pipeline implemented) | 9 | high | yes | yes | no | 27 | has defensive error handling; has execution entrypoint; writes analysis outputs; hardcoded local paths |
| freqrange.ipynb | nerve-evoked/muscle | not ported | 6 | low | yes | yes | no | 2 | writes analysis outputs; hardcoded local paths |
| gm56acclim.ipynb | nerve-evoked/muscle | not ported | 9 | high | yes | yes | yes | 33 | has defensive error handling; writes analysis outputs; hardcoded local paths |
| gm56weaklink.ipynb | nerve-evoked/muscle | not ported | 9 | high | yes | yes | yes | 36 | has defensive error handling; writes analysis outputs; hardcoded local paths |
| musclemodelrealisticVm (1).ipynb | nerve-evoked/muscle | not ported | 5 | high | no | no | no | 0 | has defensive error handling; has execution entrypoint |
| StimulusGen.ipynb | stimulus-generation | not ported | 3 | low | no | no | no | 2 | writes analysis outputs; hardcoded local paths |

## Current Pipeline Coverage vs Notebooks

- Implemented in codebase now:
  - `contracture.ipynb` -> `contracture` pipeline (partial parity)
  - `PairedContractions.ipynb` -> `nerve_evoked` pipeline (partial parity)
- Not yet ported as production pipelines:
  - `Heartbeat.ipynb` (heartbeat, readiness=9/10)
  - `rawheart.ipynb` (heartbeat, readiness=5/10)
  - `HIKSIM.ipynb` (high-k/control, readiness=5/10)
  - `control.ipynb` (high-k/control, readiness=9/10)
  - `dualhik.ipynb` (high-k/control, readiness=10/10)
  - `Untitled.ipynb` (misc, readiness=3/10)
  - `Untitled.ipynb` (misc, readiness=0/10)
  - `modelfiber.ipynb` (modeling, readiness=4/10)
  - `Muscle.ipynb` (nerve-evoked/muscle, readiness=6/10)
  - `freqrange.ipynb` (nerve-evoked/muscle, readiness=6/10)
  - `gm56acclim.ipynb` (nerve-evoked/muscle, readiness=9/10)
  - `gm56weaklink.ipynb` (nerve-evoked/muscle, readiness=9/10)
  - `musclemodelrealisticVm (1).ipynb` (nerve-evoked/muscle, readiness=5/10)
  - `StimulusGen.ipynb` (stimulus-generation, readiness=3/10)

## Conventions and Rigor Findings

- Common strengths:
  - Strong functional decomposition in several notebooks (especially ABF-heavy analyses).
  - Frequent plotting and export steps for visual/manual QA.
  - Google Sheet integration appears in most experimental analysis notebooks.
- Common blockers to direct productionization:
  - Hardcoded machine-specific paths are widespread.
  - Notebook-level side effects (load/process/save in top-level cells) are common.
  - Inconsistent naming and schema assumptions across notebooks/tabs.
  - Statistical testing is sparse and inconsistent across analyses.
- Priority recommendation for next port wave:
  1. High-readiness ABF notebooks with metadata integration and output writes.
  2. Medium-readiness notebooks tied to active lab analyses.
  3. Modeling/utility notebooks after analysis pipelines are stable.

## Questions For Next Porting Phase

1. Which unported notebooks are scientifically mandatory for the next milestone (top 3 in priority order)?
dualhik, control, and gm56weaklink
2. Should we treat `control.ipynb` as a standalone pipeline or as shared logic feeding other pipelines?
Wherever pipeline execute similar tasks they should be merged for lightweight deployment. It is however crucial that any optionality/flexibility that exists in any pipeline should be preserved or expanded upon.
3. For `dualhik.ipynb` and `HIKSIM.ipynb`, which metrics are required in production output (and which are optional)?
Assume all metrics are required for production output
4. Do you want `gm56acclim.ipynb` and `gm56weaklink.ipynb` promoted as separate pipelines or a single parameterized pipeline?
Again, wherever there are synergies across pipelines they should be merged. I want to eventually create an app that will use agentic capabilities to perform analyses from natural language and use the refactored code from these pipelines as a code base. These pipelines should encompass the ability space of the code base but in no way do I want them to run exactly as is as they are mostly piece meal and not incredibly well thought out or consistent
5. Which notebook outputs are considered publication-grade and must keep exact formatting?
None, but NEVER OVERWRITE THE BASE NOTEBOOKS
6. For each notebook, what is the canonical metadata tab (FTBath / FTMuscle / FTSpont / other)?
FT spont involves recordings of the heart, so heartbeat and rawheart. FTMuscle involves any nerve evoked pipelines. FTBath involves any bath evoked contractures and hik pipelines
7. Can we formalize a cross-notebook metadata schema map now (column aliases -> canonical fields)?
Yes
8. When legacy notebook logic conflicts with current production assumptions, what is the tie-breaker rule?
The functionality of the production pipeline is far more important than the exact logic, so long as functionality is preserved that is all that matters
9. Which analyses require statistical markers in v1 of the port, and which test family should each use by default?
All analyses that produce plots comparing conditions or that would otherwise need a statistical test should have a statistical test. to determine the default statistical test, use your judgement based on the data type and HOOK OUT TO ASK IF YOU HAVE ANY QUESTIONS. However, the codebase should be built to allow easy calling of a wide range of statistical tests for any data.
10. Should we enforce one global baseline policy, or maintain notebook/protocol-specific baseline rules?
Maintain notebook specific baseline rules as these were generally developed for use on the specific data type. However, ensure that any time windows for baseline extraction are standardized to 2 seconds across all notebooks.
11. For notebooks with no explicit stats, do you want us to add default tests now or only scaffolding hooks?
Add default tests now
12. Which notebooks should fail hard on missing metadata vs skip-and-flag behavior?
all notebooks should skip and flag
13. Do you want per-notebook golden datasets before porting each notebook, or one larger phase-gate test set?
Phase gate
14. What acceptance criterion should mark a notebook as fully ported (function parity, output parity, both)?
both
15. Should we deprecate utility/modeling notebooks (`modelfiber`, `musclemodelrealisticVm`) from the production pipeline scope?
no, modeling will ultimately be involved in the final app so it should be ported over as well.
16. Do `StimulusGen.ipynb` outputs need to be integrated into the same package/CLI now or later?
Now
17. For heartbeat analyses, do you want a dedicated `heartbeat` pipeline command in the next sprint?
Yes
18. What is the desired run order/dependency graph across pipelines when running all analyses in one sweep?
Not sure what this means
19. Should each newly ported notebook get its own visual sanity template in the VSCode gallery?
Yes
20. Which notebook should be ported immediately after current two pipelines (single next target)?
hikcontrol
