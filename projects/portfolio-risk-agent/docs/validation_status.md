# Validation status

Validation is separated by what it actually establishes.

| Component | Evidence | Interpretation |
|---|---|---|
| Native numerical kernel | Release CTest and AddressSanitizer/UndefinedBehaviorSanitizer builds | Numerical edge cases, causal timing and memory/undefined-behaviour checks |
| Independent numerical reference | [168 checks](../outputs/evaluation/evaluation_report.md) against NumPy/SciPy, analytical moments and finite differences | Agreement within stated tolerances, not investment performance |
| Agent and data boundary | Python tests with injected Responses/Yahoo clients and native integration | Schema, call budget, argument immutability, error handling and report evidence |
| Yahoo adapter | A local 2026-11-13 run retrieved 1,510 aligned observations for SPY/IEF/GLD over 2019-01-01 inclusive to 2025-01-01 exclusive and completed the six-tool workflow | One successful external integration check, not an availability guarantee; downloaded data and provider caches remain uncommitted |
| R reference | `evaluation/check_reference.R`, run in the GitHub evidence job | An independent language implementation of moments, VaR, ES and covariance; consult the CI artifact for executed results |
| Live OpenAI selection | **Not run in the checked-in evaluation** | No API credential was available; mocks and scripted workflows do not establish model accuracy |

The public tool cases were written during development. They are not a held-out benchmark. The optional live evaluator currently scores only the full-analysis contract; partial requests and refusal cases require different completion policies before live scoring. Free-text unsupported claims require separate review and are not automatically scored.

## Reproducibility

The checked-in run records a source commit plus the hashes of the actual source files, because it was produced before committing the implementation. It records a dirty source state explicitly. Production calculations are native; reference calculations do not feed the engine. A fixed seed and input snapshot support deterministic numerical reproduction in the same build environment. Cross-platform checks use tolerances; compiler, library and hardware differences can affect final bits.

API model identifiers and response IDs are recorded for actual live runs. Model tool choices, model prose and later provider revisions are not guaranteed identical across requests. Numerical agreement does not establish economic usefulness, causal validity or suitability for a real investor.
