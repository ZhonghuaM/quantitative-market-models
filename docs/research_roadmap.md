# Research roadmap

The primary study has one question: whether daily OHLCV forecasts improve on baselines and whether apparent value survives execution assumptions and costs. Its next steps should strengthen that evidence before expanding the method inventory.

## Strengthen the existing study

1. **Replace uncertain provenance with a documented dataset.** Record the provider, permitted use, instrument identifier, retrieval date, exchange calendar, adjustments, and missing-session policy. Re-running the pipeline on new data should be a separately identified experiment, not an unexplained replacement of old results.
2. **Fix development and evaluation boundaries.** Define the target, features, baselines, model-selection procedure, and execution assumptions before reserving or collecting a new evaluation period. Keep the current historical sample labelled exploratory.
3. **Test sensitivity deliberately.** Examine cost and threshold assumptions under a declared protocol; distinguish discrimination, probability quality, turnover, and investment outcomes. Include null results and comparison against credible simple alternatives.

These are planned improvements, not claims that the current dataset or strategy has passed them.

## Possible extension: lead–lag change benchmark

A lead–lag experiment is future work. It would extend the same emphasis on timing, baseline comparison, and uncertainty rather than establish a second unrelated catalogue of models.

Start with a published estimator and synthetic series with known lead–lag relationships. Include independent no-change replications, abrupt and gradual changes, and changes in dependence strength or noise that should not be mistaken for changes in the lag structure. Calibrate decision thresholds on separate simulations. Report finite-horizon false-alarm probability and detection delay across repeated runs; periodically re-estimating a network is not itself a declaration of change.

Keep lag-resolved information when evaluating changes in lag magnitude. A score that aggregates across lags may detect direction while missing a change from lag 1 to lag 3; that is a useful failure case, not evidence of a successful detector. Only after establishing the synthetic benchmark should a documented multi-asset market illustration and optional cost-aware trading evaluation be added.

## Prior work and boundaries

- **Cartea, Cucuringu and Jin, “Detecting lead–lag relationships in stock returns and portfolio strategies.”** This work already connects lead–lag estimation and directed networks to portfolio strategies. The connection between lead–lag information and trading is not a new contribution here. [Paper](https://doi.org/10.1016/j.jempfin.2026.101775).
- **Zhou et al., “DeltaLag: Learning Dynamic Lead-Lag Patterns in Financial Markets.”** DeltaLag learns dynamic, pair-specific leader and lag relationships. An “adaptive financial network” alone would not distinguish a new study. [Full paper](https://arxiv.org/html/2511.00390v1).
- **Sulem, Kenlay, Cucuringu and Dong, “Graph similarity learning for change-point detection in dynamic networks.”** The paper includes financial correlation networks and explicitly identifies directed/lead–lag graphs as an extension. Moving change detection to lead–lag networks is therefore an existing research direction. [Full paper](https://link.springer.com/article/10.1007/s10994-023-06405-x).

The proposed benchmark is a reproduction and evaluation exercise. Any later novelty claim would require a specific methodological distinction, a broader literature review, and evidence that addresses the existing methods' scope.
