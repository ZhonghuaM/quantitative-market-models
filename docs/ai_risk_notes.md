# Text-retrieval example

This supplementary example chunks a short, illustrative filing-style passage, ranks chunks with TF–IDF similarity, and assembles a brief from retrieved text. The passage is embedded in the analysis script; it is not an authenticated company filing. No language model or autonomous financial-decision system is involved. It runs as part of `--study all`, separately from the primary financial forecasting study.

The generated table records the query, retrieved text, and similarity scores. A score measures lexical similarity, not factual reliability, reader-judged relevance, or calibrated confidence. Reusing source text reduces unsupported generation but cannot establish that the source is accurate, complete, current, or appropriate for the question.

A useful future extension would use identifiable documents with stable citation spans, a labelled query set, and retrieval metrics. Generated summaries would need a separate evaluation of factual support. These extensions are not implemented in the current example.
