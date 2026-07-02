# AI and Retrieval Risk Notes

The repository includes a lightweight retrieval demo rather than an autonomous financial advice system.

Risk controls:

- Retrieval is source-grounded: outputs are built from retrieved chunks rather than unsupported generation.
- Reports keep assumptions and limitations next to results.
- Public-data download scripts avoid embedded credentials.
- Human review is expected before any research conclusion is used externally.
- The demo avoids live recommendations and presents historical analysis as hypothetical research.

Potential extensions:

- SEC filing section extraction with citation spans.
- Embedding-based retrieval with an evaluation set.
- Hallucination checks against source passages.
- Red-team prompts for unsupported financial claims.
- Human-in-the-loop review before analyst-style summaries are published.
