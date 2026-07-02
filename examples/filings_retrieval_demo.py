import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_models.nlp_retrieval import TfidfRetriever, chunk_text, source_grounded_brief


TEXT = """
The company faces liquidity risk if market conditions reduce access to credit facilities.
Revenue growth depends on customer renewals, product adoption, and successful expansion into
new enterprise accounts. Operational risks include cybersecurity incidents, supplier disruption,
regulatory change, and foreign exchange volatility. Management mitigates these risks through
cash monitoring, incident response planning, supplier diversification, and compliance controls.
"""


def main() -> None:
    question = "Which liquidity and operational risks should an analyst review?"
    chunks = chunk_text(TEXT, chunk_words=35, overlap_words=8)
    results = TfidfRetriever().fit(chunks).query(question, top_k=2)
    print(source_grounded_brief(question, results))


if __name__ == "__main__":
    main()
