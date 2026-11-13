"""Evidence-only report rendering: model-generated prose never supplies numbers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


BLUE = "#17324D"
TEAL = "#167D8D"


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EFF4F7"), colors.white]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#CBD5DF")),
            ]
        )
    )
    return table


def write_report(output: Path, results: dict, metadata: dict, config: dict, mode: str) -> list[str]:
    risk, mc, opt, back = [
        results[k] for k in ("estimate_risk", "simulate_risk", "optimise_portfolio", "backtest_var")
    ]
    original, candidate = opt["comparison"]["input"], opt["comparison"]["minimum_variance"]
    assets, value = risk["assets"], config.get("portfolio_value", 1_000_000)
    fixture = config["source"] == "fixture"
    data_label = "SYNTHETIC FIXTURE" if fixture else "YAHOO ADJUSTED-CLOSE SNAPSHOT"
    run_label = (
        "Scripted offline workflow; no LLM called"
        if mode == "scripted_workflow"
        else "OpenAI tool orchestration; model text stored separately"
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    frontier = opt["frontier"]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    ax.plot(
        [math.sqrt(p["estimated_daily_variance"] * 252) * 100 for p in frontier],
        [p["estimated_daily_return"] * 252 * 100 for p in frontier],
        color=TEAL,
        marker=".",
    )
    ax.set(
        xlabel="Estimated annual volatility (%)",
        ylabel="Estimated annual arithmetic return (%)",
        title="Training-sample efficient frontier | long-only, fully invested",
    )
    ax.grid(alpha=0.2)
    fig.savefig(figures / "efficient_frontier.png", dpi=170)
    plt.close(fig)
    forecasts = back["forecasts"]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    dates = [row["date"] for row in forecasts]
    ax.plot(
        range(len(dates)),
        [100 * row["loss"] for row in forecasts],
        color=BLUE,
        alpha=0.65,
        lw=0.7,
        label="Realised one-day loss",
    )
    ax.plot(
        range(len(dates)),
        [100 * row["var"] for row in forecasts],
        color=TEAL,
        lw=1.3,
        label="Past-only historical VaR",
    )
    breaches = [(i, r) for i, r in enumerate(forecasts) if r["breach"]]
    ax.scatter(
        [i for i, r in breaches],
        [100 * r["loss"] for i, r in breaches],
        color="#BB4444",
        s=12,
        label="Breach",
        zorder=3,
    )
    tick_indices = sorted({0, len(dates) // 2, len(dates) - 1})
    ax.set_xticks(tick_indices, [dates[i] for i in tick_indices])
    ax.set(ylabel="Loss (%)", title="Rolling one-day VaR coverage | original fixed portfolio")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.savefig(figures / "var_backtest.png", dpi=170)
    plt.close(fig)
    with (output / "portfolio_allocation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["asset", "input_weight", "minimum_variance_weight"])
        writer.writerows(zip(assets, config["weights"], opt["minimum_variance"]["weights"]))
    with (output / "efficient_frontier.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["target_daily_return", "estimated_daily_return", "estimated_daily_variance", *assets]
        )
        for row in frontier:
            writer.writerow(
                [
                    row["target_daily_return"],
                    row["estimated_daily_return"],
                    row["estimated_daily_variance"],
                    *row["weights"],
                ]
            )
    with (output / "var_backtest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["date", "var", "loss", "breach"])
        writer.writeheader()
        writer.writerows(forecasts)
    with (output / "risk_estimates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "horizon_observations", "confidence", "var_fraction", "es_fraction"]
        )
        writer.writerow(
            ["historical", 1, config["confidence"], risk["historical_var"], risk["historical_es"]]
        )
        writer.writerow(
            [
                "lognormal_monte_carlo",
                config["horizon"],
                config["confidence"],
                mc["monte_carlo_var"],
                mc["monte_carlo_es"],
            ]
        )
    rows = [["Asset", "Original", "Min-variance candidate"]] + [
        [a, _pct(w), _pct(v)]
        for a, w, v in zip(assets, config["weights"], opt["minimum_variance"]["weights"])
    ]
    uncertainty = opt.get("holdout_uncertainty", [])
    if uncertainty:
        with (output / "holdout_uncertainty.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(uncertainty[0]))
            writer.writeheader()
            writer.writerows(uncertainty)
    uncertainty_rows = [["Tail improvement", "Block", "Estimate", "95% interval"]]
    for row in uncertainty:
        if row["metric"] != "annualized_variance":
            uncertainty_rows.append(
                [
                    row["metric"].replace("historical_", "").upper(),
                    str(row["block_size"]),
                    _pct(row["estimate"]),
                    f"[{_pct(row['ci_lower'])}, {_pct(row['ci_upper'])}]",
                ]
            )
    comparison_rows = [["Holdout metric", "Original", "Candidate"]] + [
        [label, _pct(original["holdout"][key]), _pct(candidate["holdout"][key])]
        for label, key in (
            ("Annual volatility", "annualized_volatility"),
            ("Historical VaR", "historical_var"),
            ("Historical ES", "historical_es"),
            ("Total return (no costs)", "total_return"),
        )
    ]
    coverage = back["kupiec"]
    brief = (
        f"The candidate minimises estimated training-sample variance under long-only, fully invested constraints. "
        f"Holdout annual volatility is {_pct(candidate['holdout']['annualized_volatility'])}, compared with "
        f"{_pct(original['holdout']['annualized_volatility'])} for the original portfolio. "
        "This objective does not minimise every tail-risk metric or guarantee future risk reduction."
    )
    lines = [
        "# Portfolio risk evidence report",
        "",
        f"**{data_label}** | {run_label}",
        "",
        f"Confidence: {config['confidence']:.1%}. Historical horizon: one recorded observation. "
        f"Portfolio value {value:,.0f} is illustrative monetary scaling, not actual account holdings.",
        "",
        f"Historical VaR: {_pct(risk['historical_var'])}; ES: {_pct(risk['historical_es'])}. "
        f"Monte Carlo ({config['horizon']} observations, {config['paths']:,} paths): "
        f"VaR {_pct(mc['monte_carlo_var'])}; ES {_pct(mc['monte_carlo_es'])}.",
        "",
        "## Allocation and validation",
        "",
        brief,
        "",
        "| Asset | Original | Candidate |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {a} | {_pct(w)} | {_pct(v)} |"
        for a, w, v in zip(assets, config["weights"], opt["minimum_variance"]["weights"])
    ]
    lines += [
        "",
        f"Optimisation uses only the first {config['train_observations']} return observations. "
        "Holdout metrics use subsequent observations with weights fixed from training, daily rebalancing and no turnover costs. "
        "Full-sample risk estimates are descriptive and are not part of allocation fitting.",
        "",
        "![Efficient frontier](figures/efficient_frontier.png)",
        "",
        f"Rolling original-portfolio VaR: {int(coverage['breaches'])}/{int(coverage['observations'])} breaches "
        f"({coverage['observed_breach_rate']:.2%}); Kupiec asymptotic p={coverage['p_value']:.4g}. "
        "Thresholds use strictly earlier returns. Coverage alone does not establish independent breaches. "
        "The optimised candidate's holdout tail statistics are not a separate rolling VaR coverage test.",
        "",
        "![VaR coverage](figures/var_backtest.png)",
        "",
        "## Evidence and limits",
        "",
        "Numerical evidence: estimate_risk.json, simulate_risk.json, optimise_portfolio.json and backtest_var.json. "
        "Data snapshots and provenance: data/. Configuration and hashes: input_config.json and run_manifest.json. "
        "Tool outcomes: tool_audit.json and agent_run.json. Model commentary, if any, is not the source of report numbers.",
        "",
        "Historical quantiles use linear interpolation; ES is the upper-tail empirical loss integral with fractional boundary mass. "
        "Monte Carlo assumes correlated Gaussian log returns, stationary fitted moments and daily rebalancing; fat tails and regime shifts can invalidate it. "
        "The efficient frontier uses a regularised covariance and noisy historical mean returns. No trade is placed. "
        "Synthetic results test implementation, not market performance. Live market data require aligned calendars, currencies and verified adjustments.",
        "",
    ]
    if uncertainty:
        lines += [
            "## Conditional holdout uncertainty",
            "",
            "Paired moving-block bootstrap, 1,000 resamples, seed 42. Improvements are original minus candidate, "
            "so positive values favour the candidate. Weights remain fixed from training; these intervals "
            "do not include refitting, model-selection or future-regime uncertainty. Tail differences below are percentage points.",
            "",
            "| Metric | Block length | Improvement | 95% interval |",
            "|---|---:|---:|---:|",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in uncertainty_rows[1:]]
        lines += [
            "",
            "See holdout_uncertainty.csv for all metrics including annualized variance. "
            "Native variance contributions are in optimise_portfolio.json; signed contributions can reflect hedging.",
            "",
        ]
    (output / "portfolio_report.md").write_text("\n".join(lines), encoding="utf-8")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Label",
            parent=styles["Normal"],
            textColor=colors.HexColor(TEAL),
            fontSize=9,
            spaceAfter=10,
        )
    )
    styles["Title"].textColor = colors.HexColor(BLUE)
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 9

    def para(text, style="BodyText"):
        return Paragraph(escape(text), styles[style])

    story = [
        para("PORTFOLIO RISK / REPRODUCIBLE EVIDENCE", "Label"),
        para("From tools to a testable allocation", "Title"),
        para(data_label + " | " + run_label, "Label"),
        para(
            f"{config['start']} to {config['end']} (end exclusive). Confidence {config['confidence']:.1%}; historical horizon one observation."
        ),
        para("Risk snapshot", "Heading2"),
        _table(
            [
                ["Estimate", "Portfolio fraction", f"At value {value:,.0f}"],
                [
                    "Historical VaR",
                    _pct(risk["historical_var"]),
                    f"{value * risk['historical_var']:,.0f}",
                ],
                [
                    "Historical ES",
                    _pct(risk["historical_es"]),
                    f"{value * risk['historical_es']:,.0f}",
                ],
                [
                    f"MC VaR ({config['horizon']} observations)",
                    _pct(mc["monte_carlo_var"]),
                    f"{value * mc['monte_carlo_var']:,.0f}",
                ],
                [
                    f"MC ES ({config['horizon']} observations)",
                    _pct(mc["monte_carlo_es"]),
                    f"{value * mc['monte_carlo_es']:,.0f}",
                ],
            ],
            [180, 130, 170],
        ),
        Spacer(1, 16),
        para("Allocation candidate", "Heading2"),
        _table(rows, [180, 130, 170]),
        Spacer(1, 12),
        para(brief),
        para(
            "Evidence: estimate_risk.json; simulate_risk.json; optimise_portfolio.json. Monetary amounts are illustrative scaling. A lower estimated variance is not a general investment recommendation.",
            "Label",
        ),
        PageBreak(),
        para("Estimation and out-of-sample checks", "Title"),
        para(
            f"The first {config['train_observations']} return observations fit the allocation. The remaining observations evaluate fixed weights; daily rebalancing is assumed and trading costs are omitted."
        ),
        _table(comparison_rows, [180, 130, 170]),
        Spacer(1, 12),
        Image(str(figures / "efficient_frontier.png"), width=6.6 * inch, height=3.135 * inch),
        para(
            "The frontier shows the model's training-sample trade-off. It is not an out-of-sample return forecast. The covariance receives a stated diagonal ridge; expected returns remain uncertain.",
            "BodyText",
        ),
        para(
            "Evidence: optimise_portfolio.json; efficient_frontier.csv; portfolio_allocation.csv.",
            "Label",
        ),
        para("Conditional tail-risk improvement", "Heading2"),
        _table(uncertainty_rows, [125, 55, 95, 205]),
        Spacer(1, 8),
        para(
            "Original minus candidate, in percentage points. Paired moving-block intervals use 1,000 resamples with fixed fitted weights. They omit refitting and model-selection uncertainty.",
            "BodyText",
        ),
        PageBreak(),
        para("Coverage, provenance and limits", "Title"),
        para(
            f"Original-portfolio VaR: {int(coverage['breaches'])} breaches in {int(coverage['observations'])} forecasts; rate {coverage['observed_breach_rate']:.2%}, expected {coverage['expected_breach_rate']:.2%}. Kupiec asymptotic p-value {coverage['p_value']:.4g}."
        ),
        Image(str(figures / "var_backtest.png"), width=6.6 * inch, height=3.135 * inch),
        para(
            "Each one-observation forecast uses only the preceding rolling window. The test checks unconditional coverage, not breach independence. Sparse tail observations limit precision."
        ),
        para("Reproduction", "Heading2"),
        para(
            "Archived prices, returns and provider settings identify the data. input_config.json fixes parameters; tool_audit.json records execution; run_manifest.json records source, binary, environment and output hashes. Actual LLM runs also retain response/model identifiers and token use."
        ),
        para("Limits", "Heading2"),
        para(
            "Synthetic data validate software behavior, not financial performance. Monte Carlo assumes Gaussian log returns and stable moments. Missing prices are rejected, not silently filled. Live data require consistent currencies and calendars. The optimised candidate has no separate rolling coverage claim. LLM text is stored separately; all numbers here come from native tool results."
        ),
        para(
            "Evidence: backtest_var.json; data metadata; run_manifest.json. Research illustration; no trade execution.",
            "Label",
        ),
    ]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(BLUE))
        canvas.drawString(42, 24, "Portfolio risk agent | evidence-led research illustration")
        canvas.drawRightString(553, 24, f"{doc.page}")

    doc = SimpleDocTemplate(
        str(output / "portfolio_report.pdf"),
        pagesize=(595.28, 841.89),
        rightMargin=42,
        leftMargin=42,
        topMargin=38,
        bottomMargin=40,
        title="Portfolio risk evidence report",
        author="quantitative-market-models",
        invariant=1,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return [
        "holdout_uncertainty.csv",
        "portfolio_report.md",
        "portfolio_report.pdf",
        "portfolio_allocation.csv",
        "risk_estimates.csv",
        "efficient_frontier.csv",
        "var_backtest.csv",
        "figures/efficient_frontier.png",
        "figures/var_backtest.png",
    ]
