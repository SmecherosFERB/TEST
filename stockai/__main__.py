"""Linie de comandă: python -m stockai AAPL MSFT NVDA"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from .advisor import ClaudeAdvisor
from .analyzer import Analyzer, Recommendation
from .config import Settings
from .data import DataError, MarketData

CONFIDENCE = {"low": "scăzută", "medium": "medie", "high": "ridicată"}


def fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0f}"


def render(rec: Recommendation) -> str:
    s = rec.scores
    lines = [
        f"═══ {rec.ticker} · {rec.price:.2f} · {rec.as_of} ═══",
        f"Decizie: {rec.decision}  (decis de {rec.decided_by}"
        + (f", încredere {CONFIDENCE[rec.claude.confidence]})" if rec.claude else ")"),
        f"Scoruri: tehnic {fmt_score(s['technical'])} · fundamental {fmt_score(s['fundamental'])}"
        f" · sentiment {fmt_score(s['sentiment'])} · compus {fmt_score(s['composite'])}",
    ]
    if rec.odds:
        o = rec.odds
        lines.append(
            f"Istoric (doar tehnic, {o.horizon} zile): a urcat în {o.probability_up:.0%} din cazurile similare"
            f" (rata de bază {o.base_rate:.0%}, n={o.samples}, randament mediu {o.avg_return:+.1%})"
        )
    if rec.ambiguity_reasons:
        lines.append("Semnal neclar: " + "; ".join(rec.ambiguity_reasons))
    if rec.claude:
        c = rec.claude
        lines.append(f"Claude: {c.probability_up_pct}% șanse de creștere, decizie {c.decision}")
        lines.append(f"  {c.reasoning}")
        if c.key_risks:
            lines.append("  Riscuri: " + "; ".join(c.key_risks))
    elif rec.claude_error:
        lines.append(f"Claude indisponibil: {rec.claude_error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="stockai", description="Semnale BUY/SELL/HOLD pentru acțiuni")
    parser.add_argument("tickers", nargs="+", help="simboluri, ex. AAPL MSFT NVDA")
    parser.add_argument("--horizon", type=int, default=20, help="orizontul în zile de tranzacționare")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-claude", action="store_true", help="doar reguli, fără Claude")
    mode.add_argument("--always-claude", action="store_true", help="cere opinia lui Claude la fiecare acțiune")
    parser.add_argument("--json", action="store_true", help="rezultat în format JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    settings = Settings(horizon_days=args.horizon)
    claude_mode = "never" if args.no_claude else "always" if args.always_claude else "auto"
    advisor = None
    if claude_mode != "never":
        if ClaudeAdvisor.available():
            advisor = ClaudeAdvisor(model=settings.claude_model, effort=settings.claude_effort)
        else:
            print("Notă: ANTHROPIC_API_KEY lipsește, rulez doar pe reguli.", file=sys.stderr)

    analyzer = Analyzer(MarketData(), settings=settings, advisor=advisor, claude_mode=claude_mode)
    results, failed = [], False
    for ticker in args.tickers:
        try:
            rec = analyzer.analyze(ticker)
        except DataError as exc:
            print(f"Eroare: {exc}", file=sys.stderr)
            failed = True
            continue
        results.append(rec)
        if not args.json:
            print(render(rec) + "\n")

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    print("Aceasta este o unealtă de analiză, nu sfat financiar.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
