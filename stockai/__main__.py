"""Linie de comandă.

    python -m stockai AAPL MSFT NVDA          # analiză
    python -m stockai --train                 # antrenează și verifică modelul de probabilitate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from .advisor import ClaudeAdvisor
from .analyzer import Analyzer, Recommendation
from .config import Settings
from .data import DataError, MarketData
from .universe import load_universe

CONFIDENCE = {"low": "scăzută", "medium": "medie", "high": "ridicată"}
SCORE_LABELS = [
    ("technical", "tehnic"),
    ("fundamental", "fundamental"),
    ("sentiment", "sentiment"),
    ("earnings", "rezultate"),
    ("insiders", "insideri"),
    ("market", "piață"),
    ("composite", "compus"),
]


def _pl(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0f}"


def render(rec: Recommendation) -> str:
    s = rec.scores
    lines = [
        f"═══ {rec.ticker} · {rec.price:.2f} · {rec.as_of} ═══",
        f"Decizie: {rec.decision}  (decis de {rec.decided_by}"
        + (f", încredere {CONFIDENCE[rec.claude.confidence]})" if rec.claude else ")"),
        "Scoruri: " + " · ".join(f"{label} {fmt_score(s.get(key))}" for key, label in SCORE_LABELS),
    ]
    if rec.model:
        m = rec.model
        lines.append(
            f"Model statistic: {m['prob']:.0%} șanse de creștere în {m['horizon']} zile"
            f" (de obicei {m['base_rate']:.0%}; antrenat pe {m['trained'].get('tickers', '?')} acțiuni)"
        )
    if rec.honest:
        h = rec.honest
        verdict = ("peste medie, sigur statistic" if h["sure"] == "up" else "sub medie, sigur statistic"
                   if h["sure"] == "down" else "intervalul include rata obișnuită: niciun avantaj dovedit")
        lines.append(
            f"Șanse estimate: {h['p']:.0%} (interval 90%: {h['lo']:.0%}–{h['hi']:.0%}, de obicei {h['base']:.0%}) · {verdict}"
        )
    if rec.odds:
        o = rec.odds
        lines.append(
            f"  doar această acțiune: {o.probability_up:.0%} din ~{o.samples / o.horizon:.0f} cazuri independente,"
            f" randament mediu {o.avg_return:+.1%}"
        )
    x = rec.extras
    if x.get("market"):
        lines.append(f"Piața (S&P 500): {x['market']['label']}, {x['market']['return_1m']:+.1%} în ultima lună")
    if x.get("macro"):
        lines.append("Macro: " + "; ".join(x["macro"]["notes"]))
    if x.get("earnings"):
        e = x["earnings"]
        surprise = f"{e['last_surprise_pct']:+.1f}%" if e["last_surprise_pct"] is not None else "n/a"
        lines.append(
            f"Rezultate: surpriză {surprise} la raportul din {e['last_reported']} (acum {e['days_since']} zile),"
            f" estimări depășite {e['beats_last4']} din ultimele trimestre"
        )
    if x.get("insiders"):
        i = x["insiders"]
        lines.append(
            f"Insideri ({i['days']} zile): {_pl(i['buys'], 'cumpărare', 'cumpărări')} de la"
            f" {_pl(i['buyers'], 'persoană', 'persoane')} (${i['buy_value']:,}),"
            f" {_pl(i['sells'], 'vânzare', 'vânzări')} de la {_pl(i['sellers'], 'persoană', 'persoane')} (${i['sell_value']:,})"
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


def train(tickers: list[str], settings: Settings, with_earnings: bool) -> int:
    from .model import ProbabilityModel, build_dataset, walk_forward

    md = MarketData()
    market = md.market(settings.history_period)
    if market is None:
        print("Eroare: nu am putut descărca S&P 500 (SPY), necesar pentru model.", file=sys.stderr)
        return 1
    prices, earnings = {}, {}
    for k, t in enumerate(tickers, 1):
        print(f"[{k}/{len(tickers)}] {t}", file=sys.stderr)
        try:
            prices[t] = md.prices(t, settings.history_period)
        except DataError as exc:
            print(f"  sar peste {t}: {exc}", file=sys.stderr)
            continue
        if with_earnings:
            q = md.earnings(t)
            if q:
                earnings[t] = q
    data = build_dataset(prices, market["Close"], earnings, horizon=settings.horizon_days)
    report = walk_forward(data, horizon=settings.horizon_days)
    print(report.render())
    model = ProbabilityModel(settings.horizon_days).fit(data)
    model.save(settings.model_path)
    print(f"\nModel salvat în {settings.model_path} ({model.info['rows']} rânduri, {model.info['tickers']} acțiuni,"
          f" {model.info['from']} → {model.info['to']}).")
    print("Ce contează cel mai mult (pozitiv = mai multe șanse de creștere):")
    from .model import FEATURE_NAMES

    for name, coef in model.coefficients().items():
        print(f"  {FEATURE_NAMES[name]}: {coef:+.3f}")
    if report.skill <= 0:
        print("\nAtenție: modelul NU a bătut rata de bază în test. Tratează-i probabilitățile cu prudență.")
    return 0


def log_predictions(rec: Recommendation) -> None:
    """Salvează predicția ca s-o putem verifica peste ~4 săptămâni (python -m stockai --evaluate)."""
    from datetime import date

    from . import track

    if not rec.honest:
        return
    made_on = date.fromisoformat(rec.as_of)
    track.log_prediction(rec.ticker, made_on, rec.honest["p"], rec.honest["base"], "stat")
    if rec.claude:
        track.log_prediction(rec.ticker, made_on, rec.claude.probability_up_pct / 100, rec.honest["base"], "claude")


def load_model(settings: Settings):
    if not os.path.exists(settings.model_path):
        return None
    from .model import ProbabilityModel

    try:
        return ProbabilityModel.load(settings.model_path)
    except Exception as exc:
        print(f"Notă: modelul din {settings.model_path} nu a putut fi încărcat ({exc}).", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="stockai", description="Semnale BUY/SELL/HOLD pentru acțiuni")
    parser.add_argument("tickers", nargs="*", help="simboluri, ex. AAPL MSFT NVDA")
    parser.add_argument("--horizon", type=int, default=20, help="orizontul în zile de tranzacționare")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-claude", action="store_true", help="doar reguli, fără Claude")
    mode.add_argument("--always-claude", action="store_true", help="cere opinia lui Claude la fiecare acțiune")
    parser.add_argument("--json", action="store_true", help="rezultat în format JSON")
    parser.add_argument("--train", action="store_true",
                        help="antrenează modelul de probabilitate pe lista de acțiuni (sau pe simbolurile date)")
    parser.add_argument("--limit", type=int, default=None, help="la --train: doar primele N acțiuni din listă")
    parser.add_argument("--with-earnings", action="store_true",
                        help="la --train: include surprizele la rezultate (o cerere Alpha Vantage pe acțiune)")
    parser.add_argument("--evaluate", action="store_true",
                        help="verifică predicțiile salvate care au ajuns la termen și arată cât de bune au fost")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    settings = Settings(horizon_days=args.horizon)

    if args.train:
        tickers = [t.upper() for t in args.tickers] or [s["ticker"] for s in load_universe() if s["sector"] != "ETF"]
        return train(tickers[: args.limit] if args.limit else tickers, settings, args.with_earnings)
    if args.evaluate:
        from . import track

        md = MarketData()
        done = track.resolve(lambda t: md.prices(t, "1y"))
        print(f"Predicții verificate acum: {done}")
        print(track.report())
        return 0
    if not args.tickers:
        parser.error("dă cel puțin un simbol, sau folosește --train / --evaluate")

    claude_mode = "never" if args.no_claude else "always" if args.always_claude else "auto"
    advisor = None
    if claude_mode != "never":
        if ClaudeAdvisor.available():
            advisor = ClaudeAdvisor(model=settings.claude_model, effort=settings.claude_effort)
        else:
            print("Notă: ANTHROPIC_API_KEY lipsește, rulez doar pe reguli.", file=sys.stderr)
    model = load_model(settings)
    if model is None:
        print("Notă: modelul de probabilitate nu e antrenat încă (python -m stockai --train).", file=sys.stderr)

    analyzer = Analyzer(MarketData(), settings=settings, advisor=advisor, claude_mode=claude_mode, model=model)
    results, failed = [], False
    for ticker in args.tickers:
        try:
            rec = analyzer.analyze(ticker)
        except DataError as exc:
            print(f"Eroare: {exc}", file=sys.stderr)
            failed = True
            continue
        results.append(rec)
        log_predictions(rec)
        if not args.json:
            print(render(rec) + "\n")

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2, default=str))
    print("Aceasta este o unealtă de analiză, nu sfat financiar.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
