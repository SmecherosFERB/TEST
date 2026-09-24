"""Cum se ia decizia BUY / SELL / HOLD.

Principiul unui desk profesionist: decizia pornește de la dovezi verificate, nu de la un scor cu ponderi alese de mână.

1. **Statistica decide direcția.** BUY doar dacă tot intervalul de încredere de 90% e peste rata obișnuită
   („sigur statistic”), SELL doar dacă e tot sub. Estimarea vine din modelul verificat pe ani nevăzuți sau,
   fără model, din istoricul acțiunii.
2. **Regulile confirmă sau contrazic.** Scorul compus (tehnic, fundamental, știri, rezultate, insideri, piață)
   nu poate crea singur un BUY/SELL, pentru că ponderile lui nu sunt verificate; dar poate bloca unul (conflict).
3. **Porți de siguranță.** Pe date slabe nu se decide nimic. Încrederea scade înaintea unui raport trimestrial
   și când estimarea nu vine din modelul verificat.
4. **Claude arbitrează doar ce e neclar**, cu verificări: decizia lui trebuie să se potrivească cu propria
   probabilitate, nu poate întoarce o statistică sigură, iar dacă a greșit mai des decât statistica în
   predicțiile verificate, decide statistica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STRONG_RULE = 40.0  # scor compus de la care regulile „văd ceva” ce statistica nu confirmă
LEVELS = ["low", "medium", "high"]


def _lower(conf: str) -> str:
    return LEVELS[max(0, LEVELS.index(conf) - 1)]


def _pct(x: float) -> str:
    return f"{x:.0%}"


@dataclass
class Decision:
    action: str = "HOLD"
    confidence: str = "low"
    decided_by: str = "statistică"
    why: list[str] = field(default_factory=list)  # de ce această decizie
    ask: list[str] = field(default_factory=list)  # ce e neclar (motivele pentru Claude)


def stat_verdict(est: dict[str, Any] | None, min_edge: float) -> str:
    """up / down = sigur statistic; lean_up / lean_down = avantaj, dar nesigur; none = fără avantaj."""
    if not est:
        return "none"
    if est.get("sure") == "up":
        return "up"
    if est.get("sure") == "down":
        return "down"
    edge = est["p"] - est["base"]
    if edge >= min_edge:
        return "lean_up"
    if edge <= -min_edge:
        return "lean_down"
    return "none"


def base_decision(
    est: dict[str, Any] | None,
    rule: str,
    composite: float,
    min_edge: float,
    quality_grade: str | None = None,
    earnings_in_days: int | None = None,
    horizon_days: int = 28,
) -> Decision:
    if quality_grade == "slabă":
        return Decision(why=["datele de preț au probleme: nu decidem pe date nesigure"])
    if not est:
        return Decision(why=["prea puține date pentru o estimare statistică"],
                        ask=["prea puține date pentru o estimare statistică"])
    sv = stat_verdict(est, min_edge)
    interval = f"{_pct(est['lo'])}–{_pct(est['hi'])} față de {_pct(est['base'])} de obicei"
    d = Decision()
    if sv in ("up", "down"):
        action, opposite = ("BUY", "SELL") if sv == "up" else ("SELL", "BUY")
        word = "peste" if sv == "up" else "sub"
        if rule == opposite:
            d.why.append(f"statistica e sigur {word} medie ({interval}), dar regulile spun {opposite}")
            d.ask.append(f"statistica e sigur {word} medie ({interval}), dar regulile spun {opposite} (scor {composite:+.0f})")
        else:
            d.action = action
            margin = est["lo"] - est["base"] if sv == "up" else est["base"] - est["hi"]
            d.confidence = "high" if rule == action and margin >= 0.02 else "medium"
            d.why.append(f"statistica e sigur {word} medie ({interval})")
            d.why.append("regulile sunt de acord" if rule == action else "regulile sunt neutre")
    elif sv in ("lean_up", "lean_down"):
        direction = "BUY" if sv == "lean_up" else "SELL"
        d.why.append(f"statistica înclină spre {'creștere' if sv == 'lean_up' else 'scădere'} ({_pct(est['p'])}, "
                     f"interval {interval}), dar nu e sigură")
        if rule == direction:
            d.ask.append(f"statistica înclină spre {direction} și regulile sunt de acord, dar intervalul include media ({interval})")
        elif rule != "HOLD":
            d.ask.append(f"statistica înclină spre {direction}, regulile spun {rule}")
    else:
        d.why.append(f"statistica nu arată un avantaj ({_pct(est['p'])}, interval {interval})")
        if abs(composite) >= STRONG_RULE:
            d.ask.append(f"regulile dau un semnal puternic ({composite:+.0f}), dar statistica nu arată niciun avantaj ({interval})")
    if d.action != "HOLD":
        if earnings_in_days is not None and 0 <= earnings_in_days <= horizon_days:
            d.confidence = _lower(d.confidence)
            d.why.append(f"raport trimestrial în {earnings_in_days} zile: încredere mai mică")
        if est.get("source") != "model":
            d.confidence = _lower(d.confidence)
            d.why.append("estimarea vine doar din istoricul acțiunii, nu din modelul verificat")
    return d


def with_claude(
    base: Decision,
    verdict: Any,
    est: dict[str, Any] | None,
    min_edge: float,
    quality_grade: str | None = None,
    claude_worse: bool = False,
) -> Decision:
    """Integrează opinia lui Claude, cu verificări. `claude_worse`: Claude a greșit mai des decât statistica."""
    if verdict is None:
        return base
    if claude_worse:
        return Decision(base.action, base.confidence, "statistică",
                        base.why + ["în predicțiile verificate, Claude a greșit mai des decât statistica: decide statistica"],
                        base.ask)
    action, conf = verdict.decision, verdict.confidence
    p = verdict.probability_up_pct / 100 if verdict.probability_up_pct is not None else None
    base_rate = est["base"] if est else 0.5
    why = list(base.why)
    if action == "BUY" and (p is None or p < base_rate + min_edge):
        why.append(f"Claude a spus BUY, dar probabilitatea lui ({_pct(p) if p is not None else 'lipsă'}) nu e peste "
                   f"{_pct(base_rate)} de obicei: HOLD")
        action = "HOLD"
    if action == "SELL" and (p is None or p > base_rate - min_edge):
        why.append(f"Claude a spus SELL, dar probabilitatea lui ({_pct(p) if p is not None else 'lipsă'}) nu e sub "
                   f"{_pct(base_rate)} de obicei: HOLD")
        action = "HOLD"
    sv = stat_verdict(est, min_edge)
    if (sv == "up" and action == "SELL") or (sv == "down" and action == "BUY"):
        why.append("Claude contrazice o statistică sigură: HOLD")
        action = "HOLD"
    if est and p is not None and (p < est["lo"] - 0.05 or p > est["hi"] + 0.05):
        why.append(f"probabilitatea lui Claude ({_pct(p)}) iese mult din intervalul statistic: încredere scăzută")
        conf = "low"
    if quality_grade == "slabă":
        action = "HOLD"
    return Decision(action, conf, "Claude", why, base.ask)
