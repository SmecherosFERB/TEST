"""Cum se ia decizia: mereu BUY sau SELL, cu o convingere și o mărime de poziție.

Așa lucrează un desk profesionist: fiecare acțiune primește o direcție, iar forța dovezilor decide **cât** cumperi
sau vinzi. Un semnal slab nu devine „HOLD”, ci o poziție mică (sau zero, când nu există niciun avantaj).

1. **Direcția** vine din statistica verificată pe ani nevăzuți: de ce parte a ratei obișnuite e probabilitatea
   estimată. Fără niciun avantaj, departajează modelul „bate S&P 500”, apoi trade-ul cu țintă și stop, apoi regulile.
2. **Convingerea** (niciuna / scăzută / medie / ridicată): ridicată doar când tot intervalul de 90% e de o parte a
   mediei, modelul a ajutat sigur în anii de test și regulile sunt de acord. Scade înaintea unui raport trimestrial,
   când estimarea nu vine din modelul verificat, când trade-ul cu țintă și stop contrazice direcția, sau pe date slabe.
3. **Mărimea** vine din risc: la o mișcare tipică de 4 săptămâni împotriva poziției pierzi cel mult ~1% din portofoliu
   la convingere ridicată; convingerea mai mică micșorează poziția, iar fără avantaj ea e zero.
4. **Claude** arbitrează doar ce e neclar, cu verificări: direcția lui trebuie să se potrivească cu propria
   probabilitate, nu poate întoarce o statistică sigură, iar dacă a greșit mai des decât statistica, decide statistica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

STRONG_RULE = 40.0  # scor compus de la care regulile „văd ceva” ce statistica nu confirmă
LEVELS = ["none", "low", "medium", "high"]
RISK_PER_TRADE = 0.01  # cât pierzi din portofoliu dacă prețul merge o abatere tipică împotriva poziției
SIZE_BY_CONVICTION = {"none": 0.0, "low": 0.25, "medium": 0.6, "high": 1.0}
MAX_POSITION = 0.10


def _lower(conf: str) -> str:
    return LEVELS[max(0, LEVELS.index(conf) - 1)]


def _raise(conf: str) -> str:
    return LEVELS[min(len(LEVELS) - 1, LEVELS.index(conf) + 1)]


def _pct(x: float) -> str:
    return f"{x:.0%}"


def position_size(conviction: str, move: float | None) -> float | None:
    """Fracțiunea din portofoliu: risc fix pe o mișcare tipică (`move`, de ex. 0,08), scalat cu convingerea."""
    if move is None or not move > 0:
        return None
    # Rotunjire „jumătate în sus”, ca în pagină (round() din Python rotunjește spre par).
    return math.floor(min(MAX_POSITION, RISK_PER_TRADE / move * SIZE_BY_CONVICTION[conviction]) * 1e4 + 0.5) / 1e4


@dataclass
class Decision:
    action: str = "BUY"  # BUY sau SELL
    confidence: str = "none"  # convingerea: none / low / medium / high
    decided_by: str = "statistică"
    why: list[str] = field(default_factory=list)  # de ce această decizie
    ask: list[str] = field(default_factory=list)  # ce e neclar (motivele pentru Claude)
    size: float | None = None  # fracțiunea din portofoliu sugerată
    move: float | None = None  # mișcarea tipică pe orizont, folosită pentru mărime


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


def _edge(e: dict[str, Any] | None) -> float:
    return (e["p"] - e["base"]) if e else 0.0


HOLD_LABEL = {5: "o săptămână", 10: "două săptămâni", 20: "o lună", 40: "două luni", 60: "trei luni"}


def holding_plan(per_horizon: dict[int, dict[str, Any]], action: str, price: float, daily_vol: float | None,
                 helps: dict[int, bool] | None = None, standard_days: int = 20) -> dict[str, Any] | None:
    """Cât să ții: orizontul cu cel mai bun avantaj în direcția deciziei, raportat la timp (avantaj × √(20/zile)),
    cu ieșire mai devreme la țintă sau stop (o abatere tipică pe acel orizont)."""
    rows = []
    for h, e in sorted(per_horizon.items()):
        if not e:
            continue
        edge = e["p"] - e["base"] if action == "BUY" else e["base"] - e["p"]
        low = (e["lo"] - e["base"]) if action == "BUY" else (e["base"] - e["hi"])
        sure = bool((helps or {}).get(h)) and e.get("lo") is not None and low > 0
        rows.append({"days": h, "prob_up": e["p"], "base": e["base"], "edge": edge, "sure": sure, "score": edge * (20 / h) ** 0.5})
    if len(rows) < 2:
        return None
    # Fără un avantaj de cel puțin jumătate de punct pe vreun orizont, rămâne orizontul standard al deciziei.
    with_edge = [r for r in rows if r["edge"] >= 0.005]
    best = (max(with_edge, key=lambda r: r["score"]) if with_edge
            else next((r for r in rows if r["days"] == standard_days), rows[0]))
    plan = {"days": best["days"], "label": HOLD_LABEL.get(best["days"], f"{best['days']} zile"), "sure": best["sure"],
            "no_edge": not with_edge, "per_horizon": rows}
    if daily_vol and daily_vol > 0:
        w = daily_vol * best["days"] ** 0.5
        up, down = price * math.exp(w), price * math.exp(-w)
        plan["take_profit"], plan["stop_loss"] = (up, down) if action == "BUY" else (down, up)
    return plan


def base_decision(
    est: dict[str, Any] | None,
    rule: str,
    composite: float,
    min_edge: float,
    quality_grade: str | None = None,
    earnings_in_days: int | None = None,
    horizon_days: int = 28,
    trade_est: dict[str, Any] | None = None,
    beat_est: dict[str, Any] | None = None,
    move: float | None = None,
) -> Decision:
    d = Decision(move=move)
    if not est:
        d.action = "SELL" if composite < 0 else "BUY"
        d.why.append("prea puține date pentru o estimare statistică: direcția vine doar din reguli, fără poziție")
        d.ask.append("prea puține date pentru o estimare statistică")
        d.size = position_size(d.confidence, move)
        return d
    sv = stat_verdict(est, min_edge)
    interval = f"{_pct(est['lo'])}–{_pct(est['hi'])} față de {_pct(est['base'])} de obicei"
    if sv in ("up", "down"):
        d.action = "BUY" if sv == "up" else "SELL"
        opposite = "SELL" if sv == "up" else "BUY"
        word = "peste" if sv == "up" else "sub"
        margin = est["lo"] - est["base"] if sv == "up" else est["base"] - est["hi"]
        d.why.append(f"statistica e sigur {word} medie ({interval})")
        if rule == opposite:
            d.confidence = "low"
            d.why.append(f"dar regulile spun {opposite}: convingere scăzută")
            d.ask.append(f"statistica e sigur {word} medie ({interval}), dar regulile spun {opposite} (scor {composite:+.0f})")
        else:
            d.confidence = "high" if rule == d.action and margin >= 0.02 else "medium"
            d.why.append("regulile sunt de acord" if rule == d.action else "regulile sunt neutre")
    elif sv in ("lean_up", "lean_down"):
        d.action = "BUY" if sv == "lean_up" else "SELL"
        d.confidence = "low"
        d.why.append(f"statistica înclină spre {'creștere' if sv == 'lean_up' else 'scădere'} ({_pct(est['p'])}, "
                     f"interval {interval}), dar nu e sigură: convingere scăzută")
        if rule == d.action:
            d.ask.append(f"statistica înclină spre {d.action} și regulile sunt de acord, dar intervalul include media ({interval})")
        elif rule not in ("HOLD", d.action):
            d.ask.append(f"statistica înclină spre {d.action}, regulile spun {rule}")
    else:
        # Fără avantaj: direcția o departajează, pe rând, modelul „bate S&P 500”, trade-ul și regulile. Poziție zero.
        for source, value in (("statistica", _edge(est)), ("modelul „bate S&P 500”", _edge(beat_est)),
                              ("trade-ul cu țintă și stop", _edge(trade_est)), ("regulile", composite / 1000)):
            if abs(value) >= 0.005 or source == "regulile":
                d.action = "BUY" if value >= 0 else "SELL"
                d.why.append(f"statistica nu arată un avantaj ({_pct(est['p'])}, interval {interval}); "
                             f"direcția o {'dau' if source == 'regulile' else 'dă'} {source}; fără poziție")
                break
        if abs(composite) >= STRONG_RULE:
            d.ask.append(f"regulile dau un semnal puternic ({composite:+.0f}), dar statistica nu arată niciun avantaj ({interval})")
    # Al doilea filtru („meta-labeling”, López de Prado): trade-ul cu țintă și stop confirmă sau slăbește direcția.
    if d.confidence != "none" and trade_est and trade_est.get("sure"):
        against = (d.action == "BUY" and trade_est["sure"] == "down") or (d.action == "SELL" and trade_est["sure"] == "up")
        detail = f"{_pct(trade_est['p'])} șanse ca ținta să fie atinsă prima, de obicei {_pct(trade_est['base'])}"
        if against:
            d.confidence = _lower(d.confidence)
            d.why.append(f"trade-ul cu țintă și stop contrazice direcția ({detail}): convingere mai mică")
        else:
            d.confidence = _raise(d.confidence) if d.confidence == "medium" else d.confidence
            d.why.append(f"trade-ul cu țintă și stop confirmă ({detail})")
    if d.confidence != "none":
        if earnings_in_days is not None and 0 <= earnings_in_days <= horizon_days:
            d.confidence = _lower(d.confidence)
            d.why.append(f"raport trimestrial în {earnings_in_days} zile: convingere mai mică")
        if est.get("source") != "model":
            d.confidence = _lower(d.confidence)
            d.why.append("estimarea vine doar din istoricul acțiunii, nu din modelul verificat")
    if quality_grade == "slabă":
        d.confidence = "none"
        d.why.append("datele de preț au probleme: fără poziție până se corectează")
    d.size = position_size(d.confidence, move)
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
    why = list(base.why)
    if claude_worse:
        why.append("în predicțiile verificate, Claude a greșit mai des decât statistica: decide statistica")
        return Decision(base.action, base.confidence, "statistică", why, base.ask, base.size, base.move)
    p = verdict.probability_up_pct / 100 if verdict.probability_up_pct is not None else None
    base_rate = est["base"] if est else 0.5
    side = None if p is None else "BUY" if p >= base_rate + min_edge else "SELL" if p <= base_rate - min_edge else None
    if side is None:
        why.append(f"Claude nu vede un avantaj ({_pct(p) if p is not None else 'fără probabilitate'} față de "
                   f"{_pct(base_rate)} de obicei): rămâne decizia statisticii")
        return Decision(base.action, base.confidence, "statistică", why, base.ask, base.size, base.move)
    action = verdict.decision if verdict.decision in ("BUY", "SELL") else side
    if action != side:
        why.append(f"Claude a spus {action}, dar probabilitatea lui ({_pct(p)}) arată {side}: contează probabilitatea")
        action = side
    conf = {"low": "low", "medium": "medium", "high": "high"}.get(verdict.confidence, "low")
    sv = stat_verdict(est, min_edge)
    if (sv == "up" and action == "SELL") or (sv == "down" and action == "BUY"):
        why.append("Claude contrazice o statistică sigură: rămâne direcția statisticii, cu convingere scăzută")
        action, conf = ("BUY" if sv == "up" else "SELL"), "low"
    if est and (p < est["lo"] - 0.05 or p > est["hi"] + 0.05):
        why.append(f"probabilitatea lui Claude ({_pct(p)}) iese mult din intervalul statistic: convingere scăzută")
        conf = "low"
    if quality_grade == "slabă":
        conf = "none"
    return Decision(action, conf, "Claude", why, base.ask, position_size(conf, base.move), base.move)
