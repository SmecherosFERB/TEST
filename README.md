# StockAI

Platformă de analiză a acțiunilor care dă un semnal **BUY / SELL / HOLD** împreună cu
procentul istoric din spatele lui. Când regulile nu sunt sigure, cere **a doua opinie de la Claude**.

> Unealtă de analiză, nu sfat financiar. Decizia și riscul rămân la tine.

## Cum funcționează

```
prețuri (5 ani) ──► indicatori tehnici ──► scor tehnic ─┐
date fundamentale ─────────────────────► scor fundamental ├─► scor compus ─► BUY / SELL / HOLD
știri + sentiment ─────────────────────► scor sentiment ─┘            │
                                                                       ▼
                          statistică istorică: „când scorul tehnic arăta ca azi,
                          prețul a urcat în X% din cazuri în următoarele 20 de zile”
                                                                       │
                                        semnal neclar? ──► Claude (a doua opinie)
```

1. **Scor tehnic** (-100 … +100): trend (SMA50/SMA200), MACD, RSI, benzi Bollinger, volum.
2. **Scor fundamental**: P/E actual față de cel estimat, creșterea veniturilor, marja de profit, datoriile, recomandarea analiștilor.
3. **Scor sentiment**: sentimentul știrilor din Alpha Vantage, ponderat cu relevanța (doar dacă ai cheie).
4. **Scor compus**: media ponderată (50% tehnic, 25% fundamental, 25% sentiment). Peste +25 = BUY, sub -25 = SELL.
5. **Procent istoric**: câte zile din ultimii ani au avut un scor tehnic similar și cât de des a urcat prețul după 20 de zile, comparat cu rata de bază (cât de des urcă prețul în general).

### Când intervine Claude

Claude este consultat doar când regulile „nu știu ce să facă”:

- scorul compus e **neutru** (între -25 și +25);
- componentele se **contrazic** (ex. tehnic +60, sentiment -50);
- istoricul **nu are avantaj** (procentul e la mai puțin de 3 puncte de rata de bază) sau are prea puține cazuri;
- regulile spun BUY, dar **istoricul arată invers** (sau SELL și istoricul arată creștere).

Claude primește toate datele: indicatori, scoruri, statistica istorică, fundamentale și titluri de știri.
Răspunde într-un format fix (JSON validat): decizie, probabilitate de creștere, nivel de încredere,
argumentare și riscuri. Dacă Claude nu e disponibil, rămâne decizia regulilor.

## Instalare

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # apoi completează cheile
```

| Variabilă | Obligatorie | Rol |
|---|---|---|
| `ANTHROPIC_API_KEY` | nu (fără ea, doar reguli) | a doua opinie de la Claude |
| `ALPHA_VANTAGE_API_KEY` | nu | sentiment din știri (gratuit, 25 de cereri pe zi) |
| `CLAUDE_MODEL` | nu | implicit `claude-opus-5` |
| `CLAUDE_EFFORT` | nu | `low` / `medium` / `high` (implicit) / `xhigh` / `max` |

## Utilizare

```bash
python -m stockai AAPL MSFT NVDA          # Claude doar la semnalele neclare
python -m stockai AAPL --always-claude    # Claude la fiecare acțiune
python -m stockai AAPL --no-claude        # doar reguli, fără costuri API
python -m stockai AAPL --horizon 10       # orizont de 10 zile în loc de 20
python -m stockai AAPL --json             # rezultat structurat, pentru alte programe
```

Exemplu de rezultat:

```
═══ AAPL · 227.35 · 2026-09-23 ═══
Decizie: SELL  (decis de Claude, încredere medie)
Scoruri: tehnic +12 · fundamental +26 · sentiment -40 · compus +2
Istoric (doar tehnic, 20 zile): a urcat în 58% din cazurile similare (rata de bază 56%, n=240, randament mediu +1.2%)
Semnal neclar: scor compus neutru (+2, prag ±25); istoric fără avantaj: 58% față de rata de bază 56%
Claude: 41% șanse de creștere, decizie SELL
  ...
```

## Limitări (de citit)

- **Procentul istoric folosește doar scorul tehnic**, pentru că pentru fundamentale și știri nu avem istoric gratuit.
- Zilele similare din istoric **se suprapun** (ferestre de 20 de zile consecutive), deci `n` supraestimează câte cazuri independente există.
- Procentele descriu trecutul acțiunii respective. Nu garantează nimic despre viitor.
- Ponderile și pragurile sunt puncte de plecare. Trebuie ajustate după backtesting (următorul pas).

## Structura proiectului

```
stockai/
  config.py       praguri, ponderi, orizont, modelul Claude
  data.py         Yahoo Finance (prețuri, fundamentale, titluri) + Alpha Vantage (sentiment)
  indicators.py   SMA, EMA, RSI, MACD, Bollinger, volum
  scoring.py      scorurile tehnic / fundamental / sentiment / compus
  calibration.py  procentul istoric pentru scoruri similare
  advisor.py      Claude: cerere cu output structurat și fallback automat la refuz
  analyzer.py     orchestrare + detectarea semnalelor neclare
  __main__.py     linia de comandă
tests/            teste fără rețea (date sintetice, Claude simulat)
```

```bash
python -m pytest
```

## Ce urmează

- [ ] Backtest complet: randament, pierdere maximă, rata de reușită comparate cu S&P 500
- [ ] Ajustarea ponderilor și pragurilor pe baza backtestului
- [ ] Model ML (LightGBM) cu probabilități calibrate
- [ ] Dashboard (Streamlit) cu watchlist și grafice
- [ ] Alerte pe email sau Telegram la semnale puternice
- [ ] Căutare web pentru Claude (știri de ultimă oră)
