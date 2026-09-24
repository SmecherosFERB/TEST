# StockAI

Platformă de analiză a acțiunilor care dă un semnal **BUY / SELL / HOLD** împreună cu
probabilitățile din spatele lui. Când semnalele nu sunt clare, cere **a doua opinie de la Claude**.

> Unealtă de analiză, nu sfat financiar. Decizia și riscul rămân la tine.

## Cum funcționează

```
prețuri zilnice ──► indicatori ──► scor tehnic ────────┐
date fundamentale ───────────────► scor fundamental ───┤
știri + sentiment ───────────────► scor sentiment ─────┤
rezultate trimestriale ──────────► scor rezultate ─────├─► scor compus ─► BUY / SELL / HOLD
tranzacțiile insiderilor ────────► scor insideri ──────┤          │
S&P 500 + macro (VIX, credit) ───► scor piață ─────────┘          │
                                                                  ▼
     procent istoric (situații tehnice similare)  +  model statistic antrenat pe ~113 acțiuni
                                                                  │
                                   semnal neclar? ──► Claude (a doua opinie, cu toate datele)
```

| Componentă | Ce măsoară | Pondere |
|---|---|---|
| Tehnic | trend (SMA50/200), MACD, RSI, benzi Bollinger, volum | 35% |
| Fundamental | P/E actual vs. estimat, creșterea veniturilor, marja, datoriile, părerea analiștilor | 15% |
| Sentiment | tonul știrilor (Alpha Vantage), ponderat cu relevanța | 10% |
| Rezultate | surpriza față de estimări la ultimul raport (se estompează în ~3 luni), câte din ultimele 4 au depășit estimările | 15% |
| Insideri | cumpărări pe piață ale conducerii în ultimele 180 de zile (mai mulți cumpărători = semnal mai puternic) | 10% |
| Piață | trendul S&P 500 și stresul macro (VIX, prima de risc la obligațiuni, curba randamentelor) | 15% |

Componentele fără date sunt omise, iar ponderile se recalculează. Scor compus peste +25 = BUY, sub -25 = SELL.

### Probabilitățile

1. **Procent istoric**: cât de des a urcat acțiunea în 20 de zile când scorul tehnic arăta ca azi, față de rata de bază.
2. **Model statistic** (după `python -m stockai --train`): regresie logistică pe semnale cu dovezi publicate
   (momentum pe 12 luni, distanța față de maximul pe 52 de săptămâni, volatilitate, trendul pieței, surpriza la rezultate etc.),
   antrenată pe toată lista de acțiuni. Înainte de salvare e verificată **walk-forward**: pentru fiecare an, modelul învață doar
   din anii anteriori și e testat pe anul respectiv. Raportul arată dacă bate rata de bază (eroare Brier, AUC, calibrare).
   Dacă nu o bate, programul te avertizează.

### Când intervine Claude

Claude e consultat doar când regulile „nu știu ce să facă”:

- scorul compus e **neutru** (între -25 și +25);
- componentele se **contrazic** (ex. tehnic +60, piață -50);
- istoricul **nu are avantaj** sau are prea puține cazuri, ori **contrazice** regulile;
- **modelul statistic contrazice** regulile.

Claude primește tot: indicatori, scoruri, statistica istorică, probabilitatea modelului, rezultate, insideri,
trendul pieței, macro, fundamentale și știri. Răspunde într-un format fix: decizie, probabilitate, încredere,
argumente și riscuri.

## Instalare

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # apoi completează cheile
```

Toate sursele sunt gratuite și opționale. Fără o cheie, componenta respectivă lipsește și analiza continuă.

| Variabilă | Sursă | Ce adaugă | Limită gratuită |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | a doua opinie la semnalele neclare | plătit per utilizare |
| `TWELVE_DATA_API_KEY` | [Twelve Data](https://twelvedata.com) | prețuri zilnice pe ~20 de ani, fundamentale (inclusiv datorii și short interest), rezultate trimestriale, insideri | 800 cereri/zi |
| `ALPHA_VANTAGE_API_KEY` | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) | știri cu sentiment, rezultate trimestriale, insideri | 25 cereri/zi |
| `SEC_USER_AGENT` | [SEC EDGAR](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) | tranzacțiile oficiale ale insiderilor (formularele 4) | SEC nu emite chei API: pui doar un nume și un email |
| `FRED_API_KEY` | [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | macro: VIX, curba randamentelor, prima de risc | opțional; merge și fără cheie |
| `CLAUDE_MODEL`, `CLAUDE_EFFORT` | — | modelul Claude (implicit `claude-opus-5`) și cât „gândește” | — |

Datele descărcate se păstrează în `.cache/` (prețuri 12 ore, rezultate 3 zile), ca să nu consumi cereri de două ori.

## Utilizare

```bash
python -m stockai --train                 # descarcă istoricul listei, testează modelul, îl salvează
python -m stockai --train --limit 30      # doar primele 30 de acțiuni (mai rapid)
python -m stockai AAPL MSFT NVDA          # analiză; Claude doar la semnalele neclare
python -m stockai AAPL --always-claude    # Claude la fiecare acțiune
python -m stockai AAPL --no-claude        # fără costuri Claude
python -m stockai AAPL --json             # rezultat structurat
```

La prima antrenare, cu Twelve Data gratuit (8 cereri pe minut), descărcarea celor ~113 acțiuni durează cam 15 minute.
Apoi totul vine din cache.

Exemplu de rezultat:

```
═══ AAPL · 337.02 · 2026-09-23 ═══
Decizie: HOLD  (decis de Claude, încredere scăzută)
Scoruri: tehnic +37 · fundamental +42 · sentiment +18 · rezultate +47 · insideri -20 · piață +55 · compus +34
Model statistic: 58% șanse de creștere în 20 zile (de obicei 56%; antrenat pe 113 acțiuni)
Istoric (doar tehnic, 20 zile): a urcat în 61% din cazurile similare (rata de bază 57%, n=312, randament mediu +1.8%)
Piața (S&P 500): în creștere, +2.1% în ultima lună
Macro: VIX 17.3: piață calmă
Rezultate: surpriză +7.4% la raportul din 2026-07-30 (acum 56 zile), estimări depășite 4/4 din ultimele trimestre
Insideri (180 zile): 0 cumpărări de la 0 persoane ($0), 12 vânzări de la 3 persoane ($5,300,000)
...
```

## Versiunea web (`web/stockai.html`)

Aceeași logică, rescrisă în JavaScript, publicată ca pagină claude.ai. Datele vin prin conectorii Twelve Data
(principal) și Alpha Vantage (știri și rezervă) ai celui care deschide pagina, iar a doua opinie vine de la Claude,
din contul acelei persoane.

- **„Șanse mari acum”**: clasamentul acțiunilor scanate (o cerere pe acțiune), plus trendul S&P 500;
- **~115 acțiuni importante cu numele lor** în baza de date a paginii (căutare după nume sau simbol);
  lista e în `stockai/universe.json`, iar acțiunile noi analizate se adaugă singure;
- analiza completă include rezultatele trimestriale și insiderii (păstrate 7 zile în baza de date) și trendul pieței
  (o dată pe zi);
- folosește **prețuri zilnice din Twelve Data** (medii pe 50 și 200 de zile, statistică pe 10 ani); dacă Twelve Data
  nu răspunde, trece automat pe bare săptămânale din Alpha Vantage.

Costul în cereri: scanarea, 1 cerere Twelve Data pe acțiune (planul gratuit are 800 pe zi, dar maximum 8 pe minut,
deci ~8 secunde pe acțiune). Prima analiză completă a unei acțiuni: 4 cereri Twelve Data (prețuri, fundamentale,
rezultate, insideri) și 1 Alpha Vantage (știri). Datorită cache-ului, analizele următoare costă mai puțin.

## Limitări (de citit)

- Procentul istoric folosește doar scorul tehnic; modelul statistic folosește doar semnale care au istoric gratuit.
- Zilele similare din istoric **se suprapun**, deci `n` supraestimează câte cazuri independente există.
- Lista de acțiuni conține companiile mari de **azi**. Testul pe trecut e deci ușor prea optimist (survivorship bias).
- Clasificarea insiderilor din Alpha Vantage e aproximativă (nu are codul tranzacției); SEC EDGAR e exact.
- Procentele descriu trecutul. Nu garantează nimic despre viitor.

## Structura proiectului

```
stockai/
  config.py        praguri, ponderi, orizont, modelul Claude
  data.py          alege sursele (Twelve Data / Yahoo, Alpha Vantage, SEC, FRED) și le pune în cache
  sources/         clienții pentru fiecare sursă gratuită
  cache.py         cache pe disc și pauze între cereri
  indicators.py    SMA, EMA, RSI, MACD, Bollinger, volum
  scoring.py       scorurile tehnic / fundamental / sentiment / compus
  signals.py       momentum, maxim 52 săpt., rezultate, insideri, trendul pieței, macro
  calibration.py   procentul istoric pentru scoruri similare
  model.py         modelul de probabilitate și testul walk-forward
  advisor.py       Claude: cerere cu output structurat și fallback automat la refuz
  analyzer.py      orchestrare + detectarea semnalelor neclare
  universe.py      lista de ~115 acțiuni (universe.json)
  __main__.py      linia de comandă
tests/             teste fără rețea (date sintetice, surse și Claude simulate)
docs/              surse de date comparate
web/               pagina claude.ai
```

```bash
python -m pytest
```

## Ce urmează

- [ ] Backtest de portofoliu: randament, pierdere maximă, comparat cu S&P 500
- [ ] Short interest (FINRA) în model
- [ ] Grafice cu lumânări (TradingView Lightweight Charts)
- [ ] Alerte pe email sau Telegram la semnale puternice
