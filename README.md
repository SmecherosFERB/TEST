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
     procent istoric (situații tehnice similare)  +  model statistic pe ~113 acțiuni (crește? bate S&P 500?)
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
2. **Model statistic** (după `python -m stockai --train`): regresie logistică cu penalizare pe semnale cu dovezi publicate:
   momentum pe 12 luni (cu un termen separat după un an slab al pieței), revenirea după ultima lună, apropierea de
   maximul pe 52 de săptămâni, volatilitate, volum neobișnuit, trendul acțiunii și al pieței, surpriza la rezultate.
   E antrenat pe toată lista de acțiuni, pentru două întrebări: **crește prețul?** și **bate acțiunea S&P 500?**
   (`--target beat`). A doua e de obicei mai previzibilă: semnalele spun mai mult despre care acțiuni se descurcă
   mai bine decât altele decât despre direcția pieței.
3. **Verificat pe ani nevăzuți, apoi recalibrat.** Pentru fiecare an, modelul învață doar din anii anteriori și e
   testat pe anul respectiv. Pe rezultatele acestor teste recalibrăm probabilitățile: dacă ordinea dată de model
   nu a contat sigur statistic, procentele rămân aproape de medie. Raportul arată cât de des s-a întâmplat
   lucrul estimat în fiecare cincime (de la cele mai slabe 20% la cele mai bune 20% după model).

### Calitatea datelor și regulile de selecție

Înainte ca o serie de prețuri să intre în analiză sau în model, e verificată ca la un desk profesionist:

| Verificare | De ce contează | Ce se întâmplă dacă pică |
|---|---|---|
| Doar bare închise | în timpul ședinței, bara de azi are preț și volum parțiale | bara zilei curente e scoasă până la 16:15, ora New York |
| Split-uri neajustate | un salt de o zi de tip 2:1, 3:1, 10:1 falsifică momentumul și rezultatele | ferestrele din jurul saltului (un an înainte, o lună după) nu intră în model |
| Mișcări extreme reale | o prăbușire de −55% nu e un split | se păstrează, dar sunt semnalate |
| Date la zi, zile lipsă, cotații înghețate, volum lipsă, istoric scurt | date vechi sau găurite dau semnale greșite | scor de calitate 0–100; seriile „slabe” nu intră în model și nici în clasament |
| Verificare încrucișată | maximul pe 52 de săptămâni din serie trebuie să se potrivească cu cel raportat separat de Twelve Data | penalizare și avertizare |
| Lichiditate | sub ~20 mil. $ tranzacționați pe zi, costurile mănâncă avantajul | acțiunea nu intră în „Șanse mari” |
| Aceeași companie, două clase (GOOG/GOOGL) | ar număra de două ori aceleași date | în model contează o singură dată |

Alte reguli de profesionist:

- **Intrare realistă:** semnalul se calculează la închiderea zilei, dar rezultatul se măsoară de la închiderea
  de a doua zi. Nu câștigăm pe hârtie o zi pe care n-am fi putut-o tranzacționa.
- **Diversificare:** în „Șanse mari” intră cel mult două acțiuni din același sector.
- **Risc la vedere:** fiecare acțiune arată mișcarea tipică pe 4 săptămâni (din volatilitate) și beta față de
  S&P 500; acțiunile cu raport trimestrial în următoarele 4 săptămâni sunt marcate.
- **Regimul pieței:** când S&P 500 e în scădere, pagina recomandă să te uiți la „bate S&P 500”, nu la „crește”.

### Cât de reale sunt procentele

Trei lucruri fac diferența între un procent care arată bine și unul pe care te poți baza:

1. **Cazuri independente, nu zile.** Ferestrele de 20 de zile se suprapun: 20 de zile la rând valorează cam o
   singură observație. „68% din 91 de cazuri” înseamnă de fapt ~23 de cazuri independente, adică un interval de
   încredere de 90% de ~51–82%. De aceea afișăm mereu **intervalul**, iar dacă el include rata obișnuită,
   spunem clar că **nu avem un avantaj dovedit**.
2. **Media tuturor acțiunilor.** Procentul unei singure acțiuni e tras spre ce s-a întâmplat la toate acțiunile
   cu un scor similar (în pagină: acțiunile scanate; în program: modelul antrenat pe toată lista). Asta reduce
   și efectul de „privire înapoi”: acțiunile care au urcat mult în trecut par să urce „de obicei” mai des decât
   o vor face probabil în viitor.
3. **Acțiunile se mișcă împreună.** În aceeași lună, multe acțiuni urcă sau coboară odată cu piața, așa că
   1.000 de observații pot valora cât ~200 independente. Modelul măsoară acest efect și lărgește intervalele.
4. **Verificare pe viitor.** Fiecare predicție (a statisticii și a lui Claude) e salvată și verificată cu
   prețul real după ~4 săptămâni. Pagina arată panoul „Cât de bune au fost predicțiile”; în program rulezi
   `python -m stockai --evaluate`. Comparăm cu „ca de obicei” (eroarea Brier) și verificăm calibrarea: când
   am spus 60%, a urcat chiar în ~60% din cazuri? De la ~100 de predicții verificate, rezultatele devin de încredere.

### Când intervine Claude

Claude e consultat doar când regulile „nu știu ce să facă”:

- scorul compus e **neutru** (între -25 și +25);
- componentele se **contrazic** (ex. tehnic +60, piață -50);
- istoricul **nu are avantaj** sau are prea puține cazuri, ori **contrazice** regulile;
- **modelul statistic contrazice** regulile.

Claude primește și data următorului raport trimestrial: dacă acesta cade în următoarele 4 săptămâni, prețul poate sări
mult în orice direcție, iar programul și pagina te avertizează.

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
python -m stockai --train --target beat   # al doilea model: șansele de a bate S&P 500
python -m stockai AAPL MSFT NVDA          # analiză; Claude doar la semnalele neclare
python -m stockai AAPL --always-claude    # Claude la fiecare acțiune
python -m stockai AAPL --no-claude        # fără costuri Claude
python -m stockai AAPL --json             # rezultat structurat
python -m stockai --evaluate              # verifică predicțiile ajunse la termen (după ~4 săptămâni)
```

La prima antrenare, cu Twelve Data gratuit (8 cereri pe minut), descărcarea celor ~113 acțiuni durează cam 15 minute.
Apoi totul vine din cache.

Exemplu de rezultat:

```
═══ AAPL · 337.02 · 2026-09-23 ═══
Decizie: HOLD  (decis de Claude, încredere scăzută)
Scoruri: tehnic +37 · fundamental +42 · sentiment +18 · rezultate +47 · insideri -20 · piață +55 · compus +34
Model statistic: 58% șanse de creștere în 20 zile (interval 90%: 55%–61%, de obicei 56%; antrenat pe 113 acțiuni; a ajutat în test: da)
Model statistic: 54% șanse să bată S&P 500 în 20 zile (interval 90%: 51%–57%, de obicei 51%; antrenat pe 113 acțiuni; a ajutat în test: nu încă)
Istoric (doar tehnic, 20 zile): a urcat în 61% din cazurile similare (rata de bază 57%, n=312, randament mediu +1.8%)
Piața (S&P 500): în creștere, +2.1% în ultima lună
Macro: VIX 17.3: piață calmă
Rezultate: surpriză +7.4% la raportul din 2026-07-30 (acum 56 zile), estimări depășite 4/4 din ultimele trimestre
Următorul raport trimestrial: 2026-10-29 (în 35 de zile)
Insideri (180 zile): 0 cumpărări de la 0 persoane ($0), 12 vânzări de la 3 persoane ($5,300,000)
...
```

## Versiunea web (`web/stockai.html`)

Aceeași logică, rescrisă în JavaScript, publicată ca pagină claude.ai. Datele vin prin conectorii Twelve Data
(principal) și Alpha Vantage (știri și rezervă) ai celui care deschide pagina, iar a doua opinie vine de la Claude,
din contul acelei persoane.

- **„Șanse mari acum”**: clasamentul acțiunilor scanate (o cerere pe acțiune), plus trendul S&P 500;
- **modelul învățat în pagină**: la fiecare scanare, pagina păstrează din 20 în 20 de zile (ferestre care nu se
  suprapun) semnalele acțiunii și ce a urmat. De la 15 acțiuni scanate, învață din toate, se verifică an cu an pe
  ani nevăzuți și ordonează clasamentul după **„Crește”** sau **„Bate S&P 500”**. Panoul „Modelul învățat” arată
  cât de des au reușit cele mai bune 20% față de cele mai slabe 20% și ce semnale au contat;
- **~115 acțiuni importante cu numele lor** în baza de date a paginii (căutare după nume sau simbol);
  lista e în `stockai/universe.json`, iar acțiunile noi analizate se adaugă singure;
- analiza completă include rezultatele trimestriale și insiderii (păstrate 7 zile în baza de date) și trendul pieței
  (o dată pe zi);
- folosește **prețuri zilnice din Twelve Data** (medii pe 50 și 200 de zile, statistică pe 10 ani); dacă Twelve Data
  nu răspunde, trece automat pe bare săptămânale din Alpha Vantage.

Costul în cereri: scanarea, 1 cerere Twelve Data pe acțiune (planul gratuit are 800 pe zi, dar maximum 8 pe minut,
deci ~8 secunde pe acțiune). Prima analiză completă a unei acțiuni: 4 cereri Twelve Data (prețuri, fundamentale,
rezultate, insideri) și 1 Alpha Vantage (știri). O dată pe zi, pagina cere și calendarul complet al rezultatelor
(1 cerere Alpha Vantage), ca să marcheze în clasament acțiunile cu raport trimestrial în următoarele 4 săptămâni. Datorită cache-ului, analizele următoare costă mai puțin.

## Limitări (de citit)

- Procentul istoric folosește doar scorul tehnic; modelul statistic folosește doar semnale care au istoric gratuit.
- Chiar și cele mai bune modele publicate explică sub 1% din variația randamentelor lunare. Un model bun ridică
  șansele de la, de exemplu, 55% la 58–60% pentru cele mai bune acțiuni, nu la 80%.
- Zilele similare din istoric **se suprapun**, deci `n` supraestimează câte cazuri independente există.
- Lista de acțiuni conține companiile mari de **azi**. Testul pe trecut e deci ușor prea optimist (survivorship bias).
- Conectorul Twelve Data nu oferă prețuri ajustate pentru dividende, doar pentru split-uri. Pentru „bate S&P 500”
  diferența e mică (dividendele pe 4 săptămâni sunt ~0,1%), dar randamentele acțiunilor cu dividend mare sunt ușor subestimate.
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
  quality.py       calitatea datelor: split-uri, date vechi, goluri, lichiditate, bare incomplete
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
