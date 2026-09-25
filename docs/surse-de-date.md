# Surse de date pentru StockAI

Cercetare făcută în septembrie 2026. Prețurile și limitele se schimbă des, așa că verifică pagina
furnizorului înainte să plătești ceva.

## Pe scurt

| Prioritate | Ce | Cost | Ce câștigăm |
|---|---|---|---|
| 1 | **Twelve Data** prin conectorul claude.ai | gratuit (800 cereri/zi) | istoric zilnic lung în pagină, în loc de date săptămânale; toată lista de ~115 acțiuni scanată într-o zi |
| 2 | **SEC EDGAR, FRED, FINRA** în programul Python | gratuit, surse oficiale | cumpărări ale insiderilor, context macro, short interest |
| 3 | **Backtest pe date zilnice** | gratuit | aflăm care semnale chiar ajută, înainte să le dăm pondere |
| 4 (opțional) | **FMP Starter** sau **Tiingo Power** | ~22–30 $/lună | estimări și surprize la rezultate, istoric zilnic curat |
| — | **TradingView Lightweight Charts** | gratuit (Apache 2.0) | grafice cu lumânări, zoom, volum |

## Ce e deja implementat

| Sursă | Programul Python | Pagina claude.ai |
|---|---|---|
| Twelve Data: prețuri zilnice, fundamentale, rezultate, insideri | da, cu `TWELVE_DATA_API_KEY` | da (sursa principală) |
| Alpha Vantage: știri, rezultate trimestriale, insideri | da | da |
| SEC EDGAR: insideri (formularele 4) | da, cu `SEC_USER_AGENT` | nu (fără conector) |
| FRED: VIX, curba randamentelor, prima de risc | da | nu (fără conector) |
| Trendul S&P 500 (SPY) | da | da |
| Model de probabilitate verificat walk-forward și recalibrat pe anii nevăzuți | da (`--train`) | da (învață din acțiunile scanate) |
| Ținta „bate S&P 500” (randament relativ) | da (`--train --target beat`) | da |
| Verificarea calității datelor și reguli de selecție (lichiditate, sector, bare închise) | da (`quality.py`) | da |
| Semnale noi: MAX, volatilitate proprie, informație continuă, sezonalitate, medii pe 20/100 zile | da | da |
| Modelul „Trade” (ținta înaintea stopului) și combinarea prognozelor | da (`--train --target trade`) | da |
| Reviziile analiștilor (Twelve Data `eps_trend`) | da | da (analiza completă) |
| Data următorului raport trimestrial (avertizare) | da (Twelve Data, altfel calendarul Alpha Vantage) | da (calendarul complet Alpha Vantage, o cerere pe zi pentru toată lista) |

## Ce îmbunătățește realist predicția

Mai multe surse nu înseamnă automat predicții mai bune. Ce contează:

1. **Istoric zilnic lung, ajustat pentru dividende și split-uri.** Fără el, backtestul și
   procentele istorice sunt aproximative. Pagina lucrează acum pe date săptămânale, pentru că
   planul gratuit Alpha Vantage oferă doar ultimele 100 de zile.
2. **Date fără „survivorship bias”.** Dacă istoricul conține doar companiile care există azi,
   backtestul iese prea optimist. Seturi ca Sharadar includ și companiile delistate.
3. **Semnale cu dovezi publicate**, adăugate pe rând și verificate prin backtest:
   - *momentum 12-1*: acțiunile care au urcat cel mai mult în ultimele 12 luni (fără ultima lună)
     tind să continue. Recent însă rezultatele au fost amestecate: un test pe acțiuni mari a dat
     Sharpe negativ și o scădere maximă de −81%;
   - *drift după rezultate (PEAD)*: prețul continuă în direcția surprizei la rezultatele
     trimestriale. Efectul a slăbit, dar nu a dispărut;
   - *cumpărări „oportuniste” ale insiderilor* (Cohen, Malloy, Pomorski, 2012): ~0,82% pe lună
     randament anormal. Studii mai noi nu reproduc complet rezultatul;
   - *short interest* ridicat, ca semnal de prudență.
4. **Semnalele pe care le folosește modelul** (aceleași în program și în pagină):
   - *apropierea de maximul pe 52 de săptămâni* (George și Hwang, 2004): explică mare parte din
     profitul strategiilor de momentum și nu se inversează pe termen lung;
   - *revenirea după ultima lună* (Jegadeesh, 1990): acțiunile care au urcat mult într-o lună tind
     să se corecteze puțin în luna următoare;
   - *momentum care se strică după un an slab al pieței* (Daniel și Moskowitz, 2016): prăbușirile
     momentumului vin când piața revine după scăderi, de aceea modelul are un termen separat pentru acel caz;
   - *volum neobișnuit de mare* (Gervais, Kaniel și Mingelgrin, 2001): acțiunile cu volum mult peste
     normal tind să urce în luna următoare;
   - *volatilitatea și lichiditatea*: Gu, Kelly și Xiu (2020) au comparat metode de învățare automată pe
     ~30.000 de acțiuni și au găsit că momentumul, lichiditatea și volatilitatea sunt cele mai importante
     semnale. Chiar și cele mai bune modele explică doar ~0,3–0,4% din variația randamentelor lunare.
   - *beta față de S&P 500* (Frazzini și Pedersen, 2014, „Betting Against Beta”): acțiunile cu beta mare au
     randamente ajustate la risc mai mici; pentru „bate S&P 500”, beta spune și cât amplifică acțiunea mișcarea pieței.
   - *efectul MAX* (Bali, Cakici și Whitelaw, 2011): acțiunile cu o creștere extremă de o zi în ultima lună
     („loterii”) rămân în urmă; diferența între decile depășește 1% pe lună;
   - *volatilitatea proprie* (Ang, Hodrick, Xing și Zhang, 2006): volatilitate mare după scoaterea părții pieței →
     randamente slabe luna următoare;
   - *informația continuă* („frog in the pan”, Da, Gurun și Warachka, 2014): momentumul format din multe mișcări mici
     continuă mult mai bine decât cel format din câteva salturi;
   - *sezonalitatea pe lună* (Heston și Sadka, 2008): o acțiune tinde să aibă randamente relativ bune sau slabe în
     aceeași lună calendaristică, an de an;
   - *trendul pe mai multe orizonturi* (Han, Zhou și Zhu, 2016): prețul față de mediile pe 20, 100 și 200 de zile;
   - *reviziile analiștilor* (Chan, Jegadeesh și Lakonishok, 1996): estimările de profit revizuite în sus sunt urmate
     de randamente mai bune; vin gratuit din Twelve Data (`eps_trend`), în analiza completă.
5. **Randament relativ în loc de absolut.** Dacă acțiunea urcă în 4 săptămâni depinde mult de piață,
   pe care nimeni nu o prezice bine. Semnalele de mai sus spun mai mult despre care acțiuni se descurcă
   mai bine decât altele. De aceea modelul are și ținta „bate S&P 500”.
6. **Metode, nu doar semnale.**
   - *trade cu trei bariere* (López de Prado): în loc de „crește sau nu”, modelul „Trade” estimează șansa ca ținta
     (+1 abatere tipică pe 4 săptămâni) să fie atinsă înaintea stopului (−1 abatere), cu intrare la închiderea de a doua
     zi. Același model e folosit ca al doilea filtru al deciziei („meta-labeling”): când trade-ul contrazice sigur
     direcția, convingerea (și deci poziția) scade un nivel; când o confirmă, crește;
   - *mărimea poziției după risc* (volatility targeting, Moreira și Muir, 2017): poziția e invers proporțională cu
     volatilitatea acțiunii, ca o mișcare tipică împotriva ta să coste mereu cam același procent din portofoliu;
   - *cât să ții* (Jegadeesh și Titman, 1993, au comparat perioade de deținere de 3 până la 12 luni; efectul de revenire
     pe termen scurt ține o săptămână–o lună): câte un model pe 1, 2, 4, 8 și 12 săptămâni, iar orizontul ales e cel cu
     avantajul cel mai mare raportat la timp;
   - *testul de profit după costuri*: portofoliul lunar cu cele mai bune 20% acțiuni după model, comparat cu S&P 500,
     și varianta long–short, doar pe ani nevăzuți;
   - *combinarea prognozelor* (Rapach, Strauss și Zhou, 2010): prognoza finală e jumătate modelul complet, jumătate
     media modelelor cu câte un singur semnal. Pe date simulate, varianta combinată a dat mai multe BUY-uri corecte cu
     aceeași precizie, și zero alarme false pe zgomot; doar modelele simple au dat 390 de BUY-uri greșite.
7. **Ce am cântărit și nu am adoptat (încă):**
   - *arbori de decizie și rețele neuronale* (Gu, Kelly și Xiu): câștigă pe ~30.000 de acțiuni și 60 de ani; pe ~115
     acțiuni și 12 ani învață zgomotul;
   - *randamente de noapte față de zi* (Lou, Polk și Skouras, 2019): efectul e în componenta de noapte, nu clar în
     randamentul total pe 4 săptămâni;
   - *semnale din opțiuni* (skew, put/call): date plătite;
   - *calendarul rezultatelor Twelve Data*: cere planul Grow; folosim calendarul gratuit Alpha Vantage;
   - *atenția pe Google, tranzacțiile congresmenilor, RSI(2)*: fără conector sau cu dovezi slabe;
   - *ranguri între acțiuni în fiecare lună*: fără câștig dovedibil la dimensiunea noastră.
8. **Disciplina testelor multiple.** Harvey, Liu și Zhu (2016) cer un prag t > 3 pentru semnale noi, iar Hou, Xue și
   Zhang (2020) arată că 65% din 452 de anomalii publicate nu trec nici pragul obișnuit. De aceea semnalele de aici sunt
   alese dinainte din literatură (nu căutate în date), modelul trebuie să ajute sigur statistic pe ani nevăzuți ca să
   poată da BUY/SELL, iar regula de decizie e testată an cu an, fără privit înainte.
9. **Așteptări realiste.** McLean și Pontiff (2016) au studiat 97 de semnale publicate:
   randamentele scad cu 26% în afara perioadei studiate și cu 58% după publicare. Orice semnal
   „descoperit” trebuie verificat pe date noi.

## Cum se leagă o sursă nouă

- **Pagina din claude.ai** poate citi date **doar prin conectori claude.ai**. Orice altă adresă e
  blocată de securitatea paginii. Pentru pagină contează deci doar furnizorii care au conector.
- **Programul Python** poate folosi orice API cu cheie.

Conectori financiari găsiți în directorul claude.ai: Alpha Vantage (conectat), **Twelve Data**
(conectare începută, dar neterminată), Bigdata.com, FactSet, LSEG, Zacks, viaNexus, Clear Street,
Daloopa. În afară de Alpha Vantage și Twelve Data, aceștia sunt de regulă servicii profesionale,
plătite.

## Furnizori de prețuri și date de piață

| Furnizor | Plan gratuit | Plătit, de la | Puncte forte | Unde se poate folosi |
|---|---|---|---|---|
| Alpha Vantage | 25 cereri/zi, 1 pe secundă; istoricul zilnic complet e premium | Pro 49,99 $/lună (75 pe minut) | știri cu sentiment, insideri, opțiuni, macro | pagină (conectat), Python |
| **Twelve Data** | 800 cereri/zi, 8 pe minut; acțiuni SUA, forex, cripto | Grow 29 $/lună | până la 5.000 de puncte pe cerere (~20 de ani zilnic), 100+ indicatori, fundamentale, știri | **pagină (conector)**, Python |
| Massive (fost Polygon.io) | 5 cereri/minut, 2 ani istoric | Starter 29 $ (5 ani, întârziere 15 min); Developer 79 $ (10 ani); Advanced 199 $ (timp real, 20+ ani) | calitate bună, fișiere în bloc | Python |
| Financial Modeling Prep | 250 cereri/zi, date de final de zi | Starter ~22 $/lună la plata anuală (300/min, 5 ani); Premium 59 $ (30 ani); Ultimate 149 $ | fundamentale, estimări, rezultate, insideri | Python |
| Finnhub | 60 cereri/minut; cotații SUA în timp real, știri, fundamentale de bază | Premium ~12–100 $/lună | date alternative, WebSocket | Python |
| Tiingo | 1.000 cereri/zi, 50 pe oră, 500 simboluri pe lună; 30+ ani istoric zilnic | Power 30 $/lună (uz personal) | istoric zilnic lung și curat | Python |
| EODHD | 20 cereri/zi | All-In-One 99,99 $/lună (100.000/zi) | acoperire globală, fundamentale | Python |
| Alpaca | date IEX (~2,5% din volumul SUA), 7+ ani | Algo Trader Plus 99 $/lună (toate bursele, opțiuni) | aceeași platformă permite și tranzacționare de probă | Python |
| Yahoo Finance (yfinance) | gratuit, neoficial | — | ușor de folosit | Python, doar pentru teste: blocări frecvente (`YFRateLimitError` raportat și în martie 2026) |

## Surse gratuite și oficiale pentru semnale suplimentare

| Sursă | Ce oferă | Limite |
|---|---|---|
| **SEC EDGAR** | tranzacții ale insiderilor (Form 4), fundamentale XBRL, toate rapoartele companiilor | gratuit, maximum 10 cereri pe secundă, antet `User-Agent` cu email |
| **FRED** (Fed St. Louis) | 800.000+ serii macro: dobânzi, inflație, curba randamentelor, șomaj | gratuit cu cheie, 120 cereri pe minut |
| **FINRA** | short interest de două ori pe lună, volum zilnic vândut în lipsă; arhivă din 2014 | gratuit pentru uz necomercial |
| Quiver Quantitative | tranzacțiile membrilor Congresului SUA, insideri, lobby | API de la ~10 $/lună; setul de insideri cere planul de 75 $/lună |
| Sharadar (Nasdaq Data Link) | fundamentale „point-in-time”, inclusiv companii delistate, din 1998 | plătit, preț la cerere; pentru backtest serios |

## Grafice

- **TradingView Lightweight Charts**: bibliotecă open source (Apache 2.0, ~35 KB) pentru grafice
  cu lumânări, volum și zoom. Se poate încărca în pagina noastră de pe CDN-urile permise.
  Licența cere menționarea TradingView și un link către tradingview.com.
- Widget-urile TradingView (iframe) nu merg în pagina din claude.ai, fiindcă iframe-urile sunt
  blocate. Merg doar pe un site propriu.

## Surse

- Twelve Data: [prețuri](https://twelvedata.com/pricing), [istoric](https://support.twelvedata.com/en/articles/5214728-getting-historical-data), [server MCP](https://github.com/twelvedata/mcp)
- [Massive (Polygon.io)](https://massive.com/pricing), [rezumat prețuri](https://qveris.ai/guides/polygon-pricing-optimized/)
- [Financial Modeling Prep](https://site.financialmodelingprep.com/pricing-plans)
- [Finnhub](https://finnhub.io/pricing)
- [Tiingo](https://www.tiingo.com/about/pricing)
- [EODHD](https://eodhd.com/pricing)
- [Alpaca](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Alpha Vantage premium](https://www.alphavantage.co/premium/)
- [yfinance: blocări](https://github.com/ranaroussi/yfinance/issues/2480)
- [SEC EDGAR](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [FRED API](https://freeapihub.com/apis/fred-api)
- [FINRA short interest](https://www.finra.org/finra-data/browse-catalog/equity-short-interest)
- [Quiver Quantitative](https://api.quiverquant.com/pricing/)
- [Sharadar](https://data.nasdaq.com/databases/SFA)
- [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)
- [McLean și Pontiff (2016)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)
- [Cohen, Malloy, Pomorski (2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1692517)
- [Recenzie PEAD](https://www.sciencedirect.com/science/article/pii/S2214635020303750)
- [Momentum 12-1 pe acțiuni mari](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5367656)
