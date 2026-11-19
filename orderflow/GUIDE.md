# Order flow, Bloomberg ticks and sequence models

This folder turns tick data into order-flow features and benchmarks eight sequence models on two targets:
where the BTC mid-price goes over the next 30 seconds, and how much trades in that time. The data can come
from a Bloomberg Terminal (`bbg_export.py`) or, free and with true trade sides, from Hyperliquid
(`record_hyperliquid.py`). Both write the same CSV, so everything downstream is shared.

Section 7 has the measured results. Read section 1 first, because it sets the bar those results are judged against.

## 1. What to expect before you train anything

The research on this is unusually consistent, and it is worth knowing before spending a Terminal session.

* Order flow explains price moves in the same interval very well, and predicts the next interval poorly.
  Cont, Kukanov & Stoikov (J. Financial Econometrics 2014) get about 65% R^2 regressing 10-second mid changes on
  order flow imbalance measured over the same 10 seconds, across 50 US stocks. Cont, Cucuringu & Zhang (Quantitative
  Finance 2023) then forecast the next minute's return from past OFI and get a negative out-of-sample R^2 for every
  model they try (about -0.1% to -0.4%).
* Real direction skill exists at very short horizons and fades fast. Kolm, Turiel & Westray (Mathematical Finance 2023,
  115 Nasdaq stocks) find it lasts about two average price changes. TLOB (Berti & Kasneci 2025) on Binance BTC
  perpetual data falls from F1 74.7 to 41.5 between 10 and 100 snapshots (roughly 1 to 10 seconds).
* Architecture matters less than inputs and labels. LOBCAST (Prata et al., AI Review 2024) re-ran 15 deep order-book
  models: the best one on the classic FI-2010 dataset (F1 82.6) dropped to about 61 on newer Nasdaq data, and a plain
  MLP generalised among the best. TLOB's own simple MLP ties its transformer at short horizons. Order-flow inputs beat
  raw book levels (Kolm et al.; Lucchese, Pakkanen & Veraart, IJF 2024).
* The one Mamba-on-order-book study so far (ICAIF 2026, arXiv 2610.28753) trained an 8.2M-parameter Mamba-3 on
  Nasdaq message data. Its internal state encodes OFI well (R^2 0.59), yet it predicts the next mid move no better than
  a linear readout of the visible book (R^2 0.04 vs 0.036; direction 61.2% vs 61.8%).
* Volume is the target where models clearly earn their keep. Cucuringu, Li & Zhang (2025) reach an out-of-sample R^2
  of 0.62 on 15-minute volume for 469 S&P 500 stocks; Satish, Saxena & Palmer (2014) cut volume forecast error by
  24-29% against a 21-day average.
* Nobody in this literature shows profit after costs. Most papers trade at the mid with zero fees. A taker who
  crosses a one-tick spread both ways pays a full tick, and the predictable move is often half a tick.

So the realistic goals are: a volume model that clearly beats simple baselines, and a direction model that beats a
coin flip on a very short horizon but is unlikely to pay for its spread. The benchmark here tests exactly that.

## 2. What a Bloomberg Terminal actually gives you

Verified against Bloomberg's BLPAPI developer guides (2014 and 2016 editions) and university library guides. Items
marked "check" were not confirmed by a public primary source; check them on the Terminal.

| You want | On a Terminal? | How |
|---|---|---|
| Every trade with size | Yes, about 140 business days back | `IntradayTickRequest`, event `TRADE` |
| Best bid and ask with sizes (Level 1) | Yes, same window | events `BID`, `ASK` (they may come back named `BID_BEST` or `BEST_BID`; the loader maps them) |
| Which side was the aggressor | No for equities; some futures (e.g. CME) mark it with condition codes `AB`/`AS` | infer with Lee-Ready (section 3) |
| Historical order book beyond the top level | No documented route | record live depth yourself, or buy LOBSTER, Databento, LSEG Tick History |
| Live depth | Full book (`//blp/mktdepthdata`) is B-PIPE only. A plain `//blp/mktdata` subscription can return 5 aggregated levels (`BEST_BID1..5`, `BEST_BID1_SZ..`) if the exchange Level 2 entitlement is present | check |
| Minute bars | Yes, `IntradayBarRequest`, 1 to 1440 minutes, one event type per request | |

Things that will bite you:

* The Desktop API works only on the Windows PC where the Terminal is logged in (it talks to `bbcomm` on
  `localhost:8194`). Bloomberg's macOS and Linux libraries do not work with a Terminal.
* API times are UTC. Excel `BDH` intraday times are local time. Mixing them silently shifts everything by hours.
* Tick timestamps appear to be whole seconds (MathWorks documents this; no Bloomberg page says otherwise). Pull one
  minute and look before planning anything sub-second.
* Limits: a request costs securities x event types against the Terminal's daily and monthly limits, however many ticks
  come back, so pull whole sessions rather than many small slices. Hitting a limit returns a `LIMIT` error
  (`DAILY_LIMIT_REACHED`, `MONTHLY_LIMIT_REACHED`), or `#N/A Dly Lmt` / `#N/A Mth Lmt` in Excel. Monthly limits belong
  to the Terminal, so on a shared library machine other people may have used them up.
* Licence: Bloomberg's Desktop API terms are shown at `DAPI<GO>`. University guides consistently say API data must stay
  on the Terminal PC and may not be passed on raw or derived. Read the terms before you copy a CSV anywhere.

Screens worth opening (confirmed): `QR` trade and quote recap, `QRM` full tick recap (what `includeNonPlottableEvents`
reproduces in the API), `GIP` intraday chart, `AQR` average quote recap, `BTR` block trade recap (10,000 to
1,000,000 shares, 11 days back), `MMTK` market maker codes, `FLDS` to confirm field names such as `BID_SIZE`,
`ASK_SIZE` and `BEST_BID1`. Commonly quoted but unconfirmed: `QM`, `MBO`, `DEPTH`, `MD`, `VAP`, `IGP`; press HELP on
each. `DAPI` and `WAPI` hold the API documentation and examples.

Bitcoin-related tickers to try: `IBIT US Equity` (the US composite; the Nasdaq listing is probably `IBIT UQ Equity`,
check with `DES`), `FBTC US Equity`, CME Bitcoin futures `BTCA Curncy` and Micro Bitcoin `BMR Curncy` (check the
contract chain with `CT`). `XBTUSD Curncy` looks like an indicative composite with no real trade volume (check).

### Exporting

On the Terminal PC, with Python allowed:

```
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
python bbg_export.py "IBIT US Equity" 2026-10-27T13:30:00 2026-10-27T20:00:00 ibit_20261027.csv
```

`bbg_export.py` needs only `blpapi` and the standard library, so you can copy that one file to the Terminal PC. Its
row conversion is unit-tested; the part that talks to Bloomberg could not be run without a Terminal, so treat the
first export as a test and look at the file.

Without Python, use Excel. Spreadsheet Builder (Bloomberg ribbon, Import Data, Intraday Tick) writes the formula for
you; Bloomberg's own example has this shape:

```
=BDH("IBIT US Equity","TRADE,BID,ASK","10/27/2026 9:30:00 AM","10/27/2026 4:00:00 PM","Dir=V","Dts=S","Sort=A","IntrRw=True","CondCodes=S","ExchCode=S")
```

Save it as CSV with columns time, type, price (or value), size. Times are local, so train with
`--data-tz America/New_York`.

## 3. From ticks to order flow (`features.py`)

**Book state.** Walk the ticks in order and carry the latest bid price/size and ask price/size forward. Every quote
tick changes this state; trades do not.

**Order flow imbalance (OFI).** Cont, Kukanov & Stoikov's per-event contribution, with b/a the best bid/ask price,
q its size, n the state after the event and n-1 before:

```
e_n = [b_n >= b_{n-1}] q^b_n - [b_n <= b_{n-1}] q^b_{n-1} - [a_n <= a_{n-1}] q^a_n + [a_n >= a_{n-1}] q^a_{n-1}
```

In words: a bigger bid queue or a higher bid is buying pressure; a bigger ask queue or a lower ask is selling
pressure; a queue that empties counts with the opposite sign. Summed over a bar and divided by average depth, this is
the `ofi` feature. `test_ofi_matches_cont_kukanov_stoikov` checks it against hand-worked cases.

**Trade signs (Lee-Ready).** Bloomberg does not say who initiated a trade. Lee & Ready (1991): above the prevailing
mid is a buy, below is a sell, exactly at the mid use the tick rule (direction of the last price change). Published
accuracy on Nasdaq with known signs is about 81%. One lesson from this project: when a trade and the quote update it
caused carry the same timestamp, the quote must not be used to sign that trade. Hyperliquid stamps both with the same
block time, and with naive ordering 21% of trades were signed wrong. Sorting trades first on ties raised accuracy
against the true aggressor from 78.7% to 99.7%. Bloomberg's one-second stamps have the same problem, and the same fix
(sign against the quote from before that second) is applied.

**Bars.** Ticks are grouped into fixed clock bars (2 seconds in the benchmark). A long silence (overnight, a dropped
connection) starts a new session; nothing rolls, shifts or windows across sessions.

**Features per bar** (all from the past only): normalised OFI; signed trade volume over average volume; mid return;
spread in bps; queue imbalance (bid size - ask size)/(bid size + ask size), which is also what drives Stoikov's
micro-price; log volume, log trade count, log quote count; rolling volatility; time of day (sin/cos). Time of day is
dropped automatically when training covers less than 3 days: on a single session it acts as a disguised clock index
and its test values fall outside anything seen in training.

**Labels.** `y_dir` = 1 if the mid rises by more than half the current spread over the next H bars, 0 if it falls by
more, left out otherwise (too small to matter). `y_vol` = log(1 + volume traded over the next H bars). The baseline
for volume is "the last H bars repeat".

## 4. The models (`btcbot/model/archs.py`)

Every model takes a window of the last 60 bars (each bar a vector of 11 features) and returns one number: a logit for
direction or a value for volume. They share the same input projection, pre-norm residual blocks and head, and all read
out at the newest bar, so the comparison isolates the sequence-mixing layer.

**Transformer basics.** Project each bar to a d-dimensional vector (d = 32 here) and add a learned position vector.
A block is `x = x + Mixer(LayerNorm(x))` then `x = x + MLP(LayerNorm(x))`. The mixer in a transformer is self-attention:
each bar emits a query q, key k and value v; bar i's output is a weighted average of all values with weights
`softmax(q_i . k_j / sqrt(d_head))`, so each bar decides which other bars matter. Several heads do this in parallel.
Cost grows with the square of the window length L, which is irrelevant at L = 60 and matters at L = 100,000.

* **softmax**: exactly that, read out at the newest bar. The attention control everything else is measured against.
* **transformer**: the same attention, but read out from an extra learned "CLS" token, as in the BTC bot. Kept to
  show that readout choice alone can matter more than the mixer (section 7).
* **linear** (Katharopoulos et al. 2020, "Transformers are RNNs"): replace `softmax(QK^T)V` by
  `phi(Q)(phi(K)^T V) / (phi(Q) sum_j phi(K_j))` with `phi(x) = elu(x) + 1`. Computing `phi(K)^T V` first means the
  L x L weight matrix is never built: cost is linear in L. The price is blurrier attention, worse at picking out one
  exact earlier bar. Modern descendants (gated linear attention, Gated DeltaNet, Kimi Delta Attention) add gates and
  a delta rule to fix that; Qwen3-Next mixes Gated DeltaNet with ordinary attention 3:1 and adds MoE.
* **moe** (Switch Transformer, Fedus et al.; Mixtral): the MLP becomes 4 smaller expert MLPs plus a router that sends
  each bar to its top 2 experts, weighted by a softmax over the 2 chosen router scores. A balance loss
  `0.01 * E * sum_i f_i P_i` (f = share of bars sent to expert i, P = mean router probability) stops the router from
  collapsing onto one expert. MoE raises capacity without raising compute per token; it does not change how bars talk
  to each other.
* **mamba** (Gu & Dao 2023): a selective state-space model. A hidden state h (16 numbers per channel) is updated bar by
  bar, `h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t`, output `y_t = C_t . h_t + D x_t`, gated by `silu(z)`. A is a fixed
  negative decay per channel; dt, B and C are computed from the current input, so the model decides per bar how much to
  remember and what to write ("selective"). Cost is linear in L and inference is a constant-size recurrence. The trade
  off is a compressed memory: exact recall of one specific earlier item is harder than for attention. The official
  fast kernels need CUDA; this is a pure PyTorch port; its output matches a port of the official reference scan
  (`selective_scan_ref`) to within 4e-9.
* **rwkv** (Peng et al. 2023, RWKV-4): attention rewritten as a decaying running average.
  `wkv_t = (sum_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t) / (same with v -> 1)`, gated by `sigmoid(r_t)`, with a
  one-step "token shift" that mixes each bar with the previous one. Each channel has its own decay w, and u is a bonus
  for the current bar. Trains like a transformer, runs like an RNN. RWKV-6/7 make the decay data-dependent and add a
  delta rule.
* **hybrid** (Jamba, Lieber et al. 2024): Mamba layers interleaved with an attention layer, and MoE on alternate
  layers. Jamba uses 1 attention layer per 7 Mamba layers; this miniature is Mamba+MLP, attention+MLP, Mamba+MoE.
  The idea is that Mamba carries cheap memory and position while the occasional attention layer handles exact recall.
  Turing Post's view (Mamba explainer, updated Sep 2026) is that this is where Mamba ended up: as layers inside
  hybrids such as Jamba, Falcon-H1 and Nemotron-H, not as a replacement.
* **lstm**: the pre-transformer recurrent baseline, which is also what DeepLOB uses.

Correctness checks in `tests/test_archs.py`: causality of the recurrent mixers (changing the future never changes the
past), linear attention against the explicit L x L formula, the RWKV recurrence against the closed-form sum in double
precision, MoE gates and balance loss against hand values, Mamba's initialisation, and a learnability test where every
architecture must learn a rule that needs memory across the window, through the real training loop.

## 5. How the benchmark avoids fooling itself (`train.py`)

* Walk-forward: the data is cut into 4 consecutive time blocks. Block k is predicted only by models trained on blocks
  before it, with an H-bar gap so no training label can see into the test block. The last 15% of each training span
  (after another gap) is used for early stopping and calibration.
* Every architecture gets the same folds, features, label rows and 3 seeds. The table reports the seed-averaged model
  and the spread across seeds, so a lucky seed cannot pass for a better architecture.
* Baselines: logistic regression (direction) and ridge regression (volume) on the newest bar's features, and "volume
  persists". Differences against the baseline and against `softmax` come with a 90% moving-block bootstrap interval
  (blocks of 10 x H bars keep the autocorrelation). An interval that straddles 0 means no demonstrated difference.
* Cost test: trade every bar whose confidence clears a threshold, pay a full spread in and out plus 4.5 bps taker fee
  per side (Hyperliquid's base rate). Fills are at the quote at the bar close with no latency, which is optimistic.

## 6. Running it

```
# practice data (free, true trade sides): leave running as long as you like
.venv/bin/python -m orderflow.record_hyperliquid BTC data/ticks/hl_BTC.csv

# Bloomberg: run bbg_export.py on the Terminal PC (section 2), one file per day for one security

# benchmark
.venv/bin/python -m orderflow.train data/ticks/hl_BTC.csv --bar 2s --archs all --seeds 3
.venv/bin/python -m orderflow.train data/bbg/ibit_*.csv --bar 1s --fee-bps 0 --archs softmax,mamba,hybrid
.venv/bin/python -m pytest tests -q
```

Files of one run must not overlap in time (one security at a time); the script stops with a message if they do.
Outputs go to `reports/orderflow/<run>/`: `arch_comparison.csv`, `oos_predictions.csv`, `run.json`.
Each finished model is appended to `partial.csv` in the same folder, so rerunning an interrupted command picks up where
it stopped (and adding an arch or a seed to `--archs`/`--seeds` trains only the new ones). Change the data or any
setting other than `--archs`, `--seeds`, `--device` or `--out` and it starts fresh.

The same architectures are available to the BTC bot: set `model.arch` in `config.yaml` (any name in `ARCHS`). The bot's
saved models keep loading as before.

## 7. Results

RESULTS_PLACEHOLDER

## 8. Where to go next

* Volume is where the signal is. Longer histories (several days or weeks) would let time of day back in and should
  lift every model's volume R^2; the intraday U-shape is the strongest known volume predictor.
* For direction, multi-level OFI is the documented upgrade over best-level OFI (Xu, Gould & Howison; Cont, Cucuringu &
  Zhang). Hyperliquid's `l2Book` stream gives 20 levels (throttled to about 2 snapshots per second); a Terminal only
  gives depth live and only if entitled.
* Pool several instruments into one model (Sirignano & Cont found a "universal" model beats per-stock models), for
  example BTC on Hyperliquid, CME BTCA and IBIT at the same time, after making their timestamps comparable.
* Bloomberg keeps only about 140 business days of ticks, so start recording whatever you plan to study now.
