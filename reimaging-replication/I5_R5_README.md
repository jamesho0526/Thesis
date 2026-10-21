# I5/R5 data acquisition for Table IV

## What Table IV requires

Table IV in Jiang, Kelly, and Xiu, *"(Re-)Imag(in)ing Price Trends"* restricts
each rebalance date to the 500 largest U.S. stocks by market capitalization.
The `I5/R5` specification means:

- `I5`: an image made from the preceding 5 trading days of daily OHLC and
  volume information;
- `R5`: the sign of the subsequent 5-trading-day return;
- sample: 1993-01-01 through 2019-12-31;
- training/validation: 1993-2000, with a random 70/30 split;
- out-of-sample test: 2001-2019.

The paper uses CRSP daily data for NYSE, AMEX, and NASDAQ common stocks
(`SHRCD` 10/11 and `EXCHCD` 1/2/3). The download also includes shares
outstanding and adjustment factors needed for the top-500 screen and
corporate-action-adjusted price construction.

## Download

The original CRSP data and the authors' image/label files are not publicly
redistributable. A WRDS/CRSP subscription is therefore required. From this
directory:

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install wrds pandas
$env:WRDS_USERNAME = "your_wrds_username"
$env:WRDS_PASSWORD = "your_wrds_password"
uv run python .\download_crsp_i5_r5.py
```

The default output is:

```text
data\crsp_i5_r5\crsp_daily_raw.csv.gz
```

The script downloads a warm-up period beginning 1992-12-01 so the first
sample-date image can be formed, while the actual replication sample remains
1993-01-01 to 2019-12-31. Use `--dry-run` to inspect the SQL without logging
in.

## Hardware-constrained plan

The Radeon GPU is not a drop-in replacement for the paper's CUDA workflow.
For a Ryzen 7 7700 / 32 GB RAM machine, run one I5/R5 model only, use CPU
training, batch size 32--64, and start with one training run rather than the
paper's five independent CNN fits. Keep the paper's 1993--2000 versus
2001--2019 chronological split; reducing the number of random restarts changes
the replication target but is a reasonable compute concession.

The public repository contains the preprocessing/training scaffold, but not
the licensed CRSP image dataset. Do not substitute survivorship-biased current
tickers if the goal is to match Table IV.

## Yahoo Finance fallback with data-quality controls

Yahoo Finance can be used as a non-CRSP robustness/fallback dataset, but it
cannot reproduce the paper's CRSP universe or its delisting treatment. The
following command downloads a ticker and writes three files:

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install -r .\requirements-yahoo.txt
uv run python .\scripts\download_yahoo_i5_r5.py AAPL MSFT
```

For each ticker:

- `*_raw.csv.gz` preserves the Yahoo response;
- `*_clean.csv.gz` uses `Adj Close / Close` to split-adjust OHLC consistently;
- `*_rejected.csv.gz` records rows removed and the rejection reason.

Rows with missing/non-positive prices, invalid OHLC ordering, invalid volume,
or robustly extreme adjusted returns are rejected. A declared Yahoo stock
split is not rejected; it is normalized through the adjustment factor. The
default date range includes a warm-up period from 1992-12-01 and ends
exclusive at 2020-01-01, covering the requested 1993-01-01--2019-12-31
sample. Do not silently replace rejected rows with interpolation when building
I5 windows; drop any window containing a rejected date and record that decision.

## Filtering the 2000-12-29 S&P 500 list

The files in `data\sp500_1day` are company-name CSVs rather than ticker-named
files. Use the provided historical ticker list with the resolver below:

```powershell
.\.venv\Scripts\activate
uv run python .\scripts\filter_sp500_2000.py `
  --symbols ..\data\paper_replication\sp500\sp500_20001229.txt
```

The command keeps rows from 1993-01-01 through 2019-12-31 and writes a
`sp500_2000_filter_report.csv` containing every ticker, resolved Yahoo company
name, selected file, match score, and candidate ambiguities. Only matches above
the confidence threshold are copied. Review rows marked `unmatched`,
`ambiguous`, or `below_min_score` before training; do not silently count them
as valid constituents.

## Folder organization

Paper-replication inputs and outputs are kept under:

```text
data\paper_replication\
experiments\paper_replication\
```

The fixed 2000 S&P sample is under
`data\paper_replication\sp500\filtered_1day`. Yahoo data acquisition and the
futures transfer experiment are kept separately under:

```text
data\extensions\
experiments\extensions\
```

Important: a fixed S&P 500 membership list from 2000 is not identical to Table
IV's "500 largest stocks" screen. Table IV ranks the available stocks by market
capitalization at each monthly rebalance date. A fixed 2000 list is therefore a
proxy/sample restriction, not an exact replication of the table. Exact
replication also requires historical shares outstanding or market
capitalization and delisting-aware security identifiers.

## Current two-workflow implementation

The current practical workflow is deliberately split into two independent
experiments:

1. **Paper replication approximation**: build I5/R5 images from
   `data\paper_replication\sp500\filtered_1day`, including every valid file with
   observations in 1993--2019. The generated images are written to
   `data\paper_replication\sp500\images_i5_r5`.
2. **Futures extension**: build the same I5/R5 representation from
   `data\extensions\futures\futures_full_1day`, writing to
   `data\extensions\futures\images_i5_r5`.

Run either builder with:

```powershell
uv run python .\scripts\build_i5_r5.py --source sp500 `
  --source-dir ..\data\paper_replication\sp500\filtered_1day `
  --output-dir ..\data\paper_replication\sp500\images_i5_r5
```

Replace `sp500` and the paths with `futures` paths for the extension. Train with `scripts\train_i5_r5.py`; the corrected I5 pipeline uses a 32x15
input, reconstructs prices from daily returns before local scaling, draws the
MA5 and volume layout, computes training-set pixel mean/std for z-score
normalization, and trains five independent seeds by default. The final test
probabilities are the arithmetic mean of the five Softmax outputs and are
written to `ensemble_predictions.csv`.

### Applying the S&P model to futures

The S&P ensemble can be used as a frozen cross-market predictor. This does
not retrain on futures and reuses the S&P training pixel mean/std:

```powershell
uv run python .\scripts\predict_futures_with_sp500.py `
  --data-root ..\data\extensions\futures\images_i5_r5 `
  --checkpoint-dir .\experiments\paper_replication\sp500_i5_r5 `
  --output-dir .\experiments\extensions\futures_with_sp500_model
```

The script writes `futures_predictions_from_sp500.csv` and `metrics.json`.
Because futures are a time series rather than a stock cross-section, the
reported metrics include directional accuracy, signal return, approximate
annualized Sharpe, and approximate maximum drawdown.
S&P data uses the paper's 1993--2000 versus 2001--2019 split. Futures data
does not cover those years, so the trainer reports and uses a chronological
70/30 year split instead.
