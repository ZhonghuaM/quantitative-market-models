# Bundled data record

The original provider and download history of these files are unknown. This inventory documents what can be verified from the bytes; filenames and column labels are retained as identifiers, not evidence of an instrument, ticker, exchange, or source.

| File | Role | Rows | Recorded date range | Columns |
|---|---|---:|---|---|
| `vas_equity_etf.csv` | Primary signal study | 1,266 | 2018-05-21–2023-05-19 | Date, Open, High, Low, Close, Adj Close, Volume |
| `sp500_index.csv` | Supplementary risk example | 1,250 | 2013-01-22–2018-01-05 | Date, SP500 |

The dates parse as day/month/year and are ascending, with no duplicate dates, duplicate rows, or missing cells in either file. This does not establish complete coverage of any exchange calendar. The OHLCV file's `Adj Close` and `Close` columns are identical on every row; the adjustment and dividend policy remain unknown. The signal pipeline uses `Close`.

The [machine-readable manifest](manifest.json) records exact SHA-256 hashes, byte sizes, columns, dates, missing-value counts, duplicate counts, and date-gap counts. Its inventory date is the date of this record, not the original download date. In its `unverified` fields, `null` means unknown.

## Unresolved provenance

The provider, original URL, retrieval/as-of dates, instrument identity, ticker, exchange, currency, calendar, timezone, price adjustments, dividend and corporate-action treatment, missing-observation policy, original transformations, data licence, and redistribution permission have not been verified. The software's MIT licence does not establish rights in these data.

Consequently, these files support an inspectable computational example, not a source-verified financial result or a confirmed total-return benchmark. The next substantive data improvement is to run a separately documented experiment on data with a traceable source and permitted use. Do not silently assign a vendor based on the CSV format or overwrite these files with a different history under the same record.

## Optional new downloads

The independent downloader in `scripts/download_data.py` supports market CSVs, FRED series, and SEC company-facts JSON. It is not evidence that any of those providers supplied the bundled files. For example:

```bash
python scripts/download_data.py market --symbol spy.us --output data/downloaded/spy_stooq.csv
python scripts/download_data.py fred --series DGS10 --output data/downloaded/fred_dgs10.csv
```

Downloads are saved separately and are not automatically substituted into the primary study. For any new experiment, record the request, provider, retrieval time, usage terms, identifiers, adjustment policy, calendar, missing-session handling, and file hash before interpreting the results. The downloader's `sec-facts` command also accepts `--user-agent` for a real declared contact.
