"""Data layer: prices (Yahoo / Bloomberg export / synthetic) and CFTC Commitments of Traders.

All loaders return tidy pandas objects:
  prices : DataFrame, index = daily dates, columns = tickers, values = settlement price (front month)
  prices2: optional DataFrame, same shape, second-nearby contract (enables the carry factor)
  cot    : DataFrame, long format [date, ticker, comm_long, comm_short, spec_long, spec_short, oi]
"""
from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, START_DATE, UNIVERSE, TICKERS

COT_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"


# --------------------------------------------------------------------------- prices
def download_prices(start: str = START_DATE, tickers: list[str] | None = None) -> pd.DataFrame:
    """Daily front-month settlement prices from Yahoo Finance (continuous '=F' tickers).

    Caveat: Yahoo continuous series are NOT roll-adjusted -> roll days show artificial jumps.
    We neutralise them in `clean_roll_jumps`. Use a Bloomberg export for a proper study.
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    tickers = tickers or TICKERS
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False, group_by="column")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = px.reindex(columns=tickers).sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def load_bloomberg_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load a Bloomberg Terminal export (BDH of PX_SETTLE).

    Expected: first column = date, other columns named like 'CL1', 'CL2', 'GC1', 'GC2'...
    Generic 1st contract -> front price, 2nd -> second nearby (unlocks the carry factor).
    Columns are mapped back to Yahoo tickers via config.UNIVERSE[*].bbg.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.columns = [c.replace(" Comdty", "").strip().upper() for c in df.columns]
    f1, f2 = {}, {}
    for c in UNIVERSE:
        root = c.bbg.strip().upper()
        for col in df.columns:
            if col == f"{root}1":
                f1[c.yf] = df[col]
            elif col == f"{root}2":
                f2[c.yf] = df[col]
    p1 = pd.DataFrame(f1).reindex(columns=TICKERS)
    p2 = pd.DataFrame(f2).reindex(columns=TICKERS) if f2 else None
    return p1, p2


def clean_roll_jumps(px: pd.DataFrame, z: float = 6.0) -> pd.DataFrame:
    """Return daily returns with roll-induced outliers set to 0.

    A daily move larger than `z` rolling MADs is treated as a contract roll, not a P&L event.
    Crude but transparent; documented as a limitation in the report.
    """
    r = px.pct_change(fill_method=None)
    med = r.rolling(250, min_periods=60).median()
    mad = (r - med).abs().rolling(250, min_periods=60).median() * 1.4826
    jump = (r - med).abs() > z * mad
    return r.mask(jump, 0.0)


# --------------------------------------------------------------------------- COT
def _find(cols: list[str], *keys: str) -> str:
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    raise KeyError(keys)


def download_cot(start_year: int | None = None, end_year: int | None = None,
                 cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Legacy 'Futures Only' COT report, one zip per year from cftc.gov (public, no key)."""
    import requests

    start_year = start_year or int(START_DATE[:4])
    end_year = end_year or date.today().year
    code_to_ticker = {c.cot_code: c.yf for c in UNIVERSE if c.cot_code}
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in range(start_year, end_year + 1):
        f = cache_dir / f"deacot{y}.zip"
        if not f.exists() or y >= end_year:   # current year is always refreshed
            resp = requests.get(COT_URL.format(year=y), timeout=60,
                                headers={"User-Agent": "Mozilla/5.0 (research project)"})
            resp.raise_for_status()
            f.write_bytes(resp.content)
        with zipfile.ZipFile(f) as z:
            with z.open(z.namelist()[0]) as fh:
                frames.append(pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"), low_memory=False))
    raw = pd.concat(frames, ignore_index=True)
    cols = list(raw.columns)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[_find(cols, "as of date", "yyyy-mm-dd")]),
        "code": raw[_find(cols, "cftc contract market code")].astype(str).str.strip().str.zfill(6),
        "spec_long": raw[_find(cols, "noncommercial", "long (all)")],
        "spec_short": raw[_find(cols, "noncommercial", "short (all)")],
        # startswith: avoids matching the "Noncommercial ..." columns
        "comm_long": raw[[c for c in cols if c.lower().startswith("commercial positions-long")][0]],
        "comm_short": raw[[c for c in cols if c.lower().startswith("commercial positions-short")][0]],
        "oi": raw[_find(cols, "open interest (all)")],
    })
    out["ticker"] = out["code"].map(code_to_ticker)
    out = out.dropna(subset=["ticker"]).drop(columns="code")
    return out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- synthetic
def make_synthetic(n_years: int = 18, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic market with a KNOWN, weak predictive structure (for tests / offline demo).

    - monthly expected return loads on 12m momentum and on commercial hedging pressure
    - a sector factor + idiosyncratic noise; signal-to-noise is deliberately low (~realistic).
    Returns (prices, prices2, cot).
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2006-01-02", periods=n_years * 252)
    n = len(TICKERS)
    sectors = sorted({c.sector for c in UNIVERSE})
    sec_idx = np.array([sectors.index(c.sector) for c in UNIVERSE])
    vol = rng.uniform(0.15, 0.40, n) / np.sqrt(252)

    # slow-moving latent signals (AR(1) at daily freq)
    hp = np.zeros((len(days), n))       # hedging pressure (commercials net long share)
    carry = np.zeros((len(days), n))
    for t in range(1, len(days)):
        hp[t] = 0.995 * hp[t - 1] + 0.1 * rng.standard_normal(n)
        carry[t] = 0.997 * carry[t - 1] + 0.05 * rng.standard_normal(n)
    sec_shock = rng.standard_normal((len(days), len(sectors)))[:, sec_idx] * 0.6
    eps = rng.standard_normal((len(days), n))
    mu = -0.00025 * np.tanh(hp) + 0.0002 * np.tanh(carry)           # premia (daily): low hedger long share -> premium
    r = mu + vol * (0.6 * sec_shock + 0.8 * eps)
    # momentum: add persistence of past 60d returns
    r_df = pd.DataFrame(r, index=days, columns=TICKERS)
    r_df = r_df + 0.02 * r_df.rolling(60, min_periods=1).mean().shift(1).fillna(0)
    px = 100 * (1 + r_df).cumprod()
    px2 = px * np.exp(-0.25 * np.tanh(pd.DataFrame(carry, index=days, columns=TICKERS)) / 12)

    # weekly COT consistent with hp
    tue = days[days.weekday == 1]
    rows = []
    oi = rng.uniform(1e5, 8e5, n)
    for d in tue:
        i = days.get_loc(d)
        share = 0.5 + 0.2 * np.tanh(hp[i])        # commercials' long share
        cl = oi * 0.5 * share
        cs = oi * 0.5 * (1 - share)
        sl = oi * 0.3 * (1 - share) + rng.uniform(0, 1e4, n)
        ss = oi * 0.3 * share + rng.uniform(0, 1e4, n)
        for j, tk in enumerate(TICKERS):
            rows.append((d, tk, cl[j], cs[j], sl[j], ss[j], oi[j]))
    cot = pd.DataFrame(rows, columns=["date", "ticker", "comm_long", "comm_short",
                                      "spec_long", "spec_short", "oi"])
    return px, px2, cot


# --------------------------------------------------------------------------- io helpers
def save(df: pd.DataFrame, name: str, folder: Path = RAW_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.csv.gz"
    df.to_csv(p)
    return p


def load(name: str, folder: Path = RAW_DIR, parse_dates: bool = True) -> pd.DataFrame:
    return pd.read_csv(folder / f"{name}.csv.gz", index_col=0, parse_dates=parse_dates)
