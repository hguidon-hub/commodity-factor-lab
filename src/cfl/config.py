"""Universe and global settings for Commodity Factor Lab."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"


@dataclass(frozen=True)
class Commodity:
    name: str
    yf: str            # Yahoo Finance continuous front-month ticker
    bbg: str           # Bloomberg generic root (for optional terminal export, e.g. "CL1 Comdty")
    sector: str
    cot_code: str      # CFTC contract market code (legacy report). Empty = no COT data.


# Futures universe: Yahoo front-month continuous contracts (not roll-adjusted) or a Bloomberg export.
# Kept for the Bloomberg/Refinitiv extension; the default study uses ETFS below.
FUTURES_UNIVERSE: list[Commodity] = [
    # Energy
    Commodity("WTI Crude", "CL=F", "CL", "Energy", "067651"),
    Commodity("Brent Crude", "BZ=F", "CO", "Energy", ""),          # ICE: not in CFTC legacy report
    Commodity("Natural Gas", "NG=F", "NG", "Energy", "023651"),
    Commodity("Heating Oil", "HO=F", "HO", "Energy", "022651"),
    Commodity("RBOB Gasoline", "RB=F", "XB", "Energy", "111659"),
    # Metals
    Commodity("Gold", "GC=F", "GC", "Metals", "088691"),
    Commodity("Silver", "SI=F", "SI", "Metals", "084691"),
    Commodity("Copper", "HG=F", "HG", "Metals", "085692"),
    Commodity("Platinum", "PL=F", "PL", "Metals", "076651"),
    Commodity("Palladium", "PA=F", "PA", "Metals", "075651"),
    # Grains & oilseeds
    Commodity("Corn", "ZC=F", "C ", "Grains", "002602"),
    Commodity("Wheat SRW", "ZW=F", "W ", "Grains", "001602"),
    Commodity("Wheat HRW", "KE=F", "KW", "Grains", "001612"),
    Commodity("Soybeans", "ZS=F", "S ", "Grains", "005602"),
    Commodity("Soybean Oil", "ZL=F", "BO", "Grains", "007601"),
    Commodity("Soybean Meal", "ZM=F", "SM", "Grains", "026603"),
    Commodity("Oats", "ZO=F", "O ", "Grains", "004603"),
    Commodity("Rough Rice", "ZR=F", "RR", "Grains", "039601"),
    # Softs
    Commodity("Cotton", "CT=F", "CT", "Softs", "033661"),
    Commodity("Sugar No.11", "SB=F", "SB", "Softs", "080732"),
    Commodity("Coffee", "KC=F", "KC", "Softs", "083731"),
    Commodity("Cocoa", "CC=F", "CC", "Softs", "073732"),
    Commodity("Orange Juice", "OJ=F", "JO", "Softs", "040701"),
    # Livestock
    Commodity("Live Cattle", "LE=F", "LC", "Livestock", "057642"),
    Commodity("Feeder Cattle", "GF=F", "FC", "Livestock", "061641"),
    Commodity("Lean Hogs", "HE=F", "LH", "Livestock", "054642"),
]

# Default universe: single-commodity ETFs. Their prices embed the roll of the underlying futures
# (or physical holding for precious metals), so returns are investable and faithful, at the cost of
# a smaller cross-section and shorter history (most launched 2006-2011).
ETF_UNIVERSE: list[Commodity] = [
    Commodity("WTI Crude (USO)", "USO", "", "Energy", "067651"),
    Commodity("Brent Crude (BNO)", "BNO", "", "Energy", ""),          # ICE: no CFTC legacy data
    Commodity("Natural Gas (UNG)", "UNG", "", "Energy", "023651"),
    Commodity("Gasoline (UGA)", "UGA", "", "Energy", "111659"),
    Commodity("Gold (GLD)", "GLD", "", "Metals", "088691"),
    Commodity("Silver (SLV)", "SLV", "", "Metals", "084691"),
    Commodity("Copper (CPER)", "CPER", "", "Metals", "085692"),
    Commodity("Platinum (PPLT)", "PPLT", "", "Metals", "076651"),
    Commodity("Palladium (PALL)", "PALL", "", "Metals", "075651"),
    Commodity("Corn (CORN)", "CORN", "", "Grains", "002602"),
    Commodity("Wheat (WEAT)", "WEAT", "", "Grains", "001602"),
    Commodity("Soybeans (SOYB)", "SOYB", "", "Grains", "005602"),
    Commodity("Sugar (CANE)", "CANE", "", "Softs", "080732"),
]

UNIVERSE = ETF_UNIVERSE
RETURNS_ROLL_ADJUSTED = True   # ETF prices: no artificial roll jumps to filter

BY_TICKER = {c.yf: c for c in UNIVERSE}
TICKERS = [c.yf for c in UNIVERSE]
SECTOR = {c.yf: c.sector for c in UNIVERSE}

START_DATE = "2006-01-01"     # COT legacy yearly files are clean from 2006 on
MIN_TRAIN_MONTHS = 48          # first out-of-sample month needs 4y of history
TC_BPS = 5                     # one-way transaction cost per unit of turnover (bps)
