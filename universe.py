"""Defines the tradable universe: ticker lists grouped by category and asset
class, plus the derived lookups TICKERS, CATEGORY_OF and ASSET_CLASS_OF used
throughout the pipeline.

main.py takes its ticker list from TICKERS here, and asset_sizing.py uses
CATEGORY_OF and ASSET_CLASS_OF for its diversification caps. Every lookup is
derived from _CATEGORIES, so a ticker cannot exist without an assigned
category.
"""

# =========================
# EQUITIES
# =========================

TECH = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
]

GROWTH = [
    "TSLA",
    "PLTR",
    "NFLX",
    "SHOP",
    "CRWD",
]

SEMIS = [
    "NVDA",
    "AMD",
    "AVGO",
    "TSM",
    "ASML",
]

ENERGY = [
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
]

FINANCIALS = [
    "JPM",
    "GS",
    "MS",
    "BAC",
    "BLK",
]

HEALTHCARE = [
    "LLY",
    "UNH",
    "ABBV",
    "MRK",
    "JNJ",
]

INDUSTRIALS = [
    "CAT",
    "GE",
    "RTX",
    "HON",
    "DE",
]

CONSUMER = [
    "COST",
    "WMT",
    "PG",
    "KO",
    "MCD",
]

REITS = [
    "PLD",
    "AMT",
    "EQIX",
    "O",
    "SPG",
]

# =========================
# BONDS
# =========================

TREASURIES = [
    "TLT",
    "IEF",
    "VGIT",
    "GOVT",
    "SHY",
]

CORP_BONDS = [
    "LQD",
    "VCIT",
    "HYG",
    "JNK",
    "AGG",
]

# =========================
# COMMODITIES
# =========================

PRECIOUS_METALS = [
    "GLD",
    "IAU",
    "SLV",
    "SIVR",
    "GDX",
]

ENERGY_COMMODITIES = [
    "USO",
    "UNG",
    "BNO",
    "DBO",
    "XLE",
]

BROAD_COMMODITIES = [
    "DBC",
    "COMT",
    "PDBC",
    "GSG",
    "BCI",
]

# =========================
# FOREIGN EXCHANGE
# =========================

FX = [
    "FXE",  # Euro
    "FXY",  # Yen
    "FXB",  # Pound
    "FXA",  # Australian Dollar
    "UUP",  # US Dollar Index
]

# =========================
# INTERNATIONAL EQUITY
# =========================

EUROPE = [
    "EWG",
    "EWU",
    "EWQ",
    "EWI",
    "VGK",
]

JAPAN = [
    "EWJ",
    "DXJ",
    "HEWJ",
    "JPXN",
    "BBJP",
]

CHINA = [
    "FXI",
    "MCHI",
    "KWEB",
    "ASHR",
    "CQQQ",
]

EMERGING_MARKETS = [
    "EEM",
    "VWO",
    "IEMG",
    "EMXC",
    "FM",
]

# =========================
# CRYPTO PROXIES
# =========================

CRYPTO = [
    "IBIT",
    "FBTC",
    "ETHA",
    "MSTR",
    "WGMI",
]

# =========================
# CATEGORY REGISTRY
# =========================
# CATEGORY_OF, ASSET_CLASS_OF and TICKERS are all derived from this list, so
# the universe only needs editing in one place.

_CATEGORIES = [
    # (category name,        ticker list,          asset class)
    ("TECH",                TECH,                "equities"),
    ("GROWTH",              GROWTH,              "equities"),
    ("SEMIS",               SEMIS,               "equities"),
    ("ENERGY",              ENERGY,              "equities"),
    ("FINANCIALS",          FINANCIALS,          "equities"),
    ("HEALTHCARE",          HEALTHCARE,          "equities"),
    ("INDUSTRIALS",         INDUSTRIALS,         "equities"),
    ("CONSUMER",            CONSUMER,            "equities"),
    ("REITS",               REITS,               "equities"),
    ("TREASURIES",          TREASURIES,          "bonds"),
    ("CORP_BONDS",          CORP_BONDS,          "bonds"),
    ("PRECIOUS_METALS",     PRECIOUS_METALS,     "commodities"),
    ("ENERGY_COMMODITIES",  ENERGY_COMMODITIES,  "commodities"),
    ("BROAD_COMMODITIES",   BROAD_COMMODITIES,   "commodities"),
    ("FX",                  FX,                  "fx"),
    ("EUROPE",              EUROPE,              "intl_equities"),
    ("JAPAN",               JAPAN,               "intl_equities"),
    ("CHINA",               CHINA,               "intl_equities"),
    ("EMERGING_MARKETS",    EMERGING_MARKETS,    "intl_equities"),
    ("CRYPTO",              CRYPTO,              "crypto"),
]

# Derived lookups

TICKERS = [t for _, group, _ in _CATEGORIES for t in group]

# fail fast on any ticker listed in more than one category, rather than
# silently keeping whichever category was assigned last
_seen = set()
_dupes = set()
for _t in TICKERS:
    (_dupes if _t in _seen else _seen).add(_t)
if _dupes:
    raise ValueError(f"universe.py: ticker(s) listed in more than one category: {sorted(_dupes)}")

CATEGORY_OF = {t: cat for cat, group, _ in _CATEGORIES for t in group}
ASSET_CLASS_OF = {t: asset_class for _, group, asset_class in _CATEGORIES for t in group}

CATEGORIES = {cat: group for cat, group, _ in _CATEGORIES}
ASSET_CLASSES = sorted({ac for _, _, ac in _CATEGORIES})


if __name__ == '__main__':
    print(f'{len(TICKERS)} tickers across {len(_CATEGORIES)} categories / {len(ASSET_CLASSES)} asset classes\n')
    for ac in ASSET_CLASSES:
        members = [t for t in TICKERS if ASSET_CLASS_OF[t] == ac]
        print(f'{ac:15} {len(members):3} tickers')