"""clear_backtest.py — wipe old backtest data so run_backtest() re-simulates. Run once from the project root: python clear_backtest.py."""

import sqlite3

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'FAKE']  # FAKE = test_synthetic.py pollution
CLEAR_PRICE_CACHE = False  # set True to also force a fresh yfinance download instead of reusing cached prices

conn = sqlite3.connect('naissance.db')
try:
    cur = conn.cursor()
    placeholders = ','.join('?' for _ in TICKERS)

    for ticker in TICKERS:
        cur.execute("SELECT COUNT(*) FROM trade_log WHERE ticker = ? AND source = 'backtest'", (ticker,))
        n_trades = cur.fetchone()[0]
        print(f'{ticker}: {n_trades} backtest trade_log rows to delete')

    cur.execute(f"DELETE FROM trade_log WHERE ticker IN ({placeholders}) AND source = 'backtest'", TICKERS)
    print(f'\ndeleted {cur.rowcount} rows from trade_log')

    cur.execute(f"DELETE FROM backtest_results WHERE ticker IN ({placeholders})", TICKERS)
    print(f'deleted {cur.rowcount} rows from backtest_results')

    cur.execute(f"DELETE FROM system_maturity WHERE ticker IN ({placeholders})", TICKERS)
    print(f'deleted {cur.rowcount} rows from system_maturity')

    if CLEAR_PRICE_CACHE:
        cur.execute(f"DELETE FROM price_cache WHERE ticker IN ({placeholders})", TICKERS)
        print(f'deleted {cur.rowcount} rows from price_cache (forcing fresh yfinance download)')
    else:
        print('price_cache left untouched (set CLEAR_PRICE_CACHE=True to force re-download)')

    conn.commit()
finally:
    conn.close()

print('\nDone. You can now re-run:')
print('  python -m layers.backtest_engine          # single AAPL smoke test')