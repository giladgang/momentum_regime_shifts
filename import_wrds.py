import wrds
import pandas as pd
import numpy as np
import pandas_datareader.data as web

db = wrds.Connection()

# ── A) Market index level P_t ─────────────────────────────────────────────────
# vwretd = value-weighted return including dividends (monthly)
msi = db.raw_sql("""
    SELECT date, vwretd
    FROM crsp.msi
    WHERE date BETWEEN '1970-01-01' AND '2025-12-31'
    ORDER BY date
""", date_cols=['date'])

# Build cumulative price index: P_t = 100 * cumprod(1 + r)
msi['P'] = 100 * (1 + msi['vwretd']).cumprod()

# ── B) Realized volatility from CRSP daily ────────────────────────────────────
# vwretd = value-weighted daily return including dividends
dsi = db.raw_sql("""
    SELECT date, vwretd
    FROM crsp.dsi
    WHERE date BETWEEN '1970-01-01' AND '2025-12-31'
    ORDER BY date
""", date_cols=['date'])

# Monthly realized vol: sqrt(sum(r^2)) * sqrt(252), annualized
dsi['year_month'] = dsi['date'].dt.to_period('M')
rv = (
    dsi.groupby('year_month')['vwretd']
    .apply(lambda r: np.sqrt((r**2).sum()) * np.sqrt(252))
    .reset_index()
    .rename(columns={'vwretd': 'RV'})
)

# ── C) Credit spread = BAA yield - AAA yield (direct from FRED) ──────────────
fred_data = web.DataReader(['BAA', 'AAA'], 'fred', start='1970-01-01', end='2025-12-31')
fred_data.index = fred_data.index.to_period('M')
fred_data['CS'] = fred_data['BAA'] - fred_data['AAA']
cs = fred_data.reset_index().rename(columns={'DATE': 'year_month'})

db.close()

# ── Preview ───────────────────────────────────────────────────────────────────
print("\n--- Market Index (P_t) ---")
print(msi[['date', 'vwretd', 'P']].tail())

print("\n--- Realized Volatility ---")
print(rv.tail())

print("\n--- Credit Spread ---")
print(cs.tail())
