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

# ── D) VIX (direct from FRED, daily → monthly average) ───────────────────────
vix_daily = web.DataReader('VIXCLS', 'fred', start='1990-01-01', end='2025-12-31')
vix = vix_daily.resample('ME').mean()
vix.index = vix.index.to_period('M')
vix = vix.reset_index().rename(columns={'DATE': 'year_month', 'VIXCLS': 'VIX'})

# ── E) Real GDP (direct from FRED, quarterly) ────────────────────────────────
# GDPC1 = real GDP, chained 2017 dollars, quarterly
gdp = web.DataReader('GDPC1', 'fred', start='1970-01-01', end='2025-12-31')
gdp.index = gdp.index.to_period('Q')
gdp = gdp.reset_index().rename(columns={'DATE': 'quarter', 'GDPC1': 'GDP'})

db.close()

# ── Step 2: Construct monthly features ───────────────────────────────────────

# 2.2 Drawdown DD_t with L=12 rolling window
msi['year_month'] = msi['date'].dt.to_period('M')
L = 12
msi['M_t'] = msi['P'].rolling(L).max()
msi['DD']  = (msi['P'] - msi['M_t']) / msi['M_t']

# 2.3 Log realized vol
rv['VOL'] = np.log(rv['RV'])

# 2.4 VIX: log(VIX) for same scale as VOL
vix['LVIX'] = np.log(vix['VIX'])

# 2.5 GDP growth: log quarterly growth rate, forward-filled to monthly
gdp = gdp.sort_values('quarter')
gdp['GDP_g'] = np.log(gdp['GDP'] / gdp['GDP'].shift(1))
# Expand quarterly GDP growth to monthly by forward-filling
gdp_monthly = (
    gdp[['quarter', 'GDP_g']]
    .assign(year_month=gdp['quarter'].apply(lambda q: pd.period_range(q.asfreq('M', 'S'), periods=3, freq='M')))
    .explode('year_month')
    [['year_month', 'GDP_g']]
)

# Merge into monthly panel: DD, VOL, CS, LVIX, GDP_g
panel = (
    msi[['year_month', 'date', 'vwretd', 'P', 'DD']]
    .merge(rv[['year_month', 'VOL']],       on='year_month', how='left')
    .merge(cs[['year_month', 'CS']],        on='year_month', how='left')
    .merge(vix[['year_month', 'LVIX']],     on='year_month', how='left')
    .merge(gdp_monthly[['year_month', 'GDP_g']], on='year_month', how='left')
)

# 2.6 Align timing: signal z_t predicts return at t+1
panel['ret_next'] = panel['vwretd'].shift(-1)
panel = panel.dropna(subset=['DD', 'VOL', 'CS', 'ret_next'])

# ── Step 3: Train/test split ──────────────────────────────────────────────────

# Simple split: 1990–2010 train, 2011–2025 test
# (expanding window refit happens during HMM fitting, not here)
train = panel[panel['date'] <  '2011-01-01']
test  = panel[panel['date'] >= '2011-01-01']

# Standardize using train stats only (no leakage into test)
# LVIX and GDP_g are optional; only standardize what's non-null in train
features     = ['DD', 'VOL', 'CS']
features_ext = ['DD', 'VOL', 'CS', 'LVIX', 'GDP_g']

z_mean = train[features].mean()
z_std  = train[features].std()

train_z = (train[features] - z_mean) / z_std
test_z  = (test[features]  - z_mean) / z_std

# ── Preview ───────────────────────────────────────────────────────────────────
print("\n--- Monthly Panel (tail) ---")
print(panel[['date', 'P', 'DD', 'VOL', 'CS', 'LVIX', 'GDP_g', 'ret_next']].tail(10))

print(f"\nTrain: {train['date'].min().date()} → {train['date'].max().date()}  ({len(train)} months)")
print(f"Test:  {test['date'].min().date()} → {test['date'].max().date()}  ({len(test)} months)")

print("\n--- Train feature stats (raw) ---")
print(train[features_ext].describe().round(4))

print("\n--- Train feature stats (standardized, core 3) ---")
print(train_z.describe().round(4))
