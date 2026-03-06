"""
import_wrds.py
=============
Pulls raw data from WRDS (CRSP) and FRED, constructs monthly regime features,
cleans and standardizes them, and splits into train/test sets.

Features constructed
--------------------
DD     : Drawdown — how far the market is below its 12-month peak. Always ≤ 0.
         Deep negative = market in significant decline.
VOL    : Log realized volatility — annualized, from daily CRSP returns.
         High = turbulent market. Computed as log(sqrt(sum(r_d^2)) * sqrt(252)).
CS     : Credit spread — BAA minus AAA Moody's yield (%). Widens during stress.
LVIX   : Log VIX — market-implied expected volatility from options. From 1990 only.
GDP_g  : Log quarterly GDP growth. Forward-filled to monthly. Negative = recession.

Target variable
---------------
ret_next : Next month's value-weighted market return (vwretd shifted -1).
           Signal z_t observed at month-end t is used to predict ret_{t+1}.

Train/test split
----------------
Train : 1970–2010 (or 1990–2010 for VIX-dependent models)
Test  : 2011–2025
Standardization uses train mean/std only — no leakage into test.
Expanding-window refit happens during HMM fitting, not here.

Outputs
-------
panel     : Full cleaned monthly dataframe
train_z   : Standardized train features (core 3: DD, VOL, CS)
test_z    : Standardized test features (core 3: DD, VOL, CS)
features_stress.png : Sanity check plot — DD, VOL, CS
features_macro.png  : Sanity check plot — LVIX, GDP_g
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend: saves to file, no window

import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from scipy.stats import mstats

# ── Section 1: Pull raw data ──────────────────────────────────────────────────

db = wrds.Connection()

# A) CRSP monthly market index
#    vwretd = value-weighted return including dividends
#    Used to build cumulative price index P_t and drawdown DD_t
msi = db.raw_sql("""
    SELECT date, vwretd
    FROM crsp.msi
    WHERE date BETWEEN '1970-01-01' AND '2025-12-31'
    ORDER BY date
""", date_cols=['date'])

msi['P'] = 100 * (1 + msi['vwretd']).cumprod()  # P_t = 100 * prod(1 + r)

# B) CRSP daily market returns
#    Used to compute monthly realized volatility RV_t
dsi = db.raw_sql("""
    SELECT date, vwretd
    FROM crsp.dsi
    WHERE date BETWEEN '1970-01-01' AND '2025-12-31'
    ORDER BY date
""", date_cols=['date'])

dsi['year_month'] = dsi['date'].dt.to_period('M')
rv = (
    dsi.groupby('year_month')['vwretd']
    .apply(lambda r: np.sqrt((r**2).sum()) * np.sqrt(252))  # annualized RV
    .reset_index()
    .rename(columns={'vwretd': 'RV'})
)

# C) Credit spread: BAA minus AAA Moody's yield (FRED)
#    Widens when investors demand more premium for holding risky debt
fred_data = web.DataReader(['BAA', 'AAA'], 'fred', start='1970-01-01', end='2025-12-31')
fred_data.index = fred_data.index.to_period('M')
fred_data['CS'] = fred_data['BAA'] - fred_data['AAA']
cs = fred_data.reset_index().rename(columns={'DATE': 'year_month'})

# D) VIX: market-implied expected volatility (FRED, daily → monthly average)
#    Only available from 1990. Log-transformed to match scale of VOL.
vix_daily = web.DataReader('VIXCLS', 'fred', start='1990-01-01', end='2025-12-31')
vix = vix_daily.resample('ME').mean()
vix.index = vix.index.to_period('M')
vix = vix.reset_index().rename(columns={'DATE': 'year_month', 'VIXCLS': 'VIX'})

# E) Real GDP (FRED, quarterly, chained 2017 dollars)
#    Forward-filled to monthly since GDP is released once per quarter
gdp = web.DataReader('GDPC1', 'fred', start='1970-01-01', end='2025-12-31')
gdp.index = gdp.index.to_period('Q')
gdp = gdp.reset_index().rename(columns={'DATE': 'quarter', 'GDPC1': 'GDP'})

db.close()

# ── Section 2: Construct monthly features ────────────────────────────────────

# DD: drawdown over rolling 12-month window
#     DD_t = (P_t - max(P_{t-11}, ..., P_t)) / max(...)
msi['year_month'] = msi['date'].dt.to_period('M')
L = 12
msi['M_t'] = msi['P'].rolling(L).max()
msi['DD']  = (msi['P'] - msi['M_t']) / msi['M_t']

# VOL: log realized vol (log makes it more normally distributed)
rv['VOL'] = np.log(rv['RV'])

# LVIX: log VIX (same reasoning — raw VIX is right-skewed)
vix['LVIX'] = np.log(vix['VIX'])

# GDP_g: log quarter-over-quarter GDP growth, expanded to monthly
gdp = gdp.sort_values('quarter')
gdp['GDP_g'] = np.log(gdp['GDP'] / gdp['GDP'].shift(1))
gdp_monthly = (
    gdp[['quarter', 'GDP_g']]
    .assign(year_month=gdp['quarter'].apply(
        lambda q: pd.period_range(q.asfreq('M', 'S'), periods=3, freq='M')
    ))
    .explode('year_month')
    [['year_month', 'GDP_g']]
)

# Merge all features into one monthly panel
panel = (
    msi[['year_month', 'date', 'vwretd', 'P', 'DD']]
    .merge(rv[['year_month', 'VOL']],            on='year_month', how='left')
    .merge(cs[['year_month', 'CS']],             on='year_month', how='left')
    .merge(vix[['year_month', 'LVIX']],          on='year_month', how='left')
    .merge(gdp_monthly[['year_month', 'GDP_g']], on='year_month', how='left')
)

# ── Section 3: Data quality checks ───────────────────────────────────────────

# Stale CS: FRED sometimes forward-fills missing months with the prior value.
# 19 stale months is acceptable; a large cluster (>6 in a row) would be a problem.
stale_cs = cs['CS'].diff().eq(0).sum()
print(f"Stale CS months (consecutive identical): {stale_cs}")

# CRSP gaps: flag any month where the date jump exceeds ~35 days
gaps = panel['date'].diff().dt.days.gt(35)
if gaps.any():
    print(f"CRSP monthly gaps found:\n{panel.loc[gaps, 'date']}")
else:
    print("No CRSP monthly gaps found.")

# ── Section 4: Winsorize core features ───────────────────────────────────────
# Clips extreme values at 1st/99th percentile to prevent outliers (e.g. 2008
# credit spread spike) from distorting HMM regime boundaries.
# Applied before train/test split for consistent bounds across both sets.
for col in ['DD', 'VOL', 'CS']:
    panel[col] = mstats.winsorize(panel[col].astype(float), limits=[0.01, 0.01])

# ── Section 5: Align timing and drop incomplete rows ─────────────────────────
# ret_next is the target: market return in month t+1.
# Features at month t are used to predict ret_next (no look-ahead).
panel['ret_next'] = panel['vwretd'].shift(-1)
panel = panel.dropna(subset=['DD', 'VOL', 'CS', 'ret_next'])
# Note: LVIX (pre-1990) and GDP_g (first quarter) may still be NaN — intentional.

# Timing spot-check: ret_next in row t should equal vwretd in row t+1
print("\nTiming spot check (ret_next should equal next row's vwretd):")
print(panel[['date', 'vwretd', 'ret_next']].head(5).to_string(index=False))

# ── Section 6: Train/test split and standardization ──────────────────────────

train = panel[panel['date'] <  '2011-01-01']
test  = panel[panel['date'] >= '2011-01-01']

# Core 3 features used in HMM. LVIX and GDP_g are supplemental.
features     = ['DD', 'VOL', 'CS']
features_ext = ['DD', 'VOL', 'CS', 'LVIX', 'GDP_g']

# Compute mean/std on train only — apply same scale to test (no leakage)
z_mean = train[features].mean()
z_std  = train[features].std()
train_z = (train[features] - z_mean) / z_std
test_z  = (test[features]  - z_mean) / z_std

print(f"\nTrain: {train['date'].min().date()} → {train['date'].max().date()}  ({len(train)} months)")
print(f"Test:  {test['date'].min().date()} → {test['date'].max().date()}  ({len(test)} months)")
print("\nNon-null counts per feature:")
print(panel[features_ext].notna().sum())

# ── Section 7: Sanity check plots ────────────────────────────────────────────
# Saved to file only — no interactive window.
# Open features_stress.png and features_macro.png to inspect.

crises = [
    ('1973-10-01', '1974-12-01', 'Oil shock'),
    ('1987-10-01', '1987-12-01', 'Black Monday'),
    ('2000-03-01', '2002-10-01', 'Dot-com'),
    ('2007-10-01', '2009-06-01', 'GFC'),
    ('2020-02-01', '2020-05-01', 'COVID'),
]

def add_crises(ax, label=False):
    for start, end, lbl in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color='grey')
        if label:
            ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.95,
                    lbl, fontsize=7, color='grey', va='top')

# Plot 1: Market stress indicators (DD, VOL, CS)
fig1, axes1 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (col, title, color) in zip(axes1, [
    ('DD',  'Drawdown DD_t',     'steelblue'),
    ('VOL', 'Log Realized Vol',  'darkorange'),
    ('CS',  'Credit Spread (%)', 'crimson'),
]):
    ax.plot(panel['date'], panel[col], color=color, linewidth=0.8)
    ax.set_ylabel(title, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    add_crises(ax, label=(col == 'DD'))
axes1[-1].set_xlabel('Date')
fig1.suptitle('Market Stress Indicators (grey = known crisis)', fontsize=11)
plt.tight_layout()
fig1.savefig('features_stress.png', dpi=150)
plt.close(fig1)

# Plot 2: Macro indicators (LVIX, GDP_g)
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for ax, (col, title, color) in zip(axes2, [
    ('LVIX',  'Log VIX (from 1990)',    'purple'),
    ('GDP_g', 'GDP Growth (quarterly)', 'seagreen'),
]):
    ax.plot(panel['date'], panel[col], color=color, linewidth=0.8)
    ax.set_ylabel(title, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    add_crises(ax, label=(col == 'LVIX'))
axes2[-1].set_xlabel('Date')
fig2.suptitle('Macro Indicators (grey = known crisis)', fontsize=11)
plt.tight_layout()
fig2.savefig('features_macro.png', dpi=150)
plt.close(fig2)

print("\nPlots saved: features_stress.png, features_macro.png")
