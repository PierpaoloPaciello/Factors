import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, MonthBegin
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(
    layout="wide",
    page_title='Dynamic Portfolio Strategy (Europe)',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state='expanded'
)

# Custom CSS (optional)
st.markdown(
    """
    <style>
    /* Fonts, background, sidebar, headers, etc. */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
    }
    body {
        background: linear-gradient(90deg, #f0f2f6 0%, #e2e6ee 100%);
    }
    .css-1d391kg {
        background-color: #1f4e79 !important;
    }
    .css-1d391kg .css-1v3fvcr {
        color: white;
    }
    .css-1d391kg .css-1v3fvcr .css-1nnyp7m {
        color: white;
    }
    h1, h2, h3, h4 {
        color: #1f4e79;
        font-weight: 700;
    }
    p, div, label, span {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Factor Investing: Adapting to Economic Phases (Europe Version)')

# ------------------- 1) OECD Data Fetch & Prep -------------------
@st.cache_data
def get_oecd_data(countries):
    """Download CLI data from OECD, returns a DataFrame."""
    database = '@DF_CLI'
    frequency = 'M'
    indicator = 'LI..'
    unit_of_measure = 'AA...'
    start_period = '1990-01'
    
    country_code = "+".join(countries)
    query_text = f"{database}/{country_code}.{frequency}.{indicator}.{unit_of_measure}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8'
    }
    
    download = requests.get(url=url, headers=headers)
    df = pd.read_csv(io.StringIO(download.text))
    return df

countries = [
    'AUS','AUT','BEL','CAN','CHL','CZE','DNK','EST','FIN','FRA',
    'DEU','GRC','HUN','ISL','IRL','ISR','ITA','JPN','KOR','LVA',
    'LTU','LUX','MEX','NLD','NZL','NOR','POL','PRT','SVK','SVN',
    'ESP','SWE','CHE','TUR','GBR','USA'
]

with st.spinner('Fetching OECD CLI data...'):
    cli_data = get_oecd_data(countries)

pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
pivot_data.columns.name = None
pivot_data.fillna(method='ffill', inplace=True)
pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)

# SHIFT index by 1 month to account for lag in OECD CLI
pivot_data.index = pivot_data.index + pd.DateOffset(months=1)

# Calculate Diffusion Index
pivot_data_change = pivot_data.diff()
diffusion_index   = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
pivot_data['DI']  = diffusion_index
pivot_data['DI_change'] = pivot_data['DI'].diff()
pivot_data['DI_direction'] = np.where(pivot_data['DI_change'] > 0, 'Rising', 'Falling')

def determine_phase(row):
    if row['DI'] < 0.5 and row['DI_direction'] == 'Rising':
        return 'Recovery'
    elif row['DI'] >= 0.5 and row['DI_direction'] == 'Rising':
        return 'Expansion'
    elif row['DI'] >= 0.5 and row['DI_direction'] == 'Falling':
        return 'Slowdown'
    elif row['DI'] < 0.5 and row['DI_direction'] == 'Falling':
        return 'Contraction'
    else:
        return 'Unknown'

pivot_data['Phase'] = pivot_data.apply(determine_phase, axis=1)
pivot_data.dropna(subset=['Phase'], inplace=True)

# ------------------- 2) European Factor ETFs -------------------
tickers = [
    'IEVL.MI',  # Value
    'IEQU.MI',  # Quality
    'IEMO.MI',  # Momentum
    'IFSE.MI',  # Multifactor
    'MVEU.MI',  # Low Volatility
    'IESZ.MI',  # Size
    'EHF1.DE',  # High Dividend
    'LEONIA.MI' # Cash
]

etf_labels = {
    'IEVL.MI': 'Value',
    'IEQU.MI': 'Quality',
    'IEMO.MI': 'Momentum',
    'IFSE.MI': 'Multifactor',
    'MVEU.MI': 'Low Volatility',
    'IESZ.MI': 'Size',
    'EHF1.DE': 'High Dividend',
    'LEONIA.MI': 'Cash'
}

labels_df = pd.DataFrame(list(etf_labels.items()), columns=['Ticker','Label'])
labels_df['Labels'] = labels_df['Label'].apply(lambda x: [lbl.strip() for lbl in x.split(',')])

start_date = '2016-01-01'
end_date   = datetime.datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def get_etf_data(tickers, start_date, end_date):
    d = yf.download(tickers, start=start_date, end=end_date)['Close']
    return d

with st.spinner('Fetching ETF data from Yahoo Finance...'):
    data = get_etf_data(tickers, start_date, end_date)

data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)
data.index = data.index.tz_localize(None)

available_tickers = data.columns.tolist()
labels_df = labels_df[labels_df['Ticker'].isin(available_tickers)]

# ------------------- 3) Daily Returns & Phase Alignment -------------------
daily_returns = data.pct_change().fillna(0)

req_cols = ['DI','DI_change','DI_direction','Phase']
pivot_data_daily = pivot_data[req_cols].reindex(daily_returns.index, method='ffill').fillna(method='bfill')
daily_phases = pivot_data_daily['Phase']

# Map each factor to one or more ETFs
factor_etf_mapping = {}
for _, row in labels_df.iterrows():
    for factor_label in row['Labels']:
        factor_etf_mapping.setdefault(factor_label, []).append(row['Ticker'])

unique_phases = daily_phases.dropna().unique()
factor_performance = pd.DataFrame()

# Factor performance per phase
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    
    factor_cum_returns = {}
    for factor, etfs_in_factor in factor_etf_mapping.items():
        valid_etfs = [e for e in etfs_in_factor if e in daily_returns.columns]
        if not valid_etfs:
            continue
        phase_returns = daily_returns.loc[phase_dates, valid_etfs]
        cum_ret = (1 + phase_returns).cumprod()
        mean_cum = cum_ret.iloc[-1].mean() - 1
        factor_cum_returns[factor] = mean_cum
    
    df_factors = pd.DataFrame.from_dict(factor_cum_returns, orient='index', columns=[phase])
    factor_performance = pd.concat([factor_performance, df_factors], axis=1)

factor_performance.fillna(0, inplace=True)

# ------------------- 4) Build the Dynamic Portfolio -------------------
best_etfs_per_phase = {}
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    
    phase_returns = daily_returns.loc[phase_dates]
    cum_ret = (1 + phase_returns).cumprod().iloc[-1] - 1
    sorted_etfs = cum_ret.sort_values(ascending=False)
    top_etfs = sorted_etfs.head(3).index.tolist()
    best_etfs_per_phase[phase] = top_etfs

weights_df = pd.DataFrame(0, index=daily_returns.index, columns=daily_returns.columns)
for dt in daily_returns.index:
    ph = daily_phases.loc[dt]
    picks = best_etfs_per_phase.get(ph, [])
    if picks:
        w = np.repeat(1.0 / len(picks), len(picks))
        weights_df.loc[dt, picks] = w

weights_df.fillna(method='ffill', inplace=True)

latest_date = weights_df.index.max()
current_weights = weights_df.loc[latest_date]
current_weights = current_weights[current_weights>0]
etfs_in_portfolio = weights_df.columns[(weights_df!=0).any()].tolist()
weights_over_time = weights_df[etfs_in_portfolio]
mean_weights = weights_over_time.mean()
mean_weights = mean_weights[mean_weights>0]

# ------------------- 5) Portfolio Returns & Benchmarks -------------------
portfolio_returns = (daily_returns * weights_df.shift(1)).sum(axis=1)
portfolio_cum_returns = (1 + portfolio_returns).cumprod()

# STOXX 600
stoxx600 = yf.download('XSX6.MI', start=portfolio_returns.index.min(), end=end_date)['Close']
stoxx600 = stoxx600.fillna(method='ffill').dropna()
stoxx600.index = stoxx600.index.tz_localize(None)
stoxx600_returns = stoxx600.pct_change().fillna(0)
stoxx600_cum_returns = (1 + stoxx600_returns).cumprod()

# MSCI World (URTH)
msci_world = yf.download('URTH', start=portfolio_returns.index.min(), end=end_date)['Close']
msci_world = msci_world.fillna(method='ffill').dropna()
msci_world.index = msci_world.index.tz_localize(None)
msci_world_returns = msci_world.pct_change().fillna(0)
msci_world_cum_returns = (1 + msci_world_returns).cumprod()

# Align indexes
common_idx = portfolio_cum_returns.index \
    .intersection(stoxx600_cum_returns.index) \
    .intersection(msci_world_cum_returns.index)

portfolio_cum_returns_aligned   = portfolio_cum_returns.loc[common_idx]
stoxx600_cum_returns_aligned    = stoxx600_cum_returns.loc[common_idx]
msci_world_cum_returns_aligned  = msci_world_cum_returns.loc[common_idx]

portfolio_returns_aligned  = portfolio_returns.loc[common_idx]
stoxx600_returns_aligned   = stoxx600_returns.loc[common_idx]
msci_world_returns_aligned = msci_world_returns.loc[common_idx]

window_size = 252
def calculate_drawdown(cum_ret):
    peak = cum_ret.cummax()
    return (cum_ret - peak)/peak

# ------------------- Streamlit Sections -------------------
st.sidebar.title('Navigation')
sections = [
    'Methodology',
    'Portfolio Construction',
    'Mean Portfolio Evolution'
]
selected_section = st.sidebar.radio('Go to', sections)

if selected_section == 'Methodology':
    st.markdown('## Methodology')
    st.markdown('''
    1. **OECD CLI** (shifted by 1 month) → Diffusion Index → Economic phases.
    2. **European Factor ETFs** allocated by top performance each phase.
    3. **Benchmarks**: STOXX 600 (`XSX6.MI`) and MSCI World (`URTH`).
    ''')
    st.dataframe(pivot_data[['DI','DI_change','Phase']].tail(10))
    
    st.markdown('### OECD CLI Diffusion Index (1M Shift)')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot_data.index, y=pivot_data['DI'],
        mode='lines', name='DI'
    ))
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='Threshold (0.5)')
    fig.update_layout(title='Diffusion Index', hovermode='x unified')
    st.plotly_chart(fig)

elif selected_section == 'Portfolio Construction':
    st.markdown('## Factor Performance by Phase')
    st.dataframe(factor_performance.style.background_gradient(cmap='Blues'))

    st.markdown('### Factor Heatmap')
    fig_heat, ax_heat = plt.subplots(figsize=(8,5))
    sns.heatmap(factor_performance.T*100, annot=True, fmt='.2f', cmap='Blues', ax=ax_heat, cbar_kws={'label':'Return (%)'})
    st.pyplot(fig_heat)

    st.markdown('### Top 3 ETFs per Phase')
    for ph in unique_phases:
        st.write(f"{ph}: {best_etfs_per_phase.get(ph, [])}")

    st.markdown('### Portfolio Performance vs. Benchmarks')
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=portfolio_cum_returns_aligned.index,
        y=portfolio_cum_returns_aligned,
        mode='lines', name='Dynamic Portfolio'
    ))
    fig_perf.add_trace(go.Scatter(
        x=stoxx600_cum_returns_aligned.index,
        y=stoxx600_cum_returns_aligned,
        mode='lines', name='STOXX 600 (XSX6.MI)', line=dict(dash='dash')
    ))
    fig_perf.add_trace(go.Scatter(
        x=msci_world_cum_returns_aligned.index,
        y=msci_world_cum_returns_aligned,
        mode='lines', name='MSCI World (URTH)', line=dict(dash='dot')
    ))
    fig_perf.update_layout(title='Dynamic Portfolio vs. STOXX 600 & MSCI World')
    st.plotly_chart(fig_perf)

    st.markdown('### Annual Percentage Returns')
    # Squeeze each to ensure 1D
    portfolio_annual_returns = (1 + portfolio_returns_aligned).resample('Y').prod() - 1
    if isinstance(portfolio_annual_returns, pd.DataFrame):
        portfolio_annual_returns = portfolio_annual_returns.iloc[:,0]

    msci_annual_returns = (1 + msci_world_returns_aligned).resample('Y').prod() - 1
    if isinstance(msci_annual_returns, pd.DataFrame):
        msci_annual_returns = msci_annual_returns.iloc[:,0]

    stoxx_annual_returns = (1 + stoxx600_returns_aligned).resample('Y').prod() - 1
    if isinstance(stoxx_annual_returns, pd.DataFrame):
        stoxx_annual_returns = stoxx_annual_returns.iloc[:,0]

    annual_df = pd.DataFrame({
        'Dynamic Portfolio': portfolio_annual_returns,
        'MSCI World (URTH)': msci_annual_returns,
        'STOXX 600 (XSX6.MI)': stoxx_annual_returns
    })
    annual_df.index = annual_df.index.year
    annual_df.index.name='Year'

    annual_melt = annual_df.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')
    fig_annual = px.bar(
        annual_melt,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return',
        labels={'Annual Return':'Annual Return (%)'}
    )
    fig_annual.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_annual.update_layout(title='Annual Percentage Returns', yaxis_tickformat='%')
    st.plotly_chart(fig_annual)

elif selected_section == 'Mean Portfolio Evolution':
    st.markdown('## Mean Portfolio Evolution')
    weights_monthly = weights_df.resample('M').first()
    mean_weights_monthly = pd.DataFrame(index=weights_monthly.index, columns=weights_monthly.columns)

    # Build cumulative average
    for i in range(len(weights_monthly)):
        subset = weights_monthly.iloc[:i+1]
        mean_weights_monthly.iloc[i] = subset.mean()

    mean_weights_df = mean_weights_monthly.reindex(daily_returns.index, method='ffill').fillna(0)
    mean_portfolio_returns = (daily_returns * mean_weights_df.shift(1)).sum(axis=1)
    mean_portfolio_cum = (1 + mean_portfolio_returns).cumprod()

    # Align with benchmarks
    mean_common_idx = mean_portfolio_cum.index \
        .intersection(stoxx600_cum_returns.index) \
        .intersection(msci_world_cum_returns.index)
    mean_portfolio_cum_aligned = mean_portfolio_cum.loc[mean_common_idx]
    stoxx_mean_aligned         = stoxx600_cum_returns.loc[mean_common_idx]
    msci_mean_aligned          = msci_world_cum_returns.loc[mean_common_idx]

    st.markdown('### Mean Portfolio vs. Benchmarks')
    fig_mp = go.Figure()
    fig_mp.add_trace(go.Scatter(
        x=mean_portfolio_cum_aligned.index,
        y=mean_portfolio_cum_aligned,
        mode='lines', name='Mean Portfolio'
    ))
    fig_mp.add_trace(go.Scatter(
        x=stoxx_mean_aligned.index,
        y=stoxx_mean_aligned,
        mode='lines',
        name='STOXX 600',
        line=dict(dash='dash')
    ))
    fig_mp.add_trace(go.Scatter(
        x=msci_mean_aligned.index,
        y=msci_mean_aligned,
        mode='lines',
        name='MSCI World',
        line=dict(dash='dot')
    ))
    fig_mp.update_layout(title='Mean Portfolio Performance')
    st.plotly_chart(fig_mp)

    # Rolling Sharpe
    st.markdown('### Rolling Sharpe Ratio (Mean Portfolio)')
    mean_portfolio_returns_aligned = mean_portfolio_returns.loc[mean_common_idx]
    msci_mean_aligned_returns      = msci_world_returns.loc[mean_common_idx]

    mean_rolling_sharpe = mean_portfolio_returns_aligned.rolling(window_size).apply(
        lambda x: (x.mean()/x.std())*np.sqrt(252) if x.std()!=0 else np.nan
    )
    msci_rolling_sharpe = msci_mean_aligned_returns.rolling(window_size).apply(
        lambda x: (x.mean()/x.std())*np.sqrt(252) if x.std()!=0 else np.nan
    )

    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(go.Scatter(
        x=mean_rolling_sharpe.index, y=mean_rolling_sharpe,
        mode='lines', name='Mean Portfolio'
    ))
    fig_sharpe.add_trace(go.Scatter(
        x=msci_rolling_sharpe.index, y=msci_rolling_sharpe,
        mode='lines', name='MSCI World', line=dict(dash='dash')
    ))
    fig_sharpe.update_layout(title='Rolling Sharpe Ratio (Mean Portfolio)')
    st.plotly_chart(fig_sharpe)

    # Annual returns
    st.markdown('### Annual Returns (Mean Portfolio)')
    mean_portfolio_annual = (1 + mean_portfolio_returns_aligned).resample('Y').prod() - 1
    if isinstance(mean_portfolio_annual, pd.DataFrame):
        mean_portfolio_annual = mean_portfolio_annual.iloc[:,0]

    msci_annual_mean = (1 + msci_mean_aligned_returns).resample('Y').prod() - 1
    if isinstance(msci_annual_mean, pd.DataFrame):
        msci_annual_mean = msci_annual_mean.iloc[:,0]

    stoxx_annual_mean = (1 + stoxx600_returns.loc[mean_common_idx]).resample('Y').prod() - 1
    if isinstance(stoxx_annual_mean, pd.DataFrame):
        stoxx_annual_mean = stoxx_annual_mean.iloc[:,0]

    df_annual_mean = pd.DataFrame({
        'Mean Portfolio': mean_portfolio_annual,
        'MSCI World': msci_annual_mean,
        'STOXX 600': stoxx_annual_mean
    })
    df_annual_mean.index = df_annual_mean.index.year
    df_annual_mean.index.name='Year'

    melt_mean = df_annual_mean.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')
    fig_mean = px.bar(
        melt_mean,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return'
    )
    fig_mean.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_mean.update_layout(title='Annual Percentage Returns (Mean Portfolio)', yaxis_tickformat='%')
    st.plotly_chart(fig_mean)

    st.markdown('---')

