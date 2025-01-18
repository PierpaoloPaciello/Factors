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

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title='Dynamic Portfolio Strategy - Europe',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state='expanded'
)

# Apply custom CSS for fonts/background
st.markdown(
    """
    <style>
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

# Title
st.title('Factor Investing: Adapting to Economic Phases (Europe Version)')

st.markdown('''
This app showcases a **Dynamic Portfolio Strategy** using **OECD CLI** data (shifted by 1 month) for Europe (plus select global economies). 
ETFs are selected by phase, and performance is compared to **STOXX 600** and **MSCI World** benchmarks.
''')

# Sidebar
st.sidebar.title('Navigation')
sections = [
    'Methodology',
    'Portfolio Construction',
    'Mean Portfolio Evolution'
]
selected_section = st.sidebar.radio('Go to', sections)

# ========== 1) FETCH OECD DATA ==========
@st.cache_data
def get_oecd_data(countries):
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
    'AUS', 'CAN', 'FRA', 'DEU', 'ITA', 'JPN', 'DNK', 'KOR', 'MEX', 'ESP',
    'TUR', 'GBR', 'USA', 'BRA', 'CHN', 'IND', 'IDN', 'ZAF'
]

with st.spinner('Fetching OECD CLI data...'):
    cli_data = get_oecd_data(countries)

pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
pivot_data.columns.name = None
pivot_data.fillna(method='ffill', inplace=True)
pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)

# Shift by 1 month to account for CLI lag
pivot_data.index = pivot_data.index + pd.DateOffset(months=1)

# ========== 2) DIFFUSION INDEX & PHASES ==========
pivot_data_change = pivot_data.diff()
diffusion_index = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
pivot_data['DI'] = diffusion_index
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
pivot_data = pivot_data.dropna(subset=['Phase'])

# ========== 3) EUROPEAN ETFs & LABELS ==========
tickers = [
    'IEVL.MI', 'IEQU.MI', 'IEMO.MI', 'IFSE.MI',
    'IESZ.MI', 'MVEU.MI', 'EHF1.DE', 'LEONIA.MI'
]

etf_labels = {
    'IEVL.MI': 'Value',
    'IEQU.MI': 'Quality',
    'IEMO.MI': 'Momentum',
    'IFSE.MI': 'Multifactor',
    'MVEU.MI': 'Minimum Volatility',
    'IESZ.MI': 'Size',
    'EHF1.DE': 'High Dividend',
    'LEONIA.MI': 'Cash'
}

labels_df = pd.DataFrame(list(etf_labels.items()), columns=['Ticker', 'Label'])
labels_df['Labels'] = labels_df['Label'].apply(lambda x: [lbl.strip() for lbl in x.split(',')])

start_date = '2016-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def get_etf_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

with st.spinner('Fetching ETF data from Yahoo Finance...'):
    data = get_etf_data(tickers, start_date, end_date)

data = data.fillna(method='ffill').dropna()
data.index = data.index.tz_localize(None)

available_tickers = data.columns.tolist()
labels_df = labels_df[labels_df['Ticker'].isin(available_tickers)]

# ========== 4) DAILY RETURNS & PHASE ALIGNMENT ==========
daily_returns = data.pct_change().fillna(0)
daily_returns.index = daily_returns.index.tz_localize(None)

required_cols = ['DI', 'DI_change', 'DI_direction', 'Phase']
pivot_data_daily = pivot_data[required_cols].reindex(daily_returns.index, method='ffill').fillna(method='bfill')
daily_phases = pivot_data_daily['Phase']

# ========== 5) FACTOR -> ETF MAPPING ==========
factor_etf_mapping = {}
for _, row in labels_df.iterrows():
    for label in row['Labels']:
        factor_etf_mapping.setdefault(label.strip(), []).append(row['Ticker'])

# ========== 6) FACTOR PERFORMANCE PER PHASE ==========
unique_phases = daily_phases.dropna().unique()
factor_performance = pd.DataFrame()

for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    
    if phase_dates.empty:
        continue
    
    factor_cum_returns = {}
    for factor, etfs in factor_etf_mapping.items():
        etfs_in_data = [etf for etf in etfs if etf in daily_returns.columns]
        if not etfs_in_data:
            continue
        phase_returns = daily_returns.loc[phase_dates, etfs_in_data]
        cum_returns = (1 + phase_returns).cumprod()
        mean_cum_return = cum_returns.iloc[-1].mean() - 1
        factor_cum_returns[factor] = mean_cum_return
    
    factor_cum_returns_df = pd.DataFrame.from_dict(factor_cum_returns, orient='index', columns=[phase])
    factor_performance = pd.concat([factor_performance, factor_cum_returns_df], axis=1)

factor_performance.fillna(0, inplace=True)

# ========== 7) SELECT TOP ETFs PER PHASE ==========
best_etfs_per_phase = {}
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    
    phase_returns = daily_returns.loc[phase_dates]
    cum_returns = (1 + phase_returns).cumprod().iloc[-1] - 1
    sorted_etfs = cum_returns.sort_values(ascending=False)
    top_etfs = sorted_etfs.head(3).index.tolist()
    best_etfs_per_phase[phase] = top_etfs

# ========== 8) DYNAMIC PORTFOLIO WEIGHTS ==========
weights_df = pd.DataFrame(0, index=daily_returns.index, columns=daily_returns.columns)
for date in daily_returns.index:
    phase = daily_phases.loc[date]
    etfs = best_etfs_per_phase.get(phase, [])
    if not etfs:
        continue
    w = np.repeat(1/len(etfs), len(etfs))
    weights_df.loc[date, etfs] = w

weights_df.fillna(method='ffill', inplace=True)
latest_date = weights_df.index.max()
current_weights = weights_df.loc[latest_date]
current_weights = current_weights[current_weights > 0]
etfs_in_portfolio = weights_df.columns[(weights_df != 0).any()].tolist()
weights_over_time = weights_df[etfs_in_portfolio]
mean_weights = weights_over_time.mean()
mean_weights = mean_weights[mean_weights > 0]

# ========== 9) PORTFOLIO RETURNS & NEW BENCHMARKS ==========
portfolio_returns = (daily_returns * weights_df.shift(1)).sum(axis=1)
portfolio_cum_returns = (1 + portfolio_returns).cumprod()

# --- Benchmark 1: STOXX 600 (XSX6.MI) ---
STOXX_600 = yf.download('XSX6.MI', start=portfolio_returns.index.min(), end=end_date)['Close']
STOXX_600 = STOXX_600.fillna(method='ffill').dropna()
STOXX_600.index = STOXX_600.index.tz_localize(None)
STOXX_600_returns = STOXX_600.pct_change().fillna(0)
STOXX_600_cum_returns = (1 + STOXX_600_returns).cumprod()

# --- Benchmark 2: MSCI World (URTH) ---
MSCI_W = yf.download('URTH', start=portfolio_returns.index.min(), end=end_date)['Close']
MSCI_W = MSCI_W.fillna(method='ffill').dropna()
MSCI_W.index = MSCI_W.index.tz_localize(None)
MSCI_W_returns = MSCI_W.pct_change().fillna(0)
MSCI_W_cum_returns = (1 + MSCI_W_returns).cumprod()

# Ensure common index for the portfolio & each benchmark in turn
common_index_1 = portfolio_cum_returns.index.intersection(STOXX_600_cum_returns.index)
portfolio_cum_returns = portfolio_cum_returns.loc[common_index_1]
STOXX_600_cum_returns = STOXX_600_cum_returns.loc[common_index_1]

common_index_2 = portfolio_cum_returns.index.intersection(MSCI_W_cum_returns.index)
portfolio_cum_returns = portfolio_cum_returns.loc[common_index_2]
STOXX_600_cum_returns = STOXX_600_cum_returns.loc[common_index_2]
MSCI_W_cum_returns = MSCI_W_cum_returns.loc[common_index_2]

# Rolling window for Sharpe, etc.
window_size = 252
def calculate_drawdown(cum_series):
    peak = cum_series.cummax()
    return (cum_series - peak) / peak

# ========== STREAMLIT SECTIONS ==========
if selected_section == 'Methodology':
    st.markdown('## Methodology')
    st.markdown('1. **OECD CLI** data with 1-month shift\n'
                '2. **Phase detection** via Diffusion Index < 0.5 vs. ≥ 0.5, Rising/Falling\n'
                '3. **ETF selection**: top 3 by cumulative return within each phase\n'
                '4. **Benchmarks**: STOXX 600 (`XSX6.MI`) and MSCI World (`URTH`)\n')
    
    st.markdown('### Sample of Phase Data')
    st.dataframe(pivot_data[['DI','DI_change','Phase']].tail(10), use_container_width=True)
    
    st.markdown('### Diffusion Index')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot_data.index,
        y=pivot_data['DI'],
        mode='lines', name='DI',
        line=dict(color='#1f4e79')
    ))
    fig.add_hline(
        y=0.5, line_dash='dash', line_color='red',
        annotation_text='Threshold = 0.5', annotation_position='top left'
    )
    fig.update_layout(
        title='OECD CLI Diffusion Index (Shifted 1M)',
        xaxis_title='Date',
        yaxis_title='Diffusion Index',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Portfolio Construction':
    st.markdown('## Portfolio Construction & Performance')

    # 1) Factor Performance by Phase
    st.markdown('### Factor Performance by Phase')
    st.dataframe(factor_performance.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # 2) Heatmap
    st.markdown('### Factor Performance Heatmap')
    fp_t = factor_performance.T
    plt.figure(figsize=(8,5))
    sns.heatmap(fp_t*100, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Return (%)'})
    st.pyplot(plt.gcf())

    st.markdown('### Top 3 ETFs in Each Phase')
    for phase in unique_phases:
        st.write(f"**{phase}**:", best_etfs_per_phase.get(phase, []))

    # Dynamic Portfolio vs. STOXX 600
    st.markdown('### Dynamic Portfolio vs. STOXX 600')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum_returns.index, 
        y=portfolio_cum_returns, 
        mode='lines', 
        name='Dynamic Portfolio', 
        line=dict(color='steelblue')
    ))
    fig.add_trace(go.Scatter(
        x=STOXX_600_cum_returns.index, 
        y=STOXX_600_cum_returns, 
        mode='lines', 
        name='STOXX 600 (XSX6.MI)', 
        line=dict(dash='dash', color='black')
    ))
    fig.update_layout(
        title='Europe Factor Portfolio vs. STOXX 600 ETF',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        template='plotly_white',
        height=600, width=1000,
        legend=dict(title='Legend', orientation='h', x=0.1, y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dynamic Portfolio vs. STOXX 600 & MSCI World
    st.markdown('### Portfolio Performance vs. STOXX 600 & MSCI World')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum_returns.index, 
        y=portfolio_cum_returns, 
        mode='lines', 
        name='Dynamic Portfolio', 
        line=dict(color='steelblue')
    ))
    fig.add_trace(go.Scatter(
        x=MSCI_W_cum_returns.index, 
        y=MSCI_W_cum_returns, 
        mode='lines', 
        name='MSCI World (URTH)',
        line=dict(dash='dash', color='gray')
    ))
    fig.add_trace(go.Scatter(
        x=STOXX_600_cum_returns.index, 
        y=STOXX_600_cum_returns, 
        mode='lines', 
        name='STOXX 600 (XSX6.MI)',
        line=dict(dash='dash', color='black')
    ))
    fig.update_layout(
        title='Portfolio Performance vs. STOXX 600 & MSCI World',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========== Factor vs STOXX 600 by Phase ==========
    st.markdown('### Factor Performance vs. STOXX 600 by Phase')

    factor_vs_msci = pd.DataFrame()
    for phase in unique_phases:
        phase_dates = daily_phases[daily_phases == phase].index
        # Intersection with daily_returns AND STOXX_600 to ensure same dates
        phase_dates = phase_dates.intersection(daily_returns.index).intersection(STOXX_600_returns.index)
        
        if phase_dates.empty:
            continue
        
        factor_cum_returns = {}
        for factor, etfs in factor_etf_mapping.items():
            etfs_in_data = [etf for etf in etfs if etf in daily_returns.columns]
            if not etfs_in_data:
                continue
            phase_returns = daily_returns.loc[phase_dates, etfs_in_data]
            cum_returns = (1 + phase_returns).cumprod()
            mean_cum_return = cum_returns.iloc[-1].mean() - 1
            factor_cum_returns[factor] = mean_cum_return
        
        stoxx_phase_returns = STOXX_600_returns.loc[phase_dates]
        stoxx_cum_return = (1 + stoxx_phase_returns).cumprod().iloc[-1] - 1
        factor_cum_returns['STOXX 600'] = stoxx_cum_return
        
        factor_cum_returns_df = pd.DataFrame.from_dict(factor_cum_returns, orient='index', columns=[phase])
        factor_vs_msci = pd.concat([factor_vs_msci, factor_cum_returns_df], axis=1)

    # Display factor vs. STOXX per phase
    factor_vs_msci_t = factor_vs_msci.T
    for phase in factor_vs_msci_t.index:
        data_ = factor_vs_msci_t.loc[phase] * 100
        data_ = data_.dropna()
        data_sorted = data_.sort_values(ascending=False)
        
        fig_phase = px.bar(
            x=data_sorted.index,
            y=data_sorted.values,
            title=f'Factor Performance vs. STOXX 600 during {phase} Phase',
            labels={'x': 'Factor', 'y': 'Cumulative Return (%)'},
            text=data_sorted.apply(lambda x: f"{x:.2f}%")
        )
        fig_phase.update_traces(textposition='outside')
        fig_phase.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_phase, use_container_width=True)

    # Outperformance vs. STOXX 600
    factor_outperformance = factor_vs_msci.subtract(factor_vs_msci.loc['STOXX 600'], axis=1)
    factor_outperformance = factor_outperformance.drop('STOXX 600', errors='ignore')
    factor_outperformance_t = factor_outperformance.T
    
    st.markdown('#### Factor Outperformance (vs. STOXX 600) per Phase')
    st.dataframe(factor_outperformance_t.style.format('{:.2%}'), use_container_width=True)

    # ========== Rolling Sharpe & Drawdown Comparisons ==========
    st.markdown('### Rolling Sharpe Ratio')
    portfolio_rolling_sharpe = portfolio_returns.rolling(window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )
    STOXX_600_rolling_sharpe = STOXX_600_returns.rolling(window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    ).reindex(portfolio_rolling_sharpe.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_rolling_sharpe.index, y=portfolio_rolling_sharpe,
        mode='lines', name='Dynamic Portfolio'
    ))
    fig.add_trace(go.Scatter(
        x=STOXX_600_rolling_sharpe.index, y=STOXX_600_rolling_sharpe,
        mode='lines', name='STOXX 600', line=dict(dash='dash')
    ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Rolling Sharpe Ratio (252d)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Rolling Drawdown')
    portfolio_drawdown = calculate_drawdown(portfolio_cum_returns)
    stoxx_drawdown = calculate_drawdown(STOXX_600_cum_returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_drawdown.index, y=portfolio_drawdown,
        mode='lines', name='Dynamic Portfolio'
    ))
    fig.add_trace(go.Scatter(
        x=stoxx_drawdown.index, y=stoxx_drawdown,
        mode='lines', name='STOXX 600', line=dict(dash='dash')
    ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annual Returns
    st.markdown('### Annual Percentage Returns')
    portfolio_annual_returns = (1 + portfolio_returns).resample('Y').prod() - 1
    stoxx_annual_returns = (1 + STOXX_600_returns).resample('Y').prod() - 1
    msci_annual_returns = (1 + MSCI_W_returns).resample('Y').prod() - 1

    annual_df = pd.DataFrame({
        'Dynamic Portfolio': portfolio_annual_returns,
        'STOXX 600': stoxx_annual_returns,
        'MSCI World': msci_annual_returns
    })
    annual_df.index = annual_df.index.year
    annual_df.index.name = 'Year'

    melt_df = annual_df.reset_index().melt(
        id_vars='Year', var_name='Portfolio', value_name='Annual Return'
    )

    fig = px.bar(
        melt_df, x='Year', y='Annual Return', color='Portfolio',
        barmode='group', text='Annual Return', labels={'Annual Return':'Annual Return (%)'}
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(yaxis_tickformat='%', xaxis=dict(type='category'))
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Mean Portfolio Evolution':
    st.markdown('## Mean Portfolio Evolution')

    # 1) Calc monthly weights
    weights_monthly = weights_df.resample('M').first()
    
    # 2) Cumulative average
    mean_weights_monthly = pd.DataFrame(index=weights_monthly.index, columns=weights_monthly.columns)
    for i in range(len(weights_monthly)):
        mean_weights_monthly.iloc[i] = weights_monthly.iloc[:i+1].mean()
    
    # 3) Expand to daily
    mean_weights_daily = mean_weights_monthly.reindex(daily_returns.index, method='ffill').fillna(0)
    
    # 4) Mean Portfolio returns
    mean_portfolio_returns = (daily_returns * mean_weights_daily.shift(1)).sum(axis=1)
    mean_portfolio_cum_returns = (1 + mean_portfolio_returns).cumprod()
    
    # Align with STOXX & MSCI
    common_idx = mean_portfolio_cum_returns.index.intersection(STOXX_600_cum_returns.index).intersection(MSCI_W_cum_returns.index)
    mean_portfolio_cum_returns = mean_portfolio_cum_returns.loc[common_idx]
    stoxx_cum_aligned = STOXX_600_cum_returns.loc[common_idx]
    msci_cum_aligned = MSCI_W_cum_returns.loc[common_idx]

    st.markdown('### Mean Portfolio vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_portfolio_cum_returns.index, 
        y=mean_portfolio_cum_returns, 
        mode='lines', 
        name='Mean Portfolio', 
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=stoxx_cum_aligned.index, 
        y=stoxx_cum_aligned, 
        mode='lines', 
        name='STOXX 600', 
        line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=msci_cum_aligned.index, 
        y=msci_cum_aligned, 
        mode='lines', 
        name='MSCI World (URTH)', 
        line=dict(dash='dot')
    ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Sample of Mean Weights (Monthly)')
    st.dataframe(mean_weights_monthly.head(), use_container_width=True)

    st.markdown('### Evolving Mean Allocation Over Time')
    fig2 = make_subplots(rows=1, cols=1)
    for etf in mean_weights_monthly.columns[mean_weights_monthly.sum() > 0]:
        fig2.add_trace(
            go.Scatter(
                x=mean_weights_monthly.index,
                y=mean_weights_monthly[etf],
                stackgroup='one',
                mode='lines',
                name=etf
            ),
            row=1, col=1
        )
    fig2.update_layout(hovermode='x unified', yaxis_title='Weight')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('### Rolling Sharpe & Drawdown')
    mean_sharpe = mean_portfolio_returns.rolling(window_size).apply(lambda x: (x.mean()/x.std())*np.sqrt(252))
    stoxx_sharpe = STOXX_600_returns.rolling(window_size).apply(lambda x: (x.mean()/x.std())*np.sqrt(252)).reindex(mean_sharpe.index)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=mean_sharpe.index, 
        y=mean_sharpe, 
        mode='lines', 
        name='Mean Portfolio'
    ))
    fig3.add_trace(go.Scatter(
        x=stoxx_sharpe.index, 
        y=stoxx_sharpe, 
        mode='lines', 
        name='STOXX 600', 
        line=dict(dash='dash')
    ))
    fig3.update_layout(xaxis_title='Date', yaxis_title='Sharpe Ratio')
    st.plotly_chart(fig3, use_container_width=True)

    mean_drawdown = calculate_drawdown(mean_portfolio_cum_returns)
    stoxx_drawdown_aligned = calculate_drawdown(stoxx_cum_aligned)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=mean_drawdown.index, 
        y=mean_drawdown, 
        mode='lines', 
        name='Mean Portfolio'
    ))
    fig4.add_trace(go.Scatter(
        x=stoxx_drawdown_aligned.index, 
        y=stoxx_drawdown_aligned, 
        mode='lines', 
        name='STOXX 600', 
        line=dict(dash='dash')
    ))
    fig4.update_layout(xaxis_title='Date', yaxis_title='Drawdown', yaxis_tickformat='%')
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('### Annual Percentage Returns')
    mean_annual = (1+mean_portfolio_returns).resample('Y').prod()-1
    stoxx_annual = (1+STOXX_600_returns).resample('Y').prod()-1
    msci_annual = (1+MSCI_W_returns).resample('Y').prod()-1

    ann_df = pd.DataFrame({
        'Mean Portfolio': mean_annual,
        'STOXX 600': stoxx_annual,
        'MSCI World': msci_annual
    })
    ann_df.index = ann_df.index.year
    ann_df.index.name='Year'
    ann_melt = ann_df.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')

    fig5 = px.bar(
        ann_melt, x='Year', y='Annual Return', color='Portfolio',
        barmode='group', text='Annual Return'
    )
    fig5.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig5.update_layout(yaxis_tickformat='%')
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('---')
    st.markdown('**Conclusion**: The Mean Portfolio approach provides a smoother factor exposure over time, while still comparing favorably to STOXX 600 and MSCI World.')
