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

# Set page configuration with custom theme
st.set_page_config(
    layout="wide",
    page_title='Dynamic Portfolio Strategy',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state='expanded'
)

# Apply custom CSS for fonts and backgrounds
st.markdown(
    """
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
    }
    /* Background */
    body {
        background: linear-gradient(90deg, #f0f2f6 0%, #e2e6ee 100%);
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #1f4e79 !important;
    }
    .css-1d391kg .css-1v3fvcr {
        color: white;
    }
    .css-1d391kg .css-1v3fvcr .css-1nnyp7m {
        color: white;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #1f4e79;
        font-weight: 700;
    }
    /* Text */
    p, div, label, span {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Introduction
st.title('Factor Investing: Adapting to Economic Phases')

st.markdown('''
## Introduction

This app presents the **Dynamic Portfolio Strategy**, an approach designed to optimize returns by dynamically allocating assets based on economic phases.

Using the **OECD CLI Diffusion Index** as a macroeconomic indicator, this strategy identifies four distinct economic phases:
- **Recovery**: Early growth after a contraction.
- **Expansion**: Sustained growth phase.
- **Slowdown**: Decline from peak economic activity.
- **Contraction**: Widespread economic decline.

By aligning factor-based ETFs with these phases, the strategy seeks to:
1. Outperform benchmarks (MSCI World ETF (URTH), SPY (S&P 500 ETF)).
3. Minimize drawdowns during adverse market conditions.

The analysis evaluates this strategy’s performance, highlighting its ability to leverage factors such as Momentum, Quality, and Low Volatility across economic cycles.
''')

st.markdown('---')

# Sidebar for navigation
st.sidebar.title('Navigation')
sections = [
    'Methodology',
    'Portfolio Construction',
    'Mean Portfolio Evolution'
]

selected_section = st.sidebar.radio('Go to', sections)

# Function to download CLI data from OECD
@st.cache_data
def get_oecd_data(countries):
    database = '@DF_CLI'
    frequency = 'M'
    indicator = 'LI..'
    unit_of_measure = 'AA...'
    start_period = '1990-01'
    
    # Join all country codes
    country_code = "+".join(countries)
    
    # Create the query URL
    query_text = f"{database}/{country_code}.{frequency}.{indicator}.{unit_of_measure}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"
    
    headers = { 
        'User-Agent': 'Mozilla/5.0', 
        'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8' 
    }
    
    # Fetch the data
    download = requests.get(url=url, headers=headers)
    df = pd.read_csv(io.StringIO(download.text))
    
    return df

# List of countries for DI calculation
countries = ['AUS', 'AUT', 'BEL', 'CAN', 'CHL', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
             'DEU', 'GRC', 'HUN', 'ISL', 'IRL', 'ISR', 'ITA', 'JPN', 'KOR', 'LVA',
             'LTU', 'LUX', 'MEX', 'NLD', 'NZL', 'NOR', 'POL', 'PRT', 'SVK', 'SVN',
             'ESP', 'SWE', 'CHE', 'TUR', 'GBR', 'USA']

# Fetch the CLI data for all countries
with st.spinner('Fetching OECD CLI data...'):
    cli_data = get_oecd_data(countries)

# Reshape the data for easier processing
pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')

# Remove the column name to avoid KeyError when selecting columns
pivot_data.columns.name = None

# Ensure missing values are handled (fill with previous values)
pivot_data.fillna(method='ffill', inplace=True)

# Convert TIME_PERIOD to datetime and ensure timezone-naive
pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)

# --- Moved calculation of DI and Phases here ---
# Calculate Diffusion Index and Economic Phases
pivot_data_change = pivot_data.diff()
diffusion_index = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
pivot_data['DI'] = diffusion_index
pivot_data['DI_change'] = pivot_data['DI'].diff()
pivot_data['DI_direction'] = np.where(pivot_data['DI_change'] > 0, 'Rising', 'Falling')

# Define economic phases based on DI level and direction
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

# --- Common data preparation for all sections ---

# List of ETF tickers
tickers = [
    'QUAL', 'USMV', 'MTUM', 'VLUE', 'EFAV', 'ACWV',
    'LRGF', 'IMTM', 'IVLU', 'SMLF', 'ISCF',
    'SIZE', 'GLOF',
    'HDV'
]

# Mapping of ETF tickers to labels
etf_labels = {
    'QUAL': 'Quality',
    'USMV': 'Low Volatility',
    'MTUM': 'Momentum',
    'VLUE': 'Value',
    'EFAV': 'Low Volatility',
    'ACWV': 'Low Volatility',
    'LRGF': 'Multifactor',
    'IMTM': 'Momentum',
    'IVLU': 'Value',
    'SMLF': 'Size',
    'ISCF': 'Size',
    'SIZE': 'Size',
    'GLOF': 'Multifactor',
    'HDV': 'High Dividend'
}

labels_df = pd.DataFrame(list(etf_labels.items()), columns=['Ticker', 'Label'])
labels_df['Labels'] = labels_df['Label'].apply(lambda x: [label.strip() for label in x.split(',')])

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

# Prepare daily returns and phases
daily_returns = data.pct_change().fillna(0)
daily_returns.index = daily_returns.index.tz_localize(None)
pivot_data.index = pivot_data.index.tz_localize(None)

# Extract required columns
required_columns = ['DI', 'DI_change', 'DI_direction', 'Phase']
pivot_data_daily = pivot_data[required_columns]

# Reindex to match daily_returns index
pivot_data_daily = pivot_data_daily.reindex(daily_returns.index, method='ffill')
pivot_data_daily = pivot_data_daily.fillna(method='bfill')

daily_phases = pivot_data_daily['Phase']

# Map ETFs to factors
factor_etf_mapping = {}
for idx, row in labels_df.iterrows():
    for label in row['Labels']:
        factor_etf_mapping.setdefault(label.strip(), []).append(row['Ticker'])

# Initialize a DataFrame to store factor performance per phase
factor_performance = pd.DataFrame()
unique_phases = daily_phases.dropna().unique()

# Calculate factor performance per phase
for phase in unique_phases:
    # Get dates corresponding to the phase
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    
    if phase_dates.empty:
        continue
    
    # Initialize a dictionary to store cumulative returns for each factor
    factor_cum_returns = {}
    
    for factor, etfs in factor_etf_mapping.items():
        etfs_in_data = [etf for etf in etfs if etf in daily_returns.columns]
        if not etfs_in_data:
            continue
        # Get returns for ETFs during the phase
        phase_returns = daily_returns.loc[phase_dates, etfs_in_data]
        # Calculate cumulative returns
        cum_returns = (1 + phase_returns).cumprod()
        # Calculate the mean cumulative return of the factor
        mean_cum_return = cum_returns.iloc[-1].mean() - 1
        factor_cum_returns[factor] = mean_cum_return
    
    # Create a DataFrame from the cumulative returns
    factor_cum_returns_df = pd.DataFrame.from_dict(factor_cum_returns, orient='index', columns=[phase])
    
    # Append to the factor performance DataFrame
    factor_performance = pd.concat([factor_performance, factor_cum_returns_df], axis=1)

factor_performance.fillna(0, inplace=True)

# Select top ETFs per phase
best_etfs_per_phase = {}

for phase in unique_phases:
    # Get dates corresponding to the phase
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    
    if phase_dates.empty:
        continue
    
    phase_returns = daily_returns.loc[phase_dates]
    cum_returns = (1 + phase_returns).cumprod().iloc[-1] - 1
    sorted_etfs = cum_returns.sort_values(ascending=False)
    top_etfs = sorted_etfs.head(3)
    best_etfs_per_phase[phase] = top_etfs.index.tolist()

# Calculate dynamic portfolio weights
weights_df = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns).fillna(0)

# For each date, assign weights based on the phase
for date in daily_returns.index:
    # Get the economic phase for the current date
    phase = daily_phases.loc[date]
    
    # Get the top ETFs for the current phase
    etfs = best_etfs_per_phase.get(phase, [])
    
    if not etfs:
        continue
    
    # Equal weighting among selected ETFs
    weights = np.repeat(1/len(etfs), len(etfs))
    
    # Assign weights
    weights_df.loc[date, etfs] = weights

# Forward-fill the weights to handle any NaN values
weights_df.fillna(method='ffill', inplace=True)

# Prepare data for allocations over time
latest_date = weights_df.index.max()
current_weights = weights_df.loc[latest_date]
current_weights = current_weights[current_weights > 0]
etfs_in_portfolio = weights_df.columns[(weights_df != 0).any()].tolist()
weights_over_time = weights_df[etfs_in_portfolio]

# Calculate mean allocations over the entire period
mean_weights = weights_over_time.mean()
mean_weights = mean_weights[mean_weights > 0]

# --- Calculate Benchmark Data (Moved outside the if statements) ---
# Calculate portfolio daily returns
portfolio_returns = (daily_returns * weights_df.shift(1)).sum(axis=1)

# Calculate cumulative returns
portfolio_cum_returns = (1 + portfolio_returns).cumprod()

# Download MSCI World ETF data
msci_world = yf.download('URTH', start=portfolio_returns.index.min(), end=end_date)['Close']
msci_world = msci_world.fillna(method='ffill').dropna()
msci_world.index = msci_world.index.tz_localize(None)
msci_world_returns = msci_world.pct_change().fillna(0)
msci_world_cum_returns = (1 + msci_world_returns).cumprod()

# Download SPY ETF data
spy = yf.download('SPY', start=portfolio_returns.index.min(), end=end_date)['Close']
spy = spy.fillna(method='ffill').dropna()
spy.index = spy.index.tz_localize(None)
spy_returns = spy.pct_change().fillna(0)
spy_cum_returns = (1 + spy_returns).cumprod()

# Calculate rolling window size
window_size = 252  # One year of trading days

# Calculate rolling drawdown function
def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

# Now proceed with the selected section
if selected_section == 'Methodology':
    st.markdown('---')
    # Methodology
    st.markdown('''
    ## Methodology

    1. **Economic Phase Identification**:
       - The Diffusion Index (DI) is calculated monthly and categorized into four phases:
         - **Recovery**: DI < 0.5 and rising.
         - **Expansion**: DI ≥ 0.5 and rising.
         - **Slowdown**: DI ≥ 0.5 and falling.
         - **Contraction**: DI < 0.5 and falling.

    2. **ETF Selection and Portfolio Construction**:
       - ETFs are mapped to factors, and top performers are selected for each phase.
       - Portfolios are constructed using equal weights for the top 3 ETFs and rebalanced monthly based on phase transitions.

    3. **Performance Benchmarking**:
       - Portfolio performance is compared to the MSCI World ETF (URTH) and SPY using cumulative returns, Sharpe Ratios, and drawdowns.
    ''')

    st.markdown('### Economic Phases Data')
    st.dataframe(pivot_data[['DI', 'DI_change', 'Phase']].tail(15).style.background_gradient(cmap='Blues'), use_container_width=True)

    # Plot the Diffusion Index
    st.markdown('### OECD CLI Diffusion Index')
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=pivot_data.index,
            y=pivot_data['DI'],
            mode='lines',
            name='Diffusion Index',
            line=dict(color='#1f4e79', width=2)
        )
    )
    
    # Add a horizontal line at 0.5
    fig.add_hline(
        y=0.5,
        line_dash='dash',
        line_color='red',
        annotation_text='Threshold (0.5)',
        annotation_position='top left'
    )
    
    fig.update_layout(
        title='OECD CLI Diffusion Index',
        xaxis_title='Date',
        yaxis_title='Diffusion Index',
        font=dict(size=14),
        hovermode='x unified', 
        legend=dict(x=0.01, y=0.99)
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Portfolio Construction':
    st.markdown('---')
    # Factor Performance by Phase
    st.markdown('''
    ## Factor Performance by Phase

    Different factors exhibit unique performance characteristics across economic phases:
    - **Recovery**: Momentum and Quality outperform as markets rebound.
    - **Expansion**: Size and Quality thrive in growth-driven environments.
    - **Slowdown**: Low Volatility provides stability during periods of market uncertainty.
    - **Contraction**: High Dividend preserves capital during widespread economic decline.

    ### Key Insights:
    - **Factor Performance During Economic Phases** illustrates the dominance of specific factors during each phase.
    - Momentum and Multifactor strategies excel in Recovery phases, while defensive factors like Low Volatility perform well during Slowdowns.
    ''')

    # Display the factor performance
    st.markdown('### Factor Performance Data')
    st.dataframe(factor_performance.style.background_gradient(cmap='Blues'), use_container_width=True)

    # Plot heatmap of factor performance
    st.markdown('### Factor Performance During Economic Phases')
    factor_performance_t = factor_performance.T
    plt.figure(figsize=(12, 8))
    sns.heatmap(factor_performance_t * 100, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Return (%)'})
    plt.xlabel('Factor')
    plt.ylabel('Economic Phase')
    st.pyplot(plt.gcf())

    # Display top ETFs per phase
    st.markdown('---')
    st.markdown('''
    ## Portfolio Construction

    For each economic phase, the top 3 ETFs are selected based on cumulative returns within that phase.

    ### Top ETFs During Each Phase:
    ''')

    for phase in unique_phases:
        st.markdown(f"**{phase} Phase**:")
        top_etfs = best_etfs_per_phase.get(phase, [])
        for etf in top_etfs:
            st.write(f"- {etf}")

    # Resample weights to monthly frequency
    weights_monthly = weights_df.resample('M').first()

    st.markdown('### Last Portfolio Weights')
    st.dataframe(weights_monthly.tail().style.background_gradient(cmap='Blues'), use_container_width=True)

    # Align dates
    common_index = portfolio_cum_returns.index.intersection(msci_world_cum_returns.index).intersection(spy_cum_returns.index)
    portfolio_cum_returns_aligned = portfolio_cum_returns.loc[common_index]
    msci_world_cum_returns_aligned = msci_world_cum_returns.loc[common_index]
    spy_cum_returns_aligned = spy_cum_returns.loc[common_index]

    #flatten
    msci_world_cum_returns_aligned = msci_world_cum_returns_aligned.squeeze()
    spy_cum_returns_aligned = spy_cum_returns_aligned.squeeze()


    # Plot Portfolio vs. SPY and MSCI World ETF
    st.markdown('### Portfolio Performance vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_cum_returns_aligned.index, y=portfolio_cum_returns_aligned, mode='lines', name='Dynamic Portfolio', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=spy_cum_returns_aligned.index, y=spy_cum_returns_aligned, mode='lines', name='SPY ETF', line=dict(dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=msci_world_cum_returns_aligned.index, y=msci_world_cum_returns_aligned, mode='lines', name='MSCI World ETF (URTH)', line=dict(dash='dot', width=2)))
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        width=1200,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio Allocations
    st.markdown('---')
    st.markdown('''
    ## Portfolio Allocations

    The strategy dynamically adjusts allocations to align with economic conditions.

    ### Portfolio Allocations Over Time
    ''')

    # Resample weights to monthly
    weights_monthly = weights_over_time.resample('M').first()

    # Ensure the latest date is included in the monthly data
    if latest_date not in weights_monthly.index:
        latest_month_end = latest_date.replace(day=1) + MonthEnd(1)
        weights_monthly.loc[latest_month_end] = weights_over_time.loc[latest_date]

    # Create subplots with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08,
        subplot_titles=(
            'Portfolio Allocations Over Time (Monthly)',
            f'Current Portfolio Allocation as of {latest_date.strftime("%Y-%m-%d")}',
            'Mean Portfolio Allocation Over Entire Period'
        ),
        specs=[
            [{"type": "xy"}],
            [{"type": "domain"}],
            [{"type": "domain"}]
        ]
    )
    
    # Add the stacked area chart to the first subplot
    for etf in etfs_in_portfolio:
        fig.add_trace(
            go.Scatter(
                x=weights_monthly.index,
                y=weights_monthly[etf],
                mode='lines',
                stackgroup='one',
                name=etf
            ),
            row=1, col=1
        )
    
    # Update the x-axis to show monthly ticks
    fig.update_xaxes(
        row=1, col=1,
        tickformat='%Y-%m',
        tickangle=45,
        nticks=20
    )
    
    # Add the current allocation pie chart to the second subplot
    fig.add_trace(
        go.Pie(
            labels=current_weights.index,
            values=current_weights.values,
            name='Current Allocation',
            textinfo='percent+label',
            hole=0.4,
            sort=False
        ),
        row=2, col=1
    )
    
    # Add mean allocation pie chart to the third subplot
    fig.add_trace(
        go.Pie(
            labels=mean_weights.index,
            values=mean_weights.values,
            name='Mean Allocation',
            textinfo='percent+label',
            hole=0.4,
            sort=False
        ),
        row=3, col=1
    )
    
    # Update the layout
    fig.update_layout(
        height=1500,
        showlegend=True,
        hovermode='x unified',
        legend_title='ETF',
        font=dict(size=14)
    )
    
    fig.update_yaxes(title_text='Portfolio Weight', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Sharpe Ratios
    st.markdown('---')
    st.markdown('''
    ## Rolling Sharpe Ratio Comparison

    This section compares the rolling Sharpe Ratios of the Dynamic Portfolio and the MSCI World ETF.
    ''')

    # Calculate rolling Sharpe Ratios
    portfolio_rolling_sharpe = portfolio_returns.rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )

    msci_rolling_sharpe = msci_world_returns.loc[portfolio_returns.index].rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )

    # Convert both to a Series if they happen to be (n,1) DataFrames:
    portfolio_rolling_sharpe = portfolio_rolling_sharpe.squeeze()
    msci_rolling_sharpe = msci_rolling_sharpe.squeeze()

    # Plot rolling Sharpe Ratios
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_rolling_sharpe.index, y=portfolio_rolling_sharpe,
        mode='lines', name='Dynamic Portfolio', line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_rolling_sharpe.index, y=msci_rolling_sharpe,
        mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        title='Rolling Sharpe Ratio',
        xaxis_title='Date',
        yaxis_title='Rolling Sharpe Ratio',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Drawdown Graph
    st.markdown('---')
    st.markdown('''
    ## Rolling Drawdown Comparison

    This section compares the rolling drawdowns of the Dynamic Portfolio and the MSCI World ETF.
    ''')

    # Calculate rolling drawdowns
    portfolio_drawdown = calculate_drawdown(portfolio_cum_returns)
    msci_drawdown = calculate_drawdown(msci_world_cum_returns_aligned)

    # Plot rolling drawdowns
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_drawdown.index, y=portfolio_drawdown,
        mode='lines', name='Dynamic Portfolio', line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_drawdown.index, y=msci_drawdown,
        mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        title='Rolling Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annual Percentage Returns Bar Graph
    st.markdown('---')
    st.markdown('''
    ## Annual Percentage Returns

    The bar chart below displays the annual percentage returns of the Dynamic Portfolio compared to the MSCI World ETF and SPY ETF.
    ''')

    # Calculate annual returns
    portfolio_annual_returns = (1 + portfolio_returns).resample('Y').prod() - 1
    msci_annual_returns = (1 + msci_world_returns).resample('Y').prod() - 1
    spy_annual_returns = (1 + spy_returns).resample('Y').prod() - 1

    # Ensure they are Series
    portfolio_annual_returns = portfolio_annual_returns.squeeze()
    msci_annual_returns = msci_annual_returns.squeeze()
    spy_annual_returns = spy_annual_returns.squeeze()

    # Combine into a DataFrame
    annual_returns = pd.DataFrame({
        'Dynamic Portfolio': portfolio_annual_returns,
        'MSCI World ETF': msci_annual_returns,
        'SPY ETF': spy_annual_returns
    })

    # Set index name to 'Year' before resetting index
    annual_returns.index = annual_returns.index.year
    annual_returns.index.name = 'Year'

    # Melt for plotting
    annual_returns_melted = annual_returns.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')

    # Plot bar chart
    fig = px.bar(
        annual_returns_melted,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return',
        labels={'Annual Return': 'Annual Return (%)'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        title='Annual Percentage Returns',
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Conclusion
    st.markdown('---')

elif selected_section == 'Mean Portfolio Evolution':
    st.markdown('---')
    st.markdown('''
    ## Mean Portfolio Evolution

    This section explores the performance of a **Mean Portfolio** that uses the cumulative average allocation of the dynamic portfolio up to each rebalancing date. The portfolio is rebalanced monthly to maintain the evolving mean allocation.

    ### Construction of the Mean Portfolio:
    - **Cumulative Mean Allocation**: At each month, calculate the average of the dynamic portfolio's weights from inception up to that month.
    - **Rebalancing**: The portfolio is rebalanced monthly using these cumulative average weights.
    ''')

    # Resample weights_df to monthly frequency
    weights_monthly = weights_df.resample('M').first()
    
    # Create a DataFrame to store mean portfolio weights
    mean_weights_monthly = pd.DataFrame(index=weights_monthly.index, columns=weights_monthly.columns)
    
    # Calculate cumulative average weights at each month
    for i in range(len(weights_monthly)):
        weights_up_to_month = weights_monthly.iloc[:i+1]
        cumulative_avg_weights = weights_up_to_month.mean()
        mean_weights_monthly.iloc[i] = cumulative_avg_weights
    
    # Forward-fill 
    mean_weights_df = mean_weights_monthly.reindex(daily_returns.index, method='ffill').fillna(0)
    
    # Calculate mean portfolio daily returns
    mean_portfolio_returns = (daily_returns * mean_weights_df.shift(1)).sum(axis=1)
    
    # Calculate cumulative returns
    mean_portfolio_cum_returns = (1 + mean_portfolio_returns).cumprod()
    
    # Align dates with benchmarks
    common_index = mean_portfolio_cum_returns.index.intersection(msci_world_cum_returns.index).intersection(spy_cum_returns.index)
    mean_portfolio_cum_returns_aligned = mean_portfolio_cum_returns.loc[common_index]
    msci_world_cum_returns_aligned = msci_world_cum_returns.loc[common_index]
    spy_cum_returns_aligned = spy_cum_returns.loc[common_index]
    
    # Fixed variable name here
    mean_portfolio_cum_returns_aligned = mean_portfolio_cum_returns_aligned.squeeze()
    spy_cum_returns_aligned = spy_cum_returns_aligned.squeeze()
    msci_world_cum_returns_aligned = msci_world_cum_returns_aligned.squeeze()
    
    # Plot Mean Portfolio vs. SPY and MSCI World ETF
    st.markdown('### Mean Portfolio Performance vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_portfolio_cum_returns_aligned.index, 
                            y=mean_portfolio_cum_returns_aligned, 
                            mode='lines', 
                            name='Mean Portfolio', 
                            line=dict(width=3)))
    fig.add_trace(go.Scatter(x=spy_cum_returns_aligned.index, 
                            y=spy_cum_returns_aligned, 
                            mode='lines', 
                            name='SPY ETF', 
                            line=dict(dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=msci_world_cum_returns_aligned.index, 
                            y=msci_world_cum_returns_aligned, 
                            mode='lines', 
                            name='MSCI World ETF (URTH)', 
                            line=dict(dash='dot', width=2)))
    fig.update_layout(
        title='Mean Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        width=1200,
        height=600,
        legend=dict(x=0.02, y=0.98),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio Allocations
    st.markdown('### Mean Portfolio Weights Sample (Monthly Allocations)')
    st.dataframe(mean_weights_monthly.head().style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Plot the evolving mean allocations over time
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Evolving Mean Portfolio Allocations Over Time',),
        specs=[[{"type": "xy"}]]
    )
    
    for etf in mean_weights_monthly.columns[mean_weights_monthly.sum() > 0]:
        fig.add_trace(
            go.Scatter(
                x=mean_weights_monthly.index,
                y=mean_weights_monthly[etf],
                mode='lines',
                stackgroup='one',
                name=etf
            ),
            row=1, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend_title='ETF',
        font=dict(size=14)
    )
    
    fig.update_yaxes(title_text='Mean Portfolio Weight', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling Sharpe Ratios
    st.markdown('---')
    st.markdown('''
    ## Rolling Sharpe Ratio Comparison
    
    This section compares the rolling Sharpe Ratios of the Mean Portfolio and the MSCI World ETF.
    ''')
    
    # Use common_index instead of undefined mean_common_idx
    mean_returns_aligned = mean_portfolio_returns.loc[common_index]
    msci_returns_aligned = msci_world_returns.loc[common_index]
    
    # Define window_size (add this if not defined)
    window_size = 252  # Typical 1-year window for daily data
    
    # Calculate rolling Sharpe Ratios
    mean_portfolio_rolling_sharpe = mean_portfolio_returns.rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )
    
    msci_rolling_sharpe_aligned = msci_world_returns.loc[mean_portfolio_returns.index].rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )
    
    mean_portfolio_rolling_sharpe = mean_portfolio_rolling_sharpe.squeeze()
    msci_rolling_sharpe_aligned = msci_rolling_sharpe_aligned.squeeze()
    
    # Plot rolling Sharpe Ratios
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_portfolio_rolling_sharpe.index, 
        y=mean_portfolio_rolling_sharpe,
        mode='lines', 
        name='Mean Portfolio', 
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_rolling_sharpe_aligned.index, 
        y=msci_rolling_sharpe_aligned,
        mode='lines', 
        name='MSCI World ETF (URTH)', 
        line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        title='Rolling Sharpe Ratio Comparison',
        xaxis_title='Date',
        yaxis_title='Rolling Sharpe Ratio',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Drawdown Graph
    st.markdown('---')
    st.markdown('''
    ## Rolling Drawdown Comparison

    This section shows the rolling drawdown of the Mean Portfolio.
    ''')

    # Calculate rolling drawdowns
    mean_portfolio_drawdown = calculate_drawdown(mean_portfolio_cum_returns_aligned)
    msci_drawdown_aligned = calculate_drawdown(msci_world_cum_returns_aligned)

    # Plot rolling drawdowns
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_portfolio_drawdown.index, y=mean_portfolio_drawdown,
        mode='lines', name='Mean Portfolio', line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_drawdown_aligned.index, y=msci_drawdown_aligned,
        mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        title='Rolling Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annual Percentage Returns Bar Graph
    st.markdown('---')
    st.markdown('''
    ## Annual Percentage Returns

    The bar chart below displays the annual percentage returns of the Mean Portfolio compared to the MSCI World ETF and SPY ETF.
    ''')

    # Calculate annual returns
    mean_portfolio_annual_returns = (1 + mean_portfolio_returns).resample('Y').prod() - 1
    msci_annual_returns_aligned = (1 + msci_world_returns.loc[mean_portfolio_returns.index]).resample('Y').prod() - 1
    spy_annual_returns_aligned = (1 + spy_returns.loc[mean_portfolio_returns.index]).resample('Y').prod() - 1

    # Ensure they are Series
    mean_portfolio_annual_returns = mean_portfolio_annual_returns.squeeze()
    msci_annual_returns_aligned = msci_annual_returns_aligned.squeeze()
    spy_annual_returns_aligned = spy_annual_returns_aligned.squeeze()

    # Combine into a DataFrame
    annual_returns_mean_portfolio = pd.DataFrame({
        'Mean Portfolio': mean_portfolio_annual_returns,
        'MSCI World ETF': msci_annual_returns_aligned,
        'SPY ETF': spy_annual_returns_aligned
    })

    # Set index name to 'Year' before resetting index
    annual_returns_mean_portfolio.index = annual_returns_mean_portfolio.index.year
    annual_returns_mean_portfolio.index.name = 'Year'

    # Melt for plotting
    annual_returns_melted_mean_portfolio = annual_returns_mean_portfolio.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')

    # Plot bar chart
    fig = px.bar(
        annual_returns_melted_mean_portfolio,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return',
        labels={'Annual Return': 'Annual Return (%)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        title='Annual Percentage Returns',
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Conclusion
    st.markdown('---')
