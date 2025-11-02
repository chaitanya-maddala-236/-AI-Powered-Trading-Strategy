import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AI Trading Strategy",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Custom CSS for centered, beautiful UI
st.markdown("""
<style>
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Center everything */
    .main .block-container {
        max-width: 1400px;
        padding-left: 5rem;
        padding-right: 5rem;
        padding-top: 2rem;
    }
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container with white background */
    .main-content {
        background: white;
        border-radius: 30px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto;
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 2rem 2rem 2rem;
        margin-bottom: 3rem;
    }
    
    .hero h1 {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    
    .hero p {
        font-size: 1.4rem;
        color: #666;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Control cards */
    .control-card {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .control-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.8) !important;
    }
    
    /* Big action button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 1.2rem 3rem;
        border-radius: 50px;
        font-size: 1.3rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Input styling */
    .stSelectbox, .stSlider, .stDateInput {
        background: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #f8f9fa;
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .info-box h3 {
        margin-top: 0;
        color: white;
        font-size: 1.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .feature-card h3 {
        color: #667eea;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Download button */
    .download-btn {
        background: #28a745;
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .download-btn:hover {
        background: #218838;
        transform: scale(1.05);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #667eea;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 10px;
    }
    
    .stError {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main content wrapper
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class='hero'>
    <h1>🤖 AI Trading Strategy</h1>
    <p>Machine Learning Meets Quantitative Finance</p>
</div>
""", unsafe_allow_html=True)

# Predefined stock lists (NO WIKIPEDIA NEEDED!)
POPULAR_STOCKS = {
    "🚀 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM"],
    "💼 Blue Chips": ["JPM", "JNJ", "PG", "WMT", "V", "UNH", "HD", "DIS", "BA", "CAT"],
    "📈 Growth Stocks": ["TSLA", "NVDA", "AMD", "SHOP", "SQ", "ROKU", "PLTR", "COIN", "SNOW", "NET"],
    "🏆 S&P 500 Top 50": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "V", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO",
        "PEP", "COST", "AVGO", "LLY", "TMO", "MCD", "CSCO", "ACN", "ABT", "DHR",
        "NKE", "CRM", "TXN", "PM", "NEE", "ORCL", "WFC", "VZ", "BMY", "UPS",
        "MS", "RTX", "HON", "QCOM", "INTU", "LOW", "AMGN", "T", "IBM", "CAT"
    ],
    "💰 Financial Sector": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "⚡ Energy Sector": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY"],
    "🏥 Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "ABT", "DHR", "BMY", "LLY"]
}

# Control Panel
st.markdown('<p class="section-title">⚙️ Configure Your Strategy</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Basic Settings", "⚙️ Advanced Settings", "📁 Data Management"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        stock_universe = st.selectbox(
            "📊 Stock Universe",
            list(POPULAR_STOCKS.keys()),
            index=3,
            help="Choose a predefined set of stocks"
        )
        
        n_clusters = st.slider(
            "🎯 Number of Clusters",
            min_value=3,
            max_value=10,
            value=5,
            help="More clusters = finer segmentation"
        )
        
        start_date = st.date_input(
            "📅 Start Date",
            datetime(2020, 1, 1),
            help="Beginning of backtest period"
        )
    
    with col2:
        top_n_stocks = st.slider(
            "💼 Portfolio Size",
            min_value=5,
            max_value=30,
            value=15,
            help="Number of stocks to hold"
        )
        
        lookback_period = st.slider(
            "📈 Lookback Period (days)",
            min_value=100,
            max_value=500,
            value=252,
            help="Historical data window"
        )
        
        end_date = st.date_input(
            "📅 End Date",
            datetime(2024, 1, 1),
            help="End of backtest period"
        )

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        rebalance_freq = st.slider(
            "🔄 Rebalance Frequency (days)",
            min_value=30,
            max_value=180,
            value=90,
            help="How often to adjust portfolio"
        )
    
    with col2:
        min_data_threshold = st.slider(
            "📊 Min Data Completeness (%)",
            min_value=50,
            max_value=100,
            value=80,
            help="Filter stocks with insufficient data"
        )
    
    # Cache clearing button
    st.markdown("---")
    if st.button("🗑️ Clear Data Cache", help="Clear yfinance cache if experiencing errors"):
        clear_yfinance_cache()
        st.cache_data.clear()
        st.success("✅ Cache cleared! Try running again.")
        st.rerun()

with tab3:
    data_mode = st.radio(
        "📁 Data Source",
        ["Download Fresh Data", "Upload CSV File"],
        help="Choose how to get stock data"
    )
    
    if data_mode == "Upload CSV File":
        st.markdown("### 📤 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file with stock prices",
            type=['csv'],
            help="Format: Date (index) | Stock1 | Stock2 | ..."
        )
        
        if uploaded_file:
            try:
                uploaded_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                st.success(f"✅ Loaded {len(uploaded_data.columns)} stocks, {len(uploaded_data)} days")
                
                # Preview
                with st.expander("👀 Preview Data"):
                    st.dataframe(uploaded_data.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info(f"📊 Will download {len(POPULAR_STOCKS[stock_universe])} stocks from Yahoo Finance")

# Big action button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_backtest = st.button("🚀 RUN BACKTEST NOW", use_container_width=True)

if run_backtest:
    st.session_state.run = True

# Functions
def clear_yfinance_cache():
    """Clear yfinance cache to prevent database locks"""
    import shutil
    import os
    
    cache_dir = os.path.expanduser('~/.cache/py-yfinance')
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except:
            pass

@st.cache_data(show_spinner=False, ttl=3600)
def download_stock_data(tickers, start, end):
    """Download stock data with proper error handling and retry logic"""
    import time
    
    # Method 1: Try batch download first (faster)
    try:
        data = yf.download(
            tickers, 
            start=start, 
            end=end, 
            progress=False,
            threads=False,
            group_by='column'
        )
        
        if len(tickers) == 1:
            data = data['Adj Close'].to_frame(name=tickers[0])
        else:
            data = data['Adj Close']
        
        if len(data) > 0:
            data = data.dropna(axis=1, thresh=len(data)*0.8)
            return data
    except:
        pass
    
    # Method 2: Download one by one (slower but more reliable)
    st.info("⏳ Using sequential download (more reliable)...")
    
    all_data = {}
    failed = []
    
    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=start, end=end, progress=False)
            if 'Adj Close' in ticker_data.columns:
                all_data[ticker] = ticker_data['Adj Close']
            elif len(ticker_data.columns) > 0:
                all_data[ticker] = ticker_data.iloc[:, 0]
            
            time.sleep(0.1)  # Small delay to avoid rate limits
            
        except Exception as e:
            failed.append(ticker)
            continue
    
    if failed:
        st.warning(f"⚠️ Failed to download: {', '.join(failed[:5])}")
    
    if all_data:
        data = pd.DataFrame(all_data)
        data = data.dropna(axis=1, thresh=len(data)*0.8)
        return data
    
    return None

def calculate_features(data):
    """Calculate technical features"""
    features_dict = {}
    
    for ticker in data.columns:
        prices = data[ticker].dropna()
        if len(prices) < 100:
            continue
            
        returns = prices.pct_change()
        
        features_dict[ticker] = {
            'mean_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'momentum_1m': prices.pct_change(21).iloc[-1] if len(prices) > 21 else 0,
            'momentum_3m': prices.pct_change(63).iloc[-1] if len(prices) > 63 else 0,
            'momentum_6m': prices.pct_change(126).iloc[-1] if len(prices) > 126 else 0,
            'momentum_12m': prices.pct_change(252).iloc[-1] if len(prices) > 252 else 0,
            'rsi': calculate_rsi(prices),
            'max_drawdown': calculate_max_drawdown(prices)
        }
    
    return pd.DataFrame(features_dict).T

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50

def calculate_max_drawdown(prices):
    """Calculate max drawdown"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def perform_clustering(features, n_clusters):
    """Perform K-Means clustering"""
    features_clean = features.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_clean)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    features_clean['cluster'] = clusters
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return features_clean, X_pca, kmeans, scaler, pca

def calculate_portfolio_performance(data, selected_stocks):
    """Calculate returns"""
    portfolio_data = data[selected_stocks].dropna()
    portfolio_returns = portfolio_data.pct_change().mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return portfolio_returns, cumulative_returns

# Main execution
if 'run' in st.session_state and st.session_state.run:
    try:
        # Progress
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Get data
        if data_mode == "Upload CSV File" and 'uploaded_data' in locals():
            data = uploaded_data
            status_text.markdown("### ✅ Using uploaded data")
        else:
            tickers = POPULAR_STOCKS[stock_universe]
            status_text.markdown(f"### 📊 Downloading {len(tickers)} stocks...")
            progress_bar.progress(20)
            data = download_stock_data(tickers, start_date, end_date)
        
        if data is None or len(data) < 50:
            st.error("❌ Insufficient data. Please adjust parameters.")
            st.stop()
        
        status_text.markdown(f"### ✅ Processing {len(data.columns)} stocks")
        progress_bar.progress(40)
        
        # Calculate features
        features = calculate_features(data)
        progress_bar.progress(60)
        
        # Clustering
        status_text.markdown("### 🤖 Running K-Means clustering...")
        features_clustered, X_pca, kmeans, scaler, pca = perform_clustering(features, n_clusters)
        progress_bar.progress(80)
        
        # Select best cluster
        cluster_stats = features_clustered.groupby('cluster').agg({
            'mean_return': 'mean',
            'volatility': 'mean',
            'sharpe_ratio': 'mean',
            'momentum_12m': 'mean'
        })
        cluster_stats['count'] = features_clustered.groupby('cluster').size()
        best_cluster = cluster_stats['sharpe_ratio'].idxmax()
        
        cluster_stocks = features_clustered[features_clustered['cluster'] == best_cluster]
        top_stocks = cluster_stocks.nlargest(top_n_stocks, 'sharpe_ratio').index.tolist()
        
        # Portfolio performance
        portfolio_returns, cumulative_returns = calculate_portfolio_performance(data, top_stocks)
        
        # Benchmark
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False, threads=False)['Adj Close']
        sp500_returns = sp500.pct_change()
        sp500_cumulative = (1 + sp500_returns).cumprod()
        
        # Align
        common_dates = portfolio_returns.index.intersection(sp500_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        sp500_returns = sp500_returns.loc[common_dates]
        cumulative_returns = cumulative_returns.loc[common_dates]
        sp500_cumulative = sp500_cumulative.loc[common_dates]
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success("🎉 Analysis Complete!")
        
        # RESULTS
        st.markdown('<p class="section-title">📊 Performance Overview</p>', unsafe_allow_html=True)
        
        # Metrics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sp500_total_return = (sp500_cumulative.iloc[-1] - 1) * 100
        ann_return = portfolio_returns.mean() * 252 * 100
        ann_vol = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        sp500_ann_return = sp500_returns.mean() * 252 * 100
        sp500_sharpe = (sp500_returns.mean() / sp500_returns.std()) * np.sqrt(252)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%", 
                     delta=f"{total_return - sp500_total_return:.2f}% vs S&P500")
        with col2:
            st.metric("Annual Return", f"{ann_return:.2f}%",
                     delta=f"{ann_return - sp500_ann_return:.2f}% vs S&P500")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe:.3f}",
                     delta=f"{sharpe - sp500_sharpe:.3f} vs S&P500")
        with col4:
            st.metric("Volatility", f"{ann_vol:.2f}%")
        
        # Chart
        st.markdown('<p class="section-title">💹 Cumulative Returns</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index, y=(cumulative_returns - 1) * 100,
            name='AI Strategy', line=dict(color='#667eea', width=3), fill='tonexty'
        ))
        fig.add_trace(go.Scatter(
            x=sp500_cumulative.index, y=(sp500_cumulative - 1) * 100,
            name='S&P 500', line=dict(color='#764ba2', width=3, dash='dash')
        ))
        fig.update_layout(
            height=500, xaxis_title='Date', yaxis_title='Return (%)',
            hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Cluster Statistics")
            cluster_display = cluster_stats.round(4)
            cluster_display.columns = ['Return', 'Vol', 'Sharpe', 'Mom', 'Count']
            st.dataframe(cluster_display.style.background_gradient(cmap='RdYlGn', subset=['Sharpe']),
                        use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 PCA Visualization")
            fig = go.Figure()
            for cluster in range(n_clusters):
                mask = features_clustered['cluster'] == cluster
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0], y=X_pca[mask, 1],
                    mode='markers', name=f'Cluster {cluster}',
                    marker=dict(size=10, opacity=0.7)
                ))
            fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Selected stocks
        st.markdown(f"### 📋 Selected Portfolio (Cluster {best_cluster})")
        selected_df = cluster_stocks.loc[top_stocks][['mean_return', 'volatility', 'sharpe_ratio', 'momentum_12m']]
        selected_df.columns = ['Return', 'Volatility', 'Sharpe', '12M Mom']
        st.dataframe((selected_df * 100).round(2).style.background_gradient(cmap='RdYlGn', subset=['Sharpe']),
                    use_container_width=True)
        
        # Download results
        st.markdown("### 📥 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = selected_df.to_csv()
            st.download_button("📄 Download Portfolio CSV", csv, "portfolio.csv", "text/csv")
        
        with col2:
            results_csv = pd.DataFrame({
                'Date': cumulative_returns.index,
                'Strategy Return': (cumulative_returns - 1) * 100,
                'SP500 Return': (sp500_cumulative.loc[common_dates] - 1) * 100
            }).to_csv(index=False)
            st.download_button("📈 Download Returns CSV", results_csv, "returns.csv", "text/csv")
        
        with col3:
            # Save original data
            data_csv = data.to_csv()
            st.download_button("💾 Download Price Data", data_csv, "price_data.csv", "text/csv")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 See error details"):
            st.exception(e)

else:
    # Landing page
    st.markdown("""
    <div class='info-box'>
        <h3>🚀 Ready to Start?</h3>
        <p style='font-size: 1.1rem; margin-bottom: 0;'>
            Configure your strategy above and click <b>"RUN BACKTEST NOW"</b> to unleash the power of AI!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Machine Learning</h3>
            <p>K-Means clustering identifies stocks with similar risk-return profiles automatically</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 9 Indicators</h3>
            <p>Analyzes momentum, volatility, Sharpe ratio, RSI, and drawdown for each stock</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💹 Risk-Adjusted</h3>
            <p>Selects portfolios based on Sharpe ratio for optimal risk-adjusted returns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown('<p class="section-title">📖 Quick Start Guide</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Choose Your Universe
        - Select from 7 predefined stock lists
        - No Wikipedia scraping needed!
        - Includes Tech, Finance, Energy, Healthcare
        
        #### 2️⃣ Set Parameters
        - Number of clusters (3-10)
        - Portfolio size (5-30 stocks)
        - Date range for backtesting
        """)
    
    with col2:
        st.markdown("""
        #### 3️⃣ Run Analysis
        - Click the big button
        - Wait 30-60 seconds
        - Get complete results!
        
        #### 4️⃣ Download Data
        - Export portfolio picks
        - Save performance results
        - Download price data
        """)

# Close main content wrapper
st.markdown('</div>', unsafe_allow_html=True)
