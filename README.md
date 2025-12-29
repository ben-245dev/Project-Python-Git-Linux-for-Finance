📈 Professional Quant Analysis Platform

This platform is a comprehensive quantitative finance application designed to support portfolio managers with real-time market data, backtesting, portfolio optimization, and automated reporting.

Hosted 24/7 on a Linux virtual machine, the system integrates advanced quantitative models, automation via cron jobs, and an interactive Streamlit dashboard.

🔗 Live Dashboard
https://quant-dashboard-rubensbenoit.streamlit.app/

📂 Project Architecture

The repository follows a modular and scalable architecture to ensure clean code, maintainability, and efficient debugging.

├── main.py                 # Main entry point for the Streamlit application
├── backend/                # Quantitative engine
│   ├── strategies.py       # Trading strategies
│   ├── optimization.py    # Portfolio optimization logic
│   └── data_loader.py     # Market data ingestion
├── frontend/               # User interface layer
│   ├── views/              # Dashboard pages
│   └── components.py      # Reusable UI components
├── scripts/
│   └── Daily_report.py    # Automated daily reporting script
├── requirements.txt       # Project dependencies


Key libraries: Pandas, NumPy, yfinance, PyPortfolioOpt, Scikit-learn, Streamlit.

🌐 Dashboard Pages

📊 Market Dashboard

Real-Time Tracking: Live asset prices with interactive time-series visualizations

Data Sourcing: Market data refreshed every 5 minutes using financial APIs (yfinance / Finnhub)

🎮 Paper Trading

Live Simulation: Virtual trading environment to simulate orders with no financial risk

Order Management: Trades are logged using orders.py and paper_trading.db

Performance Monitoring: Real-time P&L computation

🧪 Backtest & Strategy Analysis

Strategy Evaluation: Test quantitative rules on historical market data

Performance Metrics:

Cumulative returns vs. asset prices

Sharpe Ratio

Maximum Drawdown

🧠 Portfolio Optimization

Multivariate Analysis: Portfolio management with 3+ assets simultaneously

Efficient Frontier: Optimal allocation computed using PyPortfolioOpt

Equal Weight

Custom Constraints

Risk Metrics:

Correlation matrices

Diversification impact analysis

⚡ Quant Lab

Advanced Analytics: Research sandbox powered by quant_stats.py

Forecasting Models:

ARIMA

Linear Regression

Objective: Predict short-term price movements and analyze statistical behavior

🛠 Implemented Strategies

The following quantitative strategies are available in backend/strategies.py for backtesting and analysis:

Buy & Hold	Benchmark strategy: buy the asset and hold over the full period
Momentum (SMA Crossover)	Buy/sell signals based on short-term vs. long-term moving average crossovers
RSI (Relative Strength Index)	Mean-reversion strategy detecting overbought (>70) and oversold (<30) conditions
Bollinger Bands	Trades volatility breakouts or mean reversion using dynamic bands
ML-Based Prediction	Uses historical features and models from forecasting.py to predict next-day price direction

🚀 Setup & Automation
Installation
pip install -r requirements.txt
Run the Application
streamlit run main.py

Automated Reporting

A daily performance report is automatically generated on the Linux server using a cron job:
