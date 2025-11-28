import streamlit as st

from pages import analyse, backtest, screeners, portfolio, risk

PAGES = {
    "Accueil": None,
    "Analyse & graphique": analyse.run,
    "Backtest Stratégies": backtest.run,
    "Screeners": screeners.run,
    "Portefeuille": portfolio.run,
    "Risk Management": risk.run
}

st.set_page_config(page_title="Dashboard Trading Quant", layout="wide")
st.sidebar.image("assets/logo_dashboard.png", width=180)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", list(PAGES.keys()))

if page == "Accueil":
    st.title("Bienvenue sur la plateforme quant professionnelle")
    st.markdown("""
    - Navigation par onglets/pages sur la gauche.
    - Solutions avancées d’analyse, backtesting, gestion risque.
    - Interface moderne, objets **stratégie** (momentum, bollinger, mean reversion...).
    - Plus de 5 modules prêts à l'emploi pour des cas d'usage réel/bancaire/asset management.
    """)
    st.image("assets/dashboard_cover.jpg", use_column_width=True)
else:
    # Délégation à la page correspondante
    PAGES[page]()
st.write("---")
st.caption("Powered by Streamlit, yfinance, Plotly, numpy, pandas. Design modulaire et professionnel.")
