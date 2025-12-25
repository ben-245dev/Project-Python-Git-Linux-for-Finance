import io
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import pytz
from datetime import datetime
from config import TRADING_TIMEZONES
from backend.data_loader import get_price_series

# ---------------------------------------------------------
# 1. Composants UI (Horloges, Bannières, Boutons)
# ---------------------------------------------------------

def render_trading_clocks():
    """Affiche les horloges des différentes places boursières."""
    st.subheader("Horloges Trading")
    cols = st.columns(len(TRADING_TIMEZONES))
    
    for col, (name, tz_str) in zip(cols, TRADING_TIMEZONES.items()):
        tz = pytz.timezone(tz_str)
        now_local = datetime.now(tz)
        
        # Style CSS simple pour l'affichage
        col.markdown(
            f"""
            <div style="background-color:#1E1E1E; padding:10px; border-radius:5px; text-align:center; border: 1px solid #333;">
                <h5 style="color:#00c3ff; margin:0;">{name}</h5>
                <p style="font-size:20px; font-weight:bold; margin:0;">{now_local.strftime('%H:%M')}</p>
                <small style="color:#888;">{now_local.strftime('%d/%m')}</small>
            </div>
            """, 
            unsafe_allow_html=True
        )

def render_banner(live_data: dict):
    """Affiche un bandeau défilant avec les prix en direct."""
    if not live_data:
        st.warning("Données live indisponibles (Marchés fermés ou API limit)")
        return

    items = []
    for name, v in live_data.items():
        price = v["price"]
        items.append(f"<b>{name}</b>: {price:,.2f}")
    
    text = " &nbsp;&nbsp; | &nbsp;&nbsp; ".join(items)
    
    # Marquee HTML
    st.markdown(
        f"""
        <div style="background-color:#0E1117; border-top:1px solid #333; border-bottom:1px solid #333; padding: 5px;">
            <marquee style="font-size:16px; color:#ffffff;">{text}</marquee>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_export_button(df: pd.DataFrame, filename: str = "export.csv", label="📥 Télécharger CSV"):
    """Bouton générique pour télécharger un DataFrame en CSV."""
    if df.empty:
        return
    csv_buf = io.StringIO()
    df.to_csv(csv_buf)
    st.download_button(
        label=label,
        data=csv_buf.getvalue(),
        file_name=filename,
        mime="text/csv"
    )

# ---------------------------------------------------------
# 2. Composants Graphiques (Charts)
# ---------------------------------------------------------

def render_price_chart(df: pd.DataFrame, ticker: str, chart_type: str = "Courbes"):
    """Graphique simple pour la page d'accueil."""
    if df.empty:
        st.warning("Pas de données à afficher.")
        return
    
    # On utilise get_price_series pour gérer la robustesse des données
    close_data = get_price_series(df, "Close", ticker)
    
    if chart_type == "Bougies":
        # Pour les bougies, on a besoin de Open/High/Low/Close
        # On essaie de récupérer les séries, sinon fallback sur Close
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=get_price_series(df, "Open", ticker),
                high=get_price_series(df, "High", ticker),
                low=get_price_series(df, "Low", ticker),
                close=close_data,
                name=ticker
            )])
        except:
            st.warning("Données OHLC incomplètes pour les bougies, affichage en ligne.")
            fig = go.Figure(data=[go.Scatter(x=df.index, y=close_data, mode="lines", name=ticker)])
    else:
        fig = go.Figure(data=[go.Scatter(
            x=df.index, 
            y=close_data, 
            mode="lines", 
            name=ticker,
            line=dict(color="#00c3ff")
        )])

    fig.update_layout(
        template="plotly_dark", 
        margin=dict(l=10, r=10, t=30, b=10),
        height=400,
        xaxis_title=None
    )
    st.plotly_chart(fig, use_container_width=True)

def render_advanced_chart(df: pd.DataFrame, ticker: str):
    """
    Graphique avancé (Pro) avec Subplots :
    - Row 1: Bougies + Moyennes Mobiles (EMA)
    - Row 2: RSI
    """
    # Création du subplot (Prix en haut, RSI en bas)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=(f'{ticker} Prix', 'RSI (14)'),
        row_heights=[0.7, 0.3]
    )

    # 1. Bougies (Main Chart)
    fig.add_trace(go.Candlestick(
        x=df.index, 
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='OHLC'
    ), row=1, col=1)

    # 2. Moyennes Mobiles (si calculées dans df)
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Fast'], 
            line=dict(color='orange', width=1), 
            name='EMA Fast'
        ), row=1, col=1)
        
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Slow'], 
            line=dict(color='blue', width=1), 
            name='EMA Slow'
        ), row=1, col=1)

    # 3. RSI (Subplot)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI_14'], 
            line=dict(color='purple', width=2), 
            name='RSI'
        ), row=2, col=1)
        
        # Lignes de seuil RSI (30/70)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red", opacity=0.5)
        fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green", opacity=0.5)
        
        # Zone de fond RSI
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_layout(
        template="plotly_dark", 
        height=600, 
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)