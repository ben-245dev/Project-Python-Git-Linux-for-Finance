import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf

from backend.orders import init_db, place_order, get_transaction_history, get_portfolio_positions, get_current_prices

def page_paper_trading():
    # Initialisation de la DB au chargement de la page
    init_db()

    st.title("🎮 Simulateur de Trading (Paper Trading)")
    st.markdown("---")

    # ---------------------------------------------------------
    # 1. Passage d'Ordre
    # ---------------------------------------------------------
    with st.expander("📝 Passer un nouvel ordre", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.text_input("Ticker", "AAPL").upper()
        with col2:
            action = st.selectbox("Action", ["ACHAT", "VENTE"])
        with col3:
            qty = st.number_input("Quantité", min_value=0.01, value=1.0, step=1.0)
        
        # Récupération automatique du prix indicatif pour l'UX
        current_market_price = 0.0
        if ticker:
            try:
                # On essaie de récupérer le dernier prix connu (même si marché fermé)
                t = yf.Ticker(ticker)
                # fast_info est souvent plus rapide que history pour le last_price
                if hasattr(t, 'fast_info') and t.fast_info.last_price:
                     current_market_price = t.fast_info.last_price
                else:
                    hist = t.history(period="5d") # 5d pour couvrir le weekend
                    if not hist.empty:
                        current_market_price = hist["Close"].iloc[-1]
            except:
                pass

        with col4:
            price_input = st.number_input("Prix d'exécution ($)", value=float(current_market_price), min_value=0.0, format="%.2f")

        if st.button("Valider l'Ordre"):
            if price_input > 0:
                # Appel au backend qui retourne (Succès, Message)
                success, message = place_order(ticker, action, qty, price_input)
                
                if success:
                    st.success(f"✅ {message} ({action} {qty} {ticker} @ {price_input:.2f} $)")
                    st.rerun() # Rafraîchir la page pour mettre à jour le tableau
                else:
                    st.error(f"⛔ Ordre rejeté : {message}")
            else:
                st.error("Le prix doit être supérieur à 0.")

    # ---------------------------------------------------------
    # 2. Portefeuille Actuel & P&L
    # ---------------------------------------------------------
    st.subheader("💼 Mon Portefeuille")
    
    positions_df = get_portfolio_positions()

    if not positions_df.empty:
        # Récupération des prix actuels pour calculer la PV/MV
        tickers = positions_df["Ticker"].tolist()
        
        with st.spinner("Mise à jour des valorisations..."):
            live_prices = get_current_prices(tickers)

        # Calculs des métriques
        # On map le prix live. Si pas trouvé, on garde 0 (ou on pourrait garder le PRU pour éviter le -100%)
        positions_df["Prix Actuel"] = positions_df["Ticker"].map(live_prices).fillna(0)
        
        # Sécurité visuelle : si le prix actuel est 0 (bug API), on évite d'afficher une perte de 100%
        # On peut choisir d'utiliser le PRU comme fallback temporaire ou laisser 0
        
        positions_df["Valeur Actuelle"] = positions_df["Quantité"] * positions_df["Prix Actuel"]
        positions_df["P&L ($)"] = positions_df["Valeur Actuelle"] - positions_df["Investi"]
        
        # Évite la division par zéro
        positions_df["P&L (%)"] = positions_df.apply(
            lambda x: (x["P&L ($)"] / x["Investi"] * 100) if x["Investi"] > 0 else 0, axis=1
        )

        # Totaux
        total_invested = positions_df["Investi"].sum()
        total_value = positions_df["Valeur Actuelle"].sum()
        total_pnl = total_value - total_invested
        
        # Affichage Métriques Globales
        m1, m2, m3 = st.columns(3)
        m1.metric("Valeur Totale", f"{total_value:,.2f} $")
        m2.metric("Total Investi", f"{total_invested:,.2f} $")
        m3.metric("P&L Latent Global", f"{total_pnl:+,.2f} $", 
                  delta_color="normal" if total_pnl >= 0 else "inverse")

        st.markdown("### Détail des positions")
        
        # Styling du DataFrame (Couleurs pour P&L)
        def color_pnl(val):
            color = '#00ff00' if val > 0 else '#ff4b4b' if val < 0 else 'white'
            return f'color: {color}'

        # Affichage du tableau formaté
        st.dataframe(
            positions_df.style.format({
                "Quantité": "{:.4f}",
                "PRU (Prix Moyen)": "{:.2f} $",
                "Investi": "{:.2f} $",
                "Prix Actuel": "{:.2f} $",
                "Valeur Actuelle": "{:.2f} $",
                "P&L ($)": "{:+.2f} $",
                "P&L (%)": "{:+.2f} %"
            }).map(color_pnl, subset=['P&L ($)', 'P&L (%)']),
            use_container_width=True
        )

        # Graphique Répartition (Pie Chart)
        if total_value > 0:
            fig = go.Figure(data=[go.Pie(labels=positions_df["Ticker"], values=positions_df["Valeur Actuelle"], hole=.4)])
            fig.update_layout(title="Allocation d'actifs", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Votre portefeuille est vide. Passez votre premier ordre ci-dessus pour commencer !")

    # ---------------------------------------------------------
    # 3. Historique
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("📜 Historique des Transactions"):
        history_df = get_transaction_history()
        st.dataframe(history_df, use_container_width=True)