import streamlit as st
from streamlit_autorefresh import st_autorefresh
import datetime
from backend.orders import save_user_setting, get_user_setting

st.set_page_config(
    page_title="Quantitative Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Auto-refresh every 5 minutes (300 000 ms) ---
count = st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

from frontend.views.home import page_home
from frontend.views.strategy import page_strategy
from frontend.views.portfolio import page_portfolio
from frontend.views.quant_lab import page_quant_lab
from frontend.views.paper_trading import page_paper_trading

def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/7214/7214197.png", width=50)
    st.sidebar.title("QuantApp Pro")
    
    # Last update time
    import datetime
    st.sidebar.caption(f"Last MAJ: {datetime.datetime.now().strftime('%H:%M:%S')}")

    # --- CONFIG USER EMAIL ---
    with st.sidebar.expander("📧 Configure Alerts"):
        current_email = get_user_setting("user_email") or ""
        email_input = st.text_input("Your Email to receive the report:", value=current_email)

        if st.button("Save Email"):
            if "@" in email_input and "." in email_input:
                save_user_setting("user_email", email_input)
                st.success("Email saved!")
            else:
                st.error("Invalid format.")

    
    st.sidebar.markdown("---")
    
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Market Dashboard", 
            "🎮 Paper Trading",
            "🧪 Backtest & Strategy", 
            "🧠 Portfolio Optimization",
            "⚡ Quant Lab"
        ]
    )
    
    st.sidebar.markdown("---")

    if "Dashboard" in page:
        page_home()
    elif "Paper Trading" in page:
        page_paper_trading()
    elif "Strategy" in page:
        page_strategy()
    elif "Portfolio Optimization" in page:
        page_portfolio()
    elif "Quant Lab" in page:
        page_quant_lab()

if __name__ == "__main__":
    main()
