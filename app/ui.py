# ui_utils.py
from datetime import datetime
import pytz
import streamlit as st

INDICES = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
    "Dow Jones": "^DJI",
}

TIMEZONES = {
    "New York": "America/New_York",
    "Londres": "Europe/London",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
}

TRADING_TIMEZONES = {
    "New York": "America/New_York",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
    "Sydney": "Australia/Sydney",
}


def render_trading_clocks():
    st.subheader("Horloges des principaux fuseaux de trading")
    cols = st.columns(len(TRADING_TIMEZONES))
    for col, (name, tz_str) in zip(cols, TRADING_TIMEZONES.items()):
        with col:
            tz = pytz.timezone(tz_str)
            now_local = datetime.now(tz)
            st.markdown(
                f"""
                <div style="background-color:#0e1117;padding:10px;border-radius:8px;">
                    <h4 style="color:#00c3ff;text-align:center;">{name}</h4>
                    <p style="color:#FFFFFF;font-size:22px;text-align:center;margin:0;">
                        {now_local.strftime('%H:%M:%S')}
                    </p>
                    <p style="color:#AAAAAA;font-size:12px;text-align:center;margin:0;">
                        {now_local.strftime('%Y-%m-%d')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_banner(live_data: dict):
    items = []
    for name, v in live_data.items():
        if v is None:
            continue
        price = v["price"]
        time_utc = v["time_utc"].strftime("%H:%M UTC")
        items.append(f"{name}: {price} ({time_utc})")
    text = " | ".join(items) if items else "Données indisponibles"
    st.markdown(
        f"""
        <div style="background-color:#0e1117;padding:5px 0;">
            <marquee style="color:#00c3ff; font-size:18px;">
                {text}
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )
