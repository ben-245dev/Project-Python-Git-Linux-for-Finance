import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pytz
import streamlit as st
import yfinance as yf
from scipy.stats import norm  # prêt pour extensions VaR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
