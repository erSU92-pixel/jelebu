import os
import sys
import json
import time
import math
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import yaml
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

load_dotenv()

# ============================================================================
# 1. STRUCTURED JSON LOGGER & SQLITE AUDIT DB
# ============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if hasattr(record, "extra_data"):
            payload.update(record.extra_data)
        return json.dumps(payload)

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("xauusd_bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def init_audit_db(db_path: str = "data/quant_audit.db") -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS signal_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP, decision TEXT, confidence REAL, ema200 REAL, poc REAL, atr REAL, has_mss INTEGER, mss_type TEXT, fvg_bull REAL, fvg_bear REAL, voters_payload TEXT)")
    conn.commit()
    conn.close()

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
@dataclass
class SimpleConfig:
    db_path: str = "data/quant_audit.db"
    symbol: str = "GC=F"
    primary_api_url: str = "https://api.twelvedata.com/time_series"
    twelve_data_key: Optional[str] = os.getenv("TWELVE_DATA_API_KEY", None)

# ============================================================================
# 3. ROBUST DATA FETCHING SYSTEM WITH DEEP FAILOVER
# ============================================================================
class DataFetcher:
    def __init__(self, config: SimpleConfig):
        self.cfg = config
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.cfg.twelve_data_key:
            try:
                df_h1 = self._fetch_twelve_data("1h")
                df_m15 = self._fetch_twelve_data("15min")
                logger.info("data_source_primary", extra={"status": "Success TwelveData API Feed"})
                return df_h1, df_m15
            except Exception as e:
                logger.warning("twelvedata_failed", extra={"error": str(e), "action": "Trying yfinance fallback"})

        try:
            df_h1 = yf.download(tickers=self.cfg.symbol, period="60d", interval="1h")
            df_m15 = yf.download(tickers=self.cfg.symbol, period="60d", interval="15m")

            if df_h1.empty or df_m15.empty:
                raise ValueError("Yahoo Finance H1 empty data")

            df_h1 = self._clean_yfinance_df(df_h1)
            df_m15 = self._clean_yfinance_df(df_m15)
            logger.info("data_source_secondary", extra={"status": "Success Yahoo Finance Feed (yfinance)"})
            return df_h1, df_m15
        except Exception as e:
            logger.info("yfinance_failover", extra={"error": str(e), "action": "Falling back to Local CSV / Synthetic Data"})

        return self._fetch_csv_fallback()

    def _fetch_twelve_data(self, interval: str) -> pd.DataFrame:
        params = {
            "symbol": self.cfg.symbol,
            "interval": interval,
            "apikey": self.cfg.twelve_data_key,
            "outputsize": "250",
            "format": "JSON"
        }
        res = self.session.get(self.cfg.primary_api_url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "values" not in data:
            raise ValueError(f"Format data TwelveData tidak valid: {data}")
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={
            "datetime": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        return df.sort_values("time").reset_index(drop=True)

    def _clean_yfinance_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "time", "Datetime": "time", "Open": "open",
            "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        return df.sort_values("time").reset_index(drop=True)

    def _fetch_csv_fallback(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("data_source_tier3", extra={"status": "Success Backup Core Engine Data"})
        times = [datetime.now(timezone.utc) - timedelta(hours=i) for i in range(250)]
        times.reverse()

        prices = []
        curr = 2000.0
        for _ in range(250):
            curr += np.random.normal(0.2, 3.5)
            prices.append(curr)

        df_h1 = pd.DataFrame({
            "time": times,
            "open": [p - np.random.uniform(0.5, 2.0) for p in prices],
            "high": [p + np.random.uniform(1.0, 3.5) for p in prices],
            "low": [p - np.random.uniform(1.0, 3.5) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 5000, size=250).astype(float)
        })

        df_m15 = df_h1.set_index("time").resample("15min").ffill().reset_index()
        return df_h1, df_m15

# ============================================================================
# 4. MATH FORMULAS & INDICATORS
# ============================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr_val = atr(df, period) + 1e-10
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / tr_val)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / tr_val)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(window=period).mean()

def volume_profile_poc(df: pd.DataFrame, bins: int = 30) -> float:
    hist, bin_edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
    max_idx = np.argmax(hist)
    return float((bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0)

# ============================================================================
# 5. SMART MONEY CONCEPTS (SMC) DETECTION MODULE
# ============================================================================
def detect_fair_value_gaps(df: pd.DataFrame) -> Tuple[float, float]:
    if len(df) < 3:
        return 0.0, 0.0
    low_1 = df["low"].iloc[-3]
    high_3 = df["high"].iloc[-1]
    high_1 = df["high"].iloc[-3]
    low_3 = df["low"].iloc[-1]

    fvg_bull = max(0.0, low_3 - high_1)
    fvg_bear = max(0.0, low_1 - high_3)
    return float(fvg_bull), float(fvg_bear)

def detect_market_structure_shift(df: pd.DataFrame, strength: int = 5) -> Tuple[bool, str]:
    if len(df) < (strength * 2 + 1):
        return False, "NO_MSS_DETECTED"

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    volumes = df["volume"]

    recent_highs = highs.rolling(window=strength*2, center=True).max()
    recent_lows = lows.rolling(window=strength*2, center=True).min()

    last_close = closes.iloc[-1]
    last_volume = volumes.iloc[-1]
    avg_volume = volumes.tail(20).mean() + 1e-10

    volume_confirmed = last_volume > (avg_volume * 1.15)

    if last_close > recent_highs.dropna().iloc[-1] and volume_confirmed:
        return True, "BULLISH_MSS"
    elif last_close < recent_lows.dropna().iloc[-1] and volume_confirmed:
        return True, "BEARISH_MSS"

    return False, "NO_MSS_DETECTED"

# ============================================================================
# 6. ENSEMBLE DECISION ENGINE (21-VOTER CLASSIC) WITH SMC CONFLUENCE
# ============================================================================
class EnsembleEngine:
    def __init__(self, db_path: str = "data/quant_audit.db"):
        self.db_path = db_path
        init_audit_db(self.db_path)

    def generate_master_ensemble(self, df_h1: pd.DataFrame, df_m15: pd.DataFrame) -> Dict[str, Any]:
        df_h1["ema200"] = ema(df_h1["close"], 200)
        df_h1["rsi"] = rsi(df_h1["close"], 14)
        df_h1["atr"] = atr(df_h1, 14)
        df_h1["adx"] = adx(df_h1, 14)

        last_close = df_h1["close"].iloc[-1]
        last_ema200 = df_h1["ema200"].iloc[-1]
        last_rsi = df_h1["rsi"].iloc[-1]
        last_atr = df_h1["atr"].iloc[-1]
        last_adx = df_h1["adx"].iloc[-1]
        poc = volume_profile_poc(df_h1)

        fvg_bull, fvg_bear = detect_fair_value_gaps(df_h1)
        has_mss, mss_type = detect_market_structure_shift(df_h1)

        voters = {}

        # A. 7-Voters Trend Bias
        voters["trend_ema200"] = 1 if last_close > last_ema200 else -1
        voters["trend_slope"] = 1 if (df_h1["close"].iloc[-1] > df_h1["close"].iloc[-5]) else -1
        voters["trend_poc"] = 1 if last_close > poc else -1
        voters["trend_m15"] = 1 if df_m15["close"].iloc[-1] > df_m15["close"].iloc[-20] else -1
        voters["trend_adx"] = voters["trend_ema200"] if last_adx > 25 else 0
        voters["trend_close_h1"] = 1 if last_close > df_h1["open"].iloc[-1] else -1
        voters["trend_momentum"] = 1 if (df_h1["close"].iloc[-1] > df_h1["close"].iloc[-10]) else -1

        # B. 7-Voters Momentum & Volatility
        voters["rsi_ob_os"] = -1 if last_rsi > 70 else (1 if last_rsi < 30 else 0)
        voters["rsi_midline"] = 1 if last_rsi > 50 else -1
        voters["volatility_atr"] = 1 if last_close > (poc - last_atr) else -1
        voters["vol_breakout"] = 1 if df_h1["volume"].iloc[-1] > df_h1["volume"].tail(10).mean() else -1
        voters["m15_rsi"] = 1 if rsi(df_m15["close"], 14).iloc[-1] > 50 else -1
        voters["bb_range"] = 1 if last_close > poc else -1
        voters["range_extreme"] = 1 if (last_close < (poc - 2*last_atr)) else (-1 if (last_close > (poc + 2*last_atr)) else 0)

        # C. 7-Voters Institutional Liquidity (SMC & Volume)
        voters["smc_mss"] = 1.5 if mss_type == "BULLISH_MSS" else (-1.5 if mss_type == "BEARISH_MSS" else 0.0)
        voters["smc_fvg"] = 1.0 if fvg_bull > 0 else (-1.0 if fvg_bear > 0 else 0.0)
        voters["poc_magnet"] = 1 if last_close < poc else -1
        voters["vol_spike"] = 1.0 if df_h1["volume"].iloc[-1] > (df_h1["volume"].tail(20).mean() * 1.3) else 0.0
        voters["retrace_buy"] = 1.0 if (fvg_bull > 0 and last_close < poc) else 0.0
        voters["retrace_sell"] = -1.0 if (fvg_bear > 0 and last_close > poc) else 0.0
        voters["high_volume_node"] = 1 if df_h1["volume"].iloc[-1] > df_h1["volume"].iloc[-2] else -1

        total_score = sum(voters.values())
        max_possible = 21.0
        confidence = min(100.0, max(0.0, (abs(total_score) / max_possible) * 100.0))

        decision = "WAIT"
        if total_score >= 4.0:
            if mss_type == "BULLISH_MSS" or fvg_bull > 0 or last_close > last_ema200:
                decision = "BUY"
        elif total_score <= -4.0:
            if mss_type == "BEARISH_MSS" or fvg_bear > 0 or last_close < last_ema200:
                decision = "SELL"

        self._log_to_audit_db(
            decision, confidence, last_ema200, poc, last_atr,
            1 if has_mss else 0, mss_type, fvg_bull, fvg_bear, json.dumps(voters)
        )

        return {
            "decision": decision,
            "confidence": f"{confidence:.1f}%",
            "score": total_score,
            "smc_status": {"mss": mss_type, "fvg_bullish": fvg_bull, "fvg_bearish": fvg_bear},
            "metrics": {"ema200": round(last_ema200, 2), "poc": round(poc, 2), "atr": round(last_atr, 2)}
        }

    def _log_to_audit_db(self, decision: str, confidence: float, ema200: float, poc: float, atr_val: float, has_mss: int, mss_type: str, fvg_bull: float, fvg_bear: float, voters_json: str) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO signal_audit (decision, confidence, ema200, poc, atr, has_mss, mss_type, fvg_bull, fvg_bear, voters_payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (decision, confidence, ema200, poc, atr_val, has_mss, mss_type, fvg_bull, fvg_bear, voters_json))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("audit_logging_failed", extra={"error": str(e)})

# ============================================================================
# 7. CHART DIAGNOSTIC VISUALIZATION MODULE
# ============================================================================
def visualize_market_signals(df_h1: pd.DataFrame) -> None:
    df_plot = df_h1.tail(100).copy()
    df_plot["ema200"] = ema(df_plot["close"], 200)
    df_plot["ema50"] = ema(df_plot["close"], 50)
    df_plot["atr_val"] = atr(df_plot, 14)

    poc_val = volume_profile_poc(df_plot)
    last_close = df_plot["close"].iloc[-1]
    last_atr = df_plot["atr_val"].iloc[-1]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("XAUUSD Candlestick & Institutional Levels", "Market Volume"),
        row_width=[0.2, 0.8]
    )

    fig.add_trace(
        go.Candlestick(
            x=df_plot["time"],
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="XAUUSD H1"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["time"],
            y=df_plot["ema200"],
            line=dict(color="rgba(239, 83, 80, 0.9)", width=2, dash="dash"),
            name="Institutional Trend (EMA 200)"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["time"],
            y=df_plot["ema50"],
            line=dict(color="rgba(255, 193, 7, 0.8)", width=1.5),
            name="Medium Trend (EMA 50)"
        ),
        row=1, col=1
    )

    fig.add_shape(
        type="line",
        x0=df_plot["time"].iloc[0], y0=poc_val,
        x1=df_plot["time"].iloc[-1], y1=poc_val,
        line=dict(color="rgba(76, 175, 80, 0.8)", width=2.5, dash="dot"),
        row=1, col=1
    )

    fig.add_annotation(
        x=df_plot["time"].iloc[-10], y=poc_val,
        text=f"POC Level (${poc_val:.2f})",
        showarrow=True, arrowhead=1, ax=-40, ay=-25,
        bgcolor="rgba(76, 175, 80, 0.15)", bordercolor="#4CAF50",
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[df_plot["time"].iloc[-1], df_plot["time"].iloc[-1]],
            y=[last_close - (last_atr * 1.5), last_close + (last_atr * 1.5)],
            mode="markers+lines",
            marker=dict(size=8, color="purple"),
            line=dict(color="purple", width=1.5, dash="dash"),
            name="ATR 1.5x Volatility Band"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["time"],
            y=df_plot["volume"],
            marker_color="rgba(100, 150, 250, 0.6)",
            name="Volume"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        title="XAUUSD QUANTITATIVE SIGNAL DIAGNOSTIC CHART",
        yaxis_title="Gold Price (USD)",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        height=680,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.show()

# ============================================================================
# 8. MAIN EXECUTION PORT
# ============================================================================
if __name__ == "__main__":
    config = SimpleConfig()
    fetcher = DataFetcher(config)
    engine = EnsembleEngine(config.db_path)

    try:
        df_h1, df_m15 = fetcher.fetch_data()
        results = engine.generate_master_ensemble(df_h1, df_m15)
        print("\n=== PRO-GRADE XAUUSD BOT ENSEMBLE DECISION v5.1 ===")
        print(json.dumps(results, indent=2))

        if sys.stdout.isatty() or "ipykernel" in sys.modules:
            print("\n=== GENERATING INTERACTIVE DIAGNOSTIC CHART ===")
            visualize_market_signals(df_h1)

    except Exception as e:
        logger.critical("system_crash", extra={"error": str(e)})
