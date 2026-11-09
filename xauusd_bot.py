"""
XAUUSD Quantitative Institutional Signal Bot (Master Quant Grade v3.8 - Dynamic Multi-Strategy & Smart Risk)
Architecture:
  - Dynamic Session Profiles (Asia Mean Reversion Mode vs London/NY Sniper Trend Mode)
  - Volatility Regime Switch (ATR & ADX Adaptive Volatility Tuning)
  - Multi-Timeframe Volume Confirmation (H1 Macro + M15 Micro Volume Spike Guard)
  - Smart Trailing & Break-Even Logic (Dynamic TP1, TP2, and Break-Even SL levels)
  - Pillar 1 & 2: Multi-Timeframe Confluence + Deep Forensic Gating (Liquidity Sweeps, Wick Rejections, POC)
  - Pillar 3: 21-Voter Ensemble Modules (Uniform Timeframe Consensus)
  - Pillar 4: Macro Trend Filter (EMA 200 H1 Hard Veto)
"""

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
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            payload.update(record.extra_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str = "xauusd_quant") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


LOGGER = get_logger("xauusd_bot")


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    record = logger.makeRecord(
        logger.name, logging.INFO, fn="", lno=0, msg=event, args=(), exc_info=None
    )
    record.extra_data = kwargs
    logger.handle(record)


class AuditDatabase:
    def __init__(self, db_path: str = "data/quant_audit.db") -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    action TEXT,
                    spot_price REAL,
                    total_score REAL,
                    voter_agreement REAL,
                    forensic_score REAL,
                    regime TEXT,
                    lot_size REAL,
                    risk_usd REAL,
                    notes TEXT
                )
            """)

    def record_signal(self, ensemble: Any, spot: float, levels: Optional[Dict[str, Any]]) -> None:
        try:
            with self.conn:
                self.conn.execute(
                    """INSERT INTO execution_logs (timestamp, action, spot_price, total_score, voter_agreement, forensic_score, regime, lot_size, risk_usd, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        ensemble.action,
                        spot,
                        ensemble.total_score,
                        ensemble.voter_agreement,
                        ensemble.forensic_score,
                        ensemble.regime,
                        levels["lot_size"] if levels else 0.0,
                        levels["risk_amount_usd"] if levels else 0.0,
                        "; ".join(ensemble.notes)
                    )
                )
        except Exception as e:
            log_event(LOGGER, "sqlite_log_error", error=str(e))


# ============================================================================
# 2. TECHNICAL & FORENSIC MATH LIBRARY
# ============================================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.ewm(alpha=1.0/period, adjust=False).mean() / (loss.ewm(alpha=1.0/period, adjust=False).mean() + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    plus_dm = np.where((df['high'].diff() > (-df['low'].diff())) & (df['high'].diff() > 0), df['high'].diff(), 0.0)
    minus_dm = np.where(((-df['low'].diff()) > df['high'].diff()) & ((-df['low'].diff()) > 0), -df['low'].diff(), 0.0)
    tr = atr(df, period)
    p_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/period, adjust=False).mean() / (tr + 1e-10)
    m_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/period, adjust=False).mean() / (tr + 1e-10)
    dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)
    return dx.ewm(alpha=1.0/period, adjust=False).mean(), p_di, m_di

def volume_profile_poc(df: pd.DataFrame, bins: int = 25) -> float:
    hist, edges = np.histogram(df['close'], bins=bins, weights=df['volume'])
    poc_idx = np.argmax(hist)
    return float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)


# ============================================================================
# 3. ADVANCED DYNAMIC PROFILES & GUARDS
# ============================================================================
def get_dynamic_session_profile() -> Tuple[str, bool, float, float]:
    """
    Mengembalikan profil parameter dinamis berdasarkan sesi pasar UTC:
    Format: (nama_sesi, diizinkan_trading, min_threshold, atr_multiplier)
    - Sesi Asia (00:00 - 08:00 UTC): Mode Mean Reversion / Range (Threshold 75%, ATR 1.0)
    - Sesi London & NY (08:00 - 21:00 UTC): Mode Trend Sniper (Threshold 85%, ATR 1.5)
    """
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    weekday = now_utc.weekday() # 5 = Sabtu, 6 = Minggu
    
    if weekday >= 5:
        return "WEEKEND", False, 99.0, 1.0
    
    if 0 <= hour < 8:
        return "ASIA_RANGE_MODE", True, 75.0, 1.0
    elif 8 <= hour < 21:
        return "LONDON_NY_SNIPER_MODE", True, 85.0, 1.5
    else:
        return "OFF_HOURS", False, 99.0, 1.0


def check_macro_news_guard() -> Tuple[bool, str]:
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() == 4 and 12 <= now_utc.hour <= 14:
        return False, "High-Impact News Window (NFP Risk)"
    return True, "No Major News Conflict"


# ============================================================================
# 4. MULTI-SOURCE DATA FETCHER & FAILOVER PIPELINE
# ============================================================================
def create_retry_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class DataFetcher:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.session = create_retry_session(cfg.retry_total)

    def fetch_twelvedata(self, interval: str, size: int) -> Optional[pd.DataFrame]:
        api_key = self.cfg.twelve_data_api_key
        if not api_key: return None
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={size}&apikey={api_key}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "status" in data and data["status"] == "error":
            raise ValueError(data.get("message", "TwelveData Error"))
        values = data.get("values", [])
        if not values: return None
        
        df = pd.DataFrame(values).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time", ascending=True).reset_index(drop=True)
        for c in ["open", "high", "low", "close"]: df[c] = df[c].astype(float)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0
        return df

    def fetch_csv_fallback(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        csv_path = os.getenv("OHLCV_CSV_PATH", "data/xauusd_h1.csv")
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df_h1 = pd.read_csv(csv_path)
            df_h1.columns = [c.strip().lower() for c in df_h1.columns]
            dt_c = "datetime" if "datetime" in df_h1.columns else ("time" if "time" in df_h1.columns else df_h1.columns[0])
            df_h1["time"] = pd.to_datetime(df_h1[dt_c])
            df_h1 = df_h1.sort_values("time", ascending=True).reset_index(drop=True)
            for c in ["open", "high", "low", "close"]: df_h1[c] = df_h1[c].astype(float)
            df_h1["volume"] = pd.to_numeric(df_h1["volume"], errors="coerce").fillna(1000.0) if "volume" in df_h1.columns else 1000.0
            df_m15 = df_h1.resample('15min', on='time').ffill().reset_index()
            return df_h1, df_m15
        raise RuntimeError("Local CSV fallback tidak ditemukan.")

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df_h1 = self.fetch_twelvedata("1h", 300)
            df_m15 = self.fetch_twelvedata("15min", 200)
            log_event(LOGGER, "data_source_primary", status="Success TwelveData API")
        except Exception as e:
            log_event(LOGGER, "failover_triggered", error=str(e), action="Switching to Local CSV")
            return self.fetch_csv_fallback()

        if df_h1 is None or len(df_h1) < self.cfg.min_bars:
            return self.fetch_csv_fallback()

        if df_m15 is None:
            df_m15 = df_h1.resample('15min', on='time').ffill().reset_index()

        return df_h1, df_m15

    def fetch_spot(self) -> float:
        try:
            resp = self.session.get("https://xaus.com/api/v1/spot?compact=1", timeout=5)
            data = resp.json()
            return float(data.get("xau", {}).get("price") or data.get("spot_usd_oz", 0.0))
        except Exception:
            return 0.0


# ============================================================================
# 5. 21-VOTER & MASTER ENSEMBLE ENGINE
# ============================================================================
@dataclass
class EnsembleResult:
    action: str
    total_score: float
    voter_agreement: float
    forensic_score: float
    regime: str
    poc_level: float
    notes: List[str]
    atr_value: float
    atr_multiplier: float


def run_21_voter_consensus(df: pd.DataFrame) -> Tuple[float, int, int]:
    close = df['close']
    c = close.iloc[-1]
    votes = []
    
    votes.append(1 if c > sma(close, 5).iloc[-1] else -1)
    votes.append(1 if c > sma(close, 10).iloc[-1] else -1)
    votes.append(1 if c > sma(close, 20).iloc[-1] else -1)
    votes.append(1 if c > sma(close, 50).iloc[-1] else -1)
    votes.append(1 if c > sma(close, 100).iloc[-1] else -1)
    votes.append(1 if c > sma(close, 200).iloc[-1] else -1)
    
    votes.append(1 if ema(close, 9).iloc[-1] > ema(close, 21).iloc[-1] else -1)
    votes.append(1 if ema(close, 12).iloc[-1] > ema(close, 26).iloc[-1] else -1)
    votes.append(1 if ema(close, 50).iloc[-1] > ema(close, 200).iloc[-1] else -1)
    
    r_val = rsi(close, 14).iloc[-1]
    votes.append(1 if r_val < 50 else -1)
    votes.append(1 if r_val < 40 else -1)
    votes.append(1 if r_val > 50 else -1)
    
    adx_v, p_di, m_di = adx(df, 14)
    votes.append(1 if p_di.iloc[-1] > m_di.iloc[-1] else -1)
    votes.append(1 if adx_v.iloc[-1] > 20 else -1)
    
    votes.append(1 if c > close.iloc[-2] else -1)
    votes.append(1 if c > close.iloc[-5] else -1)
    votes.append(1 if c > close.iloc[-10] else -1)
    
    highs, lows = df['high'], df['low']
    votes.append(1 if c > (highs.iloc[-5:].mean()) else -1)
    votes.append(1 if c < (lows.iloc[-5:].mean()) else -1)
    
    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
    votes.append(1 if df['volume'].iloc[-1] > vol_ma else -1)
    
    poc = volume_profile_poc(df)
    votes.append(1 if c > poc else -1)

    bull_count = sum(1 for v in votes if v == 1)
    bear_count = sum(1 for v in votes if v == -1)
    max_agreement = max(bull_count, bear_count) / len(votes)
    return float(max_agreement), bull_count, bear_count


def generate_master_ensemble(df_h1: pd.DataFrame, df_m15: pd.DataFrame, cfg: Any) -> EnsembleResult:
    close = df_h1['close']
    c_last = float(close.iloc[-1])
    curr_atr = float(atr(df_h1, 14).iloc[-1])
    
    adx_s, _, _ = adx(df_h1, 14)
    regime = "trending" if adx_s.iloc[-1] > 20 else "ranging"
    poc = volume_profile_poc(df_h1)

    # Dynamic Session Profile & Thresholds
    session_name, session_ok, min_threshold, atr_multiplier = get_dynamic_session_profile()
    news_ok, news_msg = check_macro_news_guard()

    # Pillar 4: Macro Trend Filter (EMA 200 H1)
    ema200 = ema(close, 200).iloc[-1]

    # Forensic Gating & MTF Volume Confirmation
    last = df_h1.iloc[-1]
    prev = df_h1.iloc[-21:-1]
    op, hi, lo, cl = last['open'], last['high'], last['low'], last['close']
    rng = max(hi - lo, 0.01)
    
    upper_wick = hi - max(op, cl)
    lower_wick = min(op, cl) - lo
    wick_pct = (max(upper_wick, lower_wick) / rng) * 100.0

    sweep = "NONE"
    if hi > prev['high'].max() and cl < prev['high'].max(): sweep = "BEARISH_SWEEP"
    elif lo < prev['low'].min() and cl > prev['low'].min(): sweep = "BULLISH_SWEEP"

    # Multi-Timeframe Volume Spike Confirmation (H1 & M15)
    h1_vol_ratio = last['volume'] / (df_h1['volume'].iloc[-21:-1].mean() + 1e-10)
    m15_vol_ratio = df_m15['volume'].iloc[-1] / (df_m15['volume'].iloc[-20:].mean() + 1e-10)

    f_score = 30.0
    if sweep != "NONE": f_score += 35.0
    if wick_pct > 40.0: f_score += 20.0
    if h1_vol_ratio > 1.2 or m15_vol_ratio > 1.3: f_score += 15.0
    f_score = min(f_score, 100.0)

    m15_fast = ema(df_m15['close'], 9).iloc[-1]
    m15_slow = ema(df_m15['close'], 21).iloc[-1]
    m15_conf = "BULLISH" if m15_fast > m15_slow else "BEARISH"

    voter_agreement, bull_c, bear_c = run_21_voter_consensus(df_h1)
    combined_score = (voter_agreement * 50.0) + (f_score * 0.5)

    intended_action = "BUY" if bull_c > bear_c else "SELL"

    macro_passed = True
    if intended_action == "BUY" and c_last < ema200: macro_passed = False
    elif intended_action == "SELL" and c_last > ema200: macro_passed = False

    action = "WAIT"
    notes = []

    # Evaluation with Dynamic Session Rules
    if not news_ok:
        notes.append(f"MACRO NEWS GUARD: {news_msg}.")
    elif not session_ok:
        notes.append(f"SESSION GUARD: Di luar jam operasional ({session_name}).")
    elif combined_score < min_threshold:
        notes.append(f"GATED: Confluence {combined_score:.1f}% < {min_threshold}% ({session_name}).")
    elif not macro_passed:
        notes.append("MACRO VETO: Arah berlawanan dengan EMA 200 H1.")
    elif intended_action == "BUY" and m15_conf == "BULLISH":
        action = "BUY"
        notes.append(f"APPROVED [{session_name}]: Confluence {combined_score:.1f}%")
    elif intended_action == "SELL" and m15_conf == "BEARISH":
        action = "SELL"
        notes.append(f"APPROVED [{session_name}]: Confluence {combined_score:.1f}%")
    else:
        notes.append("Mikrostruktur M15 tidak selaras.")

    return EnsembleResult(
        action=action, total_score=round(combined_score, 1), voter_agreement=round(voter_agreement, 4),
        forensic_score=round(f_score, 1), regime=regime, poc_level=round(poc, 2),
        notes=notes, atr_value=round(curr_atr, 4), atr_multiplier=atr_multiplier
    )


# ============================================================================
# 6. RISK MANAGEMENT & SMART TRAILING LEVELS
# ============================================================================
def calculate_dynamic_risk_levels(signal: str, entry_price: float, atr_value: float, atr_multiplier: float) -> Optional[Dict[str, Any]]:
    if signal not in ("BUY", "SELL"): return None
    
    sl_dist = max(atr_value * atr_multiplier, 1.0)
    risk_usd = 10000.0 * 0.01
    lot_size = round(max(risk_usd / (sl_dist * 100.0), 0.01), 2)

    if signal == "BUY":
        sl_price = entry_price - sl_dist
        tp1_price = entry_price + (sl_dist * 1.5)
        tp2_price = entry_price + (sl_dist * 2.5)
        be_price = entry_price + (sl_dist * 0.8) # Break-even trigger target
    else:
        sl_price = entry_price + sl_dist
        tp1_price = entry_price - (sl_dist * 1.5)
        tp2_price = entry_price - (sl_dist * 2.5)
        be_price = entry_price - (sl_dist * 0.8)

    return {
        "entry_price": round(entry_price, 2), "sl_price": round(sl_price, 2),
        "tp1_price": round(tp1_price, 2), "tp2_price": round(tp2_price, 2),
        "break_even_trigger": round(be_price, 2), "lot_size": lot_size
    }


def send_telegram(ensemble: EnsembleResult, spot: float, levels: Optional[Dict[str, Any]]) -> bool:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id or ":" not in token: return False
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if ensemble.action == "WAIT":
        msg = f"🟡 <b>XAUUSD QUANT v3.8 (DYNAMIC GUARD): WAIT</b>\n⏰ <code>{now}</code>\n<b>Score:</b> {ensemble.total_score}%\n<b>Reason:</b> {ensemble.notes[0]}"
    else:
        emoji = "🟢" if ensemble.action == "BUY" else "🔴"
        msg = f"""{emoji} <b>XAUUSD QUANT v3.8 ({ensemble.action} - MASTER QUANT)</b>
⏰ <code>{now}</code>
───────────────────
<b>Confluence:</b> <b>{ensemble.total_score}%</b> | <b>Regime:</b> {ensemble.regime.upper()}
<b>POC Level:</b> ${ensemble.poc_level:.2f}

<code>ENTRY : ${levels['entry_price']:.2f}
SL    : ${levels['sl_price']:.2f}
TP1   : ${levels['tp1_price']:.2f}
TP2   : ${levels['tp2_price']:.2f}
BE At : ${levels['break_even_trigger']:.2f}</code>

<b>Dynamic Lot Size:</b> <b>{levels['lot_size']} Lots</b>"""

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=8
        )
        return True
    except Exception:
        return False


class SimpleConfig:
    def __init__(self) -> None:
        self.min_bars: int = 210
        self.retry_total: int = 3
        self.twelve_data_api_key: Optional[str] = os.getenv("TWELVE_DATA_API_KEY")


def main() -> int:
    log_event(LOGGER, "bot_started", version="3.8.0-dynamic-master")
    cfg = SimpleConfig()
    db = AuditDatabase()
    fetcher = DataFetcher(cfg)

    try:
        df_h1, df_m15 = fetcher.fetch_data()
    except Exception as e:
        log_event(LOGGER, "fetch_error", error=str(e))
        print(f"ACTION: WAIT - Data unavailable: {e}")
        return 0

    ensemble = generate_master_ensemble(df_h1, df_m15, cfg)
    spot = fetcher.fetch_spot()
    if spot == 0.0: spot = float(df_h1['close'].iloc[-1])

    levels = calculate_dynamic_risk_levels(ensemble.action, spot, ensemble.atr_value, ensemble.atr_multiplier)
    db.record_signal(ensemble, spot, levels)
    sent = send_telegram(ensemble, spot, levels)

    print(f"ACTION: {ensemble.action} | Total Confluence: {ensemble.total_score}% | Telegram Sent: {sent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
