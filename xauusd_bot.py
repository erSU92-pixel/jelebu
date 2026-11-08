"""
XAUUSD Quantitative Institutional Signal Bot (Master Quant Grade - Strict Non-Collision Pipeline)
Architecture:
  - Pillar 4: Macro Trend Filter (EMA 200 H1 Hard Veto)
  - Pillar 1 & 2: Multi-Timeframe Confluence + Deep Forensic Gating (Liquidity Sweeps, Wick Rejections, POC)
  - Pillar 3: 21-Voter Ensemble Modules (Uniform Timeframe Consensus)
  - Threshold Gate: Strict Confluence >= 85%
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
from datetime import datetime, timezone

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
# 3. DATA FETCHER & PIPELINE
# ============================================================================
def create_retry_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def fetch_twelvedata(session: requests.Session, api_key: str, interval: str, size: int) -> pd.DataFrame:
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={size}&apikey={api_key}"
    resp = session.get(url, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    if "status" in data and data["status"] == "error":
        raise ValueError(f"TwelveData API Error: {data.get('message', '')}")
    values = data.get("values", [])
    if not values: raise ValueError(f"No OHLCV for {interval}")
    
    df = pd.DataFrame(values).rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time", ascending=True).reset_index(drop=True)
    
    for c in ["open", "high", "low", "close"]: 
        if c in df.columns: df[c] = df[c].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0
    return df


class DataFetcher:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.session = create_retry_session(cfg.retry_total)

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_h1, df_m15 = None, None
        if self.cfg.twelve_data_api_key:
            try:
                df_h1 = fetch_twelvedata(self.session, self.cfg.twelve_data_api_key, "1h", 300)
                df_m15 = fetch_twelvedata(self.session, self.cfg.twelve_data_api_key, "15min", 200)
            except Exception as e:
                log_event(LOGGER, "twelvedata_error", error=str(e))

        if df_h1 is None or len(df_h1) < self.cfg.min_bars:
            csv_path = os.getenv("OHLCV_CSV_PATH", "data/xauusd_h1.csv")
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df_h1 = pd.read_csv(csv_path)
                df_h1.columns = [c.strip().lower() for c in df_h1.columns]
                dt_c = "datetime" if "datetime" in df_h1.columns else ("time" if "time" in df_h1.columns else df_h1.columns[0])
                df_h1["time"] = pd.to_datetime(df_h1[dt_c])
                df_h1 = df_h1.sort_values("time", ascending=True).reset_index(drop=True)
                for c in ["open", "high", "low", "close"]: df_h1[c] = df_h1[c].astype(float)
                df_h1["volume"] = pd.to_numeric(df_h1["volume"], errors="coerce").fillna(1000.0) if "volume" in df_h1.columns else 1000.0

        if df_m15 is None and df_h1 is not None:
            df_m15 = df_h1.resample('15min', on='time').ffill().reset_index()

        if df_h1 is None or len(df_h1) < self.cfg.min_bars:
            raise RuntimeError("Data feed tidak tersedia atau kosong.")
        return df_h1, df_m15

    def fetch_spot(self) -> float:
        try:
            resp = self.session.get("https://xaus.com/api/v1/spot?compact=1", timeout=5)
            data = resp.json()
            return float(data.get("xau", {}).get("price") or data.get("spot_usd_oz", 0.0))
        except Exception:
            return 0.0


# ============================================================================
# 4. 4-PILLAR NON-COLLISION FORENSIC & 21-VOTER ENGINE
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


def run_21_voter_consensus(df: pd.DataFrame) -> Tuple[float, int, int]:
    """
    Pillar 3: 21 Independent Voter Modules on uniform timeframe.
    Returns (Agreement Ratio 0.0-1.0, Bullish Votes Count, Bearish Votes Count)
    """
    close = df['close']
    c = close.iloc[-1]
    
    votes = []
    # 21 Independent Technical Modules (Standardized parameters)
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
    
    # Momentum & Volatility modules
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
    max_votes = len(votes)
    
    max_agreement = max(bull_count, bear_count) / max_votes
    return float(max_agreement), bull_count, bear_count


def evaluate_forensic_and_mtf(df_h1: pd.DataFrame, df_m15: pd.DataFrame) -> Tuple[float, str, str]:
    """
    Pillar 1 & 2: Multi-Timeframe Confluence & Deep Forensic Gating.
    """
    last = df_h1.iloc[-1]
    prev = df_h1.iloc[-21:-1]
    op, hi, lo, cl = last['open'], last['high'], last['low'], last['close']
    rng = max(hi - lo, 0.01)
    
    upper_wick = hi - max(op, cl)
    lower_wick = min(op, cl) - lo
    max_wick = max(upper_wick, lower_wick)
    wick_pct = (max_wick / rng) * 100.0

    highest_prev = prev['high'].max()
    lowest_prev = prev['low'].min()
    sweep = "NONE"
    if hi > highest_prev and cl < highest_prev: sweep = "BEARISH_SWEEP"
    elif lo < lowest_prev and cl > lowest_prev: sweep = "BULLISH_SWEEP"

    vol_ma = df_h1['volume'].iloc[-21:-1].mean()
    vol_ratio = last['volume'] / (vol_ma + 1e-10)

    # Forensic score calculation (0 to 100 scale)
    f_score = 30.0
    if sweep != "NONE": f_score += 35.0
    if wick_pct > 40.0: f_score += 20.0
    if vol_ratio > 1.5: f_score += 15.0
    f_score = min(f_score, 100.0)

    # M15 Microstructure Confluence (Pillar 1)
    m15_fast = ema(df_m15['close'], 9).iloc[-1]
    m15_slow = ema(df_m15['close'], 21).iloc[-1]
    m15_conf = "BULLISH" if m15_fast > m15_slow else "BEARISH"

    return f_score, sweep, m15_conf


def generate_master_ensemble(df_h1: pd.DataFrame, df_m15: pd.DataFrame, cfg: Any) -> EnsembleResult:
    close = df_h1['close']
    c_last = float(close.iloc[-1])
    curr_atr = float(atr(df_h1, 14).iloc[-1])
    
    adx_s, _, _ = adx(df_h1, 14)
    regime = "trending" if adx_s.iloc[-1] > 20 else "ranging"
    poc = volume_profile_poc(df_h1)

    # Pillar 4: Macro Trend Filter (EMA 200 H1 - Hard Veto)
    ema200 = ema(close, 200).iloc[-1]
    
    # Pillar 1 & 2: Forensic & MTF
    forensic_score, sweep_type, m15_conf = evaluate_forensic_and_mtf(df_h1, df_m15)

    # Pillar 3: 21-Voter Ensemble Consensus
    voter_agreement, bull_c, bear_c = run_21_voter_consensus(df_h1)

    # Non-Collision Combined Confluence Score (Weighted average: 50% Voter + 50% Forensic)
    combined_score = (voter_agreement * 50.0) + (forensic_score * 0.5)

    # Determine potential primary direction from voters and sweep
    intended_action = "WAIT"
    if bull_c > bear_c:
        intended_action = "BUY"
    else:
        intended_action = "SELL"

    # Pillar 4 Hard Veto Check
    macro_passed = True
    if intended_action == "BUY" and c_last < ema200:
        macro_passed = False
    elif intended_action == "SELL" and c_last > ema200:
        macro_passed = False

    action = "WAIT"
    notes = []

    # Strict Threshold Gate >= 85% with Non-Collision Pipeline
    if combined_score < 85.0:
        notes.append(f"GATED: Total Confluence {combined_score:.1f}% < 85% minimum threshold.")
    elif not macro_passed:
        notes.append("MACRO VETO: Arah sinyal bertentangan dengan EMA 200 H1.")
    elif intended_action == "BUY" and m15_conf == "BULLISH":
        action = "BUY"
        notes.append(f"SNIPER BUY APPROVED: Confluence {combined_score:.1f}% (Voter: {voter_agreement*100:.0f}%, Forensic: {forensic_score:.1f}%)")
    elif intended_action == "SELL" and m15_conf == "BEARISH":
        action = "SELL"
        notes.append(f"SNIPER SELL APPROVED: Confluence {combined_score:.1f}% (Voter: {voter_agreement*100:.0f}%, Forensic: {forensic_score:.1f}%)")
    else:
        notes.append("Kondisi mikrostruktur M15 tidak selaras dengan arah konsensus.")

    return EnsembleResult(
        action=action, total_score=round(combined_score, 1), voter_agreement=round(voter_agreement, 4),
        forensic_score=round(forensic_score, 1), regime=regime, poc_level=round(poc, 2),
        notes=notes, atr_value=round(curr_atr, 4)
    )


# ============================================================================
# 5. RISK MANAGEMENT & LOT SIZING
# ============================================================================
def calculate_dynamic_risk_levels(signal: str, entry_price: float, atr_value: float) -> Optional[Dict[str, Any]]:
    if signal not in ("BUY", "SELL"): return None
    sl_dist = max(atr_value * 1.5, 1.0)
    risk_usd = 10000.0 * 0.01
    lot_size = round(max(risk_usd / (sl_dist * 100.0), 0.01), 2)

    if signal == "BUY":
        sl_price = entry_price - sl_dist
        tp1_price = entry_price + (sl_dist * 2.0)
        tp2_price = entry_price + (sl_dist * 3.0)
    else:
        sl_price = entry_price + sl_dist
        tp1_price = entry_price - (sl_dist * 2.0)
        tp2_price = entry_price - (sl_dist * 3.0)

    return {
        "entry_price": round(entry_price, 2), "sl_price": round(sl_price, 2),
        "tp1_price": round(tp1_price, 2), "tp2_price": round(tp2_price, 2), "lot_size": lot_size
    }


# ============================================================================
# 6. TELEGRAM NOTIFIER & MAIN
# ============================================================================
def send_telegram(ensemble: EnsembleResult, spot: float, levels: Optional[Dict[str, Any]]) -> bool:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id or ":" not in token: return False
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if ensemble.action == "WAIT":
        msg = f"🟡 <b>XAUUSD QUANT v3.5 (SNIPER >=85%): WAIT</b>\n⏰ <code>{now}</code>\n<b>Total Confluence:</b> {ensemble.total_score}%\n<b>Reason:</b> {ensemble.notes[0]}"
    else:
        emoji = "🟢" if ensemble.action == "BUY" else "🔴"
        msg = f"""{emoji} <b>XAUUSD QUANT v3.5 ({ensemble.action} - SNIPER GRADE)</b>
⏰ <code>{now}</code>
───────────────────
<b>Total Confluence:</b> <b>{ensemble.total_score}%</b> (&ge;85% PASSED)
<b>Voter Agreement:</b> {ensemble.voter_agreement*100:.0f}% | <b>Forensic:</b> {ensemble.forensic_score}%
<b>Regime:</b> {ensemble.regime.upper()} | <b>POC:</b> ${ensemble.poc_level:.2f}

<code>ENTRY : ${levels['entry_price']:.2f}
SL    : ${levels['sl_price']:.2f}
TP1   : ${levels['tp1_price']:.2f} (RRR 1:2)
TP2   : ${levels['tp2_price']:.2f} (RRR 1:3)</code>

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
    log_event(LOGGER, "bot_started", version="3.5.1-strict-non-collision")
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

    levels = calculate_dynamic_risk_levels(ensemble.action, spot, ensemble.atr_value)
    db.record_signal(ensemble, spot, levels)
    sent = send_telegram(ensemble, spot, levels)

    print(f"ACTION: {ensemble.action} | Total Confluence: {ensemble.total_score}% | Telegram Sent: {sent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
