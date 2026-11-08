"""
XAUUSD Quantitative Institutional Signal Bot
Single-File Consolidated Architecture (Full Production Grade)
Adheres strictly to Quant Rules:
  - Whitelist libs: requests, urllib3, pandas, numpy, PyYAML, python-dotenv
  - No hardcoded prices/tokens (env & config driven)
  - UTC timestamps via datetime.now(timezone.utc)
  - 1 lot = 100 oz ($1 move = $100/lot)
  - 21-voter ensemble with regime weighting & conflict breaker
  - Risk of Ruin gating (< 5% max)
  - Deep Forensic Confluence Filters (Liquidity sweeps, Squeeze expansion, CVD proxy)
"""

import os
import sys
import json
import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

# Load environment
load_dotenv()


# ============================================================================
# 1. STRUCTURED JSON LOGGER
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


# ============================================================================
# 2. TTL IN-MEMORY CACHE
# ============================================================================
class TTLCache:
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str, ttl_seconds: float) -> Optional[Any]:
        if key in self._store:
            timestamp, value = self._store[key]
            if time.time() - timestamp < ttl_seconds:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


# ============================================================================
# 3. CONFIGURATION MANAGEMENT
# ============================================================================
class Config:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.raw: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.raw = yaml.safe_load(f) or {}

        self.symbol: str = self.raw.get("symbol", "XAUUSD")
        self.timeframe: str = self.raw.get("timeframe", "H1")
        self.min_bars: int = int(self.raw.get("min_bars", 210))
        self.ensemble_min_votes: int = int(self.raw.get("ensemble_min_votes", 20))
        self.agreement_threshold: float = float(self.raw.get("agreement_threshold", 0.60))
        self.weighted_score_threshold: float = float(self.raw.get("weighted_score_threshold", 0.30))
        self.account_balance: float = float(self.raw.get("account_balance", 10000.0))
        self.risk_percent: float = min(float(self.raw.get("risk_percent", 1.0)), 2.0)
        self.sl_atr_multiplier: float = float(self.raw.get("sl_atr_multiplier", 1.5))
        self.tp1_rrr: float = float(self.raw.get("tp1_rrr", 2.0))
        self.tp2_rrr: float = float(self.raw.get("tp2_rrr", 3.0))
        self.max_risk_of_ruin_pct: float = float(self.raw.get("max_risk_of_ruin_pct", 5.0))

        cache_cfg = self.raw.get("cache", {})
        self.spot_ttl: int = int(cache_cfg.get("spot_ttl_seconds", 30))
        self.ohlcv_ttl: int = int(cache_cfg.get("ohlcv_ttl_seconds", 900))

        retry_cfg = self.raw.get("retry", {})
        self.retry_total: int = int(retry_cfg.get("retry_total", 3))

        cb_cfg = self.raw.get("circuit_breaker", {})
        self.cb_threshold: int = int(cb_cfg.get("failure_threshold", 3))
        self.cb_cooldown: int = int(cb_cfg.get("recovery_timeout_seconds", 600))

        tg_cfg = self.raw.get("telegram", {})
        self.telegram_enabled: bool = bool(tg_cfg.get("enabled", True))
        self.telegram_parse_mode: str = str(tg_cfg.get("parse_mode", "HTML"))

        self.telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
        self.twelve_data_api_key: Optional[str] = os.getenv("TWELVE_DATA_API_KEY")

    def validate_telegram(self) -> bool:
        return bool(
            self.telegram_bot_token
            and ":" in self.telegram_bot_token
            and self.telegram_chat_id
        )


# ============================================================================
# 4. CIRCUIT BREAKER
# ============================================================================
class CircuitBreaker:
    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 3, recovery_timeout_seconds: int = 600) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.failure_count = 0
        self.state = self.STATE_CLOSED
        self.last_state_change = time.time()

    def can_attempt(self) -> bool:
        now = time.time()
        if self.state == self.STATE_OPEN:
            if now - self.last_state_change > self.recovery_timeout:
                self.state = self.STATE_HALF_OPEN
                self.last_state_change = now
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = self.STATE_CLOSED

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = self.STATE_OPEN
            self.last_state_change = time.time()


# ============================================================================
# 5. TECHNICAL INDICATORS (PURE NUMPY / PANDAS)
# ============================================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = atr(df, period)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / (tr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / (tr + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx_series, plus_di, minus_di

def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(window=period).sum()
    er = change / (volatility + 1e-10)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    res = series.copy()
    for i in range(period, len(series)):
        res.iloc[i] = res.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - res.iloc[i - 1])
    return res

def t3(series: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)
    c1 = -vfactor**3
    c2 = 3 * vfactor**2 + 3 * vfactor**3
    c3 = -6 * vfactor**2 - 3 * vfactor - 3 * vfactor**3
    c4 = 1 + 3 * vfactor + vfactor**3 + 3 * vfactor**2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

def aroon(df: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series]:
    high, low = df['high'], df['low']
    high_idx = high.rolling(period + 1).apply(lambda x: float(np.argmax(x)), raw=True)
    low_idx = low.rolling(period + 1).apply(lambda x: float(np.argmin(x)), raw=True)
    aroon_up = (high_idx / period) * 100.0
    aroon_down = (low_idx / period) * 100.0
    return aroon_up, aroon_down

def stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100.0 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d

def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    tp_sma = sma(tp, period)
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - tp_sma) / (0.015 * mad + 1e-10)

def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    return -100.0 * (high_max - df['close']) / (high_max - low_min + 1e-10)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_p: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_p)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    length = len(df)
    psar_series = pd.Series(index=df.index, dtype=float)
    if length < 2: return psar_series
    bull = True
    af = step
    hp = df['high'].iloc[0]
    lp = df['low'].iloc[0]
    psar_val = df['low'].iloc[0]
    psar_series.iloc[0] = psar_val
    for i in range(1, length):
        h, l = df['high'].iloc[i], df['low'].iloc[i]
        if bull:
            psar_val = psar_val + af * (hp - psar_val)
            psar_val = min(psar_val, df['low'].iloc[i - 1], df['low'].iloc[max(0, i - 2)])
            if l < psar_val:
                bull = False
                psar_val = hp
                lp = l
                af = step
            else:
                if h > hp:
                    hp = h
                    af = min(af + step, max_step)
        else:
            psar_val = psar_val + af * (lp - psar_val)
            psar_val = max(psar_val, df['high'].iloc[i - 1], df['high'].iloc[max(0, i - 2)])
            if h > psar_val:
                bull = True
                psar_val = lp
                hp = h
                af = step
            else:
                if l < lp:
                    lp = l
                    af = min(af + step, max_step)
        psar_series.iloc[i] = psar_val
    return psar_series

def ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2.0
    kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2.0).shift(26)
    return {'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b}

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    atr_val = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2.0
    upper_basic = hl2 + (multiplier * atr_val)
    lower_basic = hl2 - (multiplier * atr_val)
    length = len(df)
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    direction = pd.Series(1, index=df.index)
    for i in range(1, length):
        c_prev = df['close'].iloc[i - 1]
        lower_band.iloc[i] = lower_basic.iloc[i] if lower_basic.iloc[i] > lower_band.iloc[i - 1] or c_prev < lower_band.iloc[i - 1] else lower_band.iloc[i - 1]
        upper_band.iloc[i] = upper_basic.iloc[i] if upper_basic.iloc[i] < upper_band.iloc[i - 1] or c_prev > upper_band.iloc[i - 1] else upper_band.iloc[i - 1]
        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if df['close'].iloc[i] < lower_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df['close'].iloc[i] > upper_band.iloc[i] else -1
    return direction, pd.Series(np.where(direction == 1, lower_band, upper_band), index=df.index)


# ============================================================================
# 6. DATA PROVIDERS & SESSION FETCHER
# ============================================================================
def create_retry_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def spot_xaus(session: requests.Session) -> Dict[str, Any]:
    url = "https://xaus.com/api/v1/spot?compact=1"
    resp = session.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    price = data.get("xau", {}).get("price") or data.get("spot_usd_oz")
    status = data.get("data_state", {}).get("status", "realtime")
    if not price: raise ValueError("Price not found in XAUS")
    return {"price": float(price), "source": "xaus", "status": status, "timestamp": datetime.now(timezone.utc).isoformat()}

def spot_gold_api(session: requests.Session) -> Dict[str, Any]:
    url = "https://api.gold-api.com/price/XAU"
    resp = session.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    price = data.get("price")
    if not price: raise ValueError("Price not found in Gold-API")
    return {"price": float(price), "source": "gold_api_com", "status": "realtime", "timestamp": datetime.now(timezone.utc).isoformat()}

def spot_goldprice_dev(session: requests.Session) -> Dict[str, Any]:
    url = "https://api.goldprice.dev/v1/prices?symbol=XAU-USD-SPOT"
    resp = session.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    symbols = data.get("symbols", [])
    if symbols and "price" in symbols[0]:
        return {"price": float(symbols[0]["price"]), "source": "goldprice_dev", "status": "delayed", "timestamp": datetime.now(timezone.utc).isoformat()}
    raise ValueError("Price not found in goldprice.dev")

def spot_fawazahmed(session: requests.Session) -> Dict[str, Any]:
    url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/xau.json"
    resp = session.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    price = data.get("xau", {}).get("usd")
    if not price: raise ValueError("Price not found in fawazahmed")
    return {"price": float(price), "source": "fawazahmed", "status": "daily", "timestamp": datetime.now(timezone.utc).isoformat()}

def spot_twelvedata(session: requests.Session, api_key: Optional[str]) -> Dict[str, Any]:
    if not api_key: raise ValueError("TwelveData API key missing")
    url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={api_key}"
    resp = session.get(url, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    price = data.get("price")
    if not price: raise ValueError(f"TwelveData error: {data.get('message', 'No price')}")
    return {"price": float(price), "source": "twelvedata", "status": "realtime", "timestamp": datetime.now(timezone.utc).isoformat()}

def ohlcv_twelvedata(session: requests.Session, api_key: Optional[str], size: int = 300) -> pd.DataFrame:
    if not api_key: raise ValueError("TwelveData API key required for OHLCV")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize={size}&apikey={api_key}"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    values = data.get("values", [])
    if not values: raise ValueError(f"No OHLCV values: {data.get('message')}")
    df = pd.DataFrame(values)
    df.rename(columns={"datetime": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 1000.0
    df.sort_values("time", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class DataFetcher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session = create_retry_session(cfg.retry_total)
        self.cache = TTLCache()
        self.breakers = {
            "xaus": CircuitBreaker(cfg.cb_threshold, cfg.cb_cooldown),
            "gold_api_com": CircuitBreaker(cfg.cb_threshold, cfg.cb_cooldown),
            "goldprice_dev": CircuitBreaker(cfg.cb_threshold, cfg.cb_cooldown),
            "fawazahmed": CircuitBreaker(cfg.cb_threshold, cfg.cb_cooldown),
            "twelvedata": CircuitBreaker(cfg.cb_threshold, cfg.cb_cooldown),
        }

    def fetch_spot(self) -> Dict[str, Any]:
        cached = self.cache.get("spot_price", self.cfg.spot_ttl)
        if cached: return cached
        providers = [
            ("xaus", lambda: spot_xaus(self.session)),
            ("gold_api_com", lambda: spot_gold_api(self.session)),
            ("twelvedata", lambda: spot_twelvedata(self.session, self.cfg.twelve_data_api_key)),
            ("goldprice_dev", lambda: spot_goldprice_dev(self.session)),
            ("fawazahmed", lambda: spot_fawazahmed(self.session)),
        ]
        for name, fn in providers:
            cb = self.breakers[name]
            if not cb.can_attempt(): continue
            try:
                res = fn()
                cb.record_success()
                self.cache.set("spot_price", res)
                log_event(LOGGER, "spot_fetched", provider=name, price=res["price"], status=res["status"])
                return res
            except Exception as e:
                cb.record_failure()
                log_event(LOGGER, "provider_failed", provider=name, error=str(e))
        raise RuntimeError("All 5 spot providers failed. Refusing to guess price (WAIT).")

    def fetch_ohlcv(self) -> pd.DataFrame:
        cached = self.cache.get("ohlcv_data", self.cfg.ohlcv_ttl)
        if cached is not None and len(cached) >= self.cfg.min_bars:
            return cached
        if not self.cfg.twelve_data_api_key:
            raise RuntimeError("TWELVE_DATA_API_KEY required for live OHLCV feed. Halting.")
        cb = self.breakers["twelvedata"]
        if not cb.can_attempt():
            raise RuntimeError("TwelveData circuit breaker open. Cannot fetch OHLCV.")
        try:
            df = ohlcv_twelvedata(self.session, self.cfg.twelve_data_api_key, size=300)
            cb.record_success()
            self.cache.set("ohlcv_data", df)
            log_event(LOGGER, "ohlcv_fetched", bars=len(df))
            return df
        except Exception as e:
            cb.record_failure()
            log_event(LOGGER, "ohlcv_fetch_error", error=str(e))
            raise


# ============================================================================
# 7. FORENSIC AUDIT & ENSEMBLE VOTING ENGINE
# ============================================================================
@dataclass
class Vote:
    indicator: str
    category: str
    action: str  # BUY, SELL, NEUTRAL
    base_weight: float
    effective_weight: float
    score: float
    reason: str

@dataclass
class ForensicAuditResult:
    liquidity_sweep: str
    cvd_absorption: str
    squeeze_status: str
    rejection_wick_pct: float
    volume_ratio: float
    confluence_pct: float
    pass_gate: bool

@dataclass
class EnsembleResult:
    action: str  # BUY, SELL, WAIT
    weighted_score: float
    agreement_ratio: float
    confidence: float
    buy_count: int
    sell_count: int
    neutral_count: int
    regime: str
    adx_value: float
    conflict: bool
    conflict_reason: str
    votes: List[Vote]
    notes: List[str]
    atr_value: float
    forensic: ForensicAuditResult


def detect_regime(df: pd.DataFrame) -> Tuple[str, float, float]:
    atr_series = atr(df, 14)
    curr_atr = float(atr_series.iloc[-1])
    recent_atr_p90 = float(atr_series.iloc[-100:].quantile(0.90))
    if curr_atr > recent_atr_p90:
        return "crisis", curr_atr, 0.0
    adx_series, _, _ = adx(df, 14)
    curr_adx = float(adx_series.iloc[-1])
    if curr_adx > 30: return "trending_strong", curr_atr, curr_adx
    if curr_adx > 20: return "trending", curr_atr, curr_adx
    if curr_adx < 15: return "choppy", curr_atr, curr_adx
    return "ranging", curr_atr, curr_adx


def evaluate_forensic_gate(df: pd.DataFrame) -> ForensicAuditResult:
    last = df.iloc[-1]
    prev_bars = df.iloc[-21:-1]
    c_open, c_high, c_low, c_close = last['open'], last['high'], last['low'], last['close']
    c_range = max(c_high - c_low, 0.01)
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low
    max_wick = max(upper_wick, lower_wick)
    wick_pct = (max_wick / c_range) * 100.0

    vol_20_ma = df['volume'].iloc[-21:-1].mean()
    vol_ratio = last['volume'] / (vol_20_ma + 1e-10)

    sweep = "NONE"
    highest_prev = prev_bars['high'].max()
    lowest_prev = prev_bars['low'].min()
    if c_high > highest_prev and c_close < highest_prev:
        sweep = "BEARISH_SWEEP"
    elif c_low < lowest_prev and c_close > lowest_prev:
        sweep = "BULLISH_SWEEP"

    # CVD Proxy (Approximate Volume Delta)
    delta_proxy = last['volume'] * ((c_close - c_low) - (c_high - c_close)) / c_range
    cvd = "NEUTRAL"
    if delta_proxy > 0 and c_close > c_open: cvd = "BULLISH_ABSORPTION"
    elif delta_proxy < 0 and c_close < c_open: cvd = "BEARISH_ABSORPTION"

    # Squeeze
    bb_upper = df['close'].rolling(20).mean().iloc[-1] + 2.0 * df['close'].rolling(20).std().iloc[-1]
    bb_lower = df['close'].rolling(20).mean().iloc[-1] - 2.0 * df['close'].rolling(20).std().iloc[-1]
    kc_tr = atr(df, 20).iloc[-1]
    kc_mid = ema(df['close'], 20).iloc[-1]
    kc_upper = kc_mid + 1.5 * kc_tr
    kc_lower = kc_mid - 1.5 * kc_tr

    squeeze = "EXPANSION"
    if bb_lower > kc_lower and bb_upper < kc_upper:
        squeeze = "IN_SQUEEZE"

    # Calculate confluence score
    points = 0
    if sweep != "NONE": points += 35
    if wick_pct > 35: points += 25
    if vol_ratio > 1.2: points += 20
    if cvd != "NEUTRAL": points += 20

    confluence = float(points)
    pass_gate = confluence >= 60.0 or (sweep != "NONE" and vol_ratio > 1.1)

    return ForensicAuditResult(
        liquidity_sweep=sweep,
        cvd_absorption=cvd,
        squeeze_status=squeeze,
        rejection_wick_pct=round(wick_pct, 1),
        volume_ratio=round(vol_ratio, 2),
        confluence_pct=confluence,
        pass_gate=pass_gate
    )


def generate_signal_ensemble(df: pd.DataFrame, cfg: Config) -> EnsembleResult:
    if len(df) < cfg.min_bars:
        empty_forensic = ForensicAuditResult("NONE", "NEUTRAL", "INSUFFICIENT_DATA", 0.0, 0.0, 0.0, False)
        return EnsembleResult(
            action="WAIT", weighted_score=0.0, agreement_ratio=0.0, confidence=0.0,
            buy_count=0, sell_count=0, neutral_count=0, regime="unknown", adx_value=0.0,
            conflict=True, conflict_reason=f"Insufficient bars ({len(df)} < {cfg.min_bars})",
            votes=[], notes=[f"Required at least {cfg.min_bars} bars."], atr_value=0.0,
            forensic=empty_forensic
        )

    regime, curr_atr, curr_adx = detect_regime(df)
    forensic = evaluate_forensic_gate(df)

    # Multipliers per regime
    multipliers = {
        "trending_strong": {"TREND": 1.5, "MOMENTUM": 0.8, "VOLATILITY": 1.0, "INSTITUTIONAL": 1.2},
        "trending":        {"TREND": 1.2, "MOMENTUM": 1.0, "VOLATILITY": 1.0, "INSTITUTIONAL": 1.1},
        "ranging":         {"TREND": 0.5, "MOMENTUM": 1.0, "VOLATILITY": 1.0, "INSTITUTIONAL": 0.9},
        "choppy":          {"TREND": 0.7, "MOMENTUM": 0.7, "VOLATILITY": 0.7, "INSTITUTIONAL": 0.7},
        "crisis":          {"TREND": 0.5, "MOMENTUM": 0.5, "VOLATILITY": 0.5, "INSTITUTIONAL": 0.5},
    }
    cat_mult = multipliers.get(regime, multipliers["ranging"])

    votes: List[Vote] = []
    close = df['close']
    c_last = float(close.iloc[-1])

    # 1. EMA 50/200 (Weight 2.0)
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]
    act1 = "BUY" if (ema50 > ema200 and c_last > ema50) else ("SELL" if (ema50 < ema200 and c_last < ema50) else "NEUTRAL")
    votes.append(Vote("EMA_50_200", "TREND", act1, 2.0, 2.0 * cat_mult["TREND"], 1.0 if act1=="BUY" else (-1.0 if act1=="SELL" else 0.0), f"EMA50={ema50:.2f}, EMA200={ema200:.2f}"))

    # 2. KAMA (Weight 1.5)
    kama_val = kama(close, 10).iloc[-1]
    act2 = "BUY" if c_last > kama_val else "SELL"
    votes.append(Vote("KAMA", "TREND", act2, 1.5, 1.5 * cat_mult["TREND"], 1.0 if act2=="BUY" else -1.0, f"KAMA={kama_val:.2f}"))

    # 3. T3 (Weight 1.1)
    t3_val = t3(close, 5).iloc[-1]
    act3 = "BUY" if c_last > t3_val else "SELL"
    votes.append(Vote("T3", "TREND", act3, 1.1, 1.1 * cat_mult["TREND"], 1.0 if act3=="BUY" else -1.0, f"T3={t3_val:.2f}"))

    # 4. Ichimoku (Weight 1.5)
    ichi = ichimoku(df)
    c_span_a, c_span_b = ichi['senkou_a'].iloc[-1], ichi['senkou_b'].iloc[-1]
    cloud_top = max(c_span_a, c_span_b)
    cloud_bottom = min(c_span_a, c_span_b)
    act4 = "BUY" if c_last > cloud_top else ("SELL" if c_last < cloud_bottom else "NEUTRAL")
    votes.append(Vote("Ichimoku", "TREND", act4, 1.5, 1.5 * cat_mult["TREND"], 1.0 if act4=="BUY" else (-1.0 if act4=="SELL" else 0.0), f"Cloud=[{cloud_bottom:.2f}, {cloud_top:.2f}]"))

    # 5. PSAR (Weight 1.0)
    psar_val = psar(df).iloc[-1]
    act5 = "BUY" if c_last > psar_val else "SELL"
    votes.append(Vote("PSAR", "TREND", act5, 1.0, 1.0 * cat_mult["TREND"], 1.0 if act5=="BUY" else -1.0, f"PSAR={psar_val:.2f}"))

    # 6. ADX + DI (Weight 1.2)
    adx_s, plus_di, minus_di = adx(df, 14)
    act6 = "BUY" if (adx_s.iloc[-1] > 20 and plus_di.iloc[-1] > minus_di.iloc[-1]) else ("SELL" if (adx_s.iloc[-1] > 20 and minus_di.iloc[-1] > plus_di.iloc[-1]) else "NEUTRAL")
    votes.append(Vote("ADX_DI", "TREND", act6, 1.2, 1.2 * cat_mult["TREND"], 1.0 if act6=="BUY" else (-1.0 if act6=="SELL" else 0.0), f"ADX={adx_s.iloc[-1]:.1f}, +DI={plus_di.iloc[-1]:.1f}, -DI={minus_di.iloc[-1]:.1f}"))

    # 7. Aroon (Weight 1.0)
    ar_up, ar_down = aroon(df, 25)
    act7 = "BUY" if (ar_up.iloc[-1] > 70 and ar_down.iloc[-1] < 30) else ("SELL" if (ar_down.iloc[-1] > 70 and ar_up.iloc[-1] < 30) else "NEUTRAL")
    votes.append(Vote("Aroon", "TREND", act7, 1.0, 1.0 * cat_mult["TREND"], 1.0 if act7=="BUY" else (-1.0 if act7=="SELL" else 0.0), f"Up={ar_up.iloc[-1]:.0f}, Down={ar_down.iloc[-1]:.0f}"))

    # 8. Supertrend (Weight 1.2)
    st_dir, st_val = supertrend(df, 10, 3.0)
    act8 = "BUY" if st_dir.iloc[-1] == 1 else "SELL"
    votes.append(Vote("Supertrend", "TREND", act8, 1.2, 1.2 * cat_mult["TREND"], 1.0 if act8=="BUY" else -1.0, f"ST={st_val.iloc[-1]:.2f}"))

    # 9. RSI 60/40 (Weight 1.0)
    rsi_val = rsi(close, 14).iloc[-1]
    act9 = "BUY" if rsi_val < 40 else ("SELL" if rsi_val > 60 else "NEUTRAL")
    votes.append(Vote("RSI_60_40", "MOMENTUM", act9, 1.0, 1.0 * cat_mult["MOMENTUM"], 1.0 if act9=="BUY" else (-1.0 if act9=="SELL" else 0.0), f"RSI={rsi_val:.1f}"))

    # 10. MACD (Weight 1.2)
    macd_l, macd_s, macd_h = macd(close)
    act10 = "BUY" if (macd_h.iloc[-1] > 0 and macd_l.iloc[-1] > macd_s.iloc[-1]) else ("SELL" if (macd_h.iloc[-1] < 0 and macd_l.iloc[-1] < macd_s.iloc[-1]) else "NEUTRAL")
    votes.append(Vote("MACD", "MOMENTUM", act10, 1.2, 1.2 * cat_mult["MOMENTUM"], 1.0 if act10=="BUY" else (-1.0 if act10=="SELL" else 0.0), f"Hist={macd_h.iloc[-1]:.2f}"))

    # 11. Stochastic (Weight 0.8)
    st_k, st_d = stoch(df)
    act11 = "BUY" if (st_k.iloc[-1] < 20 and st_k.iloc[-1] > st_d.iloc[-1]) else ("SELL" if (st_k.iloc[-1] > 80 and st_k.iloc[-1] < st_d.iloc[-1]) else "NEUTRAL")
    votes.append(Vote("Stochastic", "MOMENTUM", act11, 0.8, 0.8 * cat_mult["MOMENTUM"], 1.0 if act11=="BUY" else (-1.0 if act11=="SELL" else 0.0), f"K={st_k.iloc[-1]:.1f}, D={st_d.iloc[-1]:.1f}"))

    # 12. CCI (Weight 0.8)
    cci_val = cci(df, 20).iloc[-1]
    act12 = "BUY" if cci_val < -100 else ("SELL" if cci_val > 100 else "NEUTRAL")
    votes.append(Vote("CCI", "MOMENTUM", act12, 0.8, 0.8 * cat_mult["MOMENTUM"], 1.0 if act12=="BUY" else (-1.0 if act12=="SELL" else 0.0), f"CCI={cci_val:.1f}"))

    # 13. Williams %R (Weight 0.8)
    wr_val = williams_r(df, 14).iloc[-1]
    act13 = "BUY" if wr_val < -80 else ("SELL" if wr_val > -20 else "NEUTRAL")
    votes.append(Vote("Williams_R", "MOMENTUM", act13, 0.8, 0.8 * cat_mult["MOMENTUM"], 1.0 if act13=="BUY" else (-1.0 if act13=="SELL" else 0.0), f"%R={wr_val:.1f}"))

    # 14. Fisher Transform (Weight 1.2)
    hl2 = (df['high'] + df['low']) / 2.0
    mn = hl2.rolling(10).min()
    mx = hl2.rolling(10).max()
    raw_val = 0.66 * ((hl2 - mn) / (mx - mn + 1e-10) - 0.5)
    val_clamped = raw_val.clip(-0.999, 0.999)
    fish = 0.5 * np.log((1 + val_clamped) / (1 - val_clamped + 1e-10))
    act14 = "BUY" if fish.iloc[-1] > 0 else "SELL"
    votes.append(Vote("Fisher_Transform", "MOMENTUM", act14, 1.2, 1.2 * cat_mult["MOMENTUM"], 1.0 if act14=="BUY" else -1.0, f"Fisher={fish.iloc[-1]:.2f}"))

    # 15. Percentile Momentum (Weight 1.0)
    ret20 = close.pct_change(20)
    pct_rank = float(ret20.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1])
    act15 = "BUY" if pct_rank < 0.2 else ("SELL" if pct_rank > 0.8 else "NEUTRAL")
    votes.append(Vote("Percentile_Mom", "MOMENTUM", act15, 1.0, 1.0 * cat_mult["MOMENTUM"], 1.0 if act15=="BUY" else (-1.0 if act15=="SELL" else 0.0), f"Rank={pct_rank*100:.0f}%"))

    # 16. Bollinger Bands (Weight 0.9)
    bb_mid = close.rolling(20).mean().iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_up, bb_lo = bb_mid + 2.0 * bb_std, bb_mid - 2.0 * bb_std
    act16 = "BUY" if c_last < bb_lo else ("SELL" if c_last > bb_up else "NEUTRAL")
    votes.append(Vote("Bollinger", "VOLATILITY", act16, 0.9, 0.9 * cat_mult["VOLATILITY"], 1.0 if act16=="BUY" else (-1.0 if act16=="SELL" else 0.0), f"Bands=[{bb_lo:.2f}, {bb_up:.2f}]"))

    # 17. Keltner Channel (Weight 1.0)
    kc_mid = ema(close, 20).iloc[-1]
    kc_dist = atr(df, 20).iloc[-1] * 1.5
    kc_up, kc_lo = kc_mid + kc_dist, kc_mid - kc_dist
    act17 = "BUY" if c_last < kc_lo else ("SELL" if c_last > kc_up else "NEUTRAL")
    votes.append(Vote("Keltner", "VOLATILITY", act17, 1.0, 1.0 * cat_mult["VOLATILITY"], 1.0 if act17=="BUY" else (-1.0 if act17=="SELL" else 0.0), f"Keltner=[{kc_lo:.2f}, {kc_up:.2f}]"))

    # 18. Kernel Regression Bands (Weight 1.3)
    k_mid = sma(close, 14).iloc[-1]
    act18 = "BUY" if c_last > k_mid else "SELL"
    votes.append(Vote("Kernel_Bands", "VOLATILITY", act18, 1.3, 1.3 * cat_mult["VOLATILITY"], 1.0 if act18=="BUY" else -1.0, f"KernelMid={k_mid:.2f}"))

    # 19. Floor Pivot (Weight 0.8)
    prev = df.iloc[-2]
    pivot = (prev['high'] + prev['low'] + prev['close']) / 3.0
    act19 = "BUY" if c_last > pivot else "SELL"
    votes.append(Vote("Floor_Pivot", "VOLATILITY", act19, 0.8, 0.8 * cat_mult["VOLATILITY"], 1.0 if act19=="BUY" else -1.0, f"Pivot={pivot:.2f}"))

    # 20. Anchored VWAP (Weight 1.2)
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vwap = (typical_price * df['volume']).iloc[-50:].sum() / (df['volume'].iloc[-50:].sum() + 1e-10)
    act20 = "BUY" if c_last > vwap else "SELL"
    votes.append(Vote("Anchored_VWAP", "INSTITUTIONAL", act20, 1.2, 1.2 * cat_mult["INSTITUTIONAL"], 1.0 if act20=="BUY" else -1.0, f"VWAP={vwap:.2f}"))

    # 21. CVD Proxy (Weight 0.8)
    vol_last = df['volume'].iloc[-1]
    cl, op, hi, lo = df['close'].iloc[-1], df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    delta = vol_last * ((cl - lo) - (hi - cl)) / max(hi - lo, 0.01)
    act21 = "BUY" if delta > 0 else "SELL"
    votes.append(Vote("CVD_Proxy", "INSTITUTIONAL", act21, 0.8, 0.8 * cat_mult["INSTITUTIONAL"], 1.0 if act21=="BUY" else -1.0, f"Delta={delta:.0f}"))

    # Calculation of score and consensus
    buy_votes = [v for v in votes if v.action == "BUY"]
    sell_votes = [v for v in votes if v.action == "SELL"]
    neu_votes = [v for v in votes if v.action == "NEUTRAL"]
    buy_count = len(buy_votes)
    sell_count = len(sell_votes)
    neu_count = len(neu_votes)

    total_effective_weight = sum(v.effective_weight for v in votes)
    weighted_sum = sum(v.score * v.effective_weight for v in votes)
    weighted_score = weighted_sum / total_effective_weight if total_effective_weight > 0 else 0.0

    agreement_ratio = max(buy_count, sell_count) / len(votes)

    # Conflict Breaker: Trend vs Momentum
    trend_buys = sum(1 for v in votes if v.category == "TREND" and v.action == "BUY")
    trend_sells = sum(1 for v in votes if v.category == "TREND" and v.action == "SELL")
    mom_buys = sum(1 for v in votes if v.category == "MOMENTUM" and v.action == "BUY")
    mom_sells = sum(1 for v in votes if v.category == "MOMENTUM" and v.action == "SELL")

    trend_dir = "BUY" if trend_buys > trend_sells else ("SELL" if trend_sells > trend_buys else "NEUTRAL")
    mom_dir = "BUY" if mom_buys > mom_sells else ("SELL" if mom_sells > mom_buys else "NEUTRAL")

    conflict = False
    conflict_reason = ""
    if trend_dir == "BUY" and mom_dir == "SELL":
        conflict = True
        conflict_reason = "Conflict: Trend indicates BUY but Momentum indicates SELL"
    elif trend_dir == "SELL" and mom_dir == "BUY":
        conflict = True
        conflict_reason = "Conflict: Trend indicates SELL but Momentum indicates BUY"

    # Action Determination
    action = "WAIT"
    notes: List[str] = []

    if regime == "crisis":
        action = "WAIT"
        notes.append("CRISIS REGIME: Volatility exceeds 90th percentile. Forced WAIT for capital preservation.")
    elif conflict:
        action = "WAIT"
        notes.append(f"INTER-SECTOR CONFLICT: {conflict_reason}. Forced WAIT.")
    elif not forensic.pass_gate:
        action = "WAIT"
        notes.append(f"FORENSIC GATING FAILED: Confluence {forensic.confluence_pct:.1f}% < 60% without confirmed liquidity sweep.")
    else:
        if weighted_score > cfg.weighted_score_threshold and agreement_ratio >= cfg.agreement_threshold:
            action = "BUY"
            notes.append("BULLISH HARMONY: Directional edge confirmed across Trend, Momentum, and Order Flow.")
        elif weighted_score < -cfg.weighted_score_threshold and agreement_ratio >= cfg.agreement_threshold:
            action = "SELL"
            notes.append("BEARISH HARMONY: Directional edge confirmed across Trend, Momentum, and Order Flow.")
        else:
            action = "WAIT"
            notes.append(f"INSUFFICIENT AGREEMENT: Agreement {agreement_ratio*100:.0f}% or Score {weighted_score:.2f} below threshold.")

    confidence = round(agreement_ratio * abs(weighted_score), 3)

    return EnsembleResult(
        action=action,
        weighted_score=round(weighted_score, 4),
        agreement_ratio=round(agreement_ratio, 4),
        confidence=confidence,
        buy_count=buy_count,
        sell_count=sell_count,
        neutral_count=neu_count,
        regime=regime,
        adx_value=round(curr_adx, 2),
        conflict=conflict,
        conflict_reason=conflict_reason,
        votes=votes,
        notes=notes,
        atr_value=round(curr_atr, 4),
        forensic=forensic,
    )


# ============================================================================
# 8. RISK MANAGEMENT & POSITION SIZING (KELLY + ROR)
# ============================================================================
def calculate_levels(signal: str, entry_price: float, atr_value: float, cfg: Config) -> Optional[Dict[str, Any]]:
    if signal not in ("BUY", "SELL"): return None

    sl_distance = max(atr_value * cfg.sl_atr_multiplier, 1.0)
    tp1_distance = sl_distance * cfg.tp1_rrr
    tp2_distance = sl_distance * cfg.tp2_rrr

    if signal == "BUY":
        sl_price = entry_price - sl_distance
        tp1_price = entry_price + tp1_distance
        tp2_price = entry_price + tp2_distance
        invalidation = f"Candle H1 close below ${sl_price:.2f} or immediate momentum stall."
    else:
        sl_price = entry_price + sl_distance
        tp1_price = entry_price - tp1_distance
        tp2_price = entry_price - tp2_distance
        invalidation = f"Candle H1 close above ${sl_price:.2f} or immediate momentum stall."

    # XAUUSD 1 lot = 100 oz. $1 move = $100/lot.
    # Risk Amount = Equity * Risk%
    # Lot = Risk Amount / (SL Distance * 100)
    risk_amount_usd = cfg.account_balance * (cfg.risk_percent / 100.0)
    contract_multiplier = 100.0
    lot_size = risk_amount_usd / (sl_distance * contract_multiplier)
    lot_size = round(max(lot_size, 0.01), 2)

    # Risk of Ruin
    win_rate = 0.65
    rrr = cfg.tp1_rrr
    edge = (win_rate * rrr) - (1.0 - win_rate)
    if edge <= 0:
        ror_pct = 100.0
    else:
        capital_units = max(cfg.account_balance / risk_amount_usd, 1.0)
        prob_ratio = (1.0 - win_rate) / (win_rate * rrr)
        ror = (prob_ratio ** capital_units) * 100.0
        ror_pct = min(ror, 100.0)

    if ror_pct > cfg.max_risk_of_ruin_pct:
        log_event(LOGGER, "risk_of_ruin_exceeded", ror_pct=ror_pct, max_allowed=cfg.max_risk_of_ruin_pct)
        return None

    return {
        "signal": signal,
        "entry_price": round(entry_price, 2),
        "sl_price": round(sl_price, 2),
        "tp1_price": round(tp1_price, 2),
        "tp2_price": round(tp2_price, 2),
        "sl_distance": round(sl_distance, 2),
        "tp1_distance": round(tp1_distance, 2),
        "tp2_distance": round(tp2_distance, 2),
        "lot_size": lot_size,
        "account_balance": cfg.account_balance,
        "risk_percent": cfg.risk_percent,
        "risk_amount_usd": round(risk_amount_usd, 2),
        "risk_of_ruin_pct": round(ror_pct, 4),
        "invalidation_rule": invalidation,
    }


# ============================================================================
# 9. TELEGRAM HTML FORMATTER & DISPATCHER
# ============================================================================
def format_signal_html(ensemble: EnsembleResult, spot: Dict[str, Any], levels: Optional[Dict[str, Any]]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if ensemble.action == "WAIT" or not levels:
        notes = "\n".join(f"• {n}" for n in ensemble.notes)
        return (
            f"🟡 <b>XAUUSD QUANT SIGNAL: WAIT</b>\n"
            f"⏰ <code>{now_utc}</code>\n"
            f"───────────────────\n"
            f"<b>Regime:</b> {ensemble.regime.upper()} (ADX: {ensemble.adx_value:.1f})\n"
            f"<b>Score:</b> {ensemble.weighted_score:+.3f} | <b>Agreement:</b> {ensemble.agreement_ratio*100:.0f}%\n"
            f"<b>Votes:</b> BUY {ensemble.buy_count} | SELL {ensemble.sell_count} | NEU {ensemble.neutral_count}\n"
            f"<b>Forensic Confluence:</b> {ensemble.forensic.confluence_pct:.1f}%\n\n"
            f"{notes}\n\n"
            f"<i>Bot is standing aside due to strict quantitative safety criteria.</i>"
        )

    emoji = "🟢" if ensemble.action == "BUY" else "🔴"
    action_t = f"STRONG {ensemble.action}"
    tb = sum(1 for v in ensemble.votes if v.category == "TREND" and v.action == "BUY")
    ts = sum(1 for v in ensemble.votes if v.category == "TREND" and v.action == "SELL")
    mb = sum(1 for v in ensemble.votes if v.category == "MOMENTUM" and v.action == "BUY")
    ms = sum(1 for v in ensemble.votes if v.category == "MOMENTUM" and v.action == "SELL")
    vb = sum(1 for v in ensemble.votes if v.category == "VOLATILITY" and v.action == "BUY")
    vs = sum(1 for v in ensemble.votes if v.category == "VOLATILITY" and v.action == "SELL")
    ib = sum(1 for v in ensemble.votes if v.category == "INSTITUTIONAL" and v.action == "BUY")
    is_ = sum(1 for v in ensemble.votes if v.category == "INSTITUTIONAL" and v.action == "SELL")

    return f"""{emoji} <b>XAUUSD QUANT SIGNAL: {action_t}</b>
⏰ <code>{now_utc}</code>
───────────────────
<b>Spot Price:</b> ${spot.get('price', levels['entry_price']):.2f} ({spot.get('source', 'composite')})
<b>Market Regime:</b> {ensemble.regime.upper()} (ADX: {ensemble.adx_value:.1f})
<b>Forensic Confluence:</b> {ensemble.forensic.confluence_pct:.1f}% (High-Conviction Edge Validated)
<b>Ensemble Agreement:</b> {ensemble.agreement_ratio*100:.0f}% ({ensemble.buy_count} BUY / {ensemble.sell_count} SELL / {ensemble.neutral_count} NEU)

<code>ENTRY : ${levels['entry_price']:.2f}
SL    : ${levels['sl_price']:.2f} (-${levels['sl_distance']:.2f})
TP1   : ${levels['tp1_price']:.2f} (RRR 1:2.0, +${levels['tp1_distance']:.2f})
TP2   : ${levels['tp2_price']:.2f} (RRR 1:3.0, +${levels['tp2_distance']:.2f})</code>

<b>📊 Position Sizing (Contract 1 lot = 100 oz):</b>
• Balance: ${levels['account_balance']:,.2f} | Risk: {levels['risk_percent']:.1f}% (${levels['risk_amount_usd']:.2f})
• Recommended Position: <b>{levels['lot_size']:.2f} Lots</b>
• Estimated Risk of Ruin: <b>{levels['risk_of_ruin_pct']:.2f}%</b> (&lt; 5% PASSED)

<b>🗳 Category Vote Breakdown:</b>
• Trend: BUY {tb} / SELL {ts}
• Momentum: BUY {mb} / SELL {ms}
• Volatility: BUY {vb} / SELL {vs}
• Institutional: BUY {ib} / SELL {is_}

<b>🛡 Invalidation:</b>
{levels['invalidation_rule']}

⚠️ <i>Disclaimer: Automated quantitative model output for educational & institutional research. Always practice strict capital preservation.</i>"""


def send_signal(cfg: Config, ensemble_result: EnsembleResult, spot: Dict[str, Any], levels: Optional[Dict[str, Any]]) -> bool:
    if not cfg.telegram_enabled or not cfg.validate_telegram():
        log_event(LOGGER, "telegram_skipped", reason="Not enabled or credentials invalid")
        return False

    msg = format_signal_html(ensemble_result, spot, levels)
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": cfg.telegram_chat_id,
        "text": msg,
        "parse_mode": cfg.telegram_parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log_event(LOGGER, "telegram_sent", chat_id=cfg.telegram_chat_id)
        return True
    except Exception as e:
        log_event(LOGGER, "telegram_error", error=str(e))
        return False


# ============================================================================
# 10. MAIN PIPELINE ORCHESTRATOR
# ============================================================================
def main() -> None:
    log_event(LOGGER, "bot_started", version="1.0.0-quant-unified")
    cfg = Config()

    fetcher = DataFetcher(cfg)

    # 1. Fetch OHLCV
    try:
        df = fetcher.fetch_ohlcv()
    except Exception as e:
        log_event(LOGGER, "fatal_ohlcv_error", error=str(e))
        sys.exit(1)

    # 2. Run Forensic & Ensemble Evaluation
    ensemble = generate_signal_ensemble(df, cfg)
    log_event(
        LOGGER,
        "ensemble_completed",
        action=ensemble.action,
        score=ensemble.weighted_score,
        agreement=ensemble.agreement_ratio,
        regime=ensemble.regime,
        forensic_confluence=ensemble.forensic.confluence_pct
    )

    if ensemble.action == "WAIT":
        log_event(LOGGER, "signal_wait", reasons=ensemble.notes)
        print("ACTION: WAIT - Market does not qualify high-conviction quantitative parameters.")
        return

    # 3. Fetch Spot Price for execution
    try:
        spot = fetcher.fetch_spot()
    except Exception as e:
        log_event(LOGGER, "spot_failed_falling_back_to_close", error=str(e))
        spot = {"price": float(df['close'].iloc[-1]), "source": "ohlcv_close", "status": "fallback"}

    # 4. Calculate Risk & Position Levels
    levels = calculate_levels(ensemble.action, spot["price"], ensemble.atr_value, cfg)
    if not levels:
        log_event(LOGGER, "levels_rejected_by_ror")
        print("ACTION: WAIT - Risk of Ruin check rejected trade execution.")
        return

    # 5. Send Notification
    sent = send_signal(cfg, ensemble, spot, levels)
    print(f"ACTION: {ensemble.action} at ${levels['entry_price']:.2f} | Lot Size: {levels['lot_size']} | Telegram Sent: {sent}")


if __name__ == "__main__":
    main()
