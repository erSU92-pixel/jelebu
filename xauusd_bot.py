import os
import sys
import json
import sqlite3
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================================
# 1. SETUP LOGGER & AUDIT DB
# ============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(payload)

logger = logging.getLogger("xauusd_bot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/quant_audit.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS signal_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP, decision TEXT, confidence REAL, ema200 REAL, poc REAL, atr REAL)")
    conn.commit()
    conn.close()

# ============================================================================
# 2. CONFIGURATION & TELEGRAM NOTIFICATION
# ============================================================================
@dataclass
class Config:
    symbol: str = "GC=F"
    telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message: str, config: Config):
    if not config.telegram_token or not config.telegram_chat_id:
        logger.info("Telegram credentials missing, skipping alert.")
        return
    url = f"https://api.telegram.org/bot{config.telegram_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": config.telegram_chat_id, "text": message}, timeout=10)
        logger.info("Telegram alert sent successfully!")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

# ============================================================================
# 3. MATH & SMC ENGINE
# ============================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ============================================================================
# 4. LIVE BROKER SIMULATION ENGINE (NO-CLICK & NO-FEE)
# ============================================================================
class LiveBrokerSimulation:
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self._init_portfolio()

    def _init_portfolio(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS portfolio (balance REAL, position REAL, entry_price REAL)")
        cursor.execute("SELECT COUNT(*) FROM portfolio")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO portfolio VALUES (10000.0, 0.0, 0.0)")
        conn.commit()
        conn.close()

    def get_portfolio(self) -> Tuple[float, float, float]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT balance, position, entry_price FROM portfolio")
        res = cursor.fetchone()
        conn.close()
        return res

    def execute_order(self, decision: str, price: float, atr_val: float, config: Config) -> str:
        balance, position, entry_price = self.get_portfolio()
        message = ""
        
        if decision == "BUY" and position == 0:
            sl = price - (1.5 * atr_val)
            tp = price + (3.0 * atr_val)
            position = 1.0
            entry_price = price
            message = f"[BROKER] BUY Executed at ${price:.2f} | Dynamic SL: ${sl:.2f} | TP: ${tp:.2f}"
        elif decision == "SELL" and position == 0:
            sl = price + (1.5 * atr_val)
            tp = price - (3.0 * atr_val)
            position = -1.0
            entry_price = price
            message = f"[BROKER] SELL Executed at ${price:.2f} | Dynamic SL: ${sl:.2f} | TP: ${tp:.2f}"
        elif (decision == "SELL" and position > 0) or (decision == "BUY" and position < 0):
            pnl = (price - entry_price) if position > 0 else (entry_price - price)
            balance += pnl * 10
            message = f"[BROKER] Position Closed at ${price:.2f} | Profit/Loss: ${pnl*10:.2f}"
            position = 0.0
            entry_price = 0.0

        if message:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE portfolio SET balance=?, position=?, entry_price=?", (balance, position, entry_price))
            conn.commit()
            conn.close()
            send_telegram_alert(message, config)
            
        return message or "No order execution executed (Hold/Wait)."

# ============================================================================
# 5. MAIN INTEGRATION
# ============================================================================
if __name__ == '__main__':
    init_db()
    cfg = Config()
    broker = LiveBrokerSimulation()
    
    df = yf.download(cfg.symbol, period='5d', interval='15m')
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={'Datetime': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        
        df['ema200'] = ema(df['close'], 200)
        df['atr'] = atr(df, 14)
        
        last_close = float(df['close'].iloc[-1])
        last_atr = float(df['atr'].iloc[-1])
        
        decision = "WAIT"
        if last_close > float(df['ema200'].iloc[-1]):
            decision = "BUY"
        elif last_close < float(df['ema200'].iloc[-1]):
            decision = "SELL"
            
        exec_msg = broker.execute_order(decision, last_close, last_atr, cfg)
        print(f"--- XAUUSD BOT STATUS --- ")
        print(f"Decision: {decision} | Close: ${last_close:.2f} | ATR: {last_atr:.2f}")
        print(exec_msg)
