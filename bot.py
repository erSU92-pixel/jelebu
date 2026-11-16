import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Konfigurasi Parameter Pasar (Ticker Gold Futures resmi yang didukung yfinance)
TICKER_SYMBOL = "GC=F"
INTERVAL = "1h"
PERIOD = "60d"

def fetch_market_data(ticker=TICKER_SYMBOL, interval=INTERVAL, period=PERIOD, max_retries=3, delay=5):
    """Mengambil data pasar historis dari yfinance dengan mekanisme retry yang aman."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Mengunduh data pasar untuk {ticker} (Percobaan {attempt}/{max_retries})...")
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                return df
        except Exception as e:
            print(f"Gagal mengunduh data pada percobaan {attempt}: {str(e)}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise RuntimeError("Gagal total mengunduh data pasar setelah beberapa kali percobaan.")
    raise RuntimeError("Data pasar kosong atau tidak valid.")

def calculate_indicators(df):
    """Menghitung indikator multi-faktor institusional secara vektorial menggunakan Pandas dan NumPy."""
    df = df.copy()
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib '{col}' tidak ditemukan dalam dataframe.")

    # 1. EMA berlapis (EMA 50 & EMA 200)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 2. ATR (Average True Range) 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # 3. RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. Volume Velocity Filter
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']

    # 5. Swing High & Swing Low untuk struktur pasar / SMC
    window_swing = 5
    df['Swing_High'] = df['High'][(df['High'] == df['High'].rolling(window=window_swing*2+1, center=True).max())]
    df['Swing_Low'] = df['Low'][(df['Low'] == df['Low'].rolling(window=window_swing*2+1, center=True).min())]

    df = df.bfill().ffill()
    return df

def analyze_market_structure(df):
    """Menganalisis struktur pasar, konfluensi institusional, dan menghasilkan arah sinyal."""
    last_row = df.iloc[-1]
    
    close_price = float(last_row['Close'])
    ema_50 = float(last_row['EMA_50'])
    ema_200 = float(last_row['EMA_200'])
    rsi = float(last_row['RSI'])
    atr = float(last_row['ATR'])
    vol_ratio = float(last_row['Volume_Ratio'])

    recent_df = df.tail(20)
    recent_swing_highs = recent_df['High'].dropna()
    recent_swing_lows = recent_df['Low'].dropna()

    resistance_level = float(recent_swing_highs.max()) if not recent_swing_highs.empty else float(last_row['High'])
    support_level = float(recent_swing_lows.min()) if not recent_swing_lows.empty else float(last_row['Low'])

    trend_bullish = ema_50 > ema_200 and close_price > ema_50
    trend_bearish = ema_50 < ema_200 and close_price < ema_50

    signal_direction = "NEUTRAL"
    score = 50

    if trend_bullish:
        score += 15
    elif trend_bearish:
        score += 15

    if 40 <= rsi <= 60:
        score += 10
    elif (rsi < 40 and trend_bullish) or (rsi > 60 and trend_bearish):
        score += 15

    if vol_ratio > 1.2:
        score += 20
    elif vol_ratio > 1.0:
        score += 10

    confluence_score = min(max(int(score), 0), 100)

    if trend_bullish and (rsi < 65) and (vol_ratio >= 0.9):
        signal_direction = "BUY"
    elif trend_bearish and (rsi > 35) and (vol_ratio >= 0.9):
        signal_direction = "SELL"

    return {
        "direction": signal_direction,
        "close": close_price,
        "support": support_level,
        "resistance": resistance_level,
        "atr": atr,
        "confluence_score": confluence_score
    }

def generate_risk_management(analysis):
    """Menghitung level Stop Loss dan Target Profit bertingkat berbasis volatilitas ATR."""
    direction = analysis["direction"]
    entry = analysis["close"]
    atr = analysis["atr"]
    support = analysis["support"]
    resistance = analysis["resistance"]

    if direction == "BUY":
        stop_loss = support - (1.5 * atr)
        risk = entry - stop_loss
        if risk <= 0:
            risk = atr * 1.5
            stop_loss = entry - risk
        tp1 = entry + (risk * 2.5)
        tp2 = entry + (risk * 3.5)
        tp3 = entry + (risk * 4.5)
        rrr = f"1:{(tp1 - entry) / risk:.2f}"
    elif direction == "SELL":
        stop_loss = resistance + (1.5 * atr)
        risk = stop_loss - entry
        if risk <= 0:
            risk = atr * 1.5
            stop_loss = entry + risk
        tp1 = entry - (risk * 2.5)
        tp2 = entry - (risk * 3.5)
        tp3 = entry - (risk * 4.5)
        rrr = f"1:{(entry - tp1) / risk:.2f}"
    else:
        stop_loss = entry
        tp1 = entry
        tp2 = entry
        tp3 = entry
        rrr = "N/A"

    return {
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "rrr": rrr
    }

def send_telegram_signal(analysis, rm):
    """Mengirimkan sinyal terformat HTML secara aman ke kanal Telegram dengan retry logic."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("PERINGATAN: TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID tidak ditemukan di environment variables.")
        return

    timestamp_utc = pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    direction_emoji = "🟢 BUY" if analysis["direction"] == "BUY" else "🔴 SELL"

    message = (
        f"💎 <b>XAUUSD APEX INSTITUTIONAL SIGNAL</b> 💎\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Direction:</b> {direction_emoji}\n"
        f"🎯 <b>Entry Zone:</b> {rm['entry']}\n"
        f"🛑 <b>Stop Loss:</b> {rm['stop_loss']}\n"
        f"📈 <b>Take Profit 1:</b> {rm['tp1']}\n"
        f"📈 <b>Take Profit 2:</b> {rm['tp2']}\n"
        f"🏆 <b>Take Profit 3:</b> {rm['tp3']}\n"
        f"⚡ <b>Confluence Score:</b> {analysis['confluence_score']}/100\n"
        f"🛡️ <b>Risk-to-Reward:</b> {rm['rrr']}\n"
        f"🕒 <b>Timestamp (UTC):</b> {timestamp_utc}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print("Sinyal berhasil dikirim ke Telegram!")
                return
            else:
                print(f"Gagal mengirim ke Telegram (Status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"Error koneksi saat mengirim Telegram (Percobaan {attempt}): {str(e)}")
        time.sleep(3)

def main():
    try:
        print("Memulai eksekusi bot kuantitatif XAUUSD...")
        df = fetch_market_data()
        df = calculate_indicators(df)
        analysis = analyze_market_structure(df)
        
        print(f"Hasil Analisis Pasar: Arah={analysis['direction']}, Confluence Score={analysis['confluence_score']}")
        
        if analysis["direction"] in ["BUY", "SELL"] and analysis["confluence_score"] >= 60:
            rm = generate_risk_management(analysis)
            send_telegram_signal(analysis, rm)
        else:
            print("Kondisi pasar belum memenuhi ambang batas konfluensi institusional (Skor < 60 atau NEUTRAL). Sinyal tidak dikirim.")
    except Exception as e:
        print(f"CRITICAL ERROR dalam eksekusi bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
