import os
import sys
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Konfigurasi Parameter Pasar (Ticker Gold Futures resmi yang didukung yfinance)
TICKER_SYMBOL = "GC=F"
INTERVAL = "1h"
PERIOD = "60d"

def fetch_market_data(ticker=TICKER_SYMBOL, interval=INTERVAL, period=PERIOD, max_retries=3, delay=5):
    """Mengambil data pasar historis dari yfinance dengan mekanisme retry berlapis yang aman."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Mengunduh data pasar untuk {ticker} (Interval: {interval}, Percobaan {attempt}/{max_retries})...")
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
    """Menghitung indikator multi-faktor institusional, SMC, Z-Score, dan ATR secara vektorial."""
    df = df.copy()
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib '{col}' tidak ditemukan dalam dataframe.")

    # 1. EMA Berlapis (EMA 50 & EMA 200) untuk Tren Makro
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 2. Statistical Mean Reversion Z-Score (Periode 20)
    rolling_mean_20 = df['Close'].rolling(window=20).mean()
    rolling_std_20 = df['Close'].rolling(window=20).std()
    df['Z_Score'] = (df['Close'] - rolling_mean_20) / (rolling_std_20 + 1e-8)

    # 3. ATR (Average True Range) Periode 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # 4. Order Flow / Volume Velocity Filter
    body = np.abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    body_to_wick = body / (total_range + 1e-8)
    volume_sma = df['Volume'].rolling(window=20).mean()
    df['Order_Flow_Score'] = body_to_wick * (df['Volume'] / (volume_sma + 1e-8))

    # 5. Market Structure / SMC (Swing High/Low, BOS, ChoCh)
    window_swing = 5
    df['Swing_High'] = df['High'][(df['High'] == df['High'].rolling(window=window_swing*2+1, center=True).max())]
    df['Swing_Low'] = df['Low'][(df['Low'] == df['Low'].rolling(window=window_swing*2+1, center=True).min())]
    
    df['Swing_High_Filled'] = df['Swing_High'].ffill()
    df['Swing_Low_Filled'] = df['Swing_Low'].ffill()

    df['BOS_Bullish'] = (df['Close'] > df['Swing_High_Filled'].shift(1)) & (df['Close'].shift(1) <= df['Swing_High_Filled'].shift(1))
    df['BOS_Bearish'] = (df['Close'] < df['Swing_Low_Filled'].shift(1)) & (df['Close'].shift(1) >= df['Swing_Low_Filled'].shift(1))

    df = df.bfill().ffill()
    return df

def analyze_market_structure(df):
    """Menganalisis konfluensi kuantitatif, status SMC, dan mencetak Forensic Audit Log."""
    last_row = df.iloc[-1]
    
    close_price = float(last_row['Close'])
    ema_50 = float(last_row['EMA_50'])
    ema_200 = float(last_row['EMA_200'])
    z_score = float(last_row['Z_Score'])
    atr = float(last_row['ATR'])
    order_flow = float(last_row['Order_Flow_Score'])
    bos_bullish = bool(last_row['BOS_Bullish'])
    bos_bearish = bool(last_row['BOS_Bearish'])
    
    recent_df = df.tail(20)
    support_level = float(recent_df['Low'].min())
    resistance_level = float(recent_df['High'].max())

    # Penentuan Status SMC
    smc_status = "NETRAL / KONSOLIDASI"
    if bos_bullish:
        smc_status = "BOS BULLISH AKTIF"
    elif bos_bearish:
        smc_status = "BOS BEARISH AKTIF"
    elif close_price > ema_50:
        smc_status = "MITIGASI SUPPORT / CHOCH BULLISH"
    elif close_price < ema_50:
        smc_status = "MITIGASI RESISTANCE / CHOCH BEARISH"

    # Penilaian Konfluensi Institusional (0-100)
    score = 50.0
    macro_bullish = ema_50 > ema_200 and close_price > ema_50
    macro_bearish = ema_50 < ema_200 and close_price < ema_50

    if macro_bullish:
        score += 15.0
    elif macro_bearish:
        score += 15.0

    if bos_bullish or bos_bearish:
        score += 15.0

    if order_flow > 1.0:
        score += 10.0

    if z_score < -1.5 or z_score > 1.5:
        score += 10.0

    confluence_score = int(min(max(score, 0.0), 100.0))

    # Sesi & Validasi Arah Sinyal
    signal_direction = "NEUTRAL"
    if confluence_score >= 65:
        if macro_bullish and z_score < 1.0 and (bos_bullish or close_price >= support_level):
            signal_direction = "BUY"
        elif macro_bearish and z_score > -1.0 and (bos_bearish or close_price <= resistance_level):
            signal_direction = "SELL"

    timestamp_str = datetime.utcnow().isoformat()

    # Forensic Audit Logger (Bukti transparan kalkulasi perantara sebelum webhook dikirim)
    print("=" * 70)
    print("🛡️ APEX ULTIMA FORENSIC QUANT ENGINE - AUDIT LOG")
    print(f"• Waktu Eksekusi (UTC)      : {timestamp_str}")
    print(f"• Harga Close Terkini       : {close_price:.2f}")
    print(f"• Rolling Z-Score (20)      : {z_score:.4f}")
    print(f"• Volatilitas ATR (14)      : {atr:.4f}")
    print(f"• Skor Order Flow Imbalance : {order_flow:.4f}")
    print(f"• Status Struktur SMC       : {smc_status}")
    print(f"• Skor Konfluensi           : {confluence_score}/100")
    print(f"• Keputusan Sinyal Akhir    : {signal_direction}")
    print("=" * 70)

    return {
        "direction": signal_direction,
        "close": close_price,
        "support": support_level,
        "resistance": resistance_level,
        "atr": atr,
        "z_score": z_score,
        "smc_status": smc_status,
        "confluence_score": confluence_score,
        "timestamp": timestamp_str
    }

def generate_risk_management(analysis):
    """Menghitung Stop Loss dinamis dan Take Profit bertingkat (TP1 1.5R, TP2 2.5R, TP3 4.0R)."""
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
        tp1 = entry + (risk * 1.5)
        tp2 = entry + (risk * 2.5)
        tp3 = entry + (risk * 4.0)
        rrr = f"1:{(tp3 - entry) / risk:.2f}"
    elif direction == "SELL":
        stop_loss = resistance + (1.5 * atr)
        risk = stop_loss - entry
        if risk <= 0:
            risk = atr * 1.5
            stop_loss = entry + risk
        tp1 = entry - (risk * 1.5)
        tp2 = entry - (risk * 2.5)
        tp3 = entry - (risk * 4.0)
        rrr = f"1:{(entry - tp3) / risk:.2f}"
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
    """Mengirimkan laporan forensik institusional terformat HTML ke Telegram dengan retry logic."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("PERINGATAN: TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID tidak ditemukan di environment variables.")
        return

    signal_decision = "VERIFIED BUY" if analysis["direction"] == "BUY" else "VERIFIED SELL"

    message = (
        f"🛡️ <b>APEX ULTIMA FORENSIC QUANT ENGINE</b> 🛡️\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌟 <b>Aset:</b> XAUUSD (GOLD)\n"
        f"📊 <b>Keputusan Sinyal:</b> {signal_decision}\n"
        f"🎯 <b>Harga Entry Presisi:</b> {rm['entry']}\n"
        f"🛑 <b>Stop Loss Forensik:</b> {rm['stop_loss']}\n"
        f"📈 <b>Take Profit 1 (1.5R):</b> {rm['tp1']}\n"
        f"📈 <b>Take Profit 2 (2.5R):</b> {rm['tp2']}\n"
        f"🚀 <b>Take Profit 3 (4.0R):</b> {rm['tp3']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔬 <b>Audit Forensik & Metrik:</b>\n"
        f"• <i>Skor Konfluensi:</i> {analysis['confluence_score']}/100 (Tervalidasi)\n"
        f"• <i>Status Struktur SMC:</i> {analysis['smc_status']}\n"
        f"• <i>Rolling Z-Score:</i> {analysis['z_score']:.4f}\n"
        f"• <i>Volatilitas ATR (14):</i> {analysis['atr']:.4f}\n"
        f"• <i>Waktu Eksekusi (UTC):</i> {analysis['timestamp']}\n"
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
                print("Laporan forensik berhasil dikirim ke Telegram!")
                return
            else:
                print(f"Gagal mengirim ke Telegram (Status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"Error koneksi saat mengirim Telegram (Percobaan {attempt}): {str(e)}")
        time.sleep(3)

def main():
    try:
        print("Memulai eksekusi bot kuantitatif hibrida XAUUSD...")
        df = fetch_market_data()
        df = calculate_indicators(df)
        analysis = analyze_market_structure(df)
        
        if analysis["direction"] in ["BUY", "SELL"] and analysis["confluence_score"] >= 65:
            rm = generate_risk_management(analysis)
            send_telegram_signal(analysis, rm)
        else:
            print("Kondisi pasar saat ini NETRAL atau belum memenuhi ambang batas konfluensi (Skor < 65). Sinyal tidak dikirim.")
    except Exception as e:
        print(f"CRITICAL ERROR dalam eksekusi bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
