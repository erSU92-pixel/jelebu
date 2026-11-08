"""
XAUUSD Forensic Walk-Backtester (Auto-Fetch Enabled)
Skrip untuk menguji akurasi Deep Forensic Gating pada data historis.
Jika file CSV lokal kosong/tidak ada, data akan ditarik secara otomatis dari TwelveData API.
"""

import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def fetch_historical_from_twelvedata(csv_path: str = "data/xauusd_h1.csv"):
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY tidak ditemukan di environment secrets!")
    
    print("Mengambil data historis langsung dari TwelveData API...")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=5000&apikey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    
    values = data.get("values", [])
    if not values:
        raise ValueError(f"Gagal mengambil data dari API: {data}")
        
    df = pd.DataFrame(values)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Berhasil menyimpan {len(df)} baris data ke {csv_path}")

def run_forensic_backtest(csv_path: str = "data/xauusd_h1.csv"):
    # Cek apakah file tidak ada atau ukurannya 0 (kosong)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"File {csv_path} kosong atau tidak ditemukan.")
        try:
            fetch_historical_from_twelvedata(csv_path)
        except Exception as e:
            print(f"Error saat mendownload data otomatis: {e}")
            return

    print(f"Memuat data historis dari {csv_path}...")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: File CSV terbaca kosong.")
        return
        
    df.columns = [c.strip().lower() for c in df.columns]
    dt_col = "datetime" if "datetime" in df.columns else ("time" if "time" in df.columns else df.columns[0])
    df["time"] = pd.to_datetime(df[dt_col])
    df = df.sort_values("time", ascending=True).reset_index(drop=True)
    
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0

    total_bars = len(df)
    print(f"Total bar data dimuat: {total_bars} bar H1.")
    print("Menjalankan simulasi Walk-Forward Forensic Gating...\n")

    triggered_signals = 0
    successful_trades = 0
    failed_trades = 0

    # Simulasi walk-forward bar demi bar
    for i in range(50, total_bars - 5):
        subset = df.iloc[:i]
        last = subset.iloc[-1]
        prev = subset.iloc[-21:-1]
        
        op, hi, lo, cl = last['open'], last['high'], last['low'], last['close']
        rng = max(hi - lo, 0.01)
        
        upper_wick = hi - max(op, cl)
        lower_wick = min(op, cl) - lo
        max_wick = max(upper_wick, lower_wick)
        wick_pct = (max_wick / rng) * 100.0

        highest_prev = prev['high'].max()
        lowest_prev = prev['low'].min()
        sweep = "NONE"
        if hi > highest_prev and cl < highest_prev:
            sweep = "BEARISH_SWEEP"
        elif lo < lowest_prev and cl > lowest_prev:
            sweep = "BULLISH_SWEEP"

        vol_ma = subset['volume'].iloc[-21:-1].mean()
        vol_ratio = last['volume'] / (vol_ma + 1e-10)

        score = 40.0
        if sweep != "NONE": score += 30.0
        if wick_pct > 35.0: score += 15.0
        if vol_ratio > 1.2: score += 15.0

        if score >= 75.0:
            triggered_signals += 1
            future_close = df.iloc[i + 5]['close']
            if sweep == "BULLISH_SWEEP" and future_close > cl:
                successful_trades += 1
            elif sweep == "BEARISH_SWEEP" and future_close < cl:
                successful_trades += 1
            else:
                failed_trades += 1

    win_rate = (successful_trades / triggered_signals * 100.0) if triggered_signals > 0 else 0.0
    
    print("=" * 45)
    print("HASIL LAPORAN BACKTEST FORENSIK XAUUSD")
    print("=" * 45)
    print(f"Total Bar Diuji          : {total_bars}")
    print(f"Sinyal Lolos Gate (>=75%): {triggered_signals}")
    print(f"Prediksi Berhasil        : {successful_trades}")
    print(f"Prediksi Gagal           : {failed_trades}")
    print(f"Forensic Win Rate        : {win_rate:.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    run_forensic_backtest()
