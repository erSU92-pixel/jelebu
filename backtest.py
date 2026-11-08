"""
XAUUSD Forensic Walk-Backtester (Robust File Check)
Skrip pengujian dengan Ambang Batas 85% & Filter Tren Makro EMA 200.
"""

import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def fetch_historical_from_twelvedata(csv_path: str = "data/xauusd_h1.csv"):
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY tidak ditemukan di environment secrets!")
    
    print("Mengambil data historis langsung dari TwelveData API...")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=5000&apikey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    
    if "status" in data and data["status"] == "error":
        raise ValueError(f"TwelveData API Error: {data.get('message', 'Unknown error')}")
        
    values = data.get("values", [])
    if not values:
        raise ValueError(f"Format respons API tidak valid atau kosong: {data}")
        
    df = pd.DataFrame(values)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Berhasil menyimpan {len(df)} baris data ke {csv_path}")

def run_forensic_backtest(csv_path: str = "data/xauusd_h1.csv"):
    # PERBAIKAN UTAMA: Hapus file jika tidak ada atau ukurannya 0 bytes
    if os.path.exists(csv_path) and os.path.getsize(csv_path) == 0:
        print(f"File {csv_path} kosong (0 bytes), menghapus file...")
        os.remove(csv_path)

    if not os.path.exists(csv_path):
        try:
            fetch_historical_from_twelvedata(csv_path)
        except Exception as e:
            print(f"Error saat mendownload data otomatis: {e}")
            return

    print(f"Memuat data historis dari {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Gagal memparsing CSV: {e}, mencoba mengunduh ulang...")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        fetch_historical_from_twelvedata(csv_path)
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

    df['ema200'] = ema(df['close'], 200)

    total_bars = len(df)
    print(f"Total bar data dimuat: {total_bars} bar H1.")
    print("Menjalankan simulasi Walk-Forward High-Precision Gating (Threshold >= 85%)...\n")

    triggered_signals = 0
    successful_trades = 0
    failed_trades = 0

    for i in range(200, total_bars - 5):
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

        score = 30.0
        if sweep != "NONE": score += 35.0
        if wick_pct > 40.0: score += 20.0
        if vol_ratio > 1.5: score += 15.0

        macro_aligned = True
        if sweep == "BULLISH_SWEEP" and cl < last['ema200']:
            macro_aligned = False
        elif sweep == "BEARISH_SWEEP" and cl > last['ema200']:
            macro_aligned = False

        if score >= 85.0 and macro_aligned:
            triggered_signals += 1
            future_close = df.iloc[i + 5]['close']
            if sweep == "BULLISH_SWEEP" and future_close > cl:
                successful_trades += 1
            elif sweep == "BEARISH_SWEEP" and future_close < cl:
                successful_trades += 1
            else:
                failed_trades += 1

    win_rate = (successful_trades / triggered_signals * 100.0) if triggered_signals > 0 else 0.0
    
    print("=" * 50)
    print("HASIL LAPORAN HIGH-PRECISION FORENSIC BACKTEST")
    print("=" * 50)
    print(f"Total Bar Diuji          : {total_bars}")
    print(f"Sinyal Lolos Gate (>=85%): {triggered_signals}")
    print(f"Prediksi Berhasil        : {successful_trades}")
    print(f"Prediksi Gagal           : {failed_trades}")
    print(f"High-Precision Win Rate  : {win_rate:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    run_forensic_backtest()
