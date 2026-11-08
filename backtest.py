"""
XAUUSD Forensic Walk-Backtester
Skrip untuk menguji akurasi Deep Forensic Gating pada data historis.
"""

import os
import pandas as pd
import numpy as np

def run_forensic_backtest(csv_path: str = "data/xauusd_h1.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: File data historis tidak ditemukan di {csv_path}")
        print("Tip: Masukkan file CSV OHLCV H1 ke dalam folder data/ terlebih dahulu.")
        return

    print(f"Memuat data historis dari {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Standarisasi nama kolom waktu
    dt_col = "datetime" if "datetime" in df.columns else ("time" if "time" in df.columns else df.columns[0])
    df["time"] = pd.to_datetime(df[dt_col])
    df = df.sort_values("time", ascending=True).reset_index(drop=True)
    
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0

    total_bars = len(df)
    print(f"Total bar data dimuat: {total_bars} bar H1.")
    print("Menjalankan simulasi Walk-Forward Forensic Gating...\n")

    triggered_signals = 0
    successful_trades = 0
    failed_trades = 0

    # Simulasi walk-forward bar demi bar (dimulai dari bar ke-50 untuk burn-in indikator)
    for i in range(50, total_bars - 5):
        subset = df.iloc[:i]
        last = subset.iloc[-1]
        prev = subset.iloc[-21:-1]
        
        op, hi, lo, cl = last['open'], last['high'], last['low'], last['close']
        rng = max(hi - lo, 0.01)
        
        # 1. Hitung Forensik (Wick & Sweep)
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

        # Hitung skor konfluensi forensik (Sama persis dengan logika di bot utama)
        score = 40.0
        if sweep != "NONE": score += 30.0
        if wick_pct > 35.0: score += 15.0
        if vol_ratio > 1.2: score += 15.0

        # Jika lolos gate presisi >= 75%
        if score >= 75.0:
            triggered_signals += 1
            # Evaluasi pergerakan 5 candle ke depan (Forward Test)
            future_close = df.iloc[i + 5]['close']
            if sweep == "BULLISH_SWEEP" and future_close > cl:
                successful_trades += 1
            elif sweep == "BEARISH_SWEEP" and future_close < cl:
                successful_trades += 1
            else:
                failed_trades += 1

    # Laporan Hasil Backtest
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
