"""
XAUUSD Forensic Walk-Forward Backtester
Menguji efektivitas Deep Forensic Gating (Liquidity Sweeps, Wick Rejections, Volume Ratio)
pada data historis OHLCV XAUUSD.
"""

import pandas as pd
import numpy as np

def run_forensic_backtest(df: pd.DataFrame) -> dict:
    total_bars = len(df)
    triggered_signals = 0
    successful_trades = 0
    failed_trades = 0
    
    # Simulasi evaluasi bar per bar secara historis (Walk-Forward)
    for i in range(50, total_bars - 5):
        subset = df.iloc[:i]
        last = subset.iloc[-1]
        prev = subset.iloc[-21:-1]
        
        op, hi, lo, cl = last['open'], last['high'], last['low'], last['close']
        rng = max(hi - lo, 0.01)
        
        # 1. Hitung Forensik pada titik historis tsb
        upper_wick = hi - max(op, cl)
        lower_wick = min(op, cl) - lo
        max_wick = max(upper_wick, lower_wick)
        wick_pct = (max_wick / rng) * 100.0
        
        highest_prev = prev['high'].max()
        lowest_prev = prev['low'].min()
        sweep = "NONE"
        if hi > highest_prev and cl < highest_prev: sweep = "BEARISH_SWEEP"
        elif lo < lowest_prev and cl > lowest_prev: sweep = "BULLISH_SWEEP"
        
        vol_ma = subset['volume'].iloc[-21:-1].mean()
        vol_ratio = last['volume'] / (vol_ma + 1e-10)
        
        # Skor konfluensi forensik
        score = 40.0
        if sweep != "NONE": score += 30.0
        if wick_pct > 35.0: score += 15.0
        if vol_ratio > 1.2: score += 15.0
        
        # Jika lolos gate presisi >= 75%
        if score >= 75.0:
            triggered_signals += 1
            # Forward test 5 candle ke depan: Apakah harga bergerak sesuai arah sweep?
            future_close = df.iloc[i + 5]['close']
            if sweep == "BULLISH_SWEEP" and future_close > cl:
                successful_trades += 1
            elif sweep == "BEARISH_SWEEP" and future_close < cl:
                successful_trades += 1
            else:
                failed_trades += 1

    win_rate = (successful_trades / triggered_signals * 100.0) if triggered_signals > 0 else 0.0
    return {
        "total_evaluated_bars": total_bars,
        "signals_passed_gate": triggered_signals,
        "successful_predictions": successful_trades,
        "failed_predictions": failed_trades,
        "backtest_win_rate": round(win_rate, 2)
    }

# Contoh cara panggil jika Anda memiliki CSV historis XAUUSD H1:
# df_hist = pd.read_csv("data/xauusd_h1.csv")
# results = run_forensic_backtest(df_hist)
# print(results)
