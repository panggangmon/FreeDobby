import os
import sys
import pandas as pd
import numpy as np

import config


def _load_merged_dataset(horizon: int) -> pd.DataFrame:
    path = f"data/dataset_h{horizon}_universe.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Merged dataset not found: {path}. Run feature_engineering.py first.")
    df = pd.read_csv(path)
    return df


def _basic_sanity(df: pd.DataFrame):
    required = ["date", "ticker", "y_up"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # date parsing check (in case stored as string)
    if df["date"].dtype == object:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # y_up type check
    if df["y_up"].dtype != np.int64 and df["y_up"].dtype != np.int32:
        # convert if possible
        df["y_up"] = pd.to_numeric(df["y_up"], errors="coerce").astype("Int64")

    return df


def _find_bad_values(df: pd.DataFrame):
    # numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # include y_up even if nullable int
    if "y_up" not in numeric_cols and "y_up" in df.columns:
        try:
            df["y_up_num"] = pd.to_numeric(df["y_up"], errors="coerce")
            numeric_cols.append("y_up_num")
        except Exception:
            pass

    nan_counts = df[numeric_cols].isna().sum().sort_values(ascending=False)
    inf_counts = pd.Series({c: np.isinf(df[c].to_numpy(dtype=float, copy=False)).sum()
                            for c in numeric_cols if c in df.columns}).sort_values(ascending=False)

    return numeric_cols, nan_counts, inf_counts


def _ticker_stats(df: pd.DataFrame):
    g = df.groupby("ticker", dropna=False)

    # rows
    rows = g.size().rename("rows")

    # date range
    dmin = g["date"].min().rename("date_min")
    dmax = g["date"].max().rename("date_max")

    # label distribution
    ymean = g["y_up"].mean().rename("y_up_rate")  # fraction of 1s
    ysum = g["y_up"].sum().rename("y_up_ones")

    stats = pd.concat([rows, dmin, dmax, ymean, ysum], axis=1).reset_index()
    stats = stats.sort_values("rows", ascending=False).reset_index(drop=True)
    return stats


def _recommend_drops(stats: pd.DataFrame, min_rows: int, min_up_rate: float, max_up_rate: float):
    # Too few rows
    too_few = stats[stats["rows"] < min_rows].copy()

    # Label too biased
    biased = stats[(stats["y_up_rate"] < min_up_rate) | (stats["y_up_rate"] > max_up_rate)].copy()

    return too_few, biased


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    horizon = int(getattr(config, "HORIZON", 5))
    min_rows = int(getattr(config, "MIN_DATASET_ROWS", 600))

    # 라벨 분포 경고 기준(너무 한쪽으로 쏠리면 학습 안정성 떨어짐)
    # 예: y_up가 3% 미만 or 97% 초과면 경고
    min_up_rate = float(getattr(config, "MIN_UP_RATE_WARN", 0.03))
    max_up_rate = float(getattr(config, "MAX_UP_RATE_WARN", 0.97))

    df = _load_merged_dataset(horizon)
    df = _basic_sanity(df)

    _print_header("Merged Dataset Overview")
    print(f"Path: data/dataset_h{horizon}_universe.csv")
    print(f"Rows: {len(df)}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Columns: {len(df.columns)}")

    # overall label distribution
    y = pd.to_numeric(df["y_up"], errors="coerce")
    overall_rate = float(y.mean())
    _print_header("Overall Label Distribution (y_up)")
    print(f"y_up=1 rate: {overall_rate:.4f}")
    print(y.value_counts(dropna=False).to_string())

    # NaN/Inf checks
    numeric_cols, nan_counts, inf_counts = _find_bad_values(df)

    _print_header("Top NaN counts (numeric columns)")
    print(nan_counts.head(15).to_string())

    _print_header("Top Inf counts (numeric columns)")
    print(inf_counts.head(15).to_string())

    # per-ticker stats
    stats = _ticker_stats(df)
    _print_header("Per-ticker Summary (top 20 by rows)")
    print(stats.head(20).to_string(index=False))

    # recommend drops
    too_few, biased = _recommend_drops(stats, min_rows=min_rows, min_up_rate=min_up_rate, max_up_rate=max_up_rate)

    _print_header(f"Tickers with too few rows (< {min_rows})")
    if len(too_few) == 0:
        print("(none)")
    else:
        print(too_few.to_string(index=False))

    _print_header(f"Tickers with biased labels (y_up_rate < {min_up_rate} or > {max_up_rate})")
    if len(biased) == 0:
        print("(none)")
    else:
        print(biased.to_string(index=False))

    # Optional: save drop list
    drop_set = set(too_few["ticker"].tolist()) | set(biased["ticker"].tolist())
    if drop_set:
        out_path = "data/dropped_tickers.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for t in sorted(drop_set):
                f.write(t + "\n")
        _print_header("Saved")
        print(f"Drop candidates saved to: {out_path}  (count={len(drop_set)})")

    _print_header("Done")
    print("If you see many biased tickers, adjust FEE_BUFFER/HORIZON or add richer features.")
    print("If many tickers have too few rows, check data collection 기간/누락/상장기간 이슈.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
