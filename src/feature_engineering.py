# feature_engineering.py
import os
import pandas as pd
import config
from feature import add_features


# =========================
# IO / Preprocess
# =========================
def _ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def _read_daily_csv(stock_code: str) -> pd.DataFrame:
    path = f"data/{stock_code}_daily.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw daily file not found: {path} (run data_manager.py first)")

    df = pd.read_csv(path)

    # Expect columns: date, open, high, low, close, volume
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def _add_labels(df: pd.DataFrame, horizon: int, fee_buffer: float) -> pd.DataFrame:
    # Forward return
    df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
    # Binary label (classification)
    df["y_up"] = (df["fwd_ret"] > fee_buffer).astype(int)
    return df


# =========================
# Dataset builders
# =========================
def create_dataset_for_ticker(
    stock_code: str,
    horizon: int,
    fee_buffer: float,
) -> str:
    """
    Reads data/{ticker}_daily.csv and writes data/{ticker}_dataset_h{horizon}.csv
    """
    _ensure_data_dir()

    raw_df = _read_daily_csv(stock_code)
    df = raw_df.copy()

    df = add_features(df)
    df = _add_labels(df, horizon=horizon, fee_buffer=fee_buffer)

    df["ticker"] = stock_code

    # Drop rows with NaNs from rolling/shift
    df = df.dropna().reset_index(drop=True)

    out_path = f"data/{stock_code}_dataset_h{horizon}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] dataset: {stock_code} rows={len(df)} -> {out_path}")
    return out_path


def build_universe_dataset(
    horizon: int,
    min_rows: int = 800,
) -> str:
    """
    Merges all per-ticker datasets into a single file:
      data/dataset_h{horizon}_universe.csv
    """
    _ensure_data_dir()

    universe = getattr(config, "UNIVERSE", [])
    if not universe:
        raise ValueError("config.UNIVERSE is empty. Please define your 50 tickers in config.py")

    dfs = []
    kept = 0

    for ticker in universe:
        p = f"data/{ticker}_dataset_h{horizon}.csv"
        if not os.path.exists(p):
            print(f"[SKIP] missing dataset: {p}")
            continue

        try:
            tmp = pd.read_csv(p)
        except Exception as e:
            print(f"[DROP] {ticker}: failed to read ({e})")
            continue

        if len(tmp) < min_rows:
            print(f"[DROP] {ticker}: too few rows ({len(tmp)} < {min_rows})")
            continue

        dfs.append(tmp)
        kept += 1

    if not dfs:
        raise RuntimeError("No datasets available to merge. Create per-ticker datasets first.")

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    out_path = f"data/dataset_h{horizon}_universe.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] merged dataset rows={len(merged)} tickers={merged['ticker'].nunique()} -> {out_path}")
    return out_path


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    # Config defaults
    HORIZON = int(getattr(config, "HORIZON", 5))
    FEE_BUFFER = float(getattr(config, "FEE_BUFFER", 0.002))
    MIN_ROWS = int(getattr(config, "MIN_DATASET_ROWS", 800))

    universe = getattr(config, "UNIVERSE", ["005930"])

    # 1) Per-ticker datasets
    for ticker in universe:
        try:
            create_dataset_for_ticker(ticker, horizon=HORIZON, fee_buffer=FEE_BUFFER)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    # 2) Universe merged dataset (only if UNIVERSE is defined with multiple tickers)
    if len(universe) > 1:
        try:
            build_universe_dataset(horizon=HORIZON, min_rows=MIN_ROWS)
        except Exception as e:
            print(f"[ERROR] merge universe: {e}")
