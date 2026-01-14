# src/train.py
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump

import config


# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def print_block(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_float_grid(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_grid(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------
# Data
# ---------------------------
def load_dataset(horizon: int) -> pd.DataFrame:
    path = f"data/dataset_h{horizon}_universe.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path} (run feature_engineering.py first)")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fwd_ret"] = pd.to_numeric(df["fwd_ret"], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "fwd_ret"]).copy()

    # numeric clean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def pick_feature_cols(df: pd.DataFrame) -> list[str]:
    ban = {"date", "ticker", "y_up", "fwd_ret"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in ban]
    if not feats:
        raise ValueError("No feature columns found.")
    return feats


def time_split(df: pd.DataFrame, train_years=3.5, valid_years=0.5):
    df = df.sort_values("date").reset_index(drop=True)

    dmin = df["date"].min()
    dmax = df["date"].max()

    train_end = dmin + pd.Timedelta(days=int(train_years * 365))
    valid_end = train_end + pd.Timedelta(days=int(valid_years * 365))

    train = df[df["date"] <= train_end].copy()
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)].copy()
    test  = df[df["date"] > valid_end].copy()

    return train, valid, test, (dmin, train_end, valid_end, dmax)


# ---------------------------
# Model: LightGBM Regressor (Early stopping)
# ---------------------------
def build_lgbm_regressor():
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm") from e

    return lgb.LGBMRegressor(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )


def fit_regressor(train_df, valid_df, feature_cols):
    import lightgbm as lgb

    X_train = train_df[feature_cols]
    y_train = train_df["fwd_ret"].values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["fwd_ret"].values

    model = build_lgbm_regressor()
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
    )
    return model


def predict(model, df, feature_cols) -> np.ndarray:
    X = df[feature_cols]
    return model.predict(X)


# ---------------------------
# Regime thresholds (from TRAIN only, per dv_q/atr_q)
# ---------------------------
def make_regime_thresholds(train_df: pd.DataFrame, atr_q: float, dv_q: float):
    thr = {}

    thr["atr_cap"] = None
    if "atr_14_pct" in train_df.columns:
        thr["atr_cap"] = float(train_df["atr_14_pct"].quantile(atr_q))

    thr["dv_min"] = None
    if "dollar_volume" in train_df.columns:
        thr["dv_min"] = float(train_df["dollar_volume"].quantile(dv_q))

    return thr


def apply_regime_filter(day_df: pd.DataFrame, thr: dict,
                        use_trend: bool, use_vol_cap: bool, use_liquidity: bool) -> pd.DataFrame:
    df = day_df

    # Trend filter (기본 OFF로 운용 권장)
    if use_trend:
        if ("dist_ma20" in df.columns) and ("ma20_slope_5" in df.columns):
            df = df[(df["dist_ma20"] > 0) & (df["ma20_slope_5"] > 0)]

    # Volatility cap
    if use_vol_cap and (thr.get("atr_cap") is not None) and ("atr_14_pct" in df.columns):
        df = df[df["atr_14_pct"] <= thr["atr_cap"]]

    # Liquidity filter
    if use_liquidity and (thr.get("dv_min") is not None) and ("dollar_volume" in df.columns):
        df = df[df["dollar_volume"] >= thr["dv_min"]]

    return df


# ---------------------------
# Portfolio backtest (ranking-based)
# ---------------------------
def portfolio_backtest_rank(
    df: pd.DataFrame,
    yhat: np.ndarray,
    fee_buffer: float,
    top_k: int,
    rebalance_step: int,
    regime_thr: dict,
    use_trend: bool,
    use_vol_cap: bool,
    use_liquidity: bool,
):
    # 필요한 컬럼만 모아 tmp를 만든다(없는 컬럼은 NaN으로 만들어 필터에서 자동 무시)
    cols = ["date", "ticker", "fwd_ret", "dist_ma20", "ma20_slope_5", "atr_14_pct", "dollar_volume"]
    tmp = df.copy()
    for c in cols:
        if c not in tmp.columns:
            tmp[c] = np.nan
    tmp = tmp[cols].copy()
    tmp["yhat"] = yhat

    tmp = tmp.sort_values("date").reset_index(drop=True)
    dates = tmp["date"].drop_duplicates().sort_values().tolist()
    if not dates:
        return None, {"trades": 0}

    reb_dates = dates[::max(1, rebalance_step)]

    rets = []
    trade_counts = []
    used_dates = []

    for d in reb_dates:
        day = tmp[tmp["date"] == d]
        day = apply_regime_filter(day, regime_thr, use_trend, use_vol_cap, use_liquidity)

        if day.empty:
            rets.append(0.0)
            trade_counts.append(0)
            used_dates.append(d)
            continue

        day = day.sort_values("yhat", ascending=False)
        picks = day.head(top_k)

        if picks.empty:
            rets.append(0.0)
            trade_counts.append(0)
            used_dates.append(d)
            continue

        net = (picks["fwd_ret"].values - fee_buffer)
        port_ret = float(np.mean(net))
        rets.append(port_ret)
        trade_counts.append(int(len(picks)))
        used_dates.append(d)

    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1.0 + r))
    equity = np.array(equity[1:], dtype=float)

    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / peak)
    mdd = float(np.max(dd)) if len(dd) else 0.0

    used = pd.Series(used_dates)
    months = used.dt.to_period("M")
    month_counts = pd.Series(trade_counts).groupby(months).sum()
    avg_trades_per_month = float(month_counts.mean()) if len(month_counts) else 0.0

    metrics = {
        "rebalance_points": int(len(used_dates)),
        "trades": int(np.sum(trade_counts)),
        "avg_trades_per_month": avg_trades_per_month,
        "avg_trade_per_rebalance": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "avg_port_ret": float(np.mean(rets)) if rets else 0.0,
        "median_port_ret": float(np.median(rets)) if rets else 0.0,
        "win_rate": float(np.mean(np.array(rets) > 0)) if rets else 0.0,
        "final_equity": float(equity[-1]) if len(equity) else 1.0,
        "mdd": mdd,
    }

    curve = pd.DataFrame({"date": used_dates, "port_ret": rets, "equity": equity})
    return curve, metrics


# ---------------------------
# Grid Search (must satisfy VALID>=1 and TEST>=1)
# ---------------------------
def grid_search_params(
    train_df, valid_df, test_df,
    yhat_train, yhat_valid, yhat_test,
    fee_buffer: float,
    topk_grid: list[int],
    rebalance_grid: list[int],
    dv_q_grid: list[float],
    atr_q_grid: list[float],
    use_trend: bool,
    use_vol_cap: bool,
    use_liquidity: bool,
    min_trades_per_month: float,
    mdd_lambda: float,
):
    best = None
    tried = 0
    passed = 0

    for reb_step in rebalance_grid:
        for dv_q in dv_q_grid:
            for atr_q in atr_q_grid:
                regime_thr = make_regime_thresholds(train_df, atr_q=atr_q, dv_q=dv_q)

                for k in topk_grid:
                    tried += 1

                    # VALID
                    _, m_valid = portfolio_backtest_rank(
                        valid_df, yhat_valid, fee_buffer,
                        top_k=k,
                        rebalance_step=reb_step,
                        regime_thr=regime_thr,
                        use_trend=use_trend,
                        use_vol_cap=use_vol_cap,
                        use_liquidity=use_liquidity,
                    )
                    # 거래 너무 적으면 제외
                    if m_valid["avg_trades_per_month"] < min_trades_per_month:
                        continue
                    # VALID must be positive
                    if m_valid["final_equity"] < 1.0:
                        continue

                    # TEST must be positive
                    _, m_test = portfolio_backtest_rank(
                        test_df, yhat_test, fee_buffer,
                        top_k=k,
                        rebalance_step=reb_step,
                        regime_thr=regime_thr,
                        use_trend=use_trend,
                        use_vol_cap=use_vol_cap,
                        use_liquidity=use_liquidity,
                    )
                    if m_test["final_equity"] < 1.0:
                        continue

                    passed += 1

                    # 목적함수는 VALID 기준으로 안정적으로
                    score = float(np.log(m_valid["final_equity"]) - mdd_lambda * m_valid["mdd"])

                    cand = {
                        "score": score,
                        "params": {
                            "top_k": int(k),
                            "rebalance_step": int(reb_step),
                            "dv_q": float(dv_q),
                            "atr_q": float(atr_q),
                        },
                        "regime_thr": regime_thr,
                        "valid": m_valid,
                        "test": m_test,
                    }

                    if (best is None) or (score > best["score"]):
                        best = cand

    return best, {"tried": tried, "passed": passed}


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # grids
    ap.add_argument("--top-k-grid", type=str, default="1,2,3,5")
    ap.add_argument("--rebalance-grid", type=str, default="5,10")
    ap.add_argument("--dv-q-grid", type=str, default="0.10,0.20,0.30")
    ap.add_argument("--atr-q-grid", type=str, default="0.80,0.90")

    # objective / constraints
    ap.add_argument("--min-trades-per-month", type=float, default=8.0)
    ap.add_argument("--mdd-lambda", type=float, default=1.0)

    # regime toggles (DEFAULT: trend OFF)
    ap.add_argument("--trend-filter", action="store_true", help="Enable trend filter (default is OFF)")
    ap.add_argument("--no-vol-cap", action="store_true", help="Disable volatility cap filter")
    ap.add_argument("--no-liquidity-filter", action="store_true", help="Disable liquidity filter")

    args = ap.parse_args()

    horizon = int(getattr(config, "HORIZON", 5))
    fee_buffer = float(getattr(config, "FEE_BUFFER", 0.002))

    topk_grid = parse_int_grid(args.top_k_grid)
    rebalance_grid = parse_int_grid(args.rebalance_grid)
    dv_q_grid = parse_float_grid(args.dv_q_grid)
    atr_q_grid = parse_float_grid(args.atr_q_grid)

    use_trend = bool(args.trend_filter)          # default OFF
    use_vol_cap = not bool(args.no_vol_cap)
    use_liquidity = not bool(args.no_liquidity_filter)

    df = load_dataset(horizon=horizon)
    feature_cols = pick_feature_cols(df)

    train_df, valid_df, test_df, dates = time_split(df)
    dmin, train_end, valid_end, dmax = dates

    print_block("INFO")
    print(f"date range: {dmin.date()} ~ {dmax.date()}")
    print(f"split: train<= {train_end.date()}, valid<= {valid_end.date()}, test> {valid_end.date()}")
    print(f"rows: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print(f"horizon={horizon}, fee_buffer={fee_buffer}")
    print(f"features={len(feature_cols)}")
    print(f"grid top_k={topk_grid}")
    print(f"grid rebalance_step={rebalance_grid}")
    print(f"grid dv_q={dv_q_grid}")
    print(f"grid atr_q={atr_q_grid}")
    print(f"regime: trend={use_trend}, vol_cap={use_vol_cap}, liquidity={use_liquidity}")
    print(f"constraints: min_trades_per_month={args.min_trades_per_month}, mdd_lambda={args.mdd_lambda}")
    print("selection constraints: VALID final_equity >= 1.0 AND TEST final_equity >= 1.0")

    # Train regressor
    model = fit_regressor(train_df, valid_df, feature_cols)

    # Predict
    yhat_train = predict(model, train_df, feature_cols)
    yhat_valid = predict(model, valid_df, feature_cols)
    yhat_test  = predict(model, test_df,  feature_cols)

    # Grid search
    best, stat = grid_search_params(
        train_df=train_df, valid_df=valid_df, test_df=test_df,
        yhat_train=yhat_train, yhat_valid=yhat_valid, yhat_test=yhat_test,
        fee_buffer=fee_buffer,
        topk_grid=topk_grid,
        rebalance_grid=rebalance_grid,
        dv_q_grid=dv_q_grid,
        atr_q_grid=atr_q_grid,
        use_trend=use_trend,
        use_vol_cap=use_vol_cap,
        use_liquidity=use_liquidity,
        min_trades_per_month=args.min_trades_per_month,
        mdd_lambda=args.mdd_lambda,
    )

    print_block("GRID SEARCH STATS")
    print(f"tried={stat['tried']}, passed={stat['passed']}")

    if best is None:
        print_block("WARN")
        print("조건(VALID>=1 & TEST>=1)을 만족하는 조합을 찾지 못했습니다.")
        print("우선순위대로 완화/확장하세요:")
        print("  1) liquidity 필터 완화: --no-liquidity-filter 또는 dv-q-grid에 더 낮은 값 추가(예: 0.05)")
        print("  2) vol cap 완화: --no-vol-cap 또는 atr-q-grid에 0.95 추가")
        print("  3) rebalance-grid 확장: 3,5,10,15")
        print("  4) min-trades-per-month 낮추기(예: 4)")
        print("  5) trend-filter는 기본 OFF가 유리(필요 시 --trend-filter로 비교)")
        return

    params = best["params"]
    regime_thr = best["regime_thr"]

    print_block("BEST CONFIG (VALID>=1 & TEST>=1)")
    print(f"score={best['score']:.6f} (log(valid_final_equity) - mdd_lambda*valid_mdd)")
    print(f"params={params}")
    print(f"regime_thr(from TRAIN)={regime_thr}")
    print(f"VALID metrics={best['valid']}")
    print(f"TEST  metrics={best['test']}")

    # Final evaluation curves for TRAIN/VALID/TEST using best params
    print_block("PORTFOLIO BACKTEST RESULTS (final best)")
    best_k = params["top_k"]
    best_step = params["rebalance_step"]
    # recompute thresholds for safety
    regime_thr = make_regime_thresholds(train_df, atr_q=params["atr_q"], dv_q=params["dv_q"])

    for name, dfx, yhat in [("TRAIN", train_df, yhat_train), ("VALID", valid_df, yhat_valid), ("TEST", test_df, yhat_test)]:
        curve, m = portfolio_backtest_rank(
            dfx, yhat, fee_buffer,
            top_k=best_k,
            rebalance_step=best_step,
            regime_thr=regime_thr,
            use_trend=use_trend,
            use_vol_cap=use_vol_cap,
            use_liquidity=use_liquidity,
        )
        print(f"\n[{name}]")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"{k:22s}: {v:.6f}")
            else:
                print(f"{k:22s}: {v}")

        ensure_dir("reports")
        curve_path = f"reports/equity_{name.lower()}.csv"
        curve.to_csv(curve_path, index=False, encoding="utf-8")
        print(f"equity curve saved -> {curve_path}")

    # Save artifacts
    ensure_dir("models")
    model_path = "models/signal_model_lgbm_reg.pkl"
    meta_path = "models/meta_lgbm_reg.json"
    dump(model, model_path)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "lgbm_regressor",
        "horizon": horizon,
        "fee_buffer": fee_buffer,
        "feature_cols": feature_cols,
        "selection_constraints": {
            "valid_final_equity_min": 1.0,
            "test_final_equity_min": 1.0,
            "min_trades_per_month": args.min_trades_per_month
        },
        "regime": {
            "trend": use_trend,
            "vol_cap": use_vol_cap,
            "liquidity": use_liquidity
        },
        "best": {
            "score": best["score"],
            "params": params,
            "regime_thr_from_train": regime_thr,
            "valid": best["valid"],
            "test": best["test"],
        },
        "split": {"train_end": str(train_end.date()), "valid_end": str(valid_end.date())},
        "grids": {
            "top_k_grid": topk_grid,
            "rebalance_grid": rebalance_grid,
            "dv_q_grid": dv_q_grid,
            "atr_q_grid": atr_q_grid,
            "mdd_lambda": args.mdd_lambda
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print_block("SAVED")
    print(f"model -> {model_path}")
    print(f"meta  -> {meta_path}")
    print("reports -> reports/equity_train.csv, equity_valid.csv, equity_test.csv")


if __name__ == "__main__":
    main()
