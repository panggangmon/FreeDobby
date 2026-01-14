# main.py
"""
Portfolio runner (LightGBM regression ranking + regime filter + (paper) rebalancing).

Flow (1 run = 1 rebalance decision):
1) Fetch recent daily OHLCV for each ticker in universe
2) Build features for the latest date
3) Load trained model + meta(feature_cols), predict yhat (expected forward return)
4) Apply regime filters (trend + volatility + liquidity)
5) Pick top_k by yhat
6) Rebalance: sell non-target, buy/trim target to equal-weight
7) Log everything (signals/orders/portfolio snapshots) for post-review
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load

import config
from feature import add_features
from kis_client import KISClient


# ---------------------------
# Logging
# ---------------------------
def _ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def setup_logger(run_dir: str) -> logging.Logger:
    _ensure_dir(run_dir)
    log_file = os.path.join(run_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("TraderLogger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on re-run in same process
    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------
# Config defaults (safe)
# ---------------------------
def cfg(name: str, default):
    return getattr(config, name, default)


@dataclass(frozen=True)
class RunConfig:
    env: str = cfg("ENV", "paper")
    universe: List[str] = field(default_factory=lambda: cfg("UNIVERSE", ["005930"]))  # default: Samsung only
    top_k: int = int(cfg("TOP_K", 3))
    history_days: int = int(cfg("LIVE_HISTORY_DAYS", 260))  # ~1y trading days
    min_history_days: int = int(cfg("MIN_HISTORY_DAYS", 80))

    # model paths (train.py default naming)
    model_path: str = cfg("MODEL_PATH", "models/signal_model_lgbm_reg.pkl")
    meta_path: str = cfg("META_PATH", "models/meta_lgbm_reg.json")

    # execution switches
    execute_paper_orders: bool = bool(cfg("EXECUTE_PAPER_ORDERS", False))
    execute_real_orders: bool = bool(cfg("EXECUTE_REAL_ORDERS", False))

    # rebalance controls
    rebalance_every_n_days: int = int(cfg("REBALANCE_EVERY_N_DAYS", 1))
    rebalance_band: float = float(cfg("REBALANCE_BAND", 0.10))  # 10% band
    min_order_value: int = int(cfg("MIN_ORDER_VALUE", 10_000))  # KRW

    # regime filters (very conservative defaults)
    min_pred_ret: float = float(cfg("MIN_PRED_RET", 0.002))  # 0.2% expected fwd return
    max_atr_pct: float = float(cfg("MAX_ATR_PCT", 0.05))     # 5% ATR/close cap
    max_rvol_20: float = float(cfg("MAX_RVOL_20", 0.06))     # 6% daily std cap
    min_dollar_vol: float = float(cfg("MIN_DOLLAR_VOL", 3e9))  # KRW

    # cost assumptions for paper accounting (only for logs)
    slippage_bps: float = float(cfg("SLIPPAGE_BPS", 5.0))
    commission_bps: float = float(cfg("COMMISSION_BPS", 1.5))

    # output
    log_root: str = cfg("LOG_DIR", "logs")


# ---------------------------
# Helpers: parsing & features
# ---------------------------
def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None or x == "":
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        if x is None or x == "":
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return int(float(x))
    except Exception:
        return default


def _pick_first(d: dict, keys: List[str]):
    for k in keys:
        if k in d and d.get(k) not in (None, ""):
            return d.get(k)
    return None


def parse_current_price(price_data: dict) -> int:
    """
    KIS current price API returns nested 'output' usually.
    We defensively search common keys.
    """
    out = price_data.get("output") or price_data.get("output1") or price_data
    if isinstance(out, list) and out:
        out = out[0]
    if not isinstance(out, dict):
        out = price_data

    v = _pick_first(out, ["stck_prpr", "stck_clpr", "prpr", "close", "last"])
    return _safe_int(v, 0)


def to_ohlcv_df(daily_prices: List[dict]) -> pd.DataFrame:
    """
    KIS daily chart API: output2 list, each item includes:
    stck_bsop_date, stck_oprc, stck_hgpr, stck_lwpr, stck_clpr, acml_vol
    We also support already-normalized dicts with date/open/high/low/close/volume.
    """
    if not daily_prices:
        return pd.DataFrame()

    sample = daily_prices[0]
    # already normalized
    if "date" in sample and "close" in sample:
        df = pd.DataFrame(daily_prices).copy()
    else:
        rows = []
        for r in daily_prices:
            rows.append({
                "date": r.get("stck_bsop_date"),
                "open": _safe_float(r.get("stck_oprc")),
                "high": _safe_float(r.get("stck_hgpr")),
                "low":  _safe_float(r.get("stck_lwpr")),
                "close": _safe_float(r.get("stck_clpr")),
                "volume": _safe_float(r.get("acml_vol")),
            })
        df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_latest_features(df: pd.DataFrame) -> pd.Series:
    """
    Build a feature set aligned with training (feature.py).
    Inference selects only model feature_cols later.
    """
    df = add_features(df.copy())
    latest = df.iloc[-1]
    return latest


def resolve_regime_config(meta: dict, rc: RunConfig) -> dict:
    regime_meta = meta.get("regime") or {}
    best = meta.get("best") or {}
    thr = best.get("regime_thr_from_train") or {}

    # If meta is missing, keep current default behavior (all enabled).
    use_trend = bool(regime_meta.get("trend", True))
    use_vol_cap = bool(regime_meta.get("vol_cap", True))
    use_liquidity = bool(regime_meta.get("liquidity", True))

    atr_cap = _safe_float(thr.get("atr_cap"), rc.max_atr_pct)
    dv_min = _safe_float(thr.get("dv_min"), rc.min_dollar_vol)

    return {
        "use_trend": use_trend,
        "use_vol_cap": use_vol_cap,
        "use_liquidity": use_liquidity,
        "atr_cap": atr_cap,
        "dv_min": dv_min,
    }


def regime_filter(latest: pd.Series, rc: RunConfig, regime_cfg: dict) -> Tuple[bool, Dict[str, bool]]:
    use_trend = bool(regime_cfg.get("use_trend", True))
    use_vol_cap = bool(regime_cfg.get("use_vol_cap", True))
    use_liquidity = bool(regime_cfg.get("use_liquidity", True))
    atr_cap = _safe_float(regime_cfg.get("atr_cap"), rc.max_atr_pct)
    dv_min = _safe_float(regime_cfg.get("dv_min"), rc.min_dollar_vol)

    trend_ok = True
    if use_trend:
        trend_ok = bool(_safe_float(latest.get("dist_ma20"), np.nan) > 0 and _safe_float(latest.get("ma20_slope_5"), np.nan) > 0)

    vol_ok = True
    if use_vol_cap:
        vol_ok = bool(_safe_float(latest.get("atr_14_pct"), np.nan) <= atr_cap)

    liq_ok = True
    if use_liquidity:
        liq_ok = bool(_safe_float(latest.get("dollar_volume"), np.nan) >= dv_min)

    flags = {"trend_ok": trend_ok, "vol_ok": vol_ok, "liq_ok": liq_ok}
    return (trend_ok and vol_ok and liq_ok), flags


# ---------------------------
# Balance parsing & rebalance plan
# ---------------------------
@dataclass
class Holding:
    ticker: str
    qty: int
    eval_value: int = 0  # KRW
    avg_price: int = 0


def parse_balance(balance: dict) -> Tuple[int, int, Dict[str, Holding]]:
    """
    Returns: (cash, total_eval, holdings_map)
    KIS inquire-balance returns output1 (holdings) and output2 (summary). fileciteturn32file4
    """
    holdings_raw = balance.get("output1") or []
    if isinstance(holdings_raw, dict):
        holdings_raw = [holdings_raw]

    holdings: Dict[str, Holding] = {}
    for r in holdings_raw:
        ticker = r.get("pdno") or r.get("stck_shrn_iscd") or r.get("stock_code")
        if not ticker:
            continue
        qty = _safe_int(r.get("hldg_qty"), 0)
        if qty <= 0:
            continue
        eval_value = _safe_int(_pick_first(r, ["evlu_amt", "evlu_pfls_amt", "evlu_amt2"]), 0)
        avg_price = _safe_int(_pick_first(r, ["pchs_avg_pric", "avg_prvs", "pchs_pric"]), 0)
        holdings[ticker] = Holding(ticker=ticker, qty=qty, eval_value=eval_value, avg_price=avg_price)

    summary = balance.get("output2") or {}
    if isinstance(summary, list):
        summary = summary[0] if summary else {}

    cash = _safe_int(_pick_first(summary, ["dnca_tot_amt", "ord_psbl_cash", "ord_psbl_cash_amt", "cash"]), 0)
    total_eval = _safe_int(_pick_first(summary, ["tot_evlu_amt", "tot_evlu_amt2"]), 0)
    if total_eval <= 0:
        total_eval = cash

    return cash, total_eval, holdings


@dataclass
class Order:
    ticker: str
    side: str  # "buy" | "sell"
    qty: int
    price: int
    reason: str


def compute_target_values(total_equity: int, tickers: List[str]) -> Dict[str, int]:
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: int(total_equity * w) for t in tickers}


def plan_rebalance(
    universe: List[str],
    selected: List[str],
    cash: int,
    total_equity: int,
    holdings: Dict[str, Holding],
    prices: Dict[str, int],
    rc: RunConfig,
) -> List[Order]:
    """
    Equal-weight target among selected. Sell anything in universe not selected.
    Rebalance-band: only trade if current value deviates by > band.
    """
    orders: List[Order] = []
    targets = compute_target_values(total_equity, selected)

    # compute current value map (fall back to qty*price if eval_value absent)
    cur_val: Dict[str, int] = {}
    for t in universe:
        h = holdings.get(t)
        if not h:
            cur_val[t] = 0
            continue
        px = prices.get(t, 0)
        if h.eval_value > 0:
            cur_val[t] = h.eval_value
        elif px > 0:
            cur_val[t] = int(h.qty * px)
        else:
            cur_val[t] = 0

    # 1) sell non-target holdings (within universe)
    for t, h in holdings.items():
        if t not in universe:
            continue
        if t not in targets:
            px = prices.get(t, 0)
            if px <= 0:
                continue
            value = h.qty * px
            if value >= rc.min_order_value:
                orders.append(Order(ticker=t, side="sell", qty=h.qty, price=px, reason="not_in_top_k"))
            continue

    # 2) adjust target holdings to equal-weight
    for t, target_value in targets.items():
        px = prices.get(t, 0)
        if px <= 0:
            continue

        h = holdings.get(t)
        current_value = cur_val.get(t, 0)

        # if inside band, skip
        lower = int(target_value * (1.0 - rc.rebalance_band))
        upper = int(target_value * (1.0 + rc.rebalance_band))
        if lower <= current_value <= upper:
            continue

        diff = target_value - current_value
        # buy if underweight, sell if overweight
        if diff > 0:
            qty = int(diff // px)
            if qty < 1:
                continue
            est_value = qty * px
            if est_value < rc.min_order_value:
                continue
            # do not exceed available cash (soft check)
            if est_value > cash:
                qty = int(cash // px)
            if qty >= 1 and qty * px >= rc.min_order_value:
                orders.append(Order(ticker=t, side="buy", qty=qty, price=px, reason="rebalance_to_equal_weight"))
                cash -= qty * px
        else:
            if not h:
                continue
            qty = int((-diff) // px)
            qty = min(qty, h.qty)
            if qty < 1:
                continue
            if qty * px >= rc.min_order_value:
                orders.append(Order(ticker=t, side="sell", qty=qty, price=px, reason="rebalance_to_equal_weight"))

    return orders


# ---------------------------
# State: rebalance cadence
# ---------------------------
def load_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state_path: str, state: dict):
    _ensure_dir(os.path.dirname(state_path))
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def should_rebalance(state: dict, rc: RunConfig, today_ymd: str) -> bool:
    last = state.get("last_rebalance_ymd")
    if not last:
        return True
    try:
        d0 = datetime.strptime(last, "%Y%m%d").date()
        d1 = datetime.strptime(today_ymd, "%Y%m%d").date()
        delta = (d1 - d0).days
        return delta >= rc.rebalance_every_n_days
    except Exception:
        return True


# ---------------------------
# Model inference
# ---------------------------
def load_model_and_meta(rc: RunConfig, logger: logging.Logger):
    if not os.path.exists(rc.model_path):
        raise FileNotFoundError(f"Model not found: {rc.model_path}")
    model = load(rc.model_path)

    meta = {}
    if os.path.exists(rc.meta_path):
        with open(rc.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    feature_cols = meta.get("feature_cols") or meta.get("feature_columns") or []
    if not feature_cols:
        logger.warning("meta file missing feature_cols; will infer from live feature set intersection.")
    return model, meta, feature_cols


def predict_yhat(model, x_row: pd.Series, feature_cols: List[str], logger: logging.Logger) -> float:
    # Build X in the exact order expected by the model (as saved in meta)
    if feature_cols:
        X = pd.DataFrame([{c: _safe_float(x_row.get(c), np.nan) for c in feature_cols}])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        # fallback: use all numeric fields in x_row
        numeric = {k: _safe_float(v, np.nan) for k, v in x_row.items()}
        X = pd.DataFrame([numeric]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        yhat = float(model.predict(X)[0])
    except Exception as e:
        logger.error(f"Model predict failed: {e}")
        raise
    return yhat


# ---------------------------
# Main
# ---------------------------
def main():
    rc = RunConfig()

    today = datetime.now()
    today_ymd = today.strftime("%Y%m%d")
    run_dir = os.path.join(rc.log_root, f"paper_run_{today.strftime('%Y%m%d_%H%M%S')}")
    logger = setup_logger(run_dir)

    logger.info("==============================================")
    logger.info("========== Portfolio Runner (LightGBM) =======")
    logger.info(f"ENV={rc.env}  TOP_K={rc.top_k}  UNIVERSE={len(rc.universe)}")
    logger.info(f"EXECUTE_PAPER_ORDERS={rc.execute_paper_orders}  EXECUTE_REAL_ORDERS={rc.execute_real_orders}")
    logger.info("==============================================")

    state_path = os.path.join(rc.log_root, "trading_state.json")
    state = load_state(state_path)

    # Drop-reason accounting (for post-review)
    drop_counts = {
        "universe": len(rc.universe),
        "signals_built": 0,
        "insufficient_history": 0,
        "price_missing": 0,
        "signal_error": 0,
    }
    drop_samples = {k: [] for k in drop_counts.keys()}

    def _drop(key: str, ticker: str, max_samples: int = 12):
        drop_counts[key] = drop_counts.get(key, 0) + 1
        if key not in drop_samples:
            drop_samples[key] = []
        if ticker and len(drop_samples[key]) < max_samples:
            drop_samples[key].append(ticker)

    if not should_rebalance(state, rc, today_ymd):
        logger.info(f"Skip: rebalance cadence not met (every {rc.rebalance_every_n_days} days).")
        return

    # Initialize client
    client = KISClient(
        env=rc.env,
        app_key=cfg("APP_KEY", ""),
        app_secret=cfg("APP_SECRET", ""),
        cano=cfg("CANO", ""),
        acnt_prdt_cd=cfg("ACNT_PRDT_CD", ""),
    )

    # Load model
    model, meta, feature_cols = load_model_and_meta(rc, logger)
    regime_cfg = resolve_regime_config(meta, rc)
    logger.info(
        "Regime config: trend=%s vol_cap=%s liquidity=%s atr_cap=%.5f dv_min=%.0f",
        regime_cfg["use_trend"],
        regime_cfg["use_vol_cap"],
        regime_cfg["use_liquidity"],
        regime_cfg["atr_cap"],
        regime_cfg["dv_min"],
    )

    # Balance snapshot
    bal = client.get_stock_balance()
    cash, total_equity, holdings = parse_balance(bal)
    logger.info(f"Account snapshot: cash={cash:,}  total_equity={total_equity:,}  holdings={len(holdings)}")

    # Save portfolio_before
    before_rows = []
    for t, h in holdings.items():
        before_rows.append({"ticker": t, "qty": h.qty, "eval_value": h.eval_value, "avg_price": h.avg_price})
    if before_rows:
        pd.DataFrame(before_rows).sort_values("ticker").to_csv(os.path.join(run_dir, "portfolio_before.csv"), index=False)
    else:
        pd.DataFrame(columns=["ticker", "qty", "eval_value", "avg_price"]).to_csv(os.path.join(run_dir, "portfolio_before.csv"), index=False)

    # 1) Build signals
    signals = []
    prices: Dict[str, int] = {}

    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=int(rc.history_days * 1.6))).strftime("%Y%m%d")  # calendar buffer

    for ticker in rc.universe:
        try:
            daily = client.get_historical_daily_price(ticker, start_date, end_date)
            df = to_ohlcv_df(daily)
            if len(df) < rc.min_history_days:
                _drop("insufficient_history", ticker)
                logger.info(f"[{ticker}] skip: insufficient history ({len(df)} rows)")
                continue

            latest = build_latest_features(df)
            ok, flags = regime_filter(latest, rc, regime_cfg)

            # current price for planning (once)
            price_data = client.get_current_price(ticker)
            px = parse_current_price(price_data)
            prices[ticker] = px
            if px <= 0:
                _drop("price_missing", ticker)
                logger.info(f"[{ticker}] skip: missing/invalid current price (px={px})")
                continue

            yhat = predict_yhat(model, latest, feature_cols, logger)

            signals.append({
                "date": today_ymd,
                "ticker": ticker,
                "yhat": yhat,
                "price": px,
                "regime_ok": ok,
                **flags,
                # keep a few key features for debugging
                "dist_ma20": _safe_float(latest.get("dist_ma20")),
                "ma20_slope_5": _safe_float(latest.get("ma20_slope_5")),
                "atr_14_pct": _safe_float(latest.get("atr_14_pct")),
                "vol_20": _safe_float(latest.get("vol_20")),
                "dollar_volume": _safe_float(latest.get("dollar_volume")),
                "vol_z_20": _safe_float(latest.get("vol_z_20")),
                "rsi_14": _safe_float(latest.get("rsi_14")),
            })
        except Exception as e:
            _drop("signal_error", ticker)
            logger.error(f"[{ticker}] failed to build signal: {e}", exc_info=True)
            continue

    sig_df = pd.DataFrame(signals)
    if sig_df.empty:
        logger.warning("No signals generated. Exiting.")
        return

    sig_df = sig_df.sort_values("yhat", ascending=False).reset_index(drop=True)
    sig_df.to_csv(os.path.join(run_dir, "signals.csv"), index=False)


    # Drop summary (signals -> regime -> pred_cut -> selected)
    drop_counts["signals_built"] = int(len(sig_df))

    pred_ok = sig_df["yhat"] >= rc.min_pred_ret
    after_pred = sig_df[pred_ok]
    after_regime = sig_df[sig_df["regime_ok"] == True]
    eligible = sig_df[(sig_df["regime_ok"] == True) & pred_ok]

    # Overlapping reason counts (diagnostic)
    reason_counts = {
        "trend_fail": int((sig_df["trend_ok"] == False).sum()),
        "vol_fail": int((sig_df["vol_ok"] == False).sum()),
        "liq_fail": int((sig_df["liq_ok"] == False).sum()),
        "pred_below_min": int((~pred_ok).sum()),
        "regime_fail": int((sig_df["regime_ok"] == False).sum()),
    }

    # Primary reason (mutually exclusive; for quick reading)
    primary = []
    for _, r in sig_df.iterrows():
        if not bool(r.get("trend_ok")):
            primary.append("trend_fail")
        elif not bool(r.get("vol_ok")):
            primary.append("vol_fail")
        elif not bool(r.get("liq_ok")):
            primary.append("liq_fail")
        elif not bool(r.get("regime_ok")):
            primary.append("regime_fail")
        elif float(r.get("yhat", 0.0)) < rc.min_pred_ret:
            primary.append("pred_below_min")
        else:
            primary.append("eligible")

    primary_counts = pd.Series(primary).value_counts().to_dict()

    logger.info("----------- Drop summary -----------")
    logger.info(f"universe={drop_counts['universe']}  signals_built={len(sig_df)}  after_regime={len(after_regime)}  after_pred={len(after_pred)}  eligible={len(eligible)}  top_k={rc.top_k}")
    logger.info(f"drop_counts: insufficient_history={drop_counts.get('insufficient_history',0)}  price_missing={drop_counts.get('price_missing',0)}  signal_error={drop_counts.get('signal_error',0)}")
    logger.info(f"reason_counts(overlap): {reason_counts}")
    logger.info(f"primary_counts(exclusive): {primary_counts}")

    # Save as JSON for later review
    with open(os.path.join(run_dir, "drop_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": today_ymd,
                "pipeline": {
                    "universe": drop_counts["universe"],
                    "signals_built": int(len(sig_df)),
                    "after_regime": int(len(after_regime)),
                    "after_pred_cut": int(len(after_pred)),
                    "eligible": int(len(eligible)),
                    "top_k": rc.top_k,
                    "min_pred_ret": rc.min_pred_ret,
                },
                "drop_counts": drop_counts,
                "reason_counts_overlap": reason_counts,
                "primary_counts_exclusive": primary_counts,
                "drop_samples": drop_samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


    # 2) Select top_k under constraints
    cand = sig_df[(sig_df["regime_ok"] == True) & (sig_df["yhat"] >= rc.min_pred_ret)].copy()
    selected = cand.head(rc.top_k)["ticker"].tolist()

    logger.info(f"Selected ({len(selected)}/{rc.top_k}): {selected}")
    with open(os.path.join(run_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"date": today_ymd, "selected": selected}, f, ensure_ascii=False, indent=2)

    # 3) Plan rebalance orders
    orders = plan_rebalance(
        universe=rc.universe,
        selected=selected,
        cash=cash,
        total_equity=total_equity,
        holdings=holdings,
        prices=prices,
        rc=rc,
    )

    ord_rows = []
    for o in orders:
        ord_rows.append({"date": today_ymd, "ticker": o.ticker, "side": o.side, "qty": o.qty, "price": o.price, "reason": o.reason})
    pd.DataFrame(ord_rows).to_csv(os.path.join(run_dir, "orders_planned.csv"), index=False)

    if not orders:
        logger.info("No rebalance orders needed (within bands or no eligible selection).")
        state["last_rebalance_ymd"] = today_ymd
        save_state(state_path, state)
        return

    # 4) Execute (paper or real) OR log-only
    can_execute = False
    if rc.env == "paper" and rc.execute_paper_orders:
        can_execute = True
    if rc.env == "real" and rc.execute_real_orders:
        can_execute = True

    exec_rows = []
    for o in orders:
        logger.info(f"ORDER {o.side.upper()} {o.ticker} qty={o.qty} price={o.price} reason={o.reason}")
        if not can_execute:
            exec_rows.append({**o.__dict__, "executed": False, "message": "DRY_RUN"})
            continue

        try:
            if o.side == "buy":
                resp = client.buy(o.ticker, o.qty, o.price)
            else:
                resp = client.sell(o.ticker, o.qty, o.price)

            ok = bool(resp.get("success"))
            msg = resp.get("message", "")
            exec_rows.append({**o.__dict__, "executed": ok, "message": msg, "response": json.dumps(resp.get("response"), ensure_ascii=False)})
        except Exception as e:
            exec_rows.append({**o.__dict__, "executed": False, "message": str(e), "response": ""})
            logger.error(f"Order failed: {o.side} {o.ticker} - {e}", exc_info=True)

    pd.DataFrame(exec_rows).to_csv(os.path.join(run_dir, "orders_executed.csv"), index=False)

    # 5) Save after snapshot (best-effort)
    try:
        bal2 = client.get_stock_balance()
        cash2, total_equity2, holdings2 = parse_balance(bal2)
        after_rows = []
        for t, h in holdings2.items():
            after_rows.append({"ticker": t, "qty": h.qty, "eval_value": h.eval_value, "avg_price": h.avg_price})
        if after_rows:
            pd.DataFrame(after_rows).sort_values("ticker").to_csv(os.path.join(run_dir, "portfolio_after.csv"), index=False)
        else:
            pd.DataFrame(columns=["ticker", "qty", "eval_value", "avg_price"]).to_csv(os.path.join(run_dir, "portfolio_after.csv"), index=False)
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"date": today_ymd, "cash_before": cash, "equity_before": total_equity, "cash_after": cash2, "equity_after": total_equity2,
                 "selected": selected, "orders": ord_rows, "executed": can_execute},
                f,
                ensure_ascii=False, indent=2
            )
    except Exception as e:
        logger.warning(f"Could not write after snapshot: {e}")

    state["last_rebalance_ymd"] = today_ymd
    save_state(state_path, state)

    logger.info("Done.")


if __name__ == "__main__":
    main()
