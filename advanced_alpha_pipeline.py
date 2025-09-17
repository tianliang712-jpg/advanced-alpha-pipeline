# -*- coding: utf-8 -*-
# advanced_alpha_pipeline.py
# 单文件全集成（在此前版本“只增不减”）：
# TuShare优先 + AKShare兜底(>=1年历史) + 训练/预测/日志/图表/HTML报告
# + 行业/概念强度 + 中性化IC/ICIR + T+0/2/3/5日执行回测 + 阈值搜索 + 更精细涨跌停约束
# + 涨停可达性子模型 + CatBoost/深度学习（PyTorch-MLP）可选集成
# + Streamlit 前台看板（含：IC/ICIR、IC时间序列、行业/概念Top5、执行层多周期展示）
import os, warnings, base64, math, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---- 第三方包 ----
import tushare as ts
import akshare as ak

from sklearn.linear_model import LinearRegression, Ridge
import lightgbm as lgb
import xgboost as xgb

# 可选：CatBoost / PyTorch
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

# 可选：Streamlit（未装也能纯后台运行）
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ================= 配置 =================
TS_TOKEN = "452b80dde79861ca8a9cd5ae6bee2d7defc0d85b0225f56f73678902"  # 你的 TuShare Key
ts.set_token(TS_TOKEN)
pro = ts.pro_api(TS_TOKEN)

DATA_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)

# 至少1年历史（适当多抓一些）
START_DATE = (datetime.today() - timedelta(days=400)).strftime("%Y%m%d")
TOPK = 10
ROLL_TRAIN_WINDOW = 120
RANDOM_STATE = 2025
NEW_STOCK_DAYS = 60        # 上市未满N日过滤
UNIVERSE_SIZE = 600        # 每日训练/打分股票数量（可调大）

# 执行/回测参数（新增：包含 T+0 与 5日）
EXEC_HOLD_DAYS_LIST = [0, 2, 3, 5]
STOP_WIN = 0.10
STOP_LOSS = -0.05

# 阈值搜索网格
SEARCH_TOPK_GRID = [5, 10, 20]
SEARCH_PROB_TH = [0.55, 0.60, 0.65, 0.70]

# ================= 工具函数 =================
def today_str(): return datetime.today().strftime("%Y%m%d")

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ts_code_to_symbol(ts_code:str):
    """000001.SZ -> 000001；600519.SH -> 600519"""
    return ts_code.split(".")[0]

def is_chinext(ts_code:str)->bool:
    # 创业/科创（300/301/688）20% 涨跌幅
    code = ts_code.split(".")[0]
    return code.startswith("300") or code.startswith("301") or code.startswith("688")

# ========= 更精细的涨跌停约束 =========
def limit_up_threshold_by_market_name(ts_code:str, market:str, name:str)->float:
    """
    约束优先级：
    - ST: 5%
    - BJ（北交所）: 30%  （stock_basic.market == 'BJ' 时）
    - 创业/科创: 20%（300/301/688）
    - 其他A股: 10%
    """
    try:
        nm = (name or "").upper()
        if "ST" in nm:  # *ST 或 ST*
            return 0.05
    except Exception:
        pass
    try:
        if str(market).upper() == "BJ":
            return 0.30
    except Exception:
        pass
    return 0.20 if is_chinext(ts_code) else 0.10

# ================= 修复旧日志工具 =================
def repair_log():
    log_file = os.path.join(DATA_DIR, "log.csv")
    if not os.path.exists(log_file):
        print("[INFO] log.csv 不存在，无需修复")
        return
    df = pd.read_csv(log_file)
    for c in ["ret_pred", "strong_prob"]:
        if c not in df.columns:
            df[c] = np.nan
            print(f"[INFO] 已为 log.csv 补充 {c} 列")
    before = len(df)
    df = df.drop_duplicates(subset=["ts_code", "next_day"])
    after = len(df)
    if after < before:
        print(f"[INFO] 已删除重复记录 {before - after} 行")
    df["next_day"] = df["next_day"].astype(str)
    df.to_csv(log_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] log.csv 修复完成 -> {log_file}")

# ================= 数据层（TuShare优先 + AKShare兜底） =================
def fetch_stock_basic():
    return pro.stock_basic(exchange="", list_status="L",
                           fields="ts_code,symbol,name,area,industry,market,list_date")

def fetch_daily(ts_code, start_date, end_date):
    """
    优先 TuShare（含前复权因子修正 close/open/high/low），失败/为空则用 AKShare(qfq) 兜底。
    """
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if adj is not None and not adj.empty:
                df = df.merge(adj, on=["ts_code","trade_date"], how="left")
                df = df.sort_values("trade_date")
                last_adj = df["adj_factor"].iloc[-1]
                for col in ["close","open","high","low"]:
                    if col in df.columns:
                        df[col] = df[col] * df["adj_factor"] / last_adj
                return df
            else:
                df = df.sort_values("trade_date")
                return df
    except Exception as e:
        print(f"[WARN] TuShare获取失败 {ts_code}: {e}")

    try:
        sym = ts_code_to_symbol(ts_code)
        ak_df = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                   start_date=start_date, end_date=end_date,
                                   adjust="qfq")
        if ak_df is not None and not ak_df.empty:
            rename_map = {
                "日期":"trade_date","开盘":"open","收盘":"close",
                "最高":"high","最低":"low","成交量":"vol","成交额":"amount"
            }
            for k in rename_map:
                if k in ak_df.columns:
                    ak_df.rename(columns={k: rename_map[k]}, inplace=True)
            ak_df["ts_code"] = ts_code
            ak_df["trade_date"] = pd.to_datetime(ak_df["trade_date"]).dt.strftime("%Y%m%d")
            return ak_df[["ts_code","trade_date","open","high","low","close","vol","amount"]]
    except Exception as e:
        print(f"[ERROR] AKShare获取失败 {ts_code}: {e}")

    return pd.DataFrame()

# ================= 因子工程 =================
def build_features(df):
    df = df.copy()
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_5"]  = df["close"].pct_change(5)
    df["ma_5"]   = df["close"].rolling(5).mean()
    df["bias_5"] = (df["close"] - df["ma_5"])/df["ma_5"]
    df["volatility_10"] = df["ret_1"].rolling(10).std()
    if "amount" in df.columns:
        df["ln_mv"] = np.log(df["amount"] + 1)
    else:
        df["ln_mv"] = np.log((df["close"]*df.get("vol",0)).fillna(0)+1)
    df["next_ret_1"] = df["close"].pct_change(-1)
    return df.dropna()

# ====== 行业/概念强度 ======
def compute_industry_strength(panel_day:pd.DataFrame)->pd.DataFrame:
    use = panel_day.dropna(subset=["industry","ret_1"])
    if use.empty:
        return pd.DataFrame(columns=["ts_code","industry","ind_strength","ind_leader_strength"])
    grp = use.groupby("industry")["ret_1"]
    ind_strength = grp.mean().rename("ind_strength")
    leader_strength = grp.max().rename("ind_leader_strength")
    out = use[["ts_code","industry"]].drop_duplicates().merge(
        ind_strength, on="industry").merge(leader_strength,on="industry")
    return out

def fetch_ths_concepts_safe():
    try:
        return ak.stock_board_concept_name_ths()
    except Exception:
        return pd.DataFrame()

def fetch_ths_concept_members_safe(board:str):
    try:
        return ak.stock_board_concept_cons_ths(symbol=board)
    except Exception:
        return pd.DataFrame()

def ts_code_from_6_or_not(code: str) -> str:
    code = str(code)
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"

def compute_concept_strength_top5_for_day(date_str:str)->pd.DataFrame:
    concepts = fetch_ths_concepts_safe()
    if concepts.empty:
        return pd.DataFrame(columns=["concept","strength"])
    top_rows = concepts.head(60)
    rows = []
    for _, r in top_rows.iterrows():
        board = r.get("板块名称") or r.get("概念名称") or r.get("name")
        if not isinstance(board, str):
            continue
        cons = fetch_ths_concept_members_safe(board)
        if cons is None or cons.empty:
            continue
        code_col = "代码" if "代码" in cons.columns else ("股票代码" if "股票代码" in cons.columns else None)
        if code_col is None:
            continue
        codes = cons[code_col].astype(str).unique().tolist()[:25]
        rets = []
        for c in codes:
            ts_c = ts_code_from_6_or_not(c)
            d = pro.daily(ts_code=ts_c, start_date=date_str, end_date=date_str)
            if d is not None and not d.empty:
                d = d.sort_values("trade_date")
                r1 = d["close"].pct_change().iloc[-1] if len(d)>=2 else 0.0
                rets.append(r1)
        if rets:
            rows.append({"concept": board, "strength": float(np.mean(rets))})
    if not rows:
        return pd.DataFrame(columns=["concept","strength"])
    df = pd.DataFrame(rows).sort_values("strength", ascending=False).head(5)
    df.to_csv(os.path.join(DATA_DIR, "concept_top5.csv"), index=False, encoding="utf-8-sig")
    return df

# ====== 中性化IC/ICIR ======
def neutralize_series_by_industry_size(df:pd.DataFrame, ret_col="next_ret_1", size_col="ln_mv", ind_col="industry"):
    use = df.dropna(subset=[ret_col, size_col, ind_col]).copy()
    if use.empty: 
        return pd.Series([], dtype=float)
    ind_dum = pd.get_dummies(use[ind_col], prefix="ind", drop_first=True)
    X = pd.concat([ind_dum, use[[size_col]]], axis=1)
    y = use[ret_col].values
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y)
    y_hat = model.predict(X)
    resid = y - y_hat
    out = pd.Series(resid, index=use.index, name="ret_neu")
    return out

def compute_ic_series(train_df: pd.DataFrame, feature_cols: list):
    if train_df.empty:
        return pd.DataFrame(columns=["trade_date","ic"]), np.nan, np.nan
    # 与主流程一致：两树 + （可选）CatBoost + （可选）MLP → 融合
    X = train_df[feature_cols].values
    y = train_df["next_ret_1"].values

    preds_list = []

    reg1 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
    reg1.fit(X, y); preds_list.append(reg1.predict(X))

    reg2 = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
    reg2.fit(X, y); preds_list.append(reg2.predict(X))

    if CATBOOST_AVAILABLE:
        reg3 = CatBoostRegressor(
            depth=6, iterations=300, learning_rate=0.05, verbose=False, random_seed=RANDOM_STATE
        )
        reg3.fit(X, y)
        preds_list.append(reg3.predict(X))

    if TORCH_AVAILABLE:
        preds_list.append(train_torch_mlp_and_predict(X, y, X, epochs=30, hidden=64))

    stack = np.vstack(preds_list).T
    lin = LinearRegression().fit(stack, np.mean(stack, axis=1))
    score = lin.predict(stack)

    df = train_df.copy()
    df["score"] = score
    ics = []
    for d, g in df.groupby("trade_date"):
        if g["score"].nunique() <= 1 or g["next_ret_1"].nunique() <= 1:
            continue
        ics.append({"trade_date": d, "ic": g["score"].corr(g["next_ret_1"])})
    ic_df = pd.DataFrame(ics).sort_values("trade_date")
    if ic_df.empty:
        return ic_df, np.nan, np.nan
    ic_mean = float(ic_df["ic"].mean())
    ic_std  = float(ic_df["ic"].std(ddof=1)) if len(ic_df) > 1 else 0.0
    icir = (ic_mean/ic_std*np.sqrt(252)) if ic_std>0 else np.nan
    ic_df.to_csv(os.path.join(DATA_DIR,"ic_series.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9,4))
    plt.plot(pd.to_datetime(ic_df["trade_date"]), ic_df["ic"], marker="o", linewidth=1)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("每日IC（训练窗口内，融合分数 vs 次日收益）")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR,"ic_timeseries.png"))
    plt.close()
    return ic_df, ic_mean, icir

# ====== 可选：PyTorch 轻量 MLP 回归 ======
def train_torch_mlp_and_predict(X_train, y_train, X_pred, epochs=30, hidden=64):
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        def forward(self, x): return self.net(x)

    device = torch.device("cpu")
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32, device=device)
    Xte = torch.tensor(X_pred, dtype=torch.float32, device=device)

    model = MLP(Xtr.shape[1], hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        yhat = model(Xte).cpu().numpy().ravel()
    return yhat

# ====== “涨停可达性”子模型 ======
def train_untradable_model(train_df:pd.DataFrame, feature_cols:list):
    if train_df.empty: 
        return None
    df = train_df.copy()
    # 使用更精细阈值：基于 market/name
    df["limit_th"] = df.apply(lambda r: limit_up_threshold_by_market_name(
        r["ts_code"], r.get("market",""), r.get("name","")
    ), axis=1)
    df["y_limit"] = (df["next_ret_1"] >= (df["limit_th"] - 0.005)).astype(int)
    if df["y_limit"].sum() == 0:
        return None
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=RANDOM_STATE)
    X = df[feature_cols].fillna(0).values
    y = df["y_limit"].values
    clf.fit(X, y)
    return clf

# ====== 主模型/stacking（可选CatBoost/深度学习） ======
def train_models_extended(X, y_class, y_reg):
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)

    reg_list = []
    reg1 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
    reg1.fit(X, y_reg); reg_list.append(lambda Z: reg1.predict(Z))

    reg2 = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
    reg2.fit(X, y_reg); reg_list.append(lambda Z: reg2.predict(Z))

    if CATBOOST_AVAILABLE:
        reg3 = CatBoostRegressor(
            depth=6, iterations=300, learning_rate=0.05, verbose=False, random_seed=RANDOM_STATE
        )
        reg3.fit(X, y_reg)
        reg_list.append(lambda Z: reg3.predict(Z))

    if TORCH_AVAILABLE:
        # 训练一次 MLP，用闭包缓存权重后的预测函数
        yhat = train_torch_mlp_and_predict(X, y_reg, X, epochs=30, hidden=64)
        # 为了简单，这里再训一遍获取预测器（避免保存模型权重到文件）
        def torch_pred(Z):
            return train_torch_mlp_and_predict(X, y_reg, Z, epochs=10, hidden=64)
        reg_list.append(torch_pred)

    clf.fit(X, y_class)
    return clf, reg_list

def stacking_predict_extended(clf, reg_list, X):
    preds = [rfn(X) for rfn in reg_list]
    stack = np.vstack(preds).T
    lin = LinearRegression().fit(stack, np.mean(stack, axis=1))
    return clf.predict_proba(X)[:,1], lin.predict(stack)

# ====== 在线增量学习&打分 ======
def daily_update():
    basics = fetch_stock_basic()
    today = today_str()
    pred_file = os.path.join(DATA_DIR, f"predictions_{today}.csv")
    if os.path.exists(pred_file):
        print(f"[INFO] 已有 {pred_file}，跳过")
        return None, None, None, None

    universe = basics.ts_code.head(UNIVERSE_SIZE).tolist()
    frames = []
    bidx = basics.set_index("ts_code")
    for code in universe:
        d = fetch_daily(code, START_DATE, today)
        if d.empty: continue
        f = build_features(d)
        if f.empty: continue
        f["ts_code"] = code
        # 追加 name / market，供更精细涨跌停使用
        f["name"]   = bidx.loc[code,"name"] if "name" in bidx.columns else ""
        f["market"] = (bidx.loc[code,"market"] if "market" in bidx.columns else "").upper()
        f["industry"] = bidx.loc[code,"industry"] if "industry" in bidx.columns else "NA"
        list_date = bidx.loc[code,"list_date"]
        f["list_days"] = (pd.to_datetime(f["trade_date"]) - pd.to_datetime(list_date)).dt.days
        frames.append(f)
    if not frames: 
        return None, None, None, None

    panel = pd.concat(frames, ignore_index=True)

    # 行业/概念强度
    feature_base = ["ret_1","ret_5","bias_5","volatility_10","ln_mv"]
    last_day = panel["trade_date"].max()
    day_cross = panel[panel["trade_date"]==last_day].copy()
    ind_strength_df = compute_industry_strength(day_cross)
    panel = panel.merge(ind_strength_df, on=["ts_code","industry"], how="left")
    for c in ["ind_strength","ind_leader_strength"]:
        panel[c] = panel[c].fillna(0.0)

    concept_top5 = compute_concept_strength_top5_for_day(last_day)
    concept_strength_val = float(concept_top5["strength"].mean()) if (concept_top5 is not None and not concept_top5.empty) else 0.0
    panel["concept_strength"] = concept_strength_val

    # 行业Top5输出
    if not day_cross.empty:
        tmp = panel[panel["trade_date"]==last_day].dropna(subset=["industry","ind_strength"])
        if not tmp.empty:
            ind_top5 = tmp.groupby("industry")["ind_strength"].mean().reset_index().sort_values("ind_strength", ascending=False).head(5)
            ind_top5.to_csv(os.path.join(DATA_DIR,"industry_top5.csv"), index=False, encoding="utf-8-sig")

    # 中性化
    panel["ret_neu"] = np.nan
    idx_mask = panel["trade_date"].isin(panel["trade_date"].unique()[-ROLL_TRAIN_WINDOW:])
    sub = panel[idx_mask].copy()
    resid = neutralize_series_by_industry_size(sub, ret_col="next_ret_1", size_col="ln_mv", ind_col="industry")
    panel.loc[sub.index, "ret_neu"] = resid

    # 训练
    feature_cols = [c for c in feature_base+["ind_strength","ind_leader_strength","concept_strength"] if c in panel.columns]
    df_train = panel.dropna(subset=feature_cols+["next_ret_1"])
    if df_train.empty:
        print("[WARN] 训练数据为空")
        return None, None, None, None

    last_days = df_train.trade_date.unique()[-ROLL_TRAIN_WINDOW:]
    train = df_train[df_train.trade_date.isin(last_days)].copy()

    # IC序列 + IC/ICIR
    ic_df, ic_mean, icir = compute_ic_series(train, feature_cols)
    if ic_df is not None and not ic_df.empty:
        pd.DataFrame([{"ic_mean": ic_mean, "icir": icir, "window_days": len(ic_df)}]).to_csv(
            os.path.join(DATA_DIR,"ic_summary.csv"), index=False, encoding="utf-8-sig"
        )

    # 主模型 + 可选模型
    clf, reg_list = train_models_extended(
        train[feature_cols].values,
        (train["next_ret_1"]>0.01).astype(int),
        train["next_ret_1"].values
    )

    # 预测最新交易日
    pred = df_train[df_train.trade_date==last_day].copy()
    prob, preds = stacking_predict_extended(clf, reg_list, pred[feature_cols].values)
    pred["strong_prob"], pred["ret_pred_raw"] = prob, preds

    # 子模型：涨停可达性（使用更精细阈值）
    untrad_clf = train_untradable_model(train, feature_cols)
    if untrad_clf is not None:
        p_untrad = untrad_clf.predict_proba(pred[feature_cols].values)[:,1]
        pred["untrad_prob"] = p_untrad
        pred["ret_pred"] = pred["ret_pred_raw"] * (1 - pred["untrad_prob"])
    else:
        pred["untrad_prob"] = 0.0
        pred["ret_pred"] = pred["ret_pred_raw"]

    # 约束：上市未满N日
    pred = pred[(pred["list_days"]>=NEW_STOCK_DAYS)]

    picks = pred.sort_values("ret_pred",ascending=False).head(TOPK)
    picks.to_csv(pred_file,index=False,encoding="utf-8-sig")
    print(f"[INFO] {today} 预测结果已保存 -> {pred_file}")
    return picks, last_day, ic_mean, icir

# ====== 预测 vs 实际日志 ======
def update_log(preds, pred_day):
    next_day = (pd.to_datetime(pred_day) + timedelta(days=1)).strftime("%Y%m%d")
    log_file = os.path.join(DATA_DIR,"log.csv")
    rets = []
    for code in preds.ts_code:
        df = fetch_daily(code,next_day,next_day)
        if df.empty: continue
        rets.append({
            "ts_code": code,
            "pred_day": pred_day,
            "next_day": next_day,
            "real_ret": df["close"].pct_change().iloc[-1] if len(df) > 1 else 0.0,
            "ret_pred": float(preds.loc[preds.ts_code==code,"ret_pred"].iloc[0]) if "ret_pred" in preds.columns else np.nan,
            "strong_prob": float(preds.loc[preds.ts_code==code,"strong_prob"].iloc[0]) if "strong_prob" in preds.columns else np.nan
        })
    if not rets: return
    df_log = pd.DataFrame(rets)

    if os.path.exists(log_file):
        old = pd.read_csv(log_file)
        for c in ["ret_pred","strong_prob"]:
            if c not in old.columns: old[c] = np.nan
        df_log = pd.concat([old,df_log], ignore_index=True)

    df_log.to_csv(log_file,index=False,encoding="utf-8-sig")
    print(f"[INFO] 实际收益已记录 -> {log_file}")

# ====== 执行层回测（含 T+0 / 2 / 3 / 5 日） ======
def simulate_execution_from_logs(hold_days=2, stop_win=STOP_WIN, stop_loss=STOP_LOSS):
    log_file = os.path.join(DATA_DIR,"log.csv")
    if not os.path.exists(log_file):
        print("[INFO] 无 log.csv，跳过执行回测")
        return None
    df = pd.read_csv(log_file)
    if df.empty: return None

    day_rets = []
    for d, g in df.groupby("next_day"):
        rets = []
        for _, row in g.iterrows():
            code = row["ts_code"]
            start = (pd.to_datetime(row["next_day"], format="%Y%m%d")).strftime("%Y%m%d")
            end   = (pd.to_datetime(row["next_day"], format="%Y%m%d") + timedelta(days=max(hold_days-1,0))).strftime("%Y%m%d")
            k = fetch_daily(code, start, end)
            if k is None or k.empty:
                continue
            k = k.sort_values("trade_date")
            # T+0：当日开买，当日离场（或止盈止损）
            if hold_days == 0:
                entry = float(k["open"].iloc[0]) if "open" in k.columns else float(k["close"].iloc[0])
                high  = float(k["high"].iloc[0]) if "high" in k.columns else float(k["close"].iloc[0])
                low   = float(k["low"].iloc[0])  if "low"  in k.columns else float(k["close"].iloc[0])
                exitp = float(k["close"].iloc[0])
                realized = None
                if (high - entry)/entry >= stop_win: realized = stop_win
                if realized is None and (low - entry)/entry <= stop_loss: realized = stop_loss
                if realized is None: realized = (exitp - entry)/entry
                rets.append(realized)
                continue

            # 多日：第一天开盘买，日内止盈/止损，否则最后一天收盘离场
            entry_price = float(k["open"].iloc[0]) if "open" in k.columns else float(k["close"].iloc[0])
            exit_price  = float(k["close"].iloc[-1])
            realized = None
            for i in range(len(k)):
                high = float(k["high"].iloc[i]) if "high" in k.columns else float(k["close"].iloc[i])
                low  = float(k["low"].iloc[i])  if "low"  in k.columns else float(k["close"].iloc[i])
                if (high - entry_price) / entry_price >= stop_win:
                    realized = stop_win; break
                if (low - entry_price) / entry_price <= stop_loss:
                    realized = stop_loss; break
            if realized is None:
                realized = (exit_price - entry_price) / entry_price
            rets.append(realized)
        if rets:
            day_rets.append({"day": d, "ret": float(np.mean(rets))})
    if not day_rets:
        return None
    exec_df = pd.DataFrame(day_rets).sort_values("day")
    exec_df["cum"] = (1+exec_df["ret"]).cumprod()
    fn = os.path.join(DATA_DIR, f"execution_hold{hold_days}.png")
    plt.figure(figsize=(8,4))
    plt.plot(exec_df["cum"].values, label=f"执行层净值(Hold{hold_days})")
    plt.legend(); plt.tight_layout(); plt.savefig(fn); plt.close()
    print(f"[INFO] 执行层曲线已保存 -> {fn}")
    return exec_df

# ====== 阈值搜索（基于 log.csv 历史） ======
def threshold_search_from_logs():
    log_file = os.path.join(DATA_DIR,"log.csv")
    if not os.path.exists(log_file):
        print("[INFO] 无 log.csv，阈值搜索跳过")
        return None
    df = pd.read_csv(log_file).dropna(subset=["real_ret"])
    if df.empty or "strong_prob" not in df.columns:
        print("[INFO] 历史不足或缺 strong_prob，阈值搜索跳过")
        return None

    best = None
    for th in SEARCH_PROB_TH:
        for k in SEARCH_TOPK_GRID:
            day_port = []
            for d, g in df.groupby("pred_day"):
                gg = g.copy()
                gg = gg[gg["strong_prob"]>=th] if "strong_prob" in gg.columns else gg
                if gg.empty: 
                    continue
                gg = gg.sort_values("ret_pred", ascending=False).head(k) if "ret_pred" in gg.columns else gg.head(k)
                if gg.empty: 
                    continue
                day_port.append(gg["real_ret"].mean())
            if not day_port:
                continue
            r = np.array(day_port)
            ann_ret = r.mean()*252
            ann_vol = r.std()*np.sqrt(252) if r.std()>0 else np.nan
            sharpe = ann_ret/ann_vol if ann_vol and not np.isnan(ann_vol) else -9
            score = sharpe
            if (best is None) or (score > best["score"]):
                best = {"prob_th": th, "topk": k, "score": score, "ann_ret": ann_ret, "ann_vol": ann_vol}
    if best:
        print(f"[INFO] 阈值搜索最佳: prob>={best['prob_th']}, TOPK={best['topk']}, Sharpe={best['score']:.2f}")
        pd.DataFrame([best]).to_csv(os.path.join(DATA_DIR,"threshold_search.csv"), index=False, encoding="utf-8-sig")
    else:
        print("[INFO] 阈值搜索未找到可行解")
    return best

# ====== 图表/报告 ======
def generate_charts():
    log_file = os.path.join(DATA_DIR,"log.csv")
    if not os.path.exists(log_file):
        print("[INFO] 无 log.csv，跳过图表生成")
        return
    df = pd.read_csv(log_file)
    if df.empty:
        print("[INFO] log.csv 为空，跳过图表生成")
        return

    # 累计净值
    ret = df["real_ret"].fillna(0)
    cum = (1+ret).cumprod()
    plt.figure(figsize=(8,4))
    plt.plot(cum.values, label="策略净值")
    plt.title("策略累计净值")
    plt.xlabel("交易日序号")
    plt.ylabel("净值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR,"performance.png"))
    plt.close()

    # 月度热力图
    df["year"] = df["next_day"].astype(str).str[:4]
    df["month"] = df["next_day"].astype(str).str[4:6].astype(int)
    heat = df.groupby(["year","month"])["real_ret"].mean().reset_index()
    if not heat.empty:
        pv = heat.pivot(index="year", columns="month", values="real_ret")
        plt.figure(figsize=(10,4))
        sns.heatmap(pv, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
        plt.title("月度平均收益热力图")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR,"monthly_heatmap.png"))
        plt.close()

    # 基准对比：沪深300
    try:
        start = df["next_day"].min()
        end   = df["next_day"].max()
        bench = fetch_daily("000300.SH", str(start), str(end))
        if not bench.empty:
            b = bench.set_index("trade_date")["close"].pct_change().fillna(0)
            bc = (1+b).cumprod()
            idx = range(len(cum))
            plt.figure(figsize=(8,4))
            plt.plot(idx, cum.values, label="策略净值")
            plt.plot(idx, bc.values[:len(idx)], label="沪深300")
            plt.title("策略 vs 沪深300")
            plt.xlabel("交易日序号")
            plt.ylabel("净值")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR,"strategy_vs_benchmark.png"))
            plt.close()
    except Exception as e:
        print(f"[WARN] 基准图失败: {e}")

def generate_html_report(extra_imgs=[]):
    html_file = os.path.join(DATA_DIR,"report.html")
    log_file = os.path.join(DATA_DIR,"log.csv")

    html = ["<html><head><meta charset='utf-8'><title>策略报告</title></head><body>"]
    html.append("<h1>📊 策略预测报告</h1>")

    # IC/ICIR 表
    ic_sum_file = os.path.join(DATA_DIR, "ic_summary.csv")
    if os.path.exists(ic_sum_file):
        ic_sum = pd.read_csv(ic_sum_file)
        html.append("<h2>🔎 IC / ICIR 概览（训练窗口）</h2>")
        html.append(ic_sum.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    # 历史日志
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        html.append("<h2>📈 历史预测日志（最近20条）</h2>")
        html.append(df.tail(20).to_html(index=False, escape=False))

    # 市场热点
    ind_top_path = os.path.join(DATA_DIR, "industry_top5.csv")
    con_top_path = os.path.join(DATA_DIR, "concept_top5.csv")
    if os.path.exists(ind_top_path) or os.path.exists(con_top_path):
        html.append("<h2>🔥 市场热点（Top5）</h2><div style='display:flex;gap:32px;'>")
        if os.path.exists(ind_top_path):
            ind_top = pd.read_csv(ind_top_path)
            html.append("<div><h3>行业强度 Top5</h3>")
            html.append(ind_top.to_html(index=False, float_format=lambda x: f"{x:.2%}" if isinstance(x,float) else str(x)))
            html.append("</div>")
        if os.path.exists(con_top_path):
            con_top = pd.read_csv(con_top_path)
            html.append("<div><h3>概念强度 Top5</h3>")
            html.append(con_top.to_html(index=False, float_format=lambda x: f"{x:.2%}" if isinstance(x,float) else str(x)))
            html.append("</div>")
        html.append("</div>")

    # 嵌入图片（新增 T+0 / 5日曲线 + IC时间序列）
    img_list = [
        os.path.join(DATA_DIR,"ic_timeseries.png"),
        os.path.join(DATA_DIR,"performance.png"),
        os.path.join(DATA_DIR,"monthly_heatmap.png"),
        os.path.join(DATA_DIR,"strategy_vs_benchmark.png"),
        os.path.join(DATA_DIR,"execution_hold0.png"),
        os.path.join(DATA_DIR,"execution_hold2.png"),
        os.path.join(DATA_DIR,"execution_hold3.png"),
        os.path.join(DATA_DIR,"execution_hold5.png"),
    ] + list(extra_imgs or [])

    for img in img_list:
        if os.path.exists(img):
            html.append(f"<h3>{os.path.basename(img)}</h3>")
            html.append(f"<img src='data:image/png;base64,{img_to_base64(img)}' width='900'>")

    html.append("</body></html>")
    with open(html_file,"w",encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[INFO] HTML报告已生成 -> {html_file}")

# ====== 跟踪图 ======
def plot_stock_track(ts_code):
    log_file = os.path.join(DATA_DIR,"log.csv")
    if not os.path.exists(log_file): return None
    df = pd.read_csv(log_file)
    df_stock = df[df["ts_code"]==ts_code]
    if df_stock.empty: return None
    df_stock["date"] = pd.to_datetime(df_stock["next_day"], format="%Y%m%d")
    plt.figure(figsize=(8,4))
    if "ret_pred" in df_stock.columns:
        plt.plot(df_stock["date"], df_stock["ret_pred"], label="预测收益", marker="o")
    plt.plot(df_stock["date"], df_stock["real_ret"], label="实际收益", marker="x")
    plt.axhline(0,color="black",linestyle="--")
    plt.title(f"{ts_code} 预测 vs 实际收益")
    plt.xlabel("日期"); plt.ylabel("收益率")
    plt.legend(); plt.tight_layout()
    fig_file = os.path.join(DATA_DIR,f"track_{ts_code}.png")
    plt.savefig(fig_file); plt.close()
    print(f"[INFO] 单股跟踪图已保存 -> {fig_file}")
    return fig_file

def plot_topN_stocks(n=10):
    log_file = os.path.join(DATA_DIR,"log.csv")
    if not os.path.exists(log_file): return []
    df = pd.read_csv(log_file)
    if df.empty: return []
    top_stocks = df["ts_code"].value_counts().head(n).index.tolist()
    files = []
    for code in top_stocks:
        f = plot_stock_track(code)
        if f: files.append(f)
    return files

# ================= 主流程（后台） =================
def run_backend_once():
    # 1) 修复日志
    repair_log()
    # 2) 增量训练 + 当日预测（含 IC/ICIR 与 热点输出）
    picks, last_day, ic_mean, icir = daily_update()
    if picks is not None:
        cols = [c for c in ["ts_code","strong_prob","ret_pred_raw","untrad_prob","ret_pred"] if c in picks.columns]
        print("今日预测 TopK：")
        print(picks[cols])
        # 3) 写入预测 vs 实际日志
        update_log(picks,last_day)
    # 4) 执行层回测（T+0 / 2 / 3 / 5）
    for d in EXEC_HOLD_DAYS_LIST:
        simulate_execution_from_logs(hold_days=d, stop_win=STOP_WIN, stop_loss=STOP_LOSS)
    # 5) 阈值搜索
    threshold_search_from_logs()
    # 6) 图表
    generate_charts()
    # 7) 跟踪图（Top10 高频）
    extra_imgs = plot_topN_stocks(10)
    # 8) HTML 报告（含：IC/ICIR表、IC时间序列图、行业/概念Top5、执行层多周期曲线）
    generate_html_report(extra_imgs)

# ================= 前台（Streamlit） =================
def run_streamlit_ui():
    st.set_page_config(page_title="策略看板（含多周期回测与可选DL/CatBoost）", layout="wide")
    st.title("📈 策略交互看板")
    st.caption("TuShare优先 + AKShare兜底；≥1年历史；行业/概念强度、IC/ICIR、T+0/2/3/5回测、可选CatBoost与深度学习子模型。")

    if st.sidebar.button("重新运行后台流程（训练/预测/更新报表）"):
        with st.spinner("正在运行后台流程..."):
            run_backend_once()
        st.success("后台流程完成，数据已更新。")

    # 载入日志
    log_file = os.path.join(DATA_DIR,"log.csv")
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
    else:
        df_log = pd.DataFrame()

    # 载入 IC 汇总与序列
    ic_sum_file = os.path.join(DATA_DIR,"ic_summary.csv")
    ic_series_file = os.path.join(DATA_DIR,"ic_series.csv")
    ic_mean = icir = None
    if os.path.exists(ic_sum_file):
        ic_sum = pd.read_csv(ic_sum_file)
        if not ic_sum.empty:
            ic_mean = ic_sum["ic_mean"].iloc[0]
            icir = ic_sum["icir"].iloc[0]

    # 指标区
    st.subheader("📊 绩效总览")
    if df_log.empty:
        st.info("暂无日志数据。请先在侧边栏运行后台流程或用 python 方式运行本脚本。")
    else:
        ret = df_log["real_ret"].fillna(0)
        win = (ret>0).mean(); avg = ret.mean()
        ann_ret = avg*252; ann_vol = ret.std()*np.sqrt(252)
        sharpe = ann_ret/ann_vol if ann_vol>0 else 0
        ic_val = df_log["ret_pred"].corr(df_log["real_ret"]) if "ret_pred" in df_log.columns else np.nan
        if ic_mean is not None and not np.isnan(ic_mean):
            ic_val = ic_mean
        rank_ic = df_log["ret_pred"].rank().corr(df_log["real_ret"].rank()) if "ret_pred" in df_log.columns else np.nan

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("胜率", f"{win:.2%}")
        c2.metric("平均收益/日", f"{avg:.2%}")
        c3.metric("Sharpe", f"{sharpe:.2f}")
        c4.metric("IC", f"{ic_val if pd.notna(ic_val) else 0.0:.4f}")
        c5.metric("ICIR", f"{icir if (icir is not None and not np.isnan(icir)) else 0.0:.2f}")

    # 左右：净值对比 + 月度热力图
    left, right = st.columns(2)
    with left:
        st.subheader("策略 vs 沪深300")
        try:
            if not df_log.empty:
                ret = df_log["real_ret"].fillna(0)
                eq = (1+ret).cumprod()
                start, end = df_log["next_day"].min(), df_log["next_day"].max()
                bench = fetch_daily("000300.SH", str(start), str(end))
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(eq.values, label="策略净值")
                if bench is not None and not bench.empty:
                    b = bench.set_index("trade_date")["close"].pct_change().fillna(0)
                    beq = (1+b).cumprod()
                    ax.plot(beq.values[:len(eq)], label="沪深300")
                ax.legend(); ax.set_title("净值对比")
                st.pyplot(fig)
            else:
                st.info("无日志数据。")
        except Exception as e:
            st.warning(f"绘图失败：{e}")

    with right:
        st.subheader("月度热力图")
        try:
            if not df_log.empty:
                d = df_log.copy()
                d["year"] = d["next_day"].astype(str).str[:4]
                d["month"] = d["next_day"].astype(str).str[4:6].astype(int)
                pv = d.groupby(["year","month"])["real_ret"].mean().reset_index().pivot("year","month","real_ret")
                fig, ax = plt.subplots(figsize=(10,4))
                sns.heatmap(pv, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
                ax.set_title("月度平均收益热力图")
                st.pyplot(fig)
            else:
                st.info("无日志数据。")
        except Exception as e:
            st.warning(f"绘制热力图失败：{e}")

    # IC 时间序列
    st.subheader("📉 每日 IC 走势（训练窗口内）")
    if os.path.exists(ic_series_file):
        ic_df = pd.read_csv(ic_series_file)
        if not ic_df.empty:
            fig, ax = plt.subplots(figsize=(9,4))
            x = pd.to_datetime(ic_df["trade_date"])
            ax.plot(x, ic_df["ic"], marker="o", linewidth=1)
            ax.axhline(0, color="black", linestyle="--")
            ax.set_title("每日IC")
            st.pyplot(fig)
        else:
            st.info("暂无 IC 序列数据。")
    else:
        st.info("未找到 ic_series.csv，请在侧栏先运行后台流程。")

    # 最新预测
    st.subheader("📋 最新预测 TopK")
    preds_files = [f for f in os.listdir(DATA_DIR) if f.startswith("predictions_") and f.endswith(".csv")]
    if preds_files:
        preds_files.sort()
        latest_file = preds_files[-1]
        df_pred = pd.read_csv(os.path.join(DATA_DIR, latest_file))
        show_cols = [c for c in ["ts_code","strong_prob","untrad_prob","ret_pred_raw","ret_pred"] if c in df_pred.columns]
        st.caption(f"文件：{latest_file}")
        st.dataframe(df_pred[show_cols].sort_values("ret_pred", ascending=False).head(TOPK), use_container_width=True)
    else:
        st.info("未发现 predictions_*.csv 文件，请先运行后台流程。")

    # 市场热点
    st.subheader("🔥 市场热点 Top5")
    ind_top_path = os.path.join(DATA_DIR, "industry_top5.csv")
    con_top_path = os.path.join(DATA_DIR, "concept_top5.csv")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**行业强度 Top5**")
        if os.path.exists(ind_top_path):
            ind_top = pd.read_csv(ind_top_path)
            st.dataframe(ind_top, use_container_width=True)
        else:
            st.info("暂无行业Top5。")
    with cols[1]:
        st.markdown("**概念强度 Top5**")
        if os.path.exists(con_top_path):
            con_top = pd.read_csv(con_top_path)
            st.dataframe(con_top, use_container_width=True)
        else:
            st.info("暂无概念Top5（可能因网络/映射无法获取）。")

    # 执行层多周期展示（直接读取已生成的图片）
    st.subheader("🧪 执行层净值曲线（T+0 / 2 / 3 / 5）")
    imgs = []
    for d in EXEC_HOLD_DAYS_LIST:
        p = os.path.join(DATA_DIR, f"execution_hold{d}.png")
        if os.path.exists(p): imgs.append(p)
    if imgs:
        for p in imgs:
            st.image(p, caption=os.path.basename(p), use_container_width=True)
    else:
        st.info("未找到执行层曲线图片，请先运行后台流程。")

    # 单股跟踪
    st.subheader("🔎 单股跟踪")
    if not df_log.empty:
        codes = sorted(df_log["ts_code"].unique())
        sel = st.selectbox("选择股票", options=codes)
        if sel:
            d = df_log[df_log["ts_code"]==sel].copy()
            d["date"] = pd.to_datetime(d["next_day"], format="%Y%m%d")
            fig, ax = plt.subplots(figsize=(8,4))
            if "ret_pred" in d.columns:
                ax.plot(d["date"], d["ret_pred"], marker="o", label="预测收益")
            ax.plot(d["date"], d["real_ret"], marker="x", label="实际收益")
            ax.axhline(0, color="black", linestyle="--")
            ax.set_title(f"{sel} 预测 vs 实际收益"); ax.legend()
            st.pyplot(fig)
    else:
        st.info("暂无日志数据，无法展示单股跟踪。")

    st.success("✅ 看板就绪。侧栏可随时重新运行后台流程。")

# ================= 主入口 =================
def main():
    # 普通 python 运行：只跑后台
    if not STREAMLIT_AVAILABLE or ("streamlit" not in sys.argv[0].lower() and "streamlit" not in " ".join(sys.argv).lower()):
        run_backend_once()
    else:
        # Streamlit 环境：先跑后台一次，再进入看板
        run_backend_once()
        run_streamlit_ui()

if __name__=="__main__":
    main()
