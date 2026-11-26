import math
import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==========================================
# 1. 페이지 기본 설정 (최상단 배치 필수)
# ==========================================
st.set_page_config(
    page_title="FAS-D 스마트팜 자율 조향 & 진단 대시보드",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. 스마트팜 전문 대시보드 커스텀 CSS & 웹폰트
# ==========================================
SMARTFARM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+KR:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
    --sf-forest: #143f33;
    --sf-emerald: #1c674e;
    --sf-emerald-hover: #17533f;
    --sf-mint: #e7f0e6;
    --sf-bg: #f3f6f2;
    --sf-surface: #ffffff;
    --sf-border: #d5e2d3;
    --sf-text: #143f33;
    --sf-text-muted: #4b695d;
    --sf-shadow: 0 6px 20px rgba(20, 63, 51, 0.06);
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--sf-bg) !important;
    color: var(--sf-text);
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}

[data-testid="stSidebar"] {
    background-color: #ebf2ea !important;
    border-right: 1px solid var(--sf-border) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--sf-forest) !important;
}

.sf-hero {
    background: linear-gradient(135deg, #143f33 0%, #1c674e 100%);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    color: #ffffff;
    box-shadow: 0 10px 28px rgba(20, 63, 51, 0.16);
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(231, 240, 230, 0.18);
}

.sf-hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.sf-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.sf-pill-hero {
    background: rgba(231, 240, 230, 0.16);
    color: #e7f0e6;
    border: 1px solid rgba(231, 240, 230, 0.3);
}

.sf-pill-mint {
    background: var(--sf-mint);
    color: var(--sf-forest);
    border: 1px solid #c5dac2;
}

.sf-pill-emerald {
    background: var(--sf-emerald);
    color: #ffffff;
}

.sf-hero-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    line-height: 1.3;
    color: #ffffff;
}

.sf-hero-sub {
    font-size: 0.92rem;
    color: #dbe9db;
    margin: 0;
    line-height: 1.55;
}

.sf-kpi-card {
    background: var(--sf-surface);
    border: 1px solid var(--sf-border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: var(--sf-shadow);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.sf-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(20, 63, 51, 0.1);
}

.sf-kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--sf-text-muted);
    margin-bottom: 0.3rem;
}

.sf-kpi-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--sf-forest);
    font-family: 'Plus Jakarta Sans', 'Noto Sans KR', sans-serif;
}

.sf-kpi-foot {
    font-size: 0.74rem;
    color: var(--sf-emerald);
    font-weight: 600;
    margin-top: 0.25rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--sf-surface) !important;
    border: 1px solid var(--sf-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--sf-shadow) !important;
    padding: 0.85rem 1.15rem !important;
}

[data-testid="stForm"] {
    background-color: var(--sf-surface) !important;
    border: 1px solid var(--sf-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--sf-shadow) !important;
    padding: 1.4rem 1.5rem !important;
}

.sf-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid var(--sf-mint);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}

.sf-section-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--sf-forest);
    margin: 0;
}

.sf-group-label {
    background: var(--sf-mint);
    color: var(--sf-forest);
    padding: 0.42rem 0.8rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    border-left: 4px solid var(--sf-emerald);
}

html body .stApp [data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [data-baseweb="tab-list"],
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background-color: transparent !important;
    padding: 6px 0 0 0 !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid #c5dac2 !important;
    box-shadow: none !important;
    flex-wrap: wrap !important;
    align-items: flex-end !important;
    margin-bottom: 0 !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"],
html body .stApp [data-testid="stTabs"] button[data-baseweb="tab"],
[data-testid="stTabs"] button[data-baseweb="tab"],
[data-testid="stTabs"] [data-baseweb="tab"],
.stTabs button[data-baseweb="tab"],
.stTabs [data-baseweb="tab"] {
    min-height: 46px !important;
    height: auto !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.62rem 1.3rem !important;
    margin: 0 0 -2px 0 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #2f4d41 !important;
    background-color: #e4ede3 !important;
    border: 1.5px solid #c5dac2 !important;
    border-top: 3px solid transparent !important;
    border-bottom: 2px solid #c5dac2 !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
    position: relative !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"] p,
html body .stApp [data-testid="stTabs"] button[role="tab"] div,
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
    color: inherit !important;
    font-weight: inherit !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"]:hover,
[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:hover,
.stTabs button[data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    background-color: #d6e6d5 !important;
    color: #143f33 !important;
}

html body .stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"],
html body .stApp [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] [aria-selected="true"],
.stTabs button[data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #c5dac2 !important;
    border-top: 4px solid #1c674e !important;
    border-bottom: 2px solid #ffffff !important;
    color: #143f33 !important;
    font-weight: 800 !important;
    padding-top: 0.72rem !important;
    padding-bottom: 0.68rem !important;
    box-shadow: 0 -3px 10px rgba(20, 63, 51, 0.08) !important;
    z-index: 5 !important;
}

html body .stApp [data-testid="stTabs"] [data-baseweb="tab-border"],
html body .stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    background: transparent !important;
}

[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 0 !important;
    margin-top: -1px !important;
}

[data-testid="stTabs"] [data-baseweb="tab-panel"] > div > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stTabs"] [data-baseweb="tab-panel"] > div > [data-testid="stForm"] {
    border-top-left-radius: 0 !important;
    border-color: #c5dac2 !important;
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stDownloadButton > button {
    background-color: var(--sf-emerald) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.58rem 1.25rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 12px rgba(28, 103, 78, 0.2) !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    background-color: var(--sf-forest) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(20, 63, 51, 0.28) !important;
}

[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #cbdcca !important;
    background-color: #fbfdfb !important;
}

[data-testid="stFileUploader"] section {
    background-color: #f8fbf7 !important;
    border: 2px dashed #9fc0a4 !important;
    border-radius: 14px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--sf-border);
}

.sf-sidebar-card {
    background: var(--sf-surface);
    border: 1px solid var(--sf-border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(20, 63, 51, 0.04);
}

.sf-insight-card {
    background: #eef6ee;
    border: 1px solid #bfd7bf;
    border-left: 5px solid var(--sf-emerald);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: var(--sf-forest);
    font-size: 0.92rem;
    line-height: 1.6;
}
</style>
"""
st.markdown(SMARTFARM_CSS, unsafe_allow_html=True)

# ==========================================
# 3. 기준선 CSV 로드 및 DB 초기화 / 파생변수 연산
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(BASE_DIR, "cold_start_standards.csv"),
    "/mount/src/smartfarm-fasd/cold_start_standards.csv",
    "cold_start_standards.csv",
    "./cold_start_standards.csv",
    os.path.join(os.getcwd(), "cold_start_standards.csv"),
]

@st.cache_data
def load_benchmark_data():
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
                df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                return df, path
            except Exception:
                continue
    return pd.DataFrame(), None

df_bench, BENCHMARK_CSV = load_benchmark_data()
DB_NAME = os.path.join(BASE_DIR, "smartfarm_yield_swing.db")

def init_db():
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        empty_df = pd.DataFrame(columns=[
            "zone_id", "week_no", "dli_daily", "t_mean", "t_day", "t_night",
            "vpd_mean", "vpd_max", "rtr", "rh_mean", "dew_point", "cum_p_n_8w",
            "fruit_load", "t_leaf", "cwsi", "stem_diameter", "stem_shrinkage",
            "vwc", "drainage_rate", "fli", "tr_ratio", "gdd_8w", "lagged_dli_7w",
            "lagged_temp_7w", "w_avg", "y_actual", "y_target", "yield_swing_delta",
            "ma_3week", "swing_index", "grade_distribution", "brix_variance"
        ])
        empty_df.to_sql("yield_swing_data", conn, index=False, if_exists="replace")
        conn.close()

def calculate_derived_variables(df):
    df = df.sort_values(by=["zone_id", "week_no"]).reset_index(drop=True)
    df["x1_cum_dli_8w"] = df.groupby("zone_id")["dli_daily"].transform(lambda x: x.rolling(window=8, min_periods=1).sum())
    df["gdd_8w"] = df.groupby("zone_id")["t_mean"].transform(lambda x: x.rolling(window=8, min_periods=1).sum())
    df["temp_pn"] = (df["dli_daily"] * 0.85) - (df["vpd_mean"] * 1.2)
    df["cum_p_n_8w"] = df.groupby("zone_id")["temp_pn"].transform(lambda x: x.rolling(window=8, min_periods=1).sum() * 10 + 500).clip(lower=300, upper=1200)
    df = df.drop(columns=["temp_pn"])
    df["lagged_dli_7w"] = df.groupby("zone_id")["dli_daily"].shift(7).fillna(df["dli_daily"].iloc[0])
    df["lagged_temp_7w"] = df.groupby("zone_id")["t_mean"].shift(7).fillna(df["t_mean"].iloc[0])
    df["fli"] = (df["fruit_load"] / (df["dli_daily"] + 1)) * 10
    df["tr_ratio"] = 3.5 - 0.02 * df["fruit_load"] + 0.001 * df["cum_p_n_8w"]
    df["w_avg"] = (df["cum_p_n_8w"] / df["fruit_load"].replace(0, 1)).clip(lower=110, upper=180) * 12
    df["yield_swing_delta"] = df["y_actual"] - df["y_target"]
    df["ma_3week"] = df.groupby("zone_id")["y_actual"].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    df["swing_index"] = ((df["y_actual"] - df["ma_3week"]) / df["ma_3week"].replace(0, 1)) * 100
    df["grade_distribution"] = (92 - 0.3 * df["swing_index"].abs()).clip(lower=70, upper=98)
    df["brix_variance"] = 0.2 + 0.005 * df["swing_index"].abs()
    return df

FACTOR_LABELS = {
    "zone_id": "구역 ID", "week_no": "주차", "dli_daily": "일일 누적 광량",
    "t_mean": "24시간 평균온도", "t_day": "주간 온도", "t_night": "야간 온도",
    "vpd_mean": "평균 증기압차", "vpd_max": "최고 증기압차", "rtr": "광-온 비",
    "rh_mean": "평균 상대습도", "dew_point": "이슬점 온도", "fruit_load": "과일 부하량",
    "t_leaf": "작물 엽온", "cwsi": "수분 스트레스 지수", "stem_diameter": "줄기 직경",
    "stem_shrinkage": "줄기 수축량", "vwc": "슬래브 수분함량", "drainage_rate": "배액율",
    "x1_cum_dli_8w": "누적 광량 8주", "gdd_8w": "적산 온도 8주", "fli": "과일 부하 및 체력 지수",
    "cum_p_n_8w": "순광합성량 8주 적산", "lagged_dli_7w": "7주 시차 광량", "lagged_temp_7w": "7주 시차 온도",
    "tr_ratio": "T/R율 균형 지수", "w_avg": "평균 단과중", "y_actual": "실제 주간 수확량",
    "y_target": "목표 주간 수확량", "yield_swing_delta": "수확량 편차", "ma_3week": "3주 이동평균 수확량",
    "swing_index": "수확량 스윙 지수", "grade_distribution": "상품과 분포율", "brix_variance": "당도 편차",
    "st_dli_daily": "[기준] 일일 누적 광량", "st_t_mean": "[기준] 24시간 평균온도",
    "st_t_day": "[기준] 주간 온도", "st_t_night": "[기준] 야간 온도",
    "st_vpd_mean": "[기준] 평균 증기압차", "st_vpd_max": "[기준] 최고 증기압차",
    "st_rtr": "[기준] 광-온 비", "st_rh_mean": "[기준] 평균 상대습도",
    "st_dew_point": "[기준] 이슬점 온도", "st_cum_p_n_8w": "[기준] 순광합성량 8주 적산",
    "st_fruit_load": "[기준] 과일 부하량", "st_t_leaf": "[기준] 작물 엽온",
    "st_cwsi": "[기준] 수분 스트레스 지수", "st_stem_diameter": "[기준] 줄기 직경",
    "st_stem_shrinkage": "[기준] 줄기 수축량", "st_vwc": "[기준] 슬래브 수분함량",
    "st_drainage_rate": "[기준] 배액율", "st_fli": "[기준] 과일 부하 및 체력 지수",
    "st_tr_ratio": "[기준] T/R율 균형 지수", "st_gdd_8w": "[기준] 적산 온도 8주",
    "st_lagged_dli_7w": "[기준] 7주 시차 광량", "st_lagged_temp_7w": "[기준] 7주 시차 온도",
    "st_w_avg": "[기준] 평균 단과중", "st_y_actual": "[기준] 실제 주간 수확량",
    "st_y_target": "[기준] 목표 주간 수확량", "st_yield_swing_delta": "[기준] 수확량 편차",
    "st_ma_3week": "[기준] 3주 이동평균 수확량", "st_swing_index": "[기준] 수확량 스윙 지수",
    "st_grade_distribution": "[기준] 상품과 분포율", "st_brix_variance": "[기준] 당도 편차"
}

DERIVED_COL_KEYS = [
    "x1_cum_dli_8w", "gdd_8w", "cum_p_n_8w", "lagged_dli_7w",
    "lagged_temp_7w", "fli", "tr_ratio", "w_avg", "yield_swing_delta",
    "ma_3week", "swing_index", "grade_distribution", "brix_variance"
]

def style_derived_columns(df_display: pd.DataFrame, derived_keys=None):
    """4.8.1절 자동 연산(파생) 컬럼을 파란색으로 강조 표시"""
    target_keys = derived_keys if derived_keys is not None else DERIVED_COL_KEYS
    derived_labels = [
        FACTOR_LABELS.get(k, k)
        for k in target_keys
        if FACTOR_LABELS.get(k, k) in df_display.columns
    ]
    if not derived_labels:
        return df_display
    try:
        return df_display.style.set_properties(
            subset=derived_labels,
            **{"background-color": "#eff6ff", "color": "#1d4ed8"},
        )
    except Exception:
        return df_display

init_db()

def load_db_snapshot():
    if os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        try:
            df = pd.read_sql("SELECT * FROM yield_swing_data", conn)
        finally:
            conn.close()
        return df
    return pd.DataFrame()

df_snapshot = load_db_snapshot()

def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps, fpmin = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h

def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def calc_pearson_stats(series_x: pd.Series, series_y: pd.Series):
    """유효 쌍 기준 Pearson 상관계수(r), 양측 검정 p-value, 유효 표본 수(n) 계산"""
    pair_df = pd.DataFrame({"x": pd.to_numeric(series_x, errors="coerce"), "y": pd.to_numeric(series_y, errors="coerce")}).dropna()
    n = len(pair_df)
    if n < 3:
        return 0.0, 1.0, n
    x_vals = pair_df["x"].to_numpy(dtype=float)
    y_vals = pair_df["y"].to_numpy(dtype=float)
    if np.std(x_vals) == 0.0 or np.std(y_vals) == 0.0:
        return 0.0, 1.0, n
    if SCIPY_AVAILABLE:
        r_val, p_val = scipy_stats.pearsonr(x_vals, y_vals)
        if np.isnan(r_val) or np.isnan(p_val):
            return 0.0, 1.0, n
        return float(r_val), float(p_val), n
    r_val = float(np.corrcoef(x_vals, y_vals)[0, 1])
    if np.isnan(r_val):
        return 0.0, 1.0, n
    r_val = max(-1.0, min(1.0, r_val))
    if abs(r_val) >= 1.0 - 1e-15:
        return r_val, 0.0, n
    df_deg = n - 2
    t_sq = (r_val * r_val * df_deg) / (1.0 - r_val * r_val)
    p_val = _betainc(df_deg / 2.0, 0.5, df_deg / (df_deg + t_sq))
    return float(r_val), float(p_val), n

with st.sidebar:
    st.markdown(
        """
        <div class="sf-sidebar-card">
            <div style="font-size:1.05rem; font-weight:800; color:#143f33; margin-bottom:4px;">
                🌿 FAS-D Control Hub
            </div>
            <div style="font-size:0.78rem; color:#4b695d;">
                5ha 스마트팜 자율 조향 시스템<br/>실시간 연동 및 엔진 상태
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not df_bench.empty and BENCHMARK_CSV:
        st.sidebar.success(f"✅ 9S-9P 기준선 CSV 로드 ({len(df_bench)}행)")
    else:
        st.sidebar.warning("⚠️ cold_start_standards.csv 미검출")

    if PLOTLY_AVAILABLE:
        st.sidebar.success("📊 Plotly 시계열 엔진: 활성")
    else:
        st.sidebar.error("⚠️ Plotly 미설치 (기본 차트 모드)")

    if not df_snapshot.empty:
        max_w = int(df_snapshot["week_no"].max())
        zone_cnt = df_snapshot["zone_id"].nunique()
        st.markdown(
            f"""
            <div class="sf-sidebar-card">
                <div style="font-size:0.78rem; font-weight:700; color:#143f33; margin-bottom:8px;">
                    🗄️ SQLite DB 요약
                </div>
                <div style="display:flex; flex-direction:column; gap:6px; font-size:0.82rem;">
                    <div><span class="sf-pill sf-pill-mint">총 레코드: {len(df_snapshot):,}행</span></div>
                    <div><span class="sf-pill sf-pill-mint">운용 구역: {zone_cnt}개 구역</span></div>
                    <div><span class="sf-pill sf-pill-mint">최신 주차: 1 ~ {max_w}주차</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.info("💡 DB에 저장된 데이터가 없습니다. Tab 1 또는 Tab 2에서 데이터를 등록해 주세요.")

st.markdown(
    """
    <div class="sf-hero">
        <div class="sf-hero-badges">
            <span class="sf-pill sf-pill-hero">🌱 5ha SmartFarm Autonomous Steering</span>
            <span class="sf-pill sf-pill-hero">📐 4.8.1절 생리 공식 자동 연산</span>
            <span class="sf-pill sf-pill-hero">📊 9S-9P 기준선 비교 & 2단계 상관진단</span>
        </div>
        <h1 class="sf-hero-title">FAS-D 스마트팜 자율 조향 시스템 — 수확량 스윙 진단 대시보드</h1>
        <p class="sf-hero-sub">
            온실 구역별 환경·생육 원천 데이터를 기반으로 <strong>4.8.1절 파생 지표 자동 연산</strong>,
            <strong>9S-9P 관리 기준선 비교</strong>, 그리고 <strong>Macro-Micro 2단계 피어슨 상관분석</strong>을 수행합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
total_rows = len(df_snapshot)
active_zones = df_snapshot["zone_id"].nunique() if not df_snapshot.empty else 0
max_week_val = int(df_snapshot["week_no"].max()) if not df_snapshot.empty else 0
swing_alerts = (
    int((df_snapshot["swing_index"].fillna(0).abs() >= 20.0).sum())
    if not df_snapshot.empty and "swing_index" in df_snapshot.columns
    else 0
)

with k1:
    st.markdown(
        f"""
        <div class="sf-kpi-card">
            <div class="sf-kpi-label">통합 DB 저장 데이터</div>
            <div class="sf-kpi-value">{total_rows:,} <span style="font-size:0.9rem; font-weight:600;">행</span></div>
            <div class="sf-kpi-foot">SQLite 실시간 동기화</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="sf-kpi-card">
            <div class="sf-kpi-label">모니터링 온실 구역</div>
            <div class="sf-kpi-value">{active_zones} <span style="font-size:0.9rem; font-weight:600;">개 구역</span></div>
            <div class="sf-kpi-foot">5ha 재배 구역 기준</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="sf-kpi-card">
            <div class="sf-kpi-label">수집 주차 범위</div>
            <div class="sf-kpi-value">1 ~ {max_week_val} <span style="font-size:0.9rem; font-weight:600;">주차</span></div>
            <div class="sf-kpi-foot">연간 52주 사이클 트래킹</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    alert_color = "#b91c1c" if swing_alerts > 0 else "#1c674e"
    alert_desc = "±20% 임계치 초과 감지" if swing_alerts > 0 else "안정 범위 유지 중"
    st.markdown(
        f"""
        <div class="sf-kpi-card">
            <div class="sf-kpi-label">스윙 주의/위험 경보</div>
            <div class="sf-kpi-value" style="color:{alert_color};">{swing_alerts} <span style="font-size:0.9rem; font-weight:600;">건</span></div>
            <div class="sf-kpi-foot" style="color:{alert_color};">{alert_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 원천 데이터 직접 입력",
    "📂 일괄 파일 업로드 및 연산",
    "📊 DB 데이터 조회",
    "📈 구역별 시계열 및 9S-9P 비교",
    "🔍 2단계 계층적 스윙 진단",
    "🔮 ML 수확량 & 단기 스윙 예측"
])

with tab1:
    st.markdown(
        """
        <div class="sf-section-header">
            <h3 class="sf-section-title">📝 주간 센서 및 생육 측정 원천 데이터 입력</h3>
            <span class="sf-pill sf-pill-mint">4.8.1절 공식 실시간 반영</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.markdown('<div class="sf-group-label">🌡️ 구역 · 주차 · 광/온도 환경</div>', unsafe_allow_html=True)
            zone_id = st.selectbox("구역", ["Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E"])
            week_no = st.number_input("주차", min_value=1, max_value=52, value=1)
            dli_daily = st.number_input("일일 DLI (mol/m²/d)", value=25.0)
            t_mean = st.number_input("24시간 평균온도 (°C)", value=20.0)
            t_day = st.number_input("주간온도 (°C)", value=23.0)
            t_night = st.number_input("야간온도 (°C)", value=17.0)
        with col2:
            st.markdown('<div class="sf-group-label">🌿 대기 수분 · 작물 생육 지표</div>', unsafe_allow_html=True)
            vpd_mean = st.number_input("평균 VPD (kPa)", value=1.3)
            vpd_max = st.number_input("최고 VPD max (kPa)", value=1.8)
            rh_mean = st.number_input("평균 상대습도 (%)", value=75.0)
            fruit_load = st.number_input("착과수 (개/m²)", value=70.0)
            t_leaf = st.number_input("작물 엽온 (°C)", value=19.0)
            cwsi = st.number_input("수분 스트레스 지수 (CWSI)", value=0.2)
        with col3:
            st.markdown('<div class="sf-group-label">💧 근권 배액 · 줄기 · 수확량</div>', unsafe_allow_html=True)
            vwc = st.number_input("슬래브 수분함량 VWC (%)", value=60.0)
            drainage_rate = st.number_input("배액율 (%)", value=30.0)
            y_actual = st.number_input("실제수확량 (kg/m²)", value=3.5)
            y_target = st.number_input("목표수확량 (kg/m²)", value=3.0)
            stem_diameter = st.number_input("줄기직경 (mm)", value=11.0)
            stem_shrinkage = st.number_input("줄기 수축량 (mm)", value=0.4)

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("💾 데이터 저장 및 자동 연산 실행", use_container_width=True)

        if submitted:
            conn = sqlite3.connect(DB_NAME)
            df_existing = pd.read_sql("SELECT * FROM yield_swing_data", conn)
            conn.close()

            new_data = {
                "zone_id": zone_id, "week_no": week_no, "dli_daily": dli_daily,
                "t_mean": t_mean, "t_day": t_day, "t_night": t_night,
                "vpd_mean": vpd_mean, "vpd_max": vpd_max, "rtr": (dli_daily * 10) / max(t_mean, 1),
                "rh_mean": rh_mean, "dew_point": t_mean - ((100 - rh_mean) / 5),
                "fruit_load": fruit_load, "t_leaf": t_leaf, "cwsi": cwsi,
                "stem_diameter": stem_diameter, "stem_shrinkage": stem_shrinkage,
                "vwc": vwc, "drainage_rate": drainage_rate, "y_actual": y_actual,
                "y_target": y_target, "cum_p_n_8w": 0, "tr_ratio": 0, "gdd_8w": 0,
                "lagged_dli_7w": 0, "lagged_temp_7w": 0, "w_avg": 0,
                "yield_swing_delta": 0, "ma_3week": 0, "swing_index": 0,
                "grade_distribution": 0, "brix_variance": 0, "fli": 0
            }
            df_new_row = pd.DataFrame([new_data])
            if not df_existing.empty:
                mask_keep = ~((df_existing["zone_id"] == zone_id) & (df_existing["week_no"] == week_no))
                df_combined = pd.concat([df_existing[mask_keep], df_new_row], ignore_index=True)
            else:
                df_combined = df_new_row
            df_processed = calculate_derived_variables(df_combined)

            conn = sqlite3.connect(DB_NAME)
            df_processed.to_sql("yield_swing_data", conn, index=False, if_exists="replace")
            conn.close()
            st.success(f"✅ [{zone_id}] {week_no}주차 원천 데이터가 성공적으로 연산되어 DB에 저장되었습니다!")

with tab2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="sf-section-header">
                <h3 class="sf-section-title">📂 CSV 또는 Excel 파일 일괄 업로드 및 4.8.1절 자동 연산</h3>
                <span class="sf-pill sf-pill-mint">Batch ETL Pipeline</span>
            </div>
            <p style="font-size:0.9rem; color:#4b695d; margin-bottom:1rem;">
                스마트팜 센서 및 생육 측정 원천 데이터가 담긴 CSV 또는 Excel 파일을 업로드하면,
                <strong>4.8.1절 공식</strong>에 따른 파생 변수와 스윙 지수가 자동으로 연산되어 통합 DB에 반영됩니다.
            </p>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "원천 데이터 파일 업로드 (CSV 또는 Excel)",
            type=["csv", "xlsx"],
            key="tab2_file_uploader"
        )

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                if uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)

                st.markdown(
                    f"""
                    <div style="margin:0.75rem 0 0.5rem 0; display:flex; align-items:center; gap:0.5rem;">
                        <span class="sf-pill sf-pill-mint">📁 업로드된 원천 데이터 미리보기</span>
                        <span class="sf-pill sf-pill-emerald">총 {len(df_upload):,}행</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(df_upload, use_container_width=True, height=300)

                if st.button("🚀 4.8.1절 공식 적용 및 자동 연산 실행", key="tab2_calc_button", use_container_width=True):
                    df_processed = calculate_derived_variables(df_upload)
                    conn = sqlite3.connect(DB_NAME)
                    df_processed.to_sql("yield_swing_data", conn, index=False, if_exists="replace")
                    conn.close()
                    st.success(f"🎉 총 {len(df_processed):,}주차 데이터 자동 연산 및 DB 저장 완료!")

                    st.markdown("---")
                    st.markdown('<div class="sf-group-label">📊 핵심 연산 지표 요약 뷰</div>', unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.5rem; margin-bottom:0.65rem;">
                            <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.25rem 0.7rem; border-radius:9999px; font-size:0.75rem; font-weight:700; background:#f8fafc; color:#475569; border:1px solid #cbd5e1;">
                                ⬜ 원천 입력 데이터
                            </span>
                            <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.25rem 0.7rem; border-radius:9999px; font-size:0.75rem; font-weight:700; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;">
                                🔵 파란색 강조: 4.8.1절 자동 연산 데이터 (순광합성량 8주 적산, 7주 시차 광량, 평균 단과중, T/R율, 스윙 지수)
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    key_derived_cols = [
                        "zone_id", "week_no", "dli_daily", "t_mean", "cum_p_n_8w",
                        "lagged_dli_7w", "w_avg", "tr_ratio", "y_actual", "swing_index"
                    ]
                    df_summary = df_processed[key_derived_cols].rename(columns=FACTOR_LABELS)
                    styled_summary = style_derived_columns(
                        df_summary,
                        ["cum_p_n_8w", "lagged_dli_7w", "w_avg", "tr_ratio", "swing_index"],
                    )
                    st.dataframe(styled_summary, use_container_width=True, height=320)
            except Exception as e:
                st.error(f"❌ 파일 처리 중 오류 발생: {e}")
        else:
            st.info("💡 안내: 상단의 **Browse files** 버튼을 클릭하여 원천 CSV 또는 Excel 파일을 업로드해 주세요.")

with tab3:
    with st.container(border=True):
        st.markdown(
            """
            <div class="sf-section-header">
                <h3 class="sf-section-title">📊 현재 저장된 스마트팜 통합 DB 조회 (연산 결과 반영)</h3>
                <span class="sf-pill sf-pill-mint">smartfarm_yield_swing.db</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if os.path.exists(DB_NAME):
            conn = sqlite3.connect(DB_NAME)
            df_db = pd.read_sql("SELECT * FROM yield_swing_data", conn)
            conn.close()

            max_week = int(df_db["week_no"].max()) if not df_db.empty else 0
            st.success(f"✅ **[DB 연동 완료]** 총 {len(df_db):,}행의 4.8.1절 자동 연산 결과가 반영되어 있습니다. (수집 범위: 1 ~ {max_week}주차)")
            st.markdown(
                """
                <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.5rem; margin-bottom:0.6rem;">
                    <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.25rem 0.7rem; border-radius:9999px; font-size:0.75rem; font-weight:700; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;">
                        🔵 파란색 강조 열: 4.8.1절 공식 자동 연산(파생) 데이터
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            df_db_renamed = df_db.rename(columns=FACTOR_LABELS)
            styled_db = style_derived_columns(df_db_renamed)
            st.dataframe(styled_db, use_container_width=True, height=450)

            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            csv = df_db.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 현재 DB 데이터를 CSV로 다운로드",
                data=csv,
                file_name="smartfarm_yield_swing_updated.csv",
                mime="text/csv"
            )
        else:
            st.info("아직 저장된 데이터가 없습니다.")

with tab4:
    with st.container(border=True):
        st.markdown(
            """
            <div class="sf-section-header">
                <h3 class="sf-section-title">📈 구역별 변수 시계열 및 9S-9P 기준선 비교</h3>
                <span class="sf-pill sf-pill-mint">실제값(파란색) vs 9S-9P 기준선(빨간색)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not os.path.exists(DB_NAME):
            st.info("아직 저장된 실시간 데이터가 없습니다.")
        else:
            conn = sqlite3.connect(DB_NAME)
            df_chart = pd.read_sql("SELECT * FROM yield_swing_data", conn)
            conn.close()

            if df_chart.empty:
                st.info("DB에 시각화할 데이터가 없습니다.")
            else:
                df_chart = df_chart.groupby(["zone_id", "week_no"], as_index=False).mean(numeric_only=True)

                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    numeric_cols = df_chart.select_dtypes(include=[np.number]).columns.tolist()
                    if "week_no" in numeric_cols:
                        numeric_cols.remove("week_no")

                    col_options = {col: FACTOR_LABELS.get(col, col) for col in numeric_cols}
                    default_index = numeric_cols.index("swing_index") if "swing_index" in numeric_cols else 0

                    selected_col_key = st.selectbox(
                        "비교할 변수 선택",
                        options=list(col_options.keys()),
                        index=default_index,
                        format_func=lambda x: col_options[x],
                        key="tab4_var_select"
                    )
                with c2:
                    all_zones = df_chart["zone_id"].unique().tolist()
                    selected_zones = st.multiselect(
                        "비교할 구역 선택",
                        options=all_zones,
                        default=all_zones,
                        key="tab4_zone_select"
                    )

                filtered_actual = df_chart[df_chart["zone_id"].isin(selected_zones)] if selected_zones else df_chart

                if filtered_actual.empty:
                    st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
                else:
                    var_label = col_options.get(selected_col_key, selected_col_key)
                    st.markdown(f"#### 📉 **[{var_label}] 실제값(파란색) vs 9S-9P 기준선(빨간색) 비교**")

                    ref_col_expected = f"st_{selected_col_key}".strip()
                    cleaned_bench_columns = [str(c).strip().replace("\ufeff", "") for c in df_bench.columns]

                    matched_ref_col = None
                    for c in cleaned_bench_columns:
                        if c.lower() == ref_col_expected.lower():
                            matched_ref_col = c
                            break

                    has_bench_col = (matched_ref_col is not None) and (not df_bench.empty)

                    if has_bench_col:
                        # CSV 내부에 구역별로 다른 기준값이 존재하는지 확인
                        has_zone_diff = False
                        if "zone_id" in df_bench.columns:
                            zone_nunique = df_bench.groupby("week_no")[matched_ref_col].nunique()
                            has_zone_diff = (zone_nunique > 1).any()

                        bench_mode = st.radio(
                            "9S-9P 기준선 표시 방식",
                            ["단일 표준 기준선 (모든 구역 공통)", "구역별 CSV 기준선 개별 표시"],
                            index=0,
                            horizontal=True,
                            key="tab4_bench_mode",
                        )

                        if bench_mode == "단일 표준 기준선 (모든 구역 공통)":
                            if "zone_id" in df_bench.columns and "Zone_A" in df_bench["zone_id"].values:
                                df_bench_std = (
                                    df_bench[df_bench["zone_id"] == "Zone_A"][["week_no", matched_ref_col]]
                                    .groupby("week_no", as_index=False)[matched_ref_col]
                                    .mean(numeric_only=True)
                                )
                            else:
                                df_bench_std = (
                                    df_bench.groupby("week_no", as_index=False)[matched_ref_col]
                                    .mean(numeric_only=True)
                                )
                            df_bench_std = df_bench_std.rename(columns={matched_ref_col: "std_ref_val"})

                            if has_zone_diff:
                                st.caption("ℹ️ 참고: `cold_start_standards.csv` 내 구역별 기준값이 서로 달라, 토마토 공통 표준 환경(Zone_A 기준) 단일 기준선으로 통합하여 비교합니다.")

                    if has_bench_col and PLOTLY_AVAILABLE:
                        actual_palette = ["#2980b9", "#0ea5e9", "#1c674e", "#6366f1", "#0284c7"]
                        long_data = []
                        color_map = {}

                        if bench_mode == "단일 표준 기준선 (모든 구역 공통)":
                            merged_plot_df = pd.merge(
                                filtered_actual[["week_no", "zone_id", selected_col_key]],
                                df_bench_std[["week_no", "std_ref_val"]],
                                on="week_no",
                                how="left",
                            )

                            for z_idx, z in enumerate(selected_zones):
                                act_label = f"{z} (실제)"
                                color_map[act_label] = actual_palette[z_idx % len(actual_palette)]
                                sub_z = merged_plot_df[merged_plot_df["zone_id"] == z]
                                for _, row in sub_z.iterrows():
                                    if pd.notna(row[selected_col_key]):
                                        long_data.append({
                                            "주차 (Week)": row["week_no"],
                                            "구분": act_label,
                                            "값": row[selected_col_key],
                                        })

                            std_label = "9S-9P 표준 기준선"
                            color_map[std_label] = "#e74c3c"
                            actual_weeks = filtered_actual["week_no"].unique()
                            std_rows = df_bench_std[df_bench_std["week_no"].isin(actual_weeks)]
                            for _, row in std_rows.iterrows():
                                if pd.notna(row["std_ref_val"]):
                                    long_data.append({
                                        "주차 (Week)": row["week_no"],
                                        "구분": std_label,
                                        "값": row["std_ref_val"],
                                    })
                        else:
                            filtered_bench = df_bench[df_bench["zone_id"].isin(selected_zones)].copy()
                            merged_plot_df = pd.merge(
                                filtered_actual[["week_no", "zone_id", selected_col_key]],
                                filtered_bench[["week_no", "zone_id", matched_ref_col]],
                                on=["week_no", "zone_id"],
                                how="left",
                            )

                            ref_palette = ["#e74c3c", "#f97316", "#dc2626", "#fb7185", "#be123c"]
                            for z_idx, z in enumerate(selected_zones):
                                act_label = f"{z} (실제)"
                                ref_label = f"{z} (9S-9P 기준)"
                                color_map[act_label] = actual_palette[z_idx % len(actual_palette)]
                                color_map[ref_label] = ref_palette[z_idx % len(ref_palette)]

                            for _, row in merged_plot_df.iterrows():
                                w = row["week_no"]
                                z = row["zone_id"]
                                val_actual = row[selected_col_key]
                                val_ref = row[matched_ref_col]
                                if pd.notna(val_actual):
                                    long_data.append({"주차 (Week)": w, "구분": f"{z} (실제)", "값": val_actual})
                                if pd.notna(val_ref):
                                    long_data.append({"주차 (Week)": w, "구분": f"{z} (9S-9P 기준)", "값": val_ref})

                        df_plot = pd.DataFrame(long_data)

                        if not df_plot.empty:
                            fig = px.line(
                                df_plot,
                                x="주차 (Week)",
                                y="값",
                                color="구분",
                                color_discrete_map=color_map,
                                markers=True
                            )
                            if selected_col_key == "swing_index":
                                fig.add_hrect(
                                    y0=-20, y1=20,
                                    fillcolor="green", opacity=0.12,
                                    layer="below", line_width=0,
                                    annotation_text="정상 스윙 구간 (-20 ~ 20)",
                                    annotation_position="top left",
                                    annotation_font_size=11,
                                    annotation_font_color="green"
                                )
                            fig.update_layout(
                                xaxis_title="주차 (Week)",
                                yaxis_title=var_label,
                                legend_title="범례",
                                height=470,
                                plot_bgcolor="#ffffff",
                                paper_bgcolor="#ffffff",
                                font=dict(family="Plus Jakarta Sans, Noto Sans KR, sans-serif", color="#143f33"),
                                hovermode="x unified",
                                margin=dict(l=20, r=20, t=30, b=20),
                            )
                            fig.update_xaxes(showgrid=True, gridcolor="#eef4ed", zeroline=False)
                            fig.update_yaxes(showgrid=True, gridcolor="#eef4ed", zerolinecolor="#c5dac2")
                            st.plotly_chart(fig, use_container_width=True)
                            st.success("💡 안내: **파란색 계열 선과 점**은 구역별 실제 수집/연산 데이터이며, **빨간색 선과 점**은 토마토 생육 표준 9S-9P 관리기준선입니다.")

                            st.markdown("---")
                            st.markdown('<div class="sf-group-label">📋 상세 비교 데이터 테이블</div>', unsafe_allow_html=True)

                            pivot_actual_tbl = filtered_actual.pivot(index="week_no", columns="zone_id", values=selected_col_key)
                            plot_table_data = pd.DataFrame(index=sorted(filtered_actual["week_no"].unique()))

                            if bench_mode == "단일 표준 기준선 (모든 구역 공통)":
                                std_series = df_bench_std.set_index("week_no")["std_ref_val"]
                                plot_table_data["9S-9P 표준 기준선"] = std_series.reindex(plot_table_data.index)
                                for z in selected_zones:
                                    if z in pivot_actual_tbl.columns:
                                        plot_table_data[f"{z} (실제)"] = pivot_actual_tbl[z]
                            else:
                                pivot_combined = merged_plot_df.pivot(index="week_no", columns="zone_id")
                                for z in selected_zones:
                                    if (selected_col_key, z) in pivot_combined.columns:
                                        plot_table_data[f"{z} (실제)"] = pivot_combined[(selected_col_key, z)]
                                    if (matched_ref_col, z) in pivot_combined.columns:
                                        plot_table_data[f"{z} (9S-9P 기준)"] = pivot_combined[(matched_ref_col, z)]

                            st.dataframe(
                                plot_table_data.reset_index().rename(columns={"index": "주차 (Week)", "week_no": "주차 (Week)"}),
                                use_container_width=True,
                                height=350
                            )

                    elif has_bench_col and not PLOTLY_AVAILABLE:
                        st.warning("⚠️ 현재 환경에 `plotly` 패키지가 설치되어 있지 않아 Streamlit 기본 차트로 기준선과 함께 표시합니다.")
                        pivot_actual_tbl = filtered_actual.pivot(index="week_no", columns="zone_id", values=selected_col_key)
                        pivot_chart = pd.DataFrame(index=sorted(filtered_actual["week_no"].unique()))
                        if bench_mode == "단일 표준 기준선 (모든 구역 공통)":
                            std_series = df_bench_std.set_index("week_no")["std_ref_val"]
                            pivot_chart["9S-9P 표준 기준선"] = std_series.reindex(pivot_chart.index)
                            for z in selected_zones:
                                if z in pivot_actual_tbl.columns:
                                    pivot_chart[f"{z} (실제)"] = pivot_actual_tbl[z]
                        else:
                            filtered_bench = df_bench[df_bench["zone_id"].isin(selected_zones)].copy()
                            merged_plot_df = pd.merge(
                                filtered_actual[["week_no", "zone_id", selected_col_key]],
                                filtered_bench[["week_no", "zone_id", matched_ref_col]],
                                on=["week_no", "zone_id"],
                                how="left"
                            )
                            for z in selected_zones:
                                sub = merged_plot_df[merged_plot_df["zone_id"] == z].set_index("week_no")
                                if selected_col_key in sub.columns:
                                    pivot_chart[f"{z} (실제)"] = sub[selected_col_key]
                                if matched_ref_col in sub.columns:
                                    pivot_chart[f"{z} (9S-9P 기준)"] = sub[matched_ref_col]

                        st.line_chart(pivot_chart)
                        st.markdown("---")
                        st.markdown('<div class="sf-group-label">📋 상세 비교 데이터 테이블</div>', unsafe_allow_html=True)
                        st.dataframe(
                            pivot_chart.reset_index().rename(columns={"index": "주차 (Week)"}),
                            use_container_width=True,
                            height=350
                        )

                    else:
                        pivot_actual = filtered_actual.pivot(index="week_no", columns="zone_id", values=selected_col_key)
                        pivot_actual.columns = [f"{col} (실제)" for col in pivot_actual.columns]

                        st.info(f"💡 안내: 선택하신 **{var_label}** 변수에 대응하는 9S-9P `st_` 기준선 컬럼(`{ref_col_expected}`)이 `cold_start_standards.csv`에 정의되어 있지 않거나 CSV가 로드되지 않았습니다.")
                        st.line_chart(pivot_actual)

                        st.markdown("---")
                        st.markdown('<div class="sf-group-label">📋 상세 데이터 테이블</div>', unsafe_allow_html=True)
                        st.dataframe(
                            pivot_actual.reset_index().rename(columns={"week_no": "주차 (Week)"}),
                            use_container_width=True,
                            height=350
                        )

with tab5:
    with st.container(border=True):
        st.markdown(
            """
            <div class="sf-section-header">
                <h3 class="sf-section-title">🔍 FAS-D 2단계 계층적 스윙 진단 시스템</h3>
                <span class="sf-pill sf-pill-mint">Macro-Micro Pearson Diagnostic</span>
            </div>
            <p style="font-size:0.9rem; color:#4b695d; margin-bottom:1rem;">
                실제 데이터가 수집된 주차 범위 내에서 스윙 발생 주차를 탐지하고 1·2단계 피어슨 상관분석 및 7주 시차 리드타임 추적을 수행합니다.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if not os.path.exists(DB_NAME):
            st.warning("⚠️ 저장된 DB가 없습니다. 데이터를 먼저 업로드해 주세요.")
        else:
            conn = sqlite3.connect(DB_NAME)
            df_analysis = pd.read_sql("SELECT * FROM yield_swing_data", conn)
            conn.close()

            if df_analysis.empty:
                st.warning("⚠️ DB에 저장된 데이터가 없습니다.")
            else:
                df_analysis["swing_index"] = df_analysis["swing_index"].fillna(0)
                swing_alert_df = df_analysis[df_analysis["swing_index"].abs() >= 20.0].sort_values(by="week_no")

                max_collected_week = int(df_analysis["week_no"].max()) if not df_analysis.empty else 0
                st.markdown(
                    f'<div class="sf-group-label">🚨 [스윙 모니터링] 수집 데이터 범위(1 ~ {max_collected_week}주차) 중 주의/위험 스윙(±20% 이상) 발생 리스트</div>',
                    unsafe_allow_html=True,
                )

                if len(swing_alert_df) > 0:
                    st.dataframe(
                        swing_alert_df[["zone_id", "week_no", "y_actual", "ma_3week", "swing_index", "cum_p_n_8w", "tr_ratio"]].rename(columns=FACTOR_LABELS),
                        use_container_width=True,
                        height=220
                    )
                else:
                    st.info(f"현재 수집된 1 ~ {max_collected_week}주차 데이터 중 임계값(±20%)을 초과하는 스윙 주차가 없습니다.")

                st.markdown("---")
                st.markdown('<div class="sf-group-label">🎯 재배사 진단 타겟 주차 선택 및 2단계 상관분석</div>', unsafe_allow_html=True)

                zones = df_analysis["zone_id"].unique().tolist()
                if len(zones) > 0:
                    col_sel1, col_sel2 = st.columns(2, gap="medium")
                    with col_sel1:
                        selected_zone = st.selectbox("진단 구역 선택", zones, key="tab5_zone_select")
                    with col_sel2:
                        available_weeks = sorted(df_analysis[df_analysis["zone_id"] == selected_zone]["week_no"].unique().tolist())
                        if available_weeks:
                            default_idx = len(available_weeks) - 1
                            selected_week = st.selectbox(
                                f"진단 주차 선택 (수집된 주차: 1 ~ {max_collected_week}주차)",
                                available_weeks,
                                index=default_idx,
                                key="tab5_week_select"
                            )
                        else:
                            selected_week = 1
                            st.selectbox("진단 주차 선택", [1], key="tab5_week_select_empty")

                    target_rows = df_analysis[(df_analysis["zone_id"] == selected_zone) & (df_analysis["week_no"] == selected_week)]

                    if not target_rows.empty:
                        target_row = target_rows.iloc[0]
                        current_dli = target_row["dli_daily"]

                        tc1, tc2, tc3 = st.columns(3, gap="medium")
                        with tc1:
                            st.markdown(
                                f"""
                                <div class="sf-kpi-card" style="background:#f8fbf7;">
                                    <div class="sf-kpi-label">[{selected_zone}] {selected_week}주차 실제 수확량</div>
                                    <div class="sf-kpi-value">{target_row['y_actual']:.2f} <span style="font-size:0.85rem;">kg/m²</span></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with tc2:
                            sw_val = target_row["swing_index"]
                            sw_color = "#b91c1c" if abs(sw_val) >= 20.0 else "#1c674e"
                            st.markdown(
                                f"""
                                <div class="sf-kpi-card" style="background:#f8fbf7;">
                                    <div class="sf-kpi-label">수확량 스윙 지수 (Swing Index)</div>
                                    <div class="sf-kpi-value" style="color:{sw_color};">{sw_val:.2f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with tc3:
                            st.markdown(
                                f"""
                                <div class="sf-kpi-card" style="background:#f8fbf7;">
                                    <div class="sf-kpi-label">당해주차 일일광량 (DLI)</div>
                                    <div class="sf-kpi-value">{current_dli:.2f} <span style="font-size:0.85rem;">mol/m²·day</span></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                        th_col1, th_col2 = st.columns(2, gap="medium")
                        with th_col1:
                            r_threshold = st.slider(
                                "통계적 유의미성 최소 상관계수 기준 (|r| Threshold)",
                                min_value=0.20,
                                max_value=0.80,
                                value=0.40,
                                step=0.05,
                                key="tab5_r_threshold",
                            )
                        with th_col2:
                            p_threshold = st.selectbox(
                                "유의수준 임계값 (p-value < α)",
                                options=[0.05, 0.01, 0.10],
                                index=0,
                                key="tab5_p_threshold",
                            )

                        if st.button("🔬 2단계 계층적 상관분석(Macro-Micro) 실행", key="tab5_analysis_button", use_container_width=True):
                            zone_df = df_analysis[df_analysis["zone_id"] == selected_zone].sort_values("week_no")

                            st.markdown("---")
                            macro_vars = ["x1_cum_dli_8w", "gdd_8w", "fli", "cum_p_n_8w", "tr_ratio", "w_avg"]
                            macro_records = []
                            for var_key in macro_vars:
                                r_val, p_val, n_valid = calc_pearson_stats(zone_df[var_key], zone_df["swing_index"])
                                is_sig = (abs(r_val) >= r_threshold) and (p_val < p_threshold) and (n_valid >= 4)
                                macro_records.append({
                                    "var_key": var_key,
                                    "핵심 인자 설명": FACTOR_LABELS.get(var_key, var_key),
                                    "Pearson (r)": round(r_val, 4),
                                    "|r|": round(abs(r_val), 4),
                                    "p-value": round(p_val, 4),
                                    "n": n_valid,
                                    "유의성 검증": "✅ 유의미" if is_sig else "⚪ 기준 미달",
                                    "is_sig": is_sig,
                                    "_r": r_val,
                                    "_p": p_val,
                                })

                            macro_result_df = pd.DataFrame(macro_records)
                            sig_candidates = [rec for rec in macro_records if rec["is_sig"]]
                            has_significant_factor = len(sig_candidates) > 0

                            res_col1, res_col2 = st.columns(2, gap="large")

                            with res_col1:
                                st.markdown(
                                    f'<div class="sf-group-label">[1단계] Macro 상관분석 (검증 기준: |r| ≥ {r_threshold:.2f} & p &lt; {p_threshold:.2f})</div>',
                                    unsafe_allow_html=True,
                                )
                                display_macro_df = macro_result_df[["핵심 인자 설명", "Pearson (r)", "|r|", "p-value", "유의성 검증"]]
                                st.dataframe(display_macro_df, use_container_width=True, hide_index=True)

                                if has_significant_factor:
                                    best_rec = max(sig_candidates, key=lambda rec: abs(rec["_r"]))
                                    top_factor = best_rec["var_key"]
                                    top_r = best_rec["_r"]
                                    top_p = best_rec["_p"]
                                    top_factor_name = best_rec["핵심 인자 설명"]
                                    st.success(
                                        f"⚡ **1단계 진단 결과**: {selected_week}주차 스윙에 통계적으로 유의미한 영향을 미친 핵심 인자는 "
                                        f"**'{top_factor_name}'**이며, 상관계수 `r = {top_r:.4f}` (`|r| = {abs(top_r):.4f}`, `p = {top_p:.4f}`)입니다."
                                    )
                                else:
                                    max_obs = max(macro_records, key=lambda rec: abs(rec["_r"])) if macro_records else None
                                    st.warning(
                                        "⚠️ 현재 선택된 주차의 스윙은 특정 단일 인자의 지배적 영향이 아닌 복합 노이즈 또는 기상 변동에 기인하며, "
                                        "통계적으로 유의미한 주범 인자가 도출되지 않았습니다."
                                    )
                                    if max_obs is not None:
                                        st.caption(
                                            f"참고 — 최대 관측 상관계수 인자 '{max_obs['핵심 인자 설명']}': "
                                            f"r = {max_obs['_r']:.4f} (|r| = {abs(max_obs['_r']):.4f}, p = {max_obs['_p']:.4f}) → "
                                            f"임계 기준(|r| ≥ {r_threshold:.2f}, p < {p_threshold:.2f}) 미충족"
                                        )

                            with res_col2:
                                top_micro_name = ""
                                top_micro_r = 0.0
                                if has_significant_factor:
                                    st.markdown(f'<div class="sf-group-label">[2단계] Micro 원인 추적 ({top_factor_name} vs 원천 센서)</div>', unsafe_allow_html=True)
                                    micro_vars = ["dli_daily", "t_mean", "vpd_mean", "fruit_load", "vwc", "drainage_rate"]
                                    micro_records = []
                                    for m_key in micro_vars:
                                        mr_val, mp_val, _ = calc_pearson_stats(zone_df[m_key], zone_df[top_factor])
                                        micro_records.append({
                                            "원천 센서 설명": FACTOR_LABELS.get(m_key, m_key),
                                            f"Correlation (r) with {top_factor}": round(mr_val, 4),
                                            "p-value": round(mp_val, 4),
                                            "_r": mr_val,
                                        })
                                    if micro_records:
                                        best_micro = max(micro_records, key=lambda rec: abs(rec["_r"]))
                                        top_micro_name = best_micro["원천 센서 설명"]
                                        top_micro_r = best_micro["_r"]
                                    micro_df = pd.DataFrame(micro_records)[["원천 센서 설명", f"Correlation (r) with {top_factor}", "p-value"]]
                                    st.dataframe(micro_df, use_container_width=True, hide_index=True)
                                else:
                                    st.markdown('<div class="sf-group-label">[2단계] Micro 원인 추적 (실행 보류)</div>', unsafe_allow_html=True)
                                    st.info(
                                        "🛡️ **Guardrail 작동 (오진단 방지)**\n\n"
                                        "1단계 Macro 상관분석에서 통계적 유의미성 기준"
                                        f"(`|r| ≥ {r_threshold:.2f}` 및 `p < {p_threshold:.2f}`)을 충족하는 지배적 인자가 없어, "
                                        "노이즈 기반의 2단계 Micro 추적 및 7주 시차 리드타임 인사이트 생성을 중단했습니다."
                                    )

                            if has_significant_factor:
                                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                                lag_week = max(1, selected_week - 7)
                                lag_row = zone_df[zone_df["week_no"] == lag_week]

                                if len(lag_row) > 0:
                                    l_row = lag_row.iloc[0]
                                    st.markdown(
                                        f"""
                                        <div class="sf-insight-card">
                                            <strong>💡 엔지니어링 인사이트 (7주 시차 리드타임 분석)</strong><br/>
                                            현재 <strong>{selected_week}주차</strong> 수확량 스윙의 1단계 핵심 병목 인자는
                                            <strong>[ {top_factor_name} ]</strong> (<code>r = {top_r:.4f}</code>)로 진단되었습니다.<br/>
                                            이를 2단계 원천 센서 추적(최대 상관 원천 변수: <strong>{top_micro_name}</strong>, <code>r = {top_micro_r:.4f}</code>) 및
                                            작물의 생리적 발달 시차인 7주 전(<strong>{lag_week}주차</strong>) 데이터로 연결한 결과,
                                            7주 전 일일 광량(<code>dli_daily</code>)은 <code>{l_row['dli_daily']:.2f}</code>,
                                            평균온도(<code>t_mean</code>)는 <code>{l_row['t_mean']:.2f}</code>(당해주차 일일 광량 <code>{current_dli:.2f}</code>)로 나타났습니다.<br/>
                                            즉, <strong>7주 전({lag_week}주차)의 광·온도 환경 제어 편차</strong>가 누적되어
                                            현재 <strong>{selected_week}주차</strong>의 <strong>{top_factor_name}</strong> 생리적 부하 및 수확량 스윙을 유발한
                                            <strong>근본 원인(Root Cause)</strong>으로 작용했습니다.
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.info("7주 전 과거 데이터 구간이 충분하지 않습니다.")

with tab6:
    with st.container(border=True):
        st.markdown(
            """
            <div class="sf-section-header">
                <h3 class="sf-section-title">🔮 ML 수확량 & 단기 스윙 예측 (3차 고도화 · 홀드아웃 검증 + 스케일 보정)</h3>
                <span class="sf-pill sf-pill-mint">9-Feature Lag Rollout · 홀드아웃 3주 검증 · 엔탈피 공조 권고</span>
            </div>
            <p style="font-size:0.9rem; color:#4b695d; margin-bottom:1rem;">
                작물의 <strong>7주 시차 광·온도(Lag)</strong>를 포함한 9개 피처와 <strong>최대 3개 ML 모델</strong>을 동시 비교하며,
                <strong>홀드아웃 검증 오차(MAE·RMSE)</strong>, <strong>수확량 스케일 보정 옵션</strong>,
                <strong>엔탈피 공조 조향 권고</strong>, <strong>52주 누적 수확량 전망</strong>을 제공합니다.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if not SKLEARN_AVAILABLE:
            st.error("⚠️ `scikit-learn>=1.3.0` 패키지가 설치되어 있지 않습니다. `requirements.txt`를 확인한 뒤 설치해 주세요.")
        elif not os.path.exists(DB_NAME):
            st.warning("⚠️ 저장된 DB가 없습니다. Tab 1 또는 Tab 2에서 데이터를 먼저 등록해 주세요.")
        else:
            conn = sqlite3.connect(DB_NAME)
            df_ml_all = pd.read_sql("SELECT * FROM yield_swing_data", conn)
            conn.close()

            if df_ml_all.empty:
                st.warning("⚠️ DB에 저장된 데이터가 없습니다. 데이터를 먼저 업로드해 주세요.")
            else:
                ml_zones = sorted(df_ml_all["zone_id"].dropna().unique().tolist())
                sel_c1, sel_c2, sel_c3 = st.columns([1, 1, 1.1], gap="medium")
                with sel_c1:
                    ml_zone = st.selectbox("예측 대상 구역 선택", ml_zones, key="tab6_zone_select")
                zone_full_df = (
                    df_ml_all[df_ml_all["zone_id"] == ml_zone]
                    .sort_values("week_no")
                    .drop_duplicates(subset=["week_no"], keep="last")
                    .reset_index(drop=True)
                )
                zone_weeks = sorted(zone_full_df["week_no"].astype(int).unique().tolist())

                with sel_c2:
                    if zone_weeks:
                        default_w_idx = zone_weeks.index(20) if 20 in zone_weeks else len(zone_weeks) - 1
                        base_week = int(
                            st.selectbox(
                                "기준 현재 주차 선택 (선택 주차까지만 학습 데이터로 사용)",
                                zone_weeks,
                                index=default_w_idx,
                                format_func=lambda w: f"{w}주차 기준",
                                key="tab6_base_week_select",
                            )
                        )
                    else:
                        base_week = 1
                        st.selectbox("기준 현재 주차 선택", [1], key="tab6_base_week_empty")

                scale_options = {
                    1.0: "1.0배 (원본 데이터 유지 · 기본값)",
                    0.4: "0.4배 (연간 52주 누적 기준 보정)",
                    0.5: "0.5배 (반감 스케일 보정)",
                    0.25: "0.25배 (저밀도·초기 수확 보정)",
                    1.5: "1.5배 (고밀도 확장 보정)",
                }
                with sel_c3:
                    yield_scale = float(
                        st.selectbox(
                            "수확량 스케일 배율 (연간 누적 정합성 보정)",
                            options=list(scale_options.keys()),
                            index=0,
                            format_func=lambda s: scale_options[s],
                            key="tab6_yield_scale_select",
                        )
                    )

                st.caption(
                    "ℹ️ 수확량 스케일 정합성 안내: 주간 수확량(`y_actual`, `y_target`)이 52주 연간 누적 기준 대비 크게 입력된 샘플 데이터를 점검할 때 "
                    "배율(예: `0.4배`)을 선택하면 DB 원본을 변경하지 않고 20주차·52주차 누적 수확량 감각을 즉시 비교할 수 있습니다."
                )

                model_catalog = {
                    "RandomForestRegressor": {
                        "label": "RandomForestRegressor (1차 기준선 모델)",
                        "color": "#f97316",
                        "dash": "dash",
                    },
                    "HistGradientBoostingRegressor": {
                        "label": "HistGradientBoostingRegressor (히스토그램 부스팅)",
                        "color": "#16a34a",
                        "dash": "dot",
                    },
                    "ExtraTreesRegressor": {
                        "label": "ExtraTreesRegressor (엑스트라 트리 앙상블)",
                        "color": "#9333ea",
                        "dash": "dashdot",
                    },
                }

                selected_models = st.multiselect(
                    "비교할 ML 모델 선택 (1~3개 선택 · 첫 번째 모델이 대표 모델로 적용)",
                    options=list(model_catalog.keys()),
                    default=["RandomForestRegressor", "HistGradientBoostingRegressor", "ExtraTreesRegressor"],
                    max_selections=3,
                    format_func=lambda m: model_catalog[m]["label"],
                    key="tab6_model_multiselect",
                )

                zone_scaled_df = zone_full_df.copy()
                for sc_col in ["y_actual", "y_target", "ma_3week"]:
                    if sc_col in zone_scaled_df.columns:
                        zone_scaled_df[sc_col] = pd.to_numeric(zone_scaled_df[sc_col], errors="coerce") * yield_scale

                train_df = zone_scaled_df[zone_scaled_df["week_no"] <= base_week].sort_values("week_no").copy()

                if len(train_df) < 8:
                    st.warning("머신러닝 학습을 위해 최소 8주 이상의 수집 데이터가 필요합니다.")
                elif not selected_models:
                    st.info("💡 비교할 머신러닝 모델을 최소 1개 이상 선택해 주세요.")
                else:
                    feature_cols = [
                        "ma_3week",
                        "dli_daily",
                        "t_mean",
                        "vpd_mean",
                        "fruit_load",
                        "cum_p_n_8w",
                        "tr_ratio",
                        "lagged_dli_7w",
                        "lagged_temp_7w",
                    ]
                    train_ml = train_df.copy()
                    train_ml["ma_3week"] = pd.to_numeric(train_ml["ma_3week"], errors="coerce").fillna(
                        pd.to_numeric(train_ml["y_actual"], errors="coerce")
                    )
                    if "lagged_dli_7w" in train_ml.columns:
                        train_ml["lagged_dli_7w"] = pd.to_numeric(train_ml["lagged_dli_7w"], errors="coerce").fillna(
                            pd.to_numeric(train_ml["dli_daily"], errors="coerce")
                        )
                    else:
                        train_ml["lagged_dli_7w"] = pd.to_numeric(train_ml["dli_daily"], errors="coerce")

                    if "lagged_temp_7w" in train_ml.columns:
                        train_ml["lagged_temp_7w"] = pd.to_numeric(train_ml["lagged_temp_7w"], errors="coerce").fillna(
                            pd.to_numeric(train_ml["t_mean"], errors="coerce")
                        )
                    else:
                        train_ml["lagged_temp_7w"] = pd.to_numeric(train_ml["t_mean"], errors="coerce")

                    for fc in feature_cols + ["y_actual", "y_target"]:
                        train_ml[fc] = pd.to_numeric(train_ml[fc], errors="coerce")
                        if train_ml[fc].isna().all():
                            train_ml[fc] = 0.0
                        else:
                            train_ml[fc] = train_ml[fc].fillna(train_ml[fc].median())

                    X_train = train_ml[feature_cols].to_numpy(dtype=float)
                    y_train = train_ml["y_actual"].to_numpy(dtype=float)

                    # 기준선(9S-9P) 조회용 맵 구성
                    bench_lookup = {}
                    if not df_bench.empty and "week_no" in df_bench.columns:
                        if "zone_id" in df_bench.columns and ml_zone in df_bench["zone_id"].values:
                            b_sub = df_bench[df_bench["zone_id"] == ml_zone]
                        elif "zone_id" in df_bench.columns and "Zone_A" in df_bench["zone_id"].values:
                            b_sub = df_bench[df_bench["zone_id"] == "Zone_A"]
                        else:
                            b_sub = df_bench
                        b_sub = b_sub.groupby("week_no", as_index=False).mean(numeric_only=True)
                        for _, brow in b_sub.iterrows():
                            bench_lookup[int(brow["week_no"])] = brow

                    recent_tail = train_ml.tail(4)
                    recent_means = {
                        "dli_daily": float(recent_tail["dli_daily"].mean()),
                        "t_mean": float(recent_tail["t_mean"].mean()),
                        "vpd_mean": float(recent_tail["vpd_mean"].mean()),
                        "fruit_load": float(recent_tail["fruit_load"].mean()),
                    }

                    # 기준 주차에서의 표준 기준선 대비 편차 계산
                    base_bench = bench_lookup.get(base_week)
                    base_offsets = {}
                    for ek in ["dli_daily", "t_mean", "vpd_mean", "fruit_load"]:
                        st_key = f"st_{ek}"
                        if base_bench is not None and st_key in base_bench and pd.notna(base_bench[st_key]):
                            base_offsets[ek] = float(train_ml[ek].iloc[-1]) - float(base_bench[st_key])
                        else:
                            base_offsets[ek] = 0.0

                    # 목표 수확량(y_target) 1~52주 합계 계산 (스케일 배율 동일 반영)
                    target_by_week = {}
                    for _, zrow in zone_scaled_df.iterrows():
                        if pd.notna(zrow.get("y_target")):
                            target_by_week[int(zrow["week_no"])] = float(zrow["y_target"])
                    avg_train_target = float(train_ml["y_target"].mean()) if not train_ml.empty else 3.0 * yield_scale
                    season_target_sum = 0.0
                    for w_i in range(1, 53):
                        if w_i in target_by_week:
                            season_target_sum += target_by_week[w_i]
                        elif w_i in bench_lookup and "st_y_target" in bench_lookup[w_i] and pd.notna(bench_lookup[w_i]["st_y_target"]):
                            season_target_sum += float(bench_lookup[w_i]["st_y_target"]) * yield_scale
                        else:
                            season_target_sum += avg_train_target

                    max_horizon_week = max(52, base_week + 3)
                    cum_actual = float(train_ml["y_actual"].sum())
                    remaining_weeks_cnt = max(0, 52 - base_week)

                    # 미래 주차별 공통 환경/생육(7주 시차 포함) 타임라인 사전 생성
                    base_env_by_week = {}
                    for _, r_tr in train_ml.iterrows():
                        base_env_by_week[int(r_tr["week_no"])] = {
                            "dli_daily": float(r_tr["dli_daily"]),
                            "t_mean": float(r_tr["t_mean"]),
                            "vpd_mean": float(r_tr["vpd_mean"]),
                            "fruit_load": float(r_tr["fruit_load"]),
                        }

                    first_dli = float(train_ml["dli_daily"].iloc[0])
                    first_temp = float(train_ml["t_mean"].iloc[0])

                    future_env_steps = []
                    env_seq = [
                        {
                            "dli_daily": float(r["dli_daily"]),
                            "vpd_mean": float(r["vpd_mean"]),
                            "fruit_load": float(r["fruit_load"]),
                        }
                        for _, r in train_ml.iterrows()
                    ]

                    for w_idx in range(base_week + 1, max_horizon_week + 1):
                        steps_ahead = w_idx - base_week
                        decay = math.exp(-0.30 * steps_ahead)
                        cycle_week = ((w_idx - 1) % 52) + 1
                        w_bench = bench_lookup.get(cycle_week)
                        step_env = {}
                        for ek in ["dli_daily", "t_mean", "vpd_mean", "fruit_load"]:
                            st_key = f"st_{ek}"
                            if w_bench is not None and st_key in w_bench and pd.notna(w_bench[st_key]):
                                step_env[ek] = float(w_bench[st_key]) + base_offsets[ek] * decay
                            else:
                                phase = ((cycle_week - 1) / 51.0) * 2.0 * math.pi
                                base_phase = ((base_week - 1) / 51.0) * 2.0 * math.pi
                                seasonal_delta = math.sin(phase - 0.4) - math.sin(base_phase - 0.4)
                                amp_map = {"dli_daily": 2.5, "t_mean": 1.4, "vpd_mean": 0.18, "fruit_load": 6.0}
                                step_env[ek] = recent_means[ek] + amp_map[ek] * seasonal_delta

                        base_env_by_week[w_idx] = step_env
                        env_seq.append({
                            "dli_daily": step_env["dli_daily"],
                            "vpd_mean": step_env["vpd_mean"],
                            "fruit_load": step_env["fruit_load"],
                        })
                        recent_8_env = env_seq[-8:]
                        temp_pn_sum = sum((item["dli_daily"] * 0.85) - (item["vpd_mean"] * 1.2) for item in recent_8_env)
                        step_cum_pn = float(np.clip(temp_pn_sum * 10.0 + 500.0, 300.0, 1200.0))
                        step_tr = float(3.5 - 0.02 * step_env["fruit_load"] + 0.001 * step_cum_pn)

                        lag_w = w_idx - 7
                        if lag_w in base_env_by_week:
                            step_lag_dli = float(base_env_by_week[lag_w]["dli_daily"])
                            step_lag_temp = float(base_env_by_week[lag_w]["t_mean"])
                        else:
                            step_lag_dli = first_dli
                            step_lag_temp = first_temp

                        future_env_steps.append({
                            "week_no": w_idx,
                            "dli_daily": step_env["dli_daily"],
                            "t_mean": step_env["t_mean"],
                            "vpd_mean": step_env["vpd_mean"],
                            "fruit_load": step_env["fruit_load"],
                            "cum_p_n_8w": step_cum_pn,
                            "tr_ratio": step_tr,
                            "lagged_dli_7w": step_lag_dli,
                            "lagged_temp_7w": step_lag_temp,
                        })

                    def build_estimator(name: str):
                        if name == "RandomForestRegressor":
                            return RandomForestRegressor(
                                n_estimators=150,
                                max_depth=6,
                                min_samples_leaf=2,
                                random_state=42,
                            )
                        if name == "HistGradientBoostingRegressor":
                            return HistGradientBoostingRegressor(
                                max_iter=120,
                                max_depth=4,
                                min_samples_leaf=3,
                                learning_rate=0.08,
                                random_state=42,
                            )
                        return ExtraTreesRegressor(
                            n_estimators=150,
                            max_depth=6,
                            min_samples_leaf=2,
                            random_state=42,
                        )

                    def compute_feature_importances(estimator, X_mat, y_vec):
                        if hasattr(estimator, "feature_importances_"):
                            raw_imp = np.asarray(estimator.feature_importances_, dtype=float)
                        else:
                            try:
                                perm = permutation_importance(
                                    estimator,
                                    X_mat,
                                    y_vec,
                                    n_repeats=5,
                                    random_state=42,
                                )
                                raw_imp = np.maximum(0.0, np.asarray(perm.importances_mean, dtype=float))
                            except Exception:
                                raw_imp = np.ones(X_mat.shape[1], dtype=float)
                        total_imp = float(np.sum(raw_imp))
                        if total_imp <= 0.0:
                            return np.full(X_mat.shape[1], 1.0 / X_mat.shape[1], dtype=float)
                        return raw_imp / total_imp

                    # 홀드아웃 3주 검증 가능 여부 (base_week >= 12주 및 학습 데이터 >= 12행)
                    can_holdout_val = (base_week >= 12) and (len(train_ml) >= 12)
                    if can_holdout_val:
                        val_fit_ml = train_ml.iloc[:-3].copy()
                        val_eval_ml = train_ml.iloc[-3:].copy()
                        X_val_fit = val_fit_ml[feature_cols].to_numpy(dtype=float)
                        y_val_fit = val_fit_ml["y_actual"].to_numpy(dtype=float)
                        y_val_true = val_eval_ml["y_actual"].to_numpy(dtype=float)
                        val_weeks_label = f"{int(val_eval_ml['week_no'].iloc[0])}~{int(val_eval_ml['week_no'].iloc[-1])}주차"
                    else:
                        val_weeks_label = "N/A"

                    model_results = {}
                    comparison_rows = []

                    for idx_m, m_name in enumerate(selected_models):
                        est = build_estimator(m_name)
                        est.fit(X_train, y_train)

                        y_in_pred = est.predict(X_train)
                        mae_val = float(np.mean(np.abs(y_train - y_in_pred)))
                        rmse_val = float(np.sqrt(np.mean((y_train - y_in_pred) ** 2)))

                        # 마지막 3주 홀드아웃 순차 롤아웃 검증 오차 계산
                        if can_holdout_val:
                            val_est = build_estimator(m_name)
                            val_est.fit(X_val_fit, y_val_fit)
                            val_yield_hist = val_fit_ml["y_actual"].tolist()
                            val_preds = []
                            for _, v_row in val_eval_ml.iterrows():
                                v_ma3 = float(np.mean(val_yield_hist[-3:])) if val_yield_hist else float(val_fit_ml["y_actual"].mean())
                                x_val_vec = np.array([[
                                    v_ma3,
                                    float(v_row["dli_daily"]),
                                    float(v_row["t_mean"]),
                                    float(v_row["vpd_mean"]),
                                    float(v_row["fruit_load"]),
                                    float(v_row["cum_p_n_8w"]),
                                    float(v_row["tr_ratio"]),
                                    float(v_row["lagged_dli_7w"]),
                                    float(v_row["lagged_temp_7w"]),
                                ]], dtype=float)
                                v_pred = max(0.05, float(val_est.predict(x_val_vec)[0]))
                                val_preds.append(v_pred)
                                val_yield_hist.append(v_pred)
                            y_val_pred_arr = np.asarray(val_preds, dtype=float)
                            val_mae_val = float(np.mean(np.abs(y_val_true - y_val_pred_arr)))
                            val_rmse_val = float(np.sqrt(np.mean((y_val_true - y_val_pred_arr) ** 2)))
                            val_mae_disp = round(val_mae_val, 4)
                            val_rmse_disp = round(val_rmse_val, 4)
                        else:
                            val_mae_val = None
                            val_rmse_val = None
                            val_mae_disp = "N/A (데이터 부족)"
                            val_rmse_disp = "N/A (데이터 부족)"

                        yield_hist = train_ml["y_actual"].tolist()
                        m_future_rows = []

                        for f_step in future_env_steps:
                            curr_ma3 = float(np.mean(yield_hist[-3:])) if yield_hist else float(train_ml["y_actual"].mean())
                            x_vec = np.array([[
                                curr_ma3,
                                f_step["dli_daily"],
                                f_step["t_mean"],
                                f_step["vpd_mean"],
                                f_step["fruit_load"],
                                f_step["cum_p_n_8w"],
                                f_step["tr_ratio"],
                                f_step["lagged_dli_7w"],
                                f_step["lagged_temp_7w"],
                            ]], dtype=float)

                            y_pred = max(0.05, float(est.predict(x_vec)[0]))
                            denom = curr_ma3 if curr_ma3 != 0 else 1.0
                            swing_pred = ((y_pred - curr_ma3) / denom) * 100.0
                            abs_sw = abs(swing_pred)
                            if abs_sw >= 20.0:
                                sw_level = "위험"
                            elif abs_sw >= 10.0:
                                sw_level = "주의"
                            else:
                                sw_level = "안정"

                            m_future_rows.append({
                                "week_no": f_step["week_no"],
                                "ma_3week_ref": curr_ma3,
                                "y_pred": y_pred,
                                "swing_index_pred": swing_pred,
                                "swing_level": sw_level,
                                "is_alert": abs_sw >= 10.0,
                            })
                            yield_hist.append(y_pred)

                        m_future_df = pd.DataFrame(m_future_rows)
                        m_short_df = m_future_df.head(3).copy()
                        m_season_df = m_future_df[m_future_df["week_no"] <= 52].copy()

                        m_cum_future = float(m_season_df["y_pred"].sum()) if not m_season_df.empty else 0.0
                        m_total_pred = cum_actual + m_cum_future
                        m_ach_rate = (m_total_pred / season_target_sum * 100.0) if season_target_sum > 0 else 0.0
                        m_short_avg_swing = float(m_short_df["swing_index_pred"].mean()) if not m_short_df.empty else 0.0

                        imp_arr = compute_feature_importances(est, X_train, y_train)

                        model_results[m_name] = {
                            "mae": mae_val,
                            "rmse": rmse_val,
                            "val_mae": val_mae_val,
                            "val_rmse": val_rmse_val,
                            "future_df": m_future_df,
                            "short_df": m_short_df,
                            "season_df": m_season_df,
                            "cum_future": m_cum_future,
                            "total_pred": m_total_pred,
                            "achievement_rate": m_ach_rate,
                            "short_avg_swing": m_short_avg_swing,
                            "importances": imp_arr,
                        }

                        comparison_rows.append({
                            "모델명": f"{m_name} (대표)" if idx_m == 0 else m_name,
                            "학습 MAE (kg/m²)": round(mae_val, 4),
                            "학습 RMSE (kg/m²)": round(rmse_val, 4),
                            "검증 MAE (kg/m²)": val_mae_disp,
                            "검증 RMSE (kg/m²)": val_rmse_disp,
                            "향후 3주 평균 스윙 지수 (%)": round(m_short_avg_swing, 2),
                            "52주 최종 예상 총 수확량 (kg/m²)": round(m_total_pred, 2),
                            "목표 달성률 (%)": round(m_ach_rate, 1),
                        })

                    rep_model_name = selected_models[0]
                    rep_res = model_results[rep_model_name]
                    short_term_df = rep_res["short_df"]

                    # ---------------------------------------------------------
                    # 1) 잔여 시즌 누적 수확량 전망 KPI (대표 모델 기준)
                    # ---------------------------------------------------------
                    scale_badge_txt = f" · 스케일 {yield_scale:.2f}배 적용" if yield_scale != 1.0 else ""
                    st.markdown(
                        f'<div class="sf-group-label">📦 잔여 시즌 누적 수확량 전망 (1 ~ 52주차 · 대표 모델: {rep_model_name}{scale_badge_txt})</div>',
                        unsafe_allow_html=True,
                    )
                    m1, m2, m3, m4 = st.columns(4, gap="medium")
                    with m1:
                        st.metric(
                            label=f"현재까지 누적 수확량 (1~{base_week}주)",
                            value=f"{cum_actual:.2f} kg/m²",
                            delta=f"{len(train_df)}주 실측 합계",
                        )
                    with m2:
                        future_label = f"잔여 시즌 예상 수확량 ({base_week + 1}~52주)" if remaining_weeks_cnt > 0 else "잔여 시즌 예상 수확량 (완료)"
                        st.metric(
                            label=future_label,
                            value=f"{rep_res['cum_future']:.2f} kg/m²",
                            delta=f"잔여 {remaining_weeks_cnt}주 예측",
                        )
                    with m3:
                        st.metric(
                            label="최종 시즌 예상 총 수확량",
                            value=f"{rep_res['total_pred']:.2f} kg/m²",
                            delta=f"{rep_res['total_pred'] - season_target_sum:+.2f} kg/m² (vs 목표 {season_target_sum:.1f})",
                        )
                    with m4:
                        st.metric(
                            label="목표 수확량 대비 달성률",
                            value=f"{rep_res['achievement_rate']:.1f}%",
                            delta=f"{rep_res['achievement_rate'] - 100.0:+.1f}%p",
                        )

                    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

                    # ---------------------------------------------------------
                    # 2) 단기 스윙 조기 경보 (향후 3주 배지 요약 · 대표 모델 기준)
                    # ---------------------------------------------------------
                    st.markdown(
                        f'<div class="sf-group-label">🚨 단기 스윙 조기 경보 ({base_week + 1} ~ {base_week + 3}주차 예측 · 대표 모델: {rep_model_name})</div>',
                        unsafe_allow_html=True,
                    )
                    badge_cols = st.columns(3, gap="medium")
                    for idx_b, (_, s_row) in enumerate(short_term_df.iterrows()):
                        with badge_cols[idx_b]:
                            sw_lvl = s_row["swing_level"]
                            if sw_lvl == "위험":
                                card_bg, card_border, badge_bg = "#fef2f2", "#fecaca", "#dc2626"
                                badge_txt, val_color = "🚨 스윙 위험 경보", "#b91c1c"
                            elif sw_lvl == "주의":
                                card_bg, card_border, badge_bg = "#fffbeb", "#fde68a", "#d97706"
                                badge_txt, val_color = "⚠️ 스윙 주의 경보", "#b45309"
                            else:
                                card_bg, card_border, badge_bg = "#f0fdf4", "#bbf7d0", "#15803d"
                                badge_txt, val_color = "✅ 정상 조향 유지", "#143f33"

                            st.markdown(
                                f"""
                                <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:14px; padding:0.95rem 1.1rem;">
                                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.45rem;">
                                        <span style="font-size:0.85rem; font-weight:800; color:#143f33;">{int(s_row['week_no'])}주차 예측 (t+{idx_b + 1})</span>
                                        <span style="background:{badge_bg}; color:#ffffff; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.72rem; font-weight:700;">{badge_txt}</span>
                                    </div>
                                    <div style="font-size:1.25rem; font-weight:800; color:{val_color};">
                                        {s_row['y_pred']:.2f} <span style="font-size:0.8rem; font-weight:600;">kg/m²</span>
                                        <span style="font-size:0.88rem; margin-left:0.45rem;">(스윙 {s_row['swing_index_pred']:+.2f}%)</span>
                                    </div>
                                    <div style="font-size:0.75rem; color:#4b695d; margin-top:0.25rem;">
                                        직전 3주 이동평균 기준: {s_row['ma_3week_ref']:.2f} kg/m²
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # ---------------------------------------------------------
                    # 2-1) 단기 스윙 경보 연동 '엔탈피 공조 & 생육 조향 권고 가이드' (테스트 모드)
                    # ---------------------------------------------------------
                    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
                    max_abs_sw_3w = float(short_term_df["swing_index_pred"].abs().max()) if not short_term_df.empty else 0.0
                    mean_sw_3w = float(short_term_df["swing_index_pred"].mean()) if not short_term_df.empty else 0.0

                    last_obs = train_ml.iloc[-1]
                    curr_t = float(last_obs["t_mean"])
                    curr_vpd = float(last_obs["vpd_mean"])
                    curr_dli_val = float(last_obs["dli_daily"])
                    curr_fruit = float(last_obs["fruit_load"])
                    curr_lag_dli = float(last_obs["lagged_dli_7w"])
                    curr_lag_t = float(last_obs["lagged_temp_7w"])

                    ref_t = float(base_bench["st_t_mean"]) if (base_bench is not None and "st_t_mean" in base_bench and pd.notna(base_bench["st_t_mean"])) else float(train_ml["t_mean"].mean())
                    ref_vpd = float(base_bench["st_vpd_mean"]) if (base_bench is not None and "st_vpd_mean" in base_bench and pd.notna(base_bench["st_vpd_mean"])) else float(train_ml["vpd_mean"].mean())
                    ref_dli = float(base_bench["st_dli_daily"]) if (base_bench is not None and "st_dli_daily" in base_bench and pd.notna(base_bench["st_dli_daily"])) else float(train_ml["dli_daily"].mean())
                    ref_fruit = float(base_bench["st_fruit_load"]) if (base_bench is not None and "st_fruit_load" in base_bench and pd.notna(base_bench["st_fruit_load"])) else float(train_ml["fruit_load"].mean())

                    delta_t = curr_t - ref_t
                    delta_vpd = curr_vpd - ref_vpd
                    delta_dli = curr_dli_val - ref_dli
                    delta_fruit = curr_fruit - ref_fruit

                    is_danger = max_abs_sw_3w >= 20.0
                    is_caution = (max_abs_sw_3w >= 10.0) and (not is_danger)

                    if is_danger or is_caution:
                        guide_items = []

                        # 엔탈피 공조(열·수분 밸런스) 통합 진단
                        high_temp = (delta_t >= 0.8) or (curr_t >= 22.5)
                        low_temp = (delta_t <= -0.8) or (curr_t <= 17.5)
                        high_vpd = (delta_vpd >= 0.15) or (curr_vpd >= 1.55)
                        low_vpd = (delta_vpd <= -0.15) or (curr_vpd <= 0.85)

                        if high_temp and high_vpd:
                            guide_items.append(
                                f"<strong>고온·과건조 엔탈피 불균형 대응</strong>: 현재 평균온도(<code>{curr_t:.1f}°C</code>, 기준 대비 <code>{delta_t:+.1f}°C</code>)와 "
                                f"증기압차(<code>{curr_vpd:.2f} kPa</code>, 기준 대비 <code>{delta_vpd:+.2f} kPa</code>)가 모두 높습니다. "
                                "단순 냉방 환기만 가동하면 과건조가 심화되므로 <strong>고압 포그/가습 분무와 차광·냉방 환기를 병행</strong>하여 실내 엔탈피(현열+잠열)를 동시 하강시키고 기공 폐쇄를 방지하세요."
                            )
                        elif high_temp and low_vpd:
                            guide_items.append(
                                f"<strong>고온·과습(저 VPD) 엔탈피 정체 대응</strong>: 온도(<code>{curr_t:.1f}°C</code>)는 높으나 VPD(<code>{curr_vpd:.2f} kPa</code>, <code>{delta_vpd:+.2f} kPa</code>)가 낮아 증산이 억제되고 있습니다. "
                                "<strong>제습 공조 가동과 순환팬·천창 환기량 상향</strong>을 결합해 잠열 부하와 수분 정체를 동시에 배출하세요."
                            )
                        elif low_temp and low_vpd:
                            guide_items.append(
                                f"<strong>저온·과습(VPD 부족) 엔탈피 저하 대응</strong>: 평균온도(<code>{curr_t:.1f}°C</code>, <code>{delta_t:+.1f}°C</code>)와 "
                                f"VPD(<code>{curr_vpd:.2f} kPa</code>, <code>{delta_vpd:+.2f} kPa</code>)가 동반 하락했습니다. "
                                "<strong>야간 최저온도 상향 및 최소 환기·제습 보완</strong>으로 이슬점 결로를 예방하고 근권 양수분 흡수력을 회복하세요."
                            )
                        elif low_temp and high_vpd:
                            guide_items.append(
                                f"<strong>저온·과건조 엔탈피 편차 대응</strong>: 저온(<code>{curr_t:.1f}°C</code>) 구간에서 VPD(<code>{curr_vpd:.2f} kPa</code>)가 높아 난방 건조 스트레스가 우려됩니다. "
                                "<strong>보온 커튼 조기 밀폐 및 미세 가습</strong>으로 온습도 엔탈피 균형을 완만하게 보정하세요."
                            )
                        else:
                            if high_vpd:
                                guide_items.append(
                                    f"<strong>증기압차 과다(고 VPD) 수분 밸런스 조정</strong>: VPD가 <code>{curr_vpd:.2f} kPa</code>(기준 대비 <code>{delta_vpd:+.2f}</code>)로 높아 "
                                    "가습·차광 보정으로 잎 증산 스트레스를 완화하는 엔탈피 제어를 권장합니다."
                                )
                            elif low_vpd:
                                guide_items.append(
                                    f"<strong>증기압차 부족(저 VPD) 제습·환기 보완</strong>: VPD가 <code>{curr_vpd:.2f} kPa</code>(기준 대비 <code>{delta_vpd:+.2f}</code>)로 낮아 "
                                    "야간 가온 환기 및 제습 공조로 적정 증산 흐름을 확보하세요."
                                )
                            else:
                                guide_items.append(
                                    f"<strong>선행 7주 시차·생육 리듬 연동 엔탈피 미세 조향</strong>: 현재 온습도(<code>{curr_t:.1f}°C</code> / VPD <code>{curr_vpd:.2f} kPa</code>)는 기준 부근이나 "
                                    f"7주 전 광량(<code>{curr_lag_dli:.1f}</code>)·온도(<code>{curr_lag_t:.1f}°C</code>) 시차 영향으로 향후 스윙 변동이 예측됩니다. "
                                    "주야간 온도차(RTR)와 습도 편차 진폭을 최소화하는 안정형 엔탈피 공조를 유지하세요."
                                )

                        # 광량 및 착과 부하 생육 조향 권고
                        if delta_dli <= -1.5 or delta_fruit >= 3.0:
                            guide_items.append(
                                f"<strong>착과 부하 및 광-온 비(RTR) 조절</strong>: 현재 일일 광량(<code>{curr_dli_val:.1f}</code>, 기준 대비 <code>{delta_dli:+.1f}</code>) 대비 "
                                f"착과 부하(<code>{curr_fruit:.1f}개/m²</code>, 기준 대비 <code>{delta_fruit:+.1f}</code>) 부담이 커지고 있습니다. "
                                "<strong>적과를 통한 착과 부하 완화 및 24시간 평균온도 소폭 하향</strong>으로 수확량 급락 스윙을 방어하세요."
                            )
                        elif delta_dli >= 1.5 or delta_fruit <= -3.0:
                            guide_items.append(
                                f"<strong>동화산물 전류 및 착과 밸런스 확보</strong>: 광량 여유(<code>{curr_dli_val:.1f}</code>, <code>{delta_dli:+.1f}</code>) 대비 "
                                f"착과 부하(<code>{curr_fruit:.1f}개/m²</code>, <code>{delta_fruit:+.1f}</code>)를 적정 수준으로 유도하고 "
                                "주간 광합성 산물이 과실로 원활히 이동하도록 주야간 온도 교차를 확보하세요."
                            )
                        else:
                            swing_dir_msg = "하락(공급 저하)" if mean_sw_3w < 0 else "상승(숙기 집중)"
                            guide_items.append(
                                f"<strong>생육 및 수확 리듬 평준화</strong>: 향후 3주 평균 <code>{mean_sw_3w:+.2f}%</code>({swing_dir_msg}) 방향의 스윙이 예상되므로 "
                                f"현재 착과수(<code>{curr_fruit:.1f}개/m²</code>)와 8주 순광합성 적산량 추이를 점검해 수확 피크를 분산하세요."
                            )

                        adv_border = "#dc2626" if is_danger else "#d97706"
                        adv_bg = "#fff5f5" if is_danger else "#fffbeb"
                        adv_badge_bg = "#dc2626" if is_danger else "#d97706"
                        adv_title_badge = (
                            f"🚨 위험 경보 연동 (최대 |스윙| {max_abs_sw_3w:.1f}% ≥ 20%)"
                            if is_danger
                            else f"⚠️ 주의 경보 연동 (최대 |스윙| {max_abs_sw_3w:.1f}% ≥ 10%)"
                        )
                        items_html = "".join(f"<li style='margin-bottom:0.4rem;'>{g}</li>" for g in guide_items)
                        st.markdown(
                            f"""
                            <div style="background:{adv_bg}; border:1px solid {adv_border}44; border-left:5px solid {adv_border}; border-radius:12px; padding:1rem 1.2rem; color:#143f33;">
                                <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:0.5rem; margin-bottom:0.6rem;">
                                    <strong style="font-size:0.96rem;">🌡️ 엔탈피 공조 &amp; 생육 조향 권고 가이드</strong>
                                    <span style="background:{adv_badge_bg}; color:#ffffff; padding:0.2rem 0.65rem; border-radius:9999px; font-size:0.74rem; font-weight:700;">{adv_title_badge}</span>
                                </div>
                                <ul style="margin:0 0 0.65rem 1.15rem; padding:0; font-size:0.88rem; line-height:1.58;">
                                    {items_html}
                                </ul>
                                <div style="font-size:0.78rem; color:#4b695d; border-top:1px dashed #cbd5e1; padding-top:0.5rem;">
                                    ※ 현재 테스트 검증 단계로 자동/수동 공조 설비 직접 제어는 비활성화되어 있으며, 조향 권고 메시지만 제공됩니다.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background:#eef6ee; border:1px solid #bfd7bf; border-left:5px solid #1c674e; border-radius:12px; padding:1rem 1.2rem; color:#143f33;">
                                <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:0.5rem; margin-bottom:0.45rem;">
                                    <strong style="font-size:0.96rem;">🌡️ 엔탈피 공조 &amp; 생육 조향 권고 가이드</strong>
                                    <span style="background:#15803d; color:#ffffff; padding:0.2rem 0.65rem; border-radius:9999px; font-size:0.74rem; font-weight:700;">✅ 안정 범위 (최대 |스윙| {max_abs_sw_3w:.1f}% &lt; 10%)</span>
                                </div>
                                <p style="margin:0 0 0.55rem 0; font-size:0.88rem; line-height:1.58;">
                                    향후 3주 예상 스윙 지수가 모두 ±10% 이내로 예측되어 <strong>현재 엔탈피 공조(온도 <code>{curr_t:.1f}°C</code> · VPD <code>{curr_vpd:.2f} kPa</code>) 및 생육 밸런스가 안정적</strong>입니다. 현재 설정된 자율 조향 운전 모드를 유지하세요.
                                </p>
                                <div style="font-size:0.78rem; color:#4b695d; border-top:1px dashed #bfd7bf; padding-top:0.5rem;">
                                    ※ 현재 테스트 검증 단계로 자동/수동 공조 설비 직접 제어는 비활성화되어 있으며, 조향 권고 메시지만 제공됩니다.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

                    # ---------------------------------------------------------
                    # 3) 통합 시계열 차트 (실제 파란 실선 + 모델별 예측선 동시 비교)
                    # ---------------------------------------------------------
                    st.markdown(
                        f'<div class="sf-group-label">📈 [{ml_zone}] 1 ~ 52주차 수확량 실적(파란 실선) vs 선택 모델별 예측 비교</div>',
                        unsafe_allow_html=True,
                    )

                    actual_plot_df = train_ml[train_ml["week_no"] <= 52].copy()
                    if PLOTLY_AVAILABLE:
                        fig_ml = go.Figure()
                        fig_ml.add_trace(
                            go.Scatter(
                                x=actual_plot_df["week_no"],
                                y=actual_plot_df["y_actual"],
                                mode="lines+markers",
                                name="실제 수확량 (과거~기준 주차)",
                                line=dict(color="#2563eb", width=3),
                                marker=dict(size=6, color="#2563eb"),
                            )
                        )
                        for m_name in selected_models:
                            m_res = model_results[m_name]
                            m_season_df = m_res["season_df"]
                            if not m_season_df.empty and not actual_plot_df.empty:
                                last_act_w = int(actual_plot_df["week_no"].iloc[-1])
                                last_act_y = float(actual_plot_df["y_actual"].iloc[-1])
                                pred_x = [last_act_w] + m_season_df["week_no"].tolist()
                                pred_y = [last_act_y] + m_season_df["y_pred"].tolist()
                                style_cfg = model_catalog[m_name]
                                fig_ml.add_trace(
                                    go.Scatter(
                                        x=pred_x,
                                        y=pred_y,
                                        mode="lines+markers",
                                        name=f"{m_name} 예측",
                                        line=dict(color=style_cfg["color"], width=2.8, dash=style_cfg["dash"]),
                                        marker=dict(size=5, color=style_cfg["color"]),
                                    )
                                )
                        fig_ml.add_vline(
                            x=base_week,
                            line_dash="dot",
                            line_color="#1c674e",
                            annotation_text=f"예측 기준: {base_week}주차",
                            annotation_position="top left",
                        )
                        fig_ml.update_layout(
                            xaxis=dict(title="주차 (Week)", range=[1, 52], dtick=4, showgrid=True, gridcolor="#eef4ed"),
                            yaxis=dict(title="주간 수확량 (kg/m²)", showgrid=True, gridcolor="#eef4ed"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=440,
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff",
                            hovermode="x unified",
                            margin=dict(l=20, r=20, t=40, b=20),
                            font=dict(family="Plus Jakarta Sans, Noto Sans KR, sans-serif", color="#143f33"),
                        )
                        st.plotly_chart(fig_ml, use_container_width=True)
                    else:
                        chart_df_fb = pd.DataFrame({"주차": range(1, 53)}).set_index("주차")
                        chart_df_fb["실제 수확량"] = actual_plot_df.set_index("week_no")["y_actual"]
                        for m_name in selected_models:
                            m_season_df = model_results[m_name]["season_df"]
                            if not m_season_df.empty:
                                chart_df_fb[f"{m_name} 예측"] = m_season_df.set_index("week_no")["y_pred"]
                        st.line_chart(chart_df_fb)

                    # ---------------------------------------------------------
                    # 4) 모델별 성과 및 전망 비교 요약 테이블 (학습 오차 + 홀드아웃 검증 오차)
                    # ---------------------------------------------------------
                    val_caption_suffix = f"마지막 3주({val_weeks_label}) 홀드아웃 검증 포함" if can_holdout_val else "홀드아웃 검증: 12주 미만 데이터 부족"
                    st.markdown(
                        f'<div class="sf-group-label">🏆 모델별 성과(학습·홀드아웃 검증 MAE·RMSE) 및 52주 시즌 전망 비교 · {val_caption_suffix}</div>',
                        unsafe_allow_html=True,
                    )
                    comp_df = pd.DataFrame(comparison_rows)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # ---------------------------------------------------------
                    # 5) 하단: 대표 모델 향후 3주 상세 표 & 변수 중요도 막대 그래프
                    # ---------------------------------------------------------
                    bot_col1, bot_col2 = st.columns([1.15, 1], gap="large")
                    with bot_col1:
                        st.markdown(
                            f'<div class="sf-group-label">📋 향후 3주 단기 스윙 예측 상세 표 (대표 모델: {rep_model_name})</div>',
                            unsafe_allow_html=True,
                        )
                        short_disp = short_term_df.copy()
                        short_disp["예측 주차"] = short_disp["week_no"].astype(int).astype(str) + "주차"
                        short_disp["직전 3주 이동평균 (kg/m²)"] = short_disp["ma_3week_ref"].round(3)
                        short_disp["예상 수확량 (kg/m²)"] = short_disp["y_pred"].round(3)
                        short_disp["예상 스윙 지수 (%)"] = short_disp["swing_index_pred"].round(2)
                        short_disp["조향 상태 판정"] = short_disp["swing_level"].apply(
                            lambda lv: "🚨 위험 경보 (≥20%)" if lv == "위험" else ("⚠️ 주의 경보 (≥10%)" if lv == "주의" else "✅ 정상 조향 유지")
                        )
                        st.dataframe(
                            short_disp[
                                [
                                    "예측 주차",
                                    "직전 3주 이동평균 (kg/m²)",
                                    "예상 수확량 (kg/m²)",
                                    "예상 스윙 지수 (%)",
                                    "조향 상태 판정",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                    with bot_col2:
                        st.markdown(
                            f'<div class="sf-group-label">📊 주요 변수 중요도 9개 피처 (대표 모델: {rep_model_name})</div>',
                            unsafe_allow_html=True,
                        )
                        imp_df = pd.DataFrame({
                            "주요 변수": [FACTOR_LABELS.get(c, c) for c in feature_cols],
                            "중요도": rep_res["importances"],
                        }).sort_values("중요도", ascending=True)

                        if PLOTLY_AVAILABLE:
                            fig_imp = px.bar(
                                imp_df,
                                x="중요도",
                                y="주요 변수",
                                orientation="h",
                                text=imp_df["중요도"].map(lambda v: f"{v * 100:.1f}%"),
                                color="중요도",
                                color_continuous_scale=["#bbf7d0", "#1c674e"],
                            )
                            fig_imp.update_layout(
                                height=320,
                                showlegend=False,
                                coloraxis_showscale=False,
                                plot_bgcolor="#ffffff",
                                paper_bgcolor="#ffffff",
                                margin=dict(l=10, r=25, t=10, b=10),
                                xaxis_title="상대 중요도 (Feature Importance)",
                                yaxis_title="",
                                font=dict(family="Plus Jakarta Sans, Noto Sans KR, sans-serif", color="#143f33"),
                            )
                            fig_imp.update_traces(textposition="outside")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.bar_chart(imp_df.set_index("주요 변수")["중요도"])