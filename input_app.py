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
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background-color: var(--sf-surface) !important;
    padding: 0.55rem !important;
    border-radius: 16px !important;
    border: 1.5px solid #c5dac2 !important;
    box-shadow: 0 4px 14px rgba(20, 63, 51, 0.08) !important;
    flex-wrap: wrap !important;
}

.stTabs button[data-baseweb="tab"],
.stTabs [data-baseweb="tab"] {
    min-height: 44px !important;
    height: auto !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.05rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    color: var(--sf-forest) !important;
    background-color: #f3f7f2 !important;
    border: 1.5px solid #c7dcc4 !important;
    box-shadow: 0 1px 3px rgba(20, 63, 51, 0.06) !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
    color: inherit !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
}

.stTabs button[data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    background-color: #dfece0 !important;
    border-color: var(--sf-emerald) !important;
    color: var(--sf-forest) !important;
}

.stTabs button[data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #143f33 0%, #1c674e 100%) !important;
    border: 1.5px solid #143f33 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(20, 63, 51, 0.28) !important;
}

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
    height: 0 !important;
    background: transparent !important;
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 원천 데이터 직접 입력",
    "📂 일괄 파일 업로드 및 연산",
    "📊 DB 데이터 조회",
    "📈 구역별 시계열 및 9S-9P 비교 (Tab 4)",
    "🔍 2단계 계층적 스윙 진단 (Tab 5)"
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
                <h3 class="sf-section-title">📈 구역별 변수 시계열 및 9S-9P 기준선 비교 (Tab 4)</h3>
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
                <h3 class="sf-section-title">🔍 FAS-D 2단계 계층적 스윙 진단 시스템 (Tab 5)</h3>
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
                                        })
                                    micro_df = pd.DataFrame(micro_records)
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
                                            현재 <strong>{selected_week}주차</strong> 당해 센서 기록(일일 광량 <code>{current_dli:.2f}</code>)과 별개로,
                                            작물의 생리적 발달 시차인 7주 전(<strong>{lag_week}주차</strong>) 원천 데이터를 추적한 결과:
                                            <strong>{lag_week}주차 일일 광량(<code>dli_daily</code>)은 {l_row['dli_daily']}</strong>,
                                            <strong>평균온도(<code>t_mean</code>)는 {l_row['t_mean']}</strong>로 나타났습니다.
                                            이 7주 전({lag_week}주차)의 환경 제어 편차가 현재 <strong>{selected_week}주차</strong> 수확량 스윙의 근본 원인(Root Cause)으로 작용했습니다.
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.info("7주 전 과거 데이터 구간이 충분하지 않습니다.")
