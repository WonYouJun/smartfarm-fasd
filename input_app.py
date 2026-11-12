import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 1. 스크립트 위치 및 스트림릿 클라우드 전용 절대 경로 후보 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(BASE_DIR, "cold_start_standards.csv"),
    "/mount/src/smartfarm-fasd/cold_start_standards.csv",
    "cold_start_standards.csv",
    "./cold_start_standards.csv"
]

# 2. 존재하는 첫 번째 경로를 찾아 파일 로드
BENCHMARK_CSV = None
for path in possible_paths:
    if os.path.exists(path):
        BENCHMARK_CSV = path
        break

if BENCHMARK_CSV:
    df_bench = pd.read_csv(BENCHMARK_CSV)
    df_bench.columns = df_bench.columns.str.strip()
    st.sidebar.success(f"✅ CSV 로드 성공: {len(df_bench)}행")
else:
    df_bench = pd.DataFrame()

# DB 경로 설정
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

st.set_page_config(page_title="FAS-D 스마트팜 자율 조향 & 진단 대시보드", layout="wide")
init_db()

st.title("🌱 FAS-D 스마트팜 5ha 자율 조향 시스템 - 수확량 스윙 진단 대시보드")
st.markdown("센서 및 현장 측정 원천 데이터를 바탕으로 **4.8.1절 공식 자동 연산**, **2단계 계층적 피어슨 상관분석**, 그리고 **9S-9P 기준선 비교**를 수행합니다.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 원천 데이터 직접 입력", 
    "📂 일괄 파일 업로드 및 연산", 
    "📊 DB 데이터 조회", 
    "📈 구역별 시계열 및 9S-9P 비교 (Tab 4)",
    "🔍 2단계 계층적 스윙 진단 (Tab 5)"
])

with tab1:
    st.subheader("주간 센서 및 생육 측정 원천 데이터 입력")
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            zone_id = st.selectbox("구역", ["Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E"])
            week_no = st.number_input("주차", min_value=1, max_value=52, value=1)
            dli_daily = st.number_input("일일 DLI", value=25.0)
            t_mean = st.number_input("평균온도", value=20.0)
            t_day = st.number_input("주간온도", value=23.0)
            t_night = st.number_input("야간온도", value=17.0)
        with col2:
            vpd_mean = st.number_input("VPD", value=1.3)
            vpd_max = st.number_input("VPD max", value=1.8)
            rh_mean = st.number_input("상대습도", value=75.0)
            fruit_load = st.number_input("착과수", value=70.0)
            t_leaf = st.number_input("엽온", value=19.0)
            cwsi = st.number_input("CWSI", value=0.2)
        with col3:
            vwc = st.number_input("VWC", value=60.0)
            drainage_rate = st.number_input("배액율", value=30.0)
            y_actual = st.number_input("실제수확량", value=3.5)
            y_target = st.number_input("목표수확량", value=3.0)
            stem_diameter = st.number_input("줄기직경", value=11.0)
            stem_shrinkage = st.number_input("수축량", value=0.4)

        submitted = st.form_submit_button("💾 데이터 저장 및 자동 연산 실행")
        if submitted:
            conn = sqlite3.connect(DB_NAME)
            df_existing = pd.read_sql("SELECT * FROM yield_swing_data", conn)
            conn.close()

            new_data = {"zone_id": zone_id, "week_no": week_no, "dli_daily": dli_daily, "t_mean": t_mean, "t_day": t_day, "t_night": t_night, "vpd_mean": vpd_mean, "vpd_max": vpd_max, "rtr": (dli_daily * 10) / max(t_mean, 1), "rh_mean": rh_mean, "dew_point": t_mean - ((100 - rh_mean) / 5), "fruit_load": fruit_load, "t_leaf": t_leaf, "cwsi": cwsi, "stem_diameter": stem_diameter, "stem_shrinkage": stem_shrinkage, "vwc": vwc, "drainage_rate": drainage_rate, "y_actual": y_actual, "y_target": y_target, "cum_p_n_8w": 0, "tr_ratio": 0, "gdd_8w": 0, "lagged_dli_7w": 0, "lagged_temp_7w": 0, "w_avg": 0, "yield_swing_delta": 0, "ma_3week": 0, "swing_index": 0, "grade_distribution": 0, "brix_variance": 0, "fli": 0}
            df_new_row = pd.DataFrame([new_data])
            df_combined = pd.concat([df_