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
    
    # 🔍 진단용: 현재 불러온 CSV의 모든 컬럼명을 화면에 출력
    st.sidebar.write("📋 CSV 컬럼 목록:", list(df_bench.columns))
    
    st.sidebar.success(f"✅ CSV 로드 성공: {len(df_bench)}행")
else:
    st.error("🚨 지정된 모든 경로에서 cold_start_standards.csv 파일을 찾지 못했습니다.")
    st.write("현재 탐색한 경로들:", possible_paths)
    st.write("현재 디렉토리 파일 목록:", os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else "경로 없음")
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

            new_data = {
                "zone_id": zone_id, "week_no": week_no, "dli_daily": dli_daily,
                "t_mean": t_mean, "t_day": t_day, "t_night": t_night,
                "vpd_mean": vpd_mean, "vpd_max": vpd_max, "rtr": (dli_daily * 10) / max(t_mean, 1),
                "rh_mean": rh_mean, "dew_point": t_mean - ((100 - rh_mean) / 5),
                "fruit_load": fruit_load, "t_leaf": t_leaf, "cwsi": cwsi,
                "stem_diameter": stem_diameter, "stem_shrinkage": stem_shrinkage,
                "vwc": vwc, "drainage_rate": drainage_rate, "y_actual": y_actual, "y_target": y_target,
                "cum_p_n_8w": 0, "tr_ratio": 0, "gdd_8w": 0, "lagged_dli_7w": 0,
                "lagged_temp_7w": 0, "w_avg": 0, "yield_swing_delta": 0,
                "ma_3week": 0, "swing_index": 0, "grade_distribution": 0, "brix_variance": 0, "fli": 0
            }
            df_new_row = pd.DataFrame([new_data])
            df_combined = pd.concat([df_existing[df_existing["week_no"] != week_no], df_new_row], ignore_index=True)
            df_processed = calculate_derived_variables(df_combined)

            conn = sqlite3.connect(DB_NAME)
            df_processed.to_sql("yield_swing_data", conn, index=False, if_exists="replace")
            conn.close()
            st.success(f"[{zone_id}] {week_no}주차 원천 데이터가 성공적으로 연산되어 DB에 저장되었습니다!")

with tab2:
    st.subheader("CSV 또는 Excel 파일 일괄 업로드 및 4.8.1절 자동 연산")
    st.markdown("스마트팜 센서 및 생육 측정 원천 데이터가 담긴 CSV 또는 Excel 파일을 업로드하면, **4.8.1절 공식**에 따른 파생 변수와 스윙 지수가 자동으로 연산되어 통합 DB에 반영됩니다.")
    
    uploaded_file = st.file_uploader("원천 데이터 파일 업로드 (CSV 또는 Excel)", type=["csv", "xlsx"], key="tab2_file_uploader")

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"📁 **업로드된 원천 데이터 미리보기** (총 {len(df_upload)}행):")
            st.dataframe(df_upload, height=300)

            if st.button("🚀 4.8.1절 공식 적용 및 자동 연산 실행", key="tab2_calc_button"):
                df_processed = calculate_derived_variables(df_upload)
                conn = sqlite3.connect(DB_NAME)
                df_processed.to_sql("yield_swing_data", conn, index=False, if_exists="replace")
                conn.close()
                st.success(f"🎉 총 {len(df_processed)}주차 데이터 자동 연산 및 DB 저장 완료!")
                
                st.markdown("---")
                st.subheader("📊 **핵심 연산 지표 요약 뷰**")
                key_derived_cols = ["zone_id", "week_no", "dli_daily", "t_mean", "cum_p_n_8w", "lagged_dli_7w", "w_avg", "tr_ratio", "y_actual", "swing_index"]
                df_summary = df_processed[key_derived_cols].rename(columns=FACTOR_LABELS)
                st.dataframe(df_summary, height=300)
        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류 발생: {e}")
    else:
        st.info("💡 안내: 상단의 **파일 찾아보기(Browse files)** 버튼을 클릭하여 CSV 또는 Excel 파일을 업로드해 주세요.")

with tab3:
    st.subheader("현재 저장된 스마트팜 통합 DB 조회 (연산 결과 반영)")
    if os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        df_db = pd.read_sql("SELECT * FROM yield_swing_data", conn)
        conn.close()

        st.success("✅ **[DB 연동 완료]** 4.8.1절 자동 연산 결과가 반영되어 있습니다.")
        max_week = int(df_db["week_no"].max()) if not df_db.empty else 0
        st.write(f"📊 총 저장된 데이터 행 수: {len(df_db)}행 (수집된 데이터 범위: 1 ~ {max_week}주차)")

        df_db_renamed = df_db.rename(columns=FACTOR_LABELS)
        st.dataframe(df_db_renamed, use_container_width=True, height=450)

        csv = df_db.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 현재 DB 데이터를 CSV로 다운로드", data=csv, file_name="smartfarm_yield_swing_updated.csv", mime="text/csv")
    else:
        st.info("아직 저장된 데이터가 없습니다.")

with tab4:
    st.subheader("📈 구역별 변수 시계열 및 9S-9P 기준선 비교 (Tab 4)")

    # ... (기존 selectbox 및 데이터 조회 코드) ...
    
    # 예시: 사용자가 선택한 변수에 대응하는 기준선 컬럼명 매칭
    # 예컨대 'swing_index'를 선택했다면 벤치마크 CSV의 'st_swing_index'를 찾도록 연결
    selected_var = ... # 사용자가 선택한 변수 컬럼 (예: 'swing_index')
    benchmark_col = f"st_{selected_var}" if not selected_var.startswith("st_") else selected_var
    
    # 공백 제거된 컬럼 목록에서 안전하게 확인
    cleaned_columns = [c.strip() for c in df_bench.columns]
    
    if benchmark_col in cleaned_columns:
        # 정상적으로 기준선 데이터를 찾아 Plotly 그래프에 빨간색 선으로 추가
        benchmark_values = df_bench[benchmark_col]
        # fig.add_trace(go.Scatter(..., line=dict(color='red'))) 처럼 빨간색 선 추가 로직 수행
    else:
        st.info(f"안내: 선택하신 변수에 대응하는 9S-9P 기준선 컬럼({benchmark_col})이 cold_start_standards.csv에 없습니다.")
    
    st.markdown("실제 수집/연산된 센서 데이터(파란색 점/실선)와 9S-9P 기준선(빨간색 점/실선)을 나란히 시각화합니다.")

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

            df_bench = pd.DataFrame()
            if os.path.exists(BENCHMARK_CSV):
                df_bench = pd.read_csv(BENCHMARK_CSV)

            c1, c2 = st.columns(2)
            with c1:
                numeric_cols = df_chart.select_dtypes(include=[np.number]).columns.tolist()
                if "week_no" in numeric_cols:
                    numeric_cols.remove("week_no")
                
                col_options = {col: FACTOR_LABELS.get(col, col) for col in numeric_cols}
                
                default_index = 0
                if "swing_index" in numeric_cols:
                    default_index = numeric_cols.index("swing_index")

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

            if selected_zones:
                filtered_actual = df_chart[df_chart["zone_id"].isin(selected_zones)]
            else:
                filtered_actual = df_chart

            if filtered_actual.empty:
                st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
            else:
                var_label = col_options.get(selected_col_key, selected_col_key)
                st.markdown(f"### 📉 **[{var_label}] 실제값(파란색) vs 9S-9P 기준선(빨간색) 비교**")

    ref_col = f"st_{selected_col_key}".strip()
    
    # 🔍 공백을 완전히 제거한 컬럼 리스트와 비교하도록 방어 코드 강화
    cleaned_bench_columns = [str(c).strip() for c in df_bench.columns]

    ref_col = f"st_{selected_col_key}".strip()

    # 🔍 공백을 완전히 제거한 컬럼 리스트와 비교하도록 방어 코드 강화
    cleaned_bench_columns = [str(c).strip() for c in df_bench.columns]

    if ref_col in cleaned_bench_columns and not df_bench.empty and PLOTLY_AVAILABLE:
        # 혹시 모를 컬럼명 공백을 맞추기 위해 컬럼명 일괄 정제
        df_bench.columns = cleaned_bench_columns
        filtered_bench = df_bench[df_bench["zone_id"].isin(selected_zones)].copy()

        merged_plot_df = pd.merge(
            filtered_actual[["week_no", "zone_id", selected_col_key]],
            filtered_bench[["week_no", "zone_id", ref_col]],
            on=["week_no", "zone_id"],
            how="left"
        )

        long_data = []
        color_map = {}

        for idx, row in merged_plot_df.iterrows():
            w = row["week_no"]
            z = row["zone_id"]
            val_actual = row[selected_col_key]
            val_ref = row[ref_col]

            act_label = f"{z} (실제)"
            ref_label = f"{z} (9S-9P 기준)"

            if pd.notna(val_actual):
                long_data.append({
                    "주차 (Week)": w,
                    "구분": act_label,
                    "값": val_actual
                })
                color_map[act_label] = "#2980b9"

            if pd.notna(val_ref):
                long_data.append({
                    "주차 (Week)": w,
                    "구분": ref_label,
                    "값": val_ref
                })
                color_map[ref_label] = "#e74c3c"

        df_plot = pd.DataFrame(long_data)

        # 💡 이 부분의 if문과 아래 그래프 그리는 코드를 바깥쪽(왼쪽)으로 4칸 당겨주세요.
        if not df_plot.empty:
            fig = px.line(
                df_plot,
                x="주차 (Week)",
                y="값",
                color="구분",
                color_discrete_map=color_map,
                markers=True
            )    
            fig.update_layout(
                xaxis_title="주차 (Week)",
                yaxis_title=var_label,
                legend_title="범례",
                height=450,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success("💡 안내: **파란색 선과 점**은 실제 수집/연산 데이터이며, **빨간색 선과 점**은 9S-9P 관리기준표 표준 기준선입니다.")

                    st.markdown("---")
                    st.markdown("### 📋 상세 비교 데이터 테이블")
                    
                    pivot_combined = merged_plot_df.pivot(index="week_no", columns="zone_id")
                    plot_table_data = pd.DataFrame(index=pivot_combined.index)
                    for z in selected_zones:
                        if (selected_col_key, z) in pivot_combined.columns:
                            plot_table_data[f"{z} (실제)"] = pivot_combined[(selected_col_key, z)]
                        if (ref_col, z) in pivot_combined.columns:
                            plot_table_data[f"{z} (9S-9P 기준)"] = pivot_combined[(ref_col, z)]
                    
                    st.dataframe(plot_table_data.reset_index().rename(columns={"week_no": "주차 (Week)"}), use_container_width=True, height=350)

                else:
                    pivot_actual = filtered_actual.pivot(index="week_no", columns="zone_id", values=selected_col_key)
                    pivot_actual.columns = [f"{col} (실제)" for col in pivot_actual.columns]
                    
                    st.info(f"💡 안내: 선택하신 **{var_label}** 변수에 대응하는 9S-9P `st_` 기준선 컬럼(`{ref_col}`)이 `cold_start_standards.csv`에 정의되어 있지 않거나 Plotly 패키지가 없습니다.")
                    st.line_chart(pivot_actual)

                    st.markdown("---")
                    st.markdown("### 📋 상세 데이터 테이블")
                    st.dataframe(pivot_actual.reset_index().rename(columns={"week_no": "주차 (Week)"}), use_container_width=True, height=350)

with tab5:
    st.subheader("🔍 FAS-D 2단계 계층적 스윙 진단 시스템 (Tab 5)")
    st.markdown("실제 데이터가 수집된 주차 범위 내에서 스윙 발생 주차를 탐지하고 1·2단계 피어슨 상관분석을 수행합니다.")

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
            swing_alert_df = df_analysis[df_analysis["swing_index"].abs() >= 5.0].sort_values(by="week_no")
            
            max_collected_week = int(df_analysis["week_no"].max()) if not df_analysis.empty else 0
            st.markdown(f"### 🚨 [스윙 모니터링] 수집된 데이터 범위(1 ~ {max_collected_week}주차) 중 주의/위험 스윙 발생 주차 리스트")
            
            if len(swing_alert_df) > 0:
                st.dataframe(swing_alert_df[["zone_id", "week_no", "y_actual", "ma_3week", "swing_index", "cum_p_n_8w", "tr_ratio"]].rename(columns=FACTOR_LABELS), height=200)
            else:
                st.info(f"현재 수집된 1 ~ {max_collected_week}주차 데이터 중 임계값(±5%)을 초과하는 스윙 주차가 없습니다.")

            st.markdown("---")
            st.markdown("### 🎯 재배사 진단 타겟 주차 선택 및 2단계 상관분석")
            
            zones = df_analysis["zone_id"].unique().tolist()
            if len(zones) > 0:
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    selected_zone = st.selectbox("진단 구역 선택", zones, key="tab5_zone_select")
                with col_sel2:
                    available_weeks = sorted(df_analysis[df_analysis["zone_id"] == selected_zone]["week_no"].unique().tolist())
                    if available_weeks:
                        default_idx = len(available_weeks) - 1
                        selected_week = st.selectbox(f"진단 주차 선택 (수집된 주차: 1 ~ {max_collected_week}주차)", available_weeks, index=default_idx, key="tab5_week_select")
                    else:
                        selected_week = 1
                        st.selectbox("진단 주차 선택", [1], key="tab5_week_select_empty")

                target_rows = df_analysis[(df_analysis["zone_id"] == selected_zone) & (df_analysis["week_no"] == selected_week)]
                
                if not target_rows.empty:
                    target_row = target_rows.iloc[0]
                    st.info(f"📌 [{selected_zone}] {selected_week}주차 현황: 실제 수확량 = {target_row['y_actual']} kg/m², 스윙 지수 = {target_row['swing_index']:.2f}%, T/R율 = {target_row['tr_ratio']:.2f}")

                    if st.button("🔬 2단계 계층적 상관분석(Macro-Micro) 실행", key="tab5_analysis_button"):
                        zone_df = df_analysis[df_analysis["zone_id"] == selected_zone].sort_values("week_no")

                        st.markdown("---")
                        st.markdown("#### [1단계] Macro 상관분석 (핵심 생리 인자 vs 스윙 진단)")
                        macro_vars = ["x1_cum_dli_8w", "gdd_8w", "fli", "cum_p_n_8w", "tr_ratio", "w_avg", "swing_index"]
                        corr_matrix = zone_df[macro_vars].corr()
                        macro_corr = corr_matrix[["swing_index"]].drop("swing_index").fillna(0)
                        macro_corr.columns = ["corr"]
                        
                        if not macro_corr.empty:
                            top_factor = macro_corr["corr"].abs().idxmax()
                            top_r = macro_corr.loc[top_factor, "corr"]
                        else:
                            top_factor = "cum_p_n_8w"
                            top_r = 0.0
                        
                        macro_corr["핵심 인자 설명"] = [FACTOR_LABELS.get(idx, idx) for idx in macro_corr.index]
                        macro_corr = macro_corr.rename(columns={"corr": "Pearson Correlation (r) with Swing Index"})
                        macro_corr = macro_corr[["핵심 인자 설명", "Pearson Correlation (r) with Swing Index"]]

                        st.write("📈 **핵심 생리 인자별 피어슨 상관계수 결과 (T/R율 포함):**")
                        st.dataframe(macro_corr)
                        
                        top_factor_name = FACTOR_LABELS.get(top_factor, top_factor)
                        st.success(f"⚡ **1단계 진단 결과**: 이번 {selected_week}주차 스윙에 가장 큰 영향을 미친 핵심 인자는 '{top_factor_name}'이며, 상관계수 r은 {top_r:.4f}입니다.")

                        st.markdown("---")
                        st.markdown(f"#### [2단계] Micro 원인 추적 분석 ({selected_week}주차 타겟 인자 vs 원천 센서 데이터)")
                        micro_vars = ["dli_daily", "t_mean", "vpd_mean", "fruit_load", "vwc", "drainage_rate"]
                        micro_corr = zone_df[micro_vars + [top_factor]].corr()[[top_factor]].drop(top_factor).fillna(0)
                        micro_corr.columns = ["corr"]

                        micro_corr["원천 센서 설명"] = [FACTOR_LABELS.get(idx, idx) for idx in micro_corr.index]
                        micro_corr = micro_corr.rename(columns={"corr": f"Correlation (r) with {top_factor}"})
                        micro_corr = micro_corr[["원천 센서 설명", f"Correlation (r) with {top_factor}"]]

                        st.write(f"🔍 **핵심 인자({top_factor_name})와 원천 센서 데이터 간의 마이크로 상관관계:**")
                        st.dataframe(micro_corr)

                        lag_week = max(1, selected_week - 7)
                        lag_row = zone_df[zone_df["week_no"] == lag_week]
                        
                        if len(lag_row) > 0:
                            l_row = lag_row.iloc[0]
                            st.warning(f"💡 **엔지니어링 인사이트 (리드타임 분석)**: 작물의 생리적 발달 시차(7주 전)인 {lag_week}주차 원천 데이터를 추적한 결과, 일일 광량(dli_daily)은 {l_row['dli_daily']}, 평균온도(t_mean)는 {l_row['t_mean']}로 나타났습니다. 이 시기의 환경 제어 편차가 현재 {selected_week}주차 스윙의 근본 원인(Root Cause)으로 작용했습니다.")
                        else:
                            st.info("7주 전 과거 데이터 구간이 충분하지 않습니다.")
