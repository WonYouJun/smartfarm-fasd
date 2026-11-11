import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

DB_NAME = "smartfarm_yield_swing.db"

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

    df["x1_cum_dli_8w"] = df.groupby("zone_id")["dli_daily"].transform(
        lambda x: x.rolling(window=8, min_periods=1).sum()
    )
    df["gdd_8w"] = df.groupby("zone_id")["t_mean"].transform(
        lambda x: x.rolling(window=8, min_periods=1).sum()
    )

    df["temp_pn"] = (df["dli_daily"] * 0.85) - (df["vpd_mean"] * 1.2)
    df["cum_p_n_8w"] = (
        df.groupby("zone_id")["temp_pn"]
        .transform(lambda x: x.rolling(window=8, min_periods=1).sum() * 10 + 500)
        .clip(lower=300, upper=1200)
    )
    df = df.drop(columns=["temp_pn"])

    df["lagged_dli_7w"] = df.groupby("zone_id")["dli_daily"].shift(7)
    df["lagged_temp_7w"] = df.groupby("zone_id")["t_mean"].shift(7)
    df["lagged_dli_7w"] = df["lagged_dli_7w"].fillna(df["dli_daily"].iloc[0])
    df["lagged_temp_7w"] = df["lagged_temp_7w"].fillna(df["t_mean"].iloc[0])

    df["fli"] = (df["fruit_load"] / (df["dli_daily"] + 1)) * 10
    df["tr_ratio"] = 3.5 - 0.02 * df["fruit_load"] + 0.001 * df["cum_p_n_8w"]
    df["w_avg"] = (df["cum_p_n_8w"] / df["fruit_load"].replace(0, 1)) * 12
    df["w_avg"] = df["w_avg"].clip(lower=110, upper=180)

    df["yield_swing_delta"] = df["y_actual"] - df["y_target"]
    df["ma_3week"] = df.groupby("zone_id")["y_actual"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["swing_index"] = (
        (df["y_actual"] - df["ma_3week"]) / df["ma_3week"].replace(0, 1)
    ) * 100

    df["grade_distribution"] = (92 - 0.3 * df["swing_index"].abs()).clip(lower=70, upper=98)
    df["brix_variance"] = 0.2 + 0.005 * df["swing_index"].abs()

    return df

FACTOR_LABELS = {
    "zone_id": "구역 ID (zone_id)",
    "week_no": "주차 (week_no)",
    "dli_daily": "일일 누적 광량 (dli_daily)",
    "t_mean": "24시간 평균온도 (t_mean)",
    "t_day": "주간 온도 (t_day)",
    "t_night": "야간 온도 (t_night)",
    "vpd_mean": "평균 증기압차 (vpd_mean)",
    "vpd_max": "최고 증기압차 (vpd_max)",
    "rtr": "광-온 비 (rtr)",
    "rh_mean": "평균 상대습도 (rh_mean)",
    "dew_point": "이슬점 온도 (dew_point)",
    "fruit_load": "과일 부하량 (fruit_load)",
    "t_leaf": "작물 엽온 (t_leaf)",
    "cwsi": "수분 스트레스 지수 (cwsi)",
    "stem_diameter": "줄기 직경 (stem_diameter)",
    "stem_shrinkage": "줄기 수축량 (stem_shrinkage)",
    "vwc": "슬래브 수분함량 (vwc)",
    "drainage_rate": "배액율 (drainage_rate)",
    "x1_cum_dli_8w": "누적 광량 8주 (x1_cum_dli_8w)",
    "gdd_8w": "적산 온도 8주 (gdd_8w)",
    "fli": "과일 부하 및 체력 지수 (fli)",
    "cum_p_n_8w": "순광합성량 8주 적산 (cum_p_n_8w)",
    "lagged_dli_7w": "7주 시차 광량 (lagged_dli_7w)",
    "lagged_temp_7w": "7주 시차 온도 (lagged_temp_7w)",
    "tr_ratio": "T/R율 균형 지수 (tr_ratio)",
    "w_avg": "평균 단과중 (w_avg)",
    "y_actual": "실제 주간 수확량 (y_actual)",
    "y_target": "목표 주간 수확량 (y_target)",
    "yield_swing_delta": "수확량 편차 (yield_swing_delta)",
    "ma_3week": "3주 이동평균 수확량 (ma_3week)",
    "swing_index": "수확량 스윙 지수 (swing_index)",
    "grade_distribution": "상품과 분포율 (grade_distribution)",
    "brix_variance": "당도 편차 (brix_variance)"
}

st.set_page_config(page_title="FAS-D 스마트팜 자율 조향 & 진단 대시보드", layout="wide")
init_db()

st.title("🌱 FAS-D 스마트팜 5ha 자율 조향 시스템 - 수확량 스윙 진단 대시보드")
st.markdown("센서 및 현장 측정 원천 데이터를 바탕으로 **4.8.1절 공식 자동 연산** 및 **2단계 계층적 피어슨 상관분석(Macro-Micro)**을 수행합니다.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 원천 데이터 직접 입력", 
    "📂 일괄 파일 업로드 및 연산", 
    "📊 DB 데이터 조회", 
    "🔍 2단계 계층적 스윙 진단 (Tab 4)"
])

with tab1:
    st.subheader("주간 센서 및 생육 측정 원천 데이터 입력")
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            zone_id = st.selectbox("구역", ["Zone_A", "Zone_B", "Zone_C"])
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
            st.success(f"[{zone_id}] {week_no}주차 원천 데이터가 성공적으로 저장되었습니다!")

with tab2:
    st.subheader("CSV 또는 Excel 파일 일괄 업로드 및 4.8.1절 자동 연산")
    uploaded_file = st.file_uploader("원천 데이터 파일 업로드", type=["csv", "xlsx"], key="tab2_file_uploader")

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"📁 **업로드된 원천 데이터 미리보기** (총 {len(df_upload)}주차):")
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

with tab3:
    st.subheader("현재 저장된 스마트팜 통합 DB 조회 (연산 결과 반영)")
    if os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        df_db = pd.read_sql("SELECT * FROM yield_swing_data", conn)
        conn.close()

        st.success("✅ **[DB 연동 완료]** 4.8.1절 자동 연산 결과가 반영되어 있습니다.")
        st.markdown("💡 **표 시각화 안내**: **파란색 굵은 글씨**는 자동으로 연산된 파생 변수 및 스윙 지수입니다.")
        
        st.write(f"📊 총 저장된 데이터 행 수: {len(df_db)}행")

        derived_cols = [
            "x1_cum_dli_8w", "gdd_8w", "cum_p_n_8w", "lagged_dli_7w", "lagged_temp_7w",
            "fli", "tr_ratio", "w_avg", "yield_swing_delta", "ma_3week", 
            "swing_index", "grade_distribution", "brix_variance", "rtr", "dew_point"
        ]

        def highlight_derived(col):
            if any(k in col.name for k in derived_cols):
                return ['color: #0066cc; font-weight: bold;' for _ in col]
            else:
                return ['' for _ in col]

        df_db_renamed = df_db.rename(columns=FACTOR_LABELS)
        styled_df = df_db_renamed.style.apply(highlight_derived, axis=0)
        st.dataframe(styled_df, height=450)

        csv = df_db.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 현재 DB 데이터를 CSV로 다운로드",
            data=csv,
            file_name="smartfarm_yield_swing_updated.csv",
            mime="text/csv",
        )
    else:
        st.info("아직 저장된 데이터가 없습니다.")

with tab4:
    st.subheader("🔍 FAS-D 2단계 계층적 스윙 진단 시스템")
    st.markdown("스윙 발생 주차 탐지 및 1단계(Macro)·2단계(Micro) 피어슨 상관분석을 수행합니다.")

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
            swing_alert_df = df_analysis[df_analysis["swing_index"].abs() >= 5.0].sort_values(by="swing_index")
            
            st.markdown("### 🚨 [스윙 모니터링] 주의/위험 스윙 발생 주차 리스트")
            if len(swing_alert_df) > 0:
                st.dataframe(swing_alert_df[["zone_id", "week_no", "y_actual", "ma_3week", "swing_index", "cum_p_n_8w", "tr_ratio"]].rename(columns=FACTOR_LABELS), height=200)
            else:
                st.info("현재 임계값(±5%)을 초과하는 스윙 주차가 없습니다.")
                st.dataframe(df_analysis[["zone_id", "week_no", "y_actual", "swing_index"]].rename(columns=FACTOR_LABELS), height=150)

            st.markdown("---")
            st.markdown("### 🎯 재배사 진단 타겟 주차 선택 및 2단계 상관분석")
            
            zones = df_analysis["zone_id"].unique()
            if len(zones) > 0:
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    selected_zone = st.selectbox("진단 구역 선택", zones, key="tab4_zone_select")
                with col_sel2:
                    available_weeks = df_analysis[df_analysis["zone_id"] == selected_zone]["week_no"].tolist()
                    selected_week = st.selectbox("진단 주차 선택", available_weeks, index=min(9, max(0, len(available_weeks)-1)), key="tab4_week_select")

                target_rows = df_analysis[(df_analysis["zone_id"] == selected_zone) & (df_analysis["week_no"] == selected_week)]
                
                if not target_rows.empty:
                    target_row = target_rows.iloc[0]
                    st.info(f"📌 [{selected_zone}] {selected_week}주차 현황: 실제 수확량 = {target_row['y_actual']} kg/m², 스윙 지수 = {target_row['swing_index']:.2f}%, T/R율 = {target_row['tr_ratio']:.2f}")

                    if st.button("🔬 2단계 계층적 상관분석(Macro-Micro) 실행", key="tab4_analysis_button"):
                        zone_df = df_analysis[df_analysis["zone_id"] == selected_zone].sort_values("week_no")

                        st.markdown("---")
                        st.markdown("#### [1단계] Macro 상관분석 (핵심 생리 인자 vs 스윙 지수)")
                        macro_vars = ["x1_cum_dli_8w", "gdd_8w", "fli", "cum_p_n_8w", "tr_ratio", "w_avg", "swing_index"]
                        macro_corr = zone_df[macro_vars].corr()[["swing_index"]].drop("swing_index").fillna(0)
                        macro_corr.columns = ["Pearson Correlation (r) with Swing Index"]
                        
                        top_factor = macro_corr["Pearson Correlation (r) with Swing Index"].abs().idxmax() if not macro_corr.empty else "cum_p_n_8w"
                        top_r = macro_corr.loc[top_factor, "Pearson Correlation (r) with Swing Index"] if top_factor in macro_corr.index else 0.0
                        
                        macro_corr["핵심 인자 설명"] = macro_corr.index.map(FACTOR_LABELS)
                        macro_corr = macro_corr[["핵심 인자 설명", "Pearson Correlation (r) with Swing Index"]]

                        st.write("📈 **핵심 생리 인자별 피어슨 상관계수 결과 (T/R율 포함):**")
                        st.dataframe(macro_corr)
                        
                        top_factor_name = FACTOR_LABELS.get(top_factor, top_factor)
                        st.success(f"⚡ **1단계 진단 결과**: 이번 {selected_week}주차 스윙에 가장 큰 영향을 미친 핵심 인자는 '{top_factor_name}'이며, 상관계수 r은 {top_r:.4f}입니다.")

                        st.markdown("---")
                        st.markdown(f"#### [2단계] Micro 원인 추적 분석 ({selected_week}주차 타겟 인자 vs 원천 센서 데이터)")
                        micro_vars = ["dli_daily", "t_mean", "vpd_mean", "fruit_load", "vwc", "drainage_rate"]
                        micro_corr = zone_df[micro_vars + [top_factor]].corr()[[top_factor]].drop(top_factor).fillna(0)
                        micro_corr.columns = [f"Correlation (r) with {top_factor}"]

                        micro_corr["원천 센서 설명"] = [FACTOR_LABELS.get(idx, idx) for idx in micro_corr.index]
                        micro_corr = micro_corr[["원천 센서 설명", f"Correlation (r) with {top_factor}"]]

                        st.write(f"🔍 **핵심 인자({top_factor_name})와 원천 센서 데이터 간의 마이크로 상관관계:**")
                        st.dataframe(micro_corr)

                        lag_week = max(1, selected_week - 7)
                        lag_row = zone_df[zone_df["week_no"] == lag_week]
                        
                        if len(lag_row) > 0:
                            l_row = lag_row.iloc[0]
                            st.warning(f"💡 **엔지니어링 인사이트 (리드타임 분석)**: 작물의 생리적 발달 시차(7주 전)인 {lag_week}주차의 원천 데이터를 추적한 결과, 일일 광량(dli_daily)은 {l_row['dli_daily']}, 평균온도(t_mean)는 {l_row['t_mean']}로 나타났습니다. 이 시기의 환경 제어 부족이 현재 {selected_week}주차 스윙의 근본 원인(Root Cause)으로 작용했습니다.")
                        else:
                            st.info("7주 전 과거 데이터 구간이 충분하지 않습니다.")