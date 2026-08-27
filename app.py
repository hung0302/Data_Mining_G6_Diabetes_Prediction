import streamlit as st
import pandas as pd
import xgboost as xgb
import gspread
from google.oauth2.service_account import Credentials
import io

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Diabetes Prediction", page_icon="📊", layout="wide")
st.title("Diabetes Risk Prediction")
st.caption("XGBoost Prediction Demo - Hỗ trợ nhập lẻ & Tải file hàng loạt")

# Danh sách đúng 21 cột feature mô hình cần
col_names = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", 
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", 
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    try:
        model = xgb.Booster()
        model.load_model("diabetes_model.json")
        return model
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file 'diabetes_model.json'. Vui lòng kiểm tra lại.")
        st.stop()

model = load_model()

# =========================================================
# GOOGLE SHEET CONNECTION
# =========================================================
@st.cache_resource
def connect_google_sheet():
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=scopes
        )
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(st.secrets["google_sheet"]["sheet_id"])
        worksheet = spreadsheet.worksheet(st.secrets["google_sheet"]["worksheet"])
        return worksheet
    except Exception as e:
        st.error("Lỗi kết nối Google Sheet. Vui lòng kiểm tra lại cấu hình và quyền truy cập.")
        st.exception(e)
        st.stop()

worksheet = connect_google_sheet()

# =========================================================
# GIAO DIỆN CHIA TAB
# =========================================================
tab1, tab2 = st.tabs(["📝 Nhập lẻ từng bệnh nhân", "📂 Tải file hàng loạt"])

# ---------------------------------------------------------
# TAB 1: NHẬP LẺ
# ---------------------------------------------------------
with tab1:
    st.subheader("Hồ sơ bệnh nhân mới")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highbp = st.selectbox("HighBP (Huyết áp cao)", options=["No", "Yes"])
            highchol = st.selectbox("HighChol (Cholesterol cao)", options=["No", "Yes"])
            cholcheck = st.selectbox("CholCheck (Đã kiểm tra Chol)", options=["No", "Yes"])
            bmi = st.number_input("BMI (Chỉ số khối cơ thể)", min_value=10.0, max_value=100.0, value=25.0)
            smoker = st.selectbox("Smoker (Hút thuốc)", options=["No", "Yes"])
            stroke = st.selectbox("Stroke (Đột quỵ)", options=["No", "Yes"])
            heart_disease = st.selectbox("HeartDiseaseorAttack (Bệnh tim)", options=["No", "Yes"])
            
        with col2:
            phys_activity = st.selectbox("PhysActivity (Tập thể dục)", options=["No", "Yes"])
            fruits = st.selectbox("Fruits (Ăn hoa quả)", options=["No", "Yes"])
            veggies = st.selectbox("Veggies (Ăn rau)", options=["No", "Yes"])
            hvy_alcohol = st.selectbox("HvyAlcoholConsump (Nghiện rượu)", options=["No", "Yes"])
            any_healthcare = st.selectbox("AnyHealthcare (Có BHYT)", options=["No", "Yes"])
            nodoc_cost = st.selectbox("NoDocbcCost (Bỏ khám do phí)", options=["No", "Yes"])
            gen_hlth = st.number_input("GenHlth (Sức khỏe chung 1-5)", min_value=1, max_value=5, value=3)
            
        with col3:
            ment_hlth = st.number_input("MentHlth (Ngày SK tâm thần kém 0-30)", min_value=0, max_value=30, value=0)
            phys_hlth = st.number_input("PhysHlth (Ngày SK thể chất kém 0-30)", min_value=0, max_value=30, value=0)
            diff_walk = st.selectbox("DiffWalk (Khó đi lại)", options=["No", "Yes"])
            sex = st.selectbox("Sex (Giới tính)", options=["Nữ (0)", "Nam (1)"])
            age = st.number_input("Age (Nhóm tuổi 1-13)", min_value=1, max_value=13, value=5)
            education = st.number_input("Education (Học vấn 1-6)", min_value=1, max_value=6, value=4)
            income = st.number_input("Income (Thu nhập 1-8)", min_value=1, max_value=8, value=5)
            
        submit_button = st.form_submit_button("Dự đoán và Lưu Google Sheet", type="primary")

    if submit_button:
        def yn_to_int(val):
            return 1 if val == "Yes" else 0
            
        features = [
            yn_to_int(highbp), yn_to_int(highchol), yn_to_int(cholcheck), float(bmi),
            yn_to_int(smoker), yn_to_int(stroke), yn_to_int(heart_disease),
            yn_to_int(phys_activity), yn_to_int(fruits), yn_to_int(veggies),
            yn_to_int(hvy_alcohol), yn_to_int(any_healthcare), yn_to_int(nodoc_cost),
            int(gen_hlth), int(ment_hlth), int(phys_hlth), yn_to_int(diff_walk),
            1 if sex == "Nam (1)" else 0,
            int(age), int(education), int(income)
        ]
        
        try:
            df_new = pd.DataFrame([features], columns=col_names)
            dtrain = xgb.DMatrix(df_new)
            probability = float(model.predict(dtrain)[0])
            
            st.success("Đã phân tích xong dữ liệu của bệnh nhân!")
            st.markdown("---")
            colA, colB = st.columns(2)
            colA.metric(label="Xác suất mắc bệnh tiểu đường", value=f"{probability*100:.2f}%")
            if probability > 0.5:
                colB.error("Cảnh báo: Nguy cơ CAO")
            else:
                colB.success("Tuyệt vời: Nguy cơ THẤP")
                
            headers = worksheet.row_values(1)
            row_to_append = [""] * len(headers)
            for col_name, val in zip(df_new.columns, features):
                if col_name in headers:
                    idx = headers.index(col_name)
                    row_to_append[idx] = val
                    
            if "Probability" in headers:
                idx = headers.index("Probability")
                row_to_append[idx] = round(probability, 4)
            else:
                worksheet.update_cell(1, len(headers)+1, "Probability")
                row_to_append.append(round(probability, 4))
                
            worksheet.append_row(row_to_append, value_input_option="RAW")
            st.info("✅ Toàn bộ thông tin bệnh án và kết quả đã được tự động lưu mới vào Google Sheet.")
            
        except Exception as e:
            st.error("Lỗi khi xử lý dữ liệu hoặc ghi Google Sheet.")
            st.exception(e)

# ---------------------------------------------------------
# TAB 2: TẢI FILE HÀNG LOẠT
# ---------------------------------------------------------
with tab2:
    st.subheader("Dự đoán hàng loạt qua CSV/Excel")
    
    # Nút tải file mẫu
    template_df = pd.DataFrame(columns=col_names)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải File CSV Mẫu (Template)",
        data=csv_template,
        file_name="diabetes_template.csv",
        mime="text/csv",
        help="Tải file này về, điền dữ liệu của các bệnh nhân rồi tải lên để dự đoán."
    )
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn file CSV/Excel của bạn vào đây", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
                
            # Kiểm tra xem có đủ cột không
            missing_cols = [col for col in col_names if col not in df_upload.columns]
            if missing_cols:
                st.error(f"File của bạn đang thiếu các cột bắt buộc sau: {missing_cols}")
                st.stop()
                
            st.success(f"Đã đọc thành công {len(df_upload)} bệnh nhân. Đang xử lý...")
            
            # Predict
            X_batch = df_upload[col_names].copy()
            # Đảm bảo toàn bộ là số
            for col in col_names:
                X_batch[col] = pd.to_numeric(X_batch[col], errors="coerce")
                
            if X_batch.isnull().any().any():
                st.warning("Có một số ô dữ liệu bị thiếu hoặc không phải dạng số, đã được bỏ qua hoặc thay bằng NaN.")
                
            dtrain_batch = xgb.DMatrix(X_batch)
            probabilities = model.predict(dtrain_batch)
            
            # Lưu kết quả
            df_upload["Probability"] = probabilities
            df_upload["Risk_Pct"] = (probabilities * 100).round(2)
            
            st.dataframe(df_upload[["Probability", "Risk_Pct"] + col_names], use_container_width=True)
            
            if st.button("Lưu toàn bộ danh sách này lên Google Sheet", type="primary"):
                with st.spinner("Đang đẩy dữ liệu lên Google Sheet..."):
                    headers = worksheet.row_values(1)
                    
                    # Nếu chưa có cột Probability trên Sheet thì thêm vào
                    if "Probability" not in headers:
                        worksheet.update_cell(1, len(headers)+1, "Probability")
                        headers.append("Probability")
                        
                    # Chuẩn bị dữ liệu theo mảng 2 chiều (nhiều hàng) để đẩy lên 1 lượt
                    rows_to_append = []
                    for idx, row in df_upload.iterrows():
                        single_row = [""] * len(headers)
                        for col_name in col_names:
                            if col_name in headers:
                                h_idx = headers.index(col_name)
                                single_row[h_idx] = row[col_name]
                        # Thêm probability
                        if "Probability" in headers:
                            h_idx = headers.index("Probability")
                            single_row[h_idx] = round(float(row["Probability"]), 4)
                            
                        rows_to_append.append(single_row)
                    
                    # Dùng append_rows để ném 1 cục lên Sheet cho nhanh, thay vì loop từng dòng
                    worksheet.append_rows(rows_to_append, value_input_option="RAW")
                    st.success(f"✅ Đã lưu thành công {len(rows_to_append)} bệnh nhân vào Google Sheet!")
                    
        except Exception as e:
            st.error("Lỗi khi đọc file hoặc xử lý mô hình. Hãy kiểm tra lại file của bạn (nếu dùng Excel, hãy chắc chắn bạn đã cài thư viện openpyxl).")
            st.exception(e)
