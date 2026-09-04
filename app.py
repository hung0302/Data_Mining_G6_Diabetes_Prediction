import streamlit as st
import pandas as pd
import xgboost as xgb
import gspread
from google.oauth2.service_account import Credentials
import io
import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Diabetes Prediction", page_icon="📊", layout="wide")
st.title("Diabetes Risk Prediction")
st.caption("XGBoost Prediction Demo - Single Entry & Batch Upload Supported")

# 21 feature columns required by the model
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
        st.error("Error: Model file 'diabetes_model.json' not found. Please check.")
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
        st.error("Error connecting to Google Sheet. Please check your credentials and permissions.")
        st.exception(e)
        st.stop()

worksheet = connect_google_sheet()

# =========================================================
# TABS INTERFACE
# =========================================================
tab1, tab2 = st.tabs(["📝 Single Patient Entry", "📂 Batch File Upload"])

# ---------------------------------------------------------
# TAB 1: SINGLE ENTRY
# ---------------------------------------------------------
with tab1:
    st.subheader("New Patient Profile")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highbp = st.selectbox("HighBP", options=["No", "Yes"])
            highchol = st.selectbox("HighChol", options=["No", "Yes"])
            cholcheck = st.selectbox("CholCheck (Checked in last 5 years)", options=["No", "Yes"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
            smoker = st.selectbox("Smoker", options=["No", "Yes"])
            stroke = st.selectbox("Stroke", options=["No", "Yes"])
            heart_disease = st.selectbox("HeartDiseaseorAttack", options=["No", "Yes"])
            
        with col2:
            phys_activity = st.selectbox("PhysActivity", options=["No", "Yes"])
            fruits = st.selectbox("Fruits", options=["No", "Yes"])
            veggies = st.selectbox("Veggies", options=["No", "Yes"])
            hvy_alcohol = st.selectbox("HvyAlcoholConsump", options=["No", "Yes"])
            any_healthcare = st.selectbox("AnyHealthcare", options=["No", "Yes"])
            nodoc_cost = st.selectbox("NoDocbcCost", options=["No", "Yes"])
            gen_hlth = st.number_input("GenHlth (1-5 scale)", min_value=1, max_value=5, value=3)
            
        with col3:
            ment_hlth = st.number_input("MentHlth (Poor mental health days 0-30)", min_value=0, max_value=30, value=0)
            phys_hlth = st.number_input("PhysHlth (Poor physical health days 0-30)", min_value=0, max_value=30, value=0)
            diff_walk = st.selectbox("DiffWalk (Difficulty walking)", options=["No", "Yes"])
            sex = st.selectbox("Sex", options=["Female (0)", "Male (1)"])
            current_year = datetime.date.today().year
            birth_year = st.number_input("Birth Year", min_value=1900, max_value=current_year, value=1990)
            age_years = current_year - birth_year
            if age_years < 25: age = 1
            elif age_years <= 29: age = 2
            elif age_years <= 34: age = 3
            elif age_years <= 39: age = 4
            elif age_years <= 44: age = 5
            elif age_years <= 49: age = 6
            elif age_years <= 54: age = 7
            elif age_years <= 59: age = 8
            elif age_years <= 64: age = 9
            elif age_years <= 69: age = 10
            elif age_years <= 74: age = 11
            elif age_years <= 79: age = 12
            else: age = 13

            education_options = {
                "Never attended school or only kindergarten": 1,
                "Grades 1 through 8 (Elementary)": 2,
                "Grades 9 through 11 (Some high school)": 3,
                "Grade 12 or GED (High school graduate)": 4,
                "College 1 year to 3 years (Some college)": 5,
                "College 4 years or more (College graduate)": 6
            }
            education_text = st.selectbox("Education Level", options=list(education_options.keys()), index=3)
            education = education_options[education_text]

            income_value = st.number_input("Annual Income (USD)", min_value=0, value=50000, step=1000)
            if income_value < 10000: income = 1
            elif income_value < 15000: income = 2
            elif income_value < 20000: income = 3
            elif income_value < 25000: income = 4
            elif income_value < 35000: income = 5
            elif income_value < 50000: income = 6
            elif income_value < 75000: income = 7
            else: income = 8
            
        submit_button = st.form_submit_button("Predict & Save to Google Sheet", type="primary")

    if submit_button:
        def yn_to_int(val):
            return 1 if val == "Yes" else 0
            
        features = [
            yn_to_int(highbp), yn_to_int(highchol), yn_to_int(cholcheck), float(bmi),
            yn_to_int(smoker), yn_to_int(stroke), yn_to_int(heart_disease),
            yn_to_int(phys_activity), yn_to_int(fruits), yn_to_int(veggies),
            yn_to_int(hvy_alcohol), yn_to_int(any_healthcare), yn_to_int(nodoc_cost),
            int(gen_hlth), int(ment_hlth), int(phys_hlth), yn_to_int(diff_walk),
            1 if sex == "Male (1)" else 0,
            int(age), int(education), int(income)
        ]
        
        try:
            df_new = pd.DataFrame([features], columns=col_names)
            dtrain = xgb.DMatrix(df_new)
            probability = float(model.predict(dtrain)[0])
            
            st.success("Patient data analysis completed!")
            st.markdown("---")
            colA, colB = st.columns(2)
            colA.metric(label="Diabetes Probability", value=f"{probability*100:.2f}%")
            if probability > 0.5:
                colB.error("Warning: HIGH Risk")
            else:
                colB.success("Great: LOW Risk")
                
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
            st.info("✅ All patient information and results have been successfully saved to Google Sheet.")
            
        except Exception as e:
            st.error("Error processing data or saving to Google Sheet.")
            st.exception(e)

# ---------------------------------------------------------
# TAB 2: BATCH FILE UPLOAD
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch Prediction via CSV/Excel")
    
    # Template download button
    template_df = pd.DataFrame(columns=col_names)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="diabetes_template.csv",
        mime="text/csv",
        help="Download this file, fill in the patient data, and upload it for prediction."
    )
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader("Drag and drop or select your CSV/Excel file here", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
                
            # Check for missing columns
            missing_cols = [col for col in col_names if col not in df_upload.columns]
            if missing_cols:
                st.error(f"Your file is missing the following required columns: {missing_cols}")
                st.stop()
                
            st.success(f"Successfully read {len(df_upload)} patients. Processing...")
            
            # Predict
            X_batch = df_upload[col_names].copy()
            # Ensure everything is numeric
            for col in col_names:
                X_batch[col] = pd.to_numeric(X_batch[col], errors="coerce")
                
            if X_batch.isnull().any().any():
                st.warning("Some data cells are missing or non-numeric, they have been skipped or replaced with NaN.")
                
            dtrain_batch = xgb.DMatrix(X_batch)
            probabilities = model.predict(dtrain_batch)
            
            # Save results
            df_upload["Probability"] = probabilities
            df_upload["Risk_Pct"] = (probabilities * 100).round(2)
            
            st.dataframe(df_upload[["Probability", "Risk_Pct"] + col_names], use_container_width=True)
            
            if st.button("Save this entire list to Google Sheet", type="primary"):
                with st.spinner("Pushing data to Google Sheet..."):
                    headers = worksheet.row_values(1)
                    
                    # Add Probability column to sheet if it doesn't exist
                    if "Probability" not in headers:
                        worksheet.update_cell(1, len(headers)+1, "Probability")
                        headers.append("Probability")
                        
                    # Prepare 2D array data to push at once
                    rows_to_append = []
                    for idx, row in df_upload.iterrows():
                        single_row = [""] * len(headers)
                        for col_name in col_names:
                            if col_name in headers:
                                h_idx = headers.index(col_name)
                                single_row[h_idx] = row[col_name]
                        # Add probability
                        if "Probability" in headers:
                            h_idx = headers.index("Probability")
                            single_row[h_idx] = round(float(row["Probability"]), 4)
                            
                        rows_to_append.append(single_row)
                    
                    # Use append_rows to push all data to the sheet at once
                    worksheet.append_rows(rows_to_append, value_input_option="RAW")
                    st.success(f"✅ Successfully saved {len(rows_to_append)} patients to Google Sheet!")
                    
        except Exception as e:
            st.error("Error reading file or processing model. Please check your file (if using Excel, ensure openpyxl is installed).")
            st.exception(e)
