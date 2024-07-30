import streamlit as st
import pickle
import pandas as pd
import os

# Load models
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Ensure the model files are in the same directory as the Streamlit app
log_reg_model_path = os.path.join(os.path.dirname(__file__), "log_reg_model.pkl")
svm_model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")

log_reg_model = load_model(log_reg_model_path)
svm_model = load_model(svm_model_path)

# Get feature names
def get_feature_names():
    feature_names = [
        'ID', 'CR_PROD_CNT_IL', 'AMOUNT_RUB_CLO_PRC', 'APP_REGISTR_RGN_CODE',
        'TURNOVER_DYNAMIC_IL_1M', 'CNT_TRAN_AUT_TENDENCY1M', 'SUM_TRAN_AUT_TENDENCY1M',
        'AMOUNT_RUB_SUP_PRC', 'SUM_TRAN_AUT_TENDENCY3M', 'REST_DYNAMIC_FDEP_1M',
        'REST_DYNAMIC_CC_3M', 'MED_DEBT_PRC_YWZ', 'LDEAL_ACT_DAYS_PCT_TR3',
        'LDEAL_ACT_DAYS_PCT_AAVG', 'LDEAL_DELINQ_PER_MAXYWZ', 'TURNOVER_DYNAMIC_CC_3M',
        'LDEAL_ACT_DAYS_PCT_TR', 'LDEAL_ACT_DAYS_PCT_TR4', 'LDEAL_ACT_DAYS_PCT_CURR',
        'CNT_TRAN_AUT_TENDENCY3M', 'REST_DYNAMIC_SAVE_3M', 'CR_PROD_CNT_VCU', 
        'REST_AVG_CUR', 'CNT_TRAN_MED_TENDENCY1M', 'AMOUNT_RUB_NAS_PRC', 
        'TRANS_COUNT_SUP_PRC', 'CNT_TRAN_CLO_TENDENCY1M', 'SUM_TRAN_MED_TENDENCY1M',
        'TRANS_COUNT_NAS_PRC', 'CR_PROD_CNT_TOVR', 'CR_PROD_CNT_PIL', 
        'SUM_TRAN_CLO_TENDENCY1M', 'TURNOVER_CC', 'TRANS_COUNT_ATM_PRC', 
        'AMOUNT_RUB_ATM_PRC', 'TURNOVER_PAYM', 'AGE', 'CNT_TRAN_MED_TENDENCY3M', 
        'CR_PROD_CNT_CC', 'SUM_TRAN_MED_TENDENCY3M', 'REST_DYNAMIC_FDEP_3M', 
        'REST_DYNAMIC_IL_1M', 'SUM_TRAN_CLO_TENDENCY3M', 'LDEAL_TENOR_MAX', 
        'LDEAL_YQZ_CHRG', 'CR_PROD_CNT_CCFP', 'DEAL_YQZ_IR_MAX', 'LDEAL_YQZ_COM', 
        'DEAL_YQZ_IR_MIN', 'CNT_TRAN_CLO_TENDENCY3M', 'REST_DYNAMIC_CUR_1M', 
        'REST_AVG_PAYM', 'LDEAL_TENOR_MIN', 'LDEAL_AMT_MONTH', 
        'LDEAL_GRACE_DAYS_PCT_MED', 'REST_DYNAMIC_CUR_3M', 'CNT_TRAN_SUP_TENDENCY3M', 
        'TURNOVER_DYNAMIC_CUR_1M', 'REST_DYNAMIC_PAYM_3M', 'SUM_TRAN_SUP_TENDENCY3M', 
        'REST_DYNAMIC_IL_3M', 'CNT_TRAN_ATM_TENDENCY3M', 'CNT_TRAN_ATM_TENDENCY1M', 
        'TURNOVER_DYNAMIC_IL_3M', 'SUM_TRAN_ATM_TENDENCY3M', 
        'DEAL_GRACE_DAYS_ACC_S1X1', 'AVG_PCT_MONTH_TO_PCLOSE', 'DEAL_YWZ_IR_MIN', 
        'SUM_TRAN_SUP_TENDENCY1M', 'DEAL_YWZ_IR_MAX', 'SUM_TRAN_ATM_TENDENCY1M', 
        'REST_DYNAMIC_PAYM_1M', 'CNT_TRAN_SUP_TENDENCY1M', 'DEAL_GRACE_DAYS_ACC_AVG', 
        'TURNOVER_DYNAMIC_CUR_3M', 'MAX_PCLOSE_DATE', 'LDEAL_YQZ_PC', 
        'CLNT_SETUP_TENOR', 'DEAL_GRACE_DAYS_ACC_MAX', 'TURNOVER_DYNAMIC_PAYM_3M', 
        'LDEAL_DELINQ_PER_MAXYQZ', 'TURNOVER_DYNAMIC_PAYM_1M', 'CLNT_SALARY_VALUE', 
        'TRANS_AMOUNT_TENDENCY3M', 'MED_DEBT_PRC_YQZ', 'TRANS_CNT_TENDENCY3M', 
        'LDEAL_USED_AMT_AVG_YQZ', 'REST_DYNAMIC_CC_1M', 'LDEAL_USED_AMT_AVG_YWZ', 
        'TURNOVER_DYNAMIC_CC_1M', 'AVG_PCT_DEBT_TO_DEAL_AMT', 
        'LDEAL_ACT_DAYS_ACC_PCT_AVG', 'CLNT_TRUST_RELATION', 'APP_MARITAL_STATUS',
        'APP_KIND_OF_PROP_HABITATION', 'CLNT_JOB_POSITION_TYPE', 'APP_DRIVING_LICENSE',
        'APP_EDUCATION', 'APP_TRAVEL_PASS', 'APP_CAR', 'APP_POSITION_TYPE',
        'APP_EMP_TYPE', 'APP_COMP_TYPE', 'PACK'
    ]
    return feature_names

# Streamlit app
st.title('Customer Prediction App')

# Feature names
features = get_feature_names()

# Collect user input
user_input = {}
for feature in features:
    user_input[feature] = st.text_input(feature, value=0)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button('Predict'):
    logreg_prediction = log_reg_model.predict(input_df)
    logreg_prediction_proba = log_reg_model.predict_proba(input_df)
    
    svm_prediction = svm_model.predict(input_df)
    svm_prediction_proba = svm_model.predict_proba(input_df)

    # Display predictions
    st.write("### Logistic Regression Prediction")
    st.write(logreg_prediction[0])
    st.write("### Logistic Regression Prediction Probability")
    st.write(logreg_prediction_proba[0])

    st.write("### SVM Prediction")
    st.write(svm_prediction[0])
    st.write("### SVM Prediction Probability")
    st.write(svm_prediction_proba[0])
