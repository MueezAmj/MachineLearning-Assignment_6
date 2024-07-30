from flask import Flask, request, jsonify
import pickle
import pandas as pd
import traceback

app = Flask(__name__)

# Load models
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

log_reg_model = load_model("log_reg_model.pkl")
svm_model = load_model("svm_model.pkl")

def get_feature_names():
    # Add all feature names, including both numerical and categorical
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

@app.route('/')
def home():
    return "Model API is running."

@app.route('/predict/logreg', methods=['POST'])
def predict_logreg():
    try:
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame(data)
            feature_names = get_feature_names()
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Add missing feature with default value
            prediction = log_reg_model.predict(df)
            prediction_proba = log_reg_model.predict_proba(df)
            return jsonify({'prediction': prediction.tolist(), 'prediction_proba': prediction_proba.tolist()})
        else:
            return jsonify({'error': 'Request content type must be application/json'}), 415
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    try:
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame(data)
            feature_names = get_feature_names()
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Add missing feature with default value
            prediction = svm_model.predict(df)
            prediction_proba = svm_model.predict_proba(df)
            return jsonify({'prediction': prediction.tolist(), 'prediction_proba': prediction_proba.tolist()})
        else:
            return jsonify({'error': 'Request content type must be application/json'}), 415
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/features', methods=['GET'])
def get_features():
    features = get_feature_names()
    return jsonify({'features': features})

if __name__ == '__main__':
    app.run(debug=True)
