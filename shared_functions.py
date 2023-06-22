
import numpy as np
import pandas as pd
import requests
from io import StringIO
import random

# import plotly.express as px  # interactive charts
# import plotly.graph_objects as go
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import Normalize
from datetime import datetime, timedelta


import streamlit as st  # 🎈 data web app development
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


# Load dataset
url = "https://app-api-eu-west-3.milesinthesky.education/leaderboard/geordie_history?groupName=Blue Fox"
payload = {}
headers = {
  'x-api-key': '4A1ZxPSude5K1HjyaZ5y67Mrzkv4R8KW4OJtyaYH'
}
response = requests.request("GET", url, headers=headers, data=payload)
df = pd.read_csv(StringIO(response.text))


st.set_page_config(
    page_title="J3 Dashboard",
    page_icon="✅",
    layout="wide",
)

# @st.cache_data # decorator runs function once and caches data - avoids downloading dataset again & again
def get_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(response.text))
    df.columns = df.columns.str.strip()
    df['timestamp'] = df['timestamp'].apply(pd.to_datetime)
    return df



# ---------FUNCTIONS--------------------------------------------------------------------------------------------------------------- #
def calc_costs(y_true, y_pred, amt_credit):
    """
    Finds the costs.

    Parameters:
    y_true (np.ndarray): True labels of the instances.
    y_probs (np.ndarray): Predicted probabilities for the positive class.
    amt_credit (pd.Series): Credit amount for each instance.

    Returns:
    float: The cost.
    """
    # since not same index ...
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    # Obtain the FP_indices and FN_indices
    false_positives_mask = (y_pred.values == 1) & (y_true.values == 0)
    FP_indices = y_pred[false_positives_mask].index
    false_negatives_mask = (y_pred.values == 0) & (y_true.values == 1)
    FN_indices =y_pred[false_negatives_mask].index

    # Calculate the cost for false positives and for false negatives
    FP_cost = np.sum(-0.25 * amt_credit.iloc[FP_indices])
    FN_cost = np.sum(-1 * amt_credit.iloc[FN_indices])

    # Calculate the overall cost
    cost = FP_cost + FN_cost

    return float(cost)

def find_metric(group_name, metric):
    for item in leaderboard_data:
        if item['model_name'] == group_name:
            return item[metric]
    return None

def create_fake_model(fraction_A, fraction_B, model_df,true_df,added_days,added_hours,model):
    num_rows_A = int(len(model_df) * fraction_A)
    num_rows_B = int(len(true_df) * fraction_B)
    sample_A = model_df.head(n=num_rows_A)
    sample_B = true_df.tail(n=num_rows_B)
    C = pd.concat([sample_A, sample_B], ignore_index=True)
    C['MODEL_IDENTIFIER']= model
    C['timestamp']= C['timestamp'] + pd.Timedelta(days=round(random.uniform(0, added_days)),hours=round(random.uniform(0, added_hours)))
    return C

def calculate_cost(row):
    if row['TARGET'] == row['PREDICTED_TARGET']:
        return 0
    elif row['TARGET'] == 1 and row['PREDICTED_TARGET'] == 0:
        return -1 * row['AMT_CREDIT']
    elif row['TARGET'] == 0 and row['PREDICTED_TARGET'] == 1:
        return -0.25 * row['AMT_CREDIT']

def calculate_cost_FN(row):
    if row['TARGET'] == 1 and row['PREDICTED_TARGET'] == 0:
        return -1 * row['AMT_CREDIT']
    else:
        return 0

def calculate_cost_FP(row):
    if row['TARGET'] == 0 and row['PREDICTED_TARGET'] == 1:
        return -0.25 * row['AMT_CREDIT']
    else:
        return 0

def create_model_dict(df):
    df.columns = df.columns.str.strip()
    df.sort_values(by='timestamp')
    model_dict = {}
    utopia_df = df.iloc[:len(df)//len(df['MODEL_IDENTIFIER'].unique())].copy()
    random_assignment_df = df.iloc[:len(df)//len(df['MODEL_IDENTIFIER'].unique())].copy()


    # Creating best/worst models
    key_utopia = "Utopia"
    utopia_df["PREDICTED_TARGET"] = df['TARGET']
    utopia_df['COST'] =  utopia_df.apply(calculate_cost, axis=1)
    utopia_df['COST_FP'] =  utopia_df.apply(calculate_cost_FP, axis=1)
    utopia_df['COST_FN'] =  utopia_df.apply(calculate_cost_FN, axis=1)
    utopia_df['MODEL_IDENTIFIER']= 'model_utopia'
    model_dict[key_utopia] = utopia_df
    key_random = "Random Assignment"
    random_assignment_df["PREDICTED_TARGET"] = np.random.choice([0, 1], size=len(random_assignment_df))
    random_assignment_df['COST'] =  random_assignment_df.apply(calculate_cost, axis=1)
    random_assignment_df['COST_FN'] =  random_assignment_df.apply(calculate_cost_FN, axis=1)
    random_assignment_df['COST_FP'] =  random_assignment_df.apply(calculate_cost_FP, axis=1)
    random_assignment_df['MODEL_IDENTIFIER']= 'model_random'
    model_dict[key_random] = random_assignment_df
    # Creating fake models
    temp_3=create_fake_model(0.85, 0.15,df,utopia_df, 0,24, 'Model_X')
    temp_4=create_fake_model(0.5, 0.5,df, utopia_df,1,20,'Model_Y')
    temp_5=create_fake_model(0.3, 0.7,df, utopia_df, 2,24, 'Model_Z')
    df = pd.concat([df, temp_3], ignore_index=True)
    df = pd.concat([df, temp_4], ignore_index=True)
    df = pd.concat([df, temp_5], ignore_index=True)

    df['COST'] = df.apply(calculate_cost, axis=1)
    df['RATIO_CC'] = df['COST'] / df['AMT_CREDIT']
    df['COST_FN'] = df.apply(calculate_cost_FN, axis=1)
    df['COST_FP'] = df.apply(calculate_cost_FP, axis=1)

    df.sort_values(by='timestamp', inplace=True)
    i=0
    for model in df['MODEL_IDENTIFIER'].unique():
        i+=1
        model_dict[f'Model {i}']= df[df['MODEL_IDENTIFIER'] == model]


    return model_dict, df


# ---------FUNCTIONS--------------------------------------------------------------------------------------------------------------- #


df = get_data()
df_mapping, df = create_model_dict(df)

leaderboard_data = []
for model in df_mapping.keys():
    model_df = df_mapping[model]
    # Model metrics
    y_pred = model_df['PREDICTED_TARGET']
    y_true = model_df['TARGET']
    amt_credit = model_df["AMT_CREDIT"]
    leaderboard_data.append({'model_name': model,
                             'formal_name': model_df['MODEL_IDENTIFIER'].unique()[0],
                             'start': min(model_df['timestamp']),
                             'end':max(model_df['timestamp']),
                             'cost': calc_costs(y_true, y_pred, amt_credit),
                             'cost_check': np.sum(model_df["COST"]),
                              'cf_matrix': confusion_matrix(y_true, y_pred ),
                              "roc": roc_curve(y_true, y_pred),
                                    'auc': roc_auc_score(y_true, y_pred),
                              'precision_recall': precision_recall_curve(y_true, y_pred),
                              "accuracy_score":accuracy_score(y_true, y_pred),
                              'calibration_curve': calibration_curve(y_true, y_pred, n_bins = 10),
                              'f1_score': f1_score(y_true, y_pred),
                              "tot_credit_request": np.sum(model_df["AMT_CREDIT"]),
                              "count_loans_request": model_df.shape[0],
                              "cost_credit_ratio":-100*(calc_costs(y_true, y_pred, amt_credit)/np.sum(model_df["AMT_CREDIT"])),
                              "count_errors": model_df['ERROR'].value_counts().get(1, 0),
                              'cost_FN': np.sum(model_df["COST_FN"]),
                              'cost_FP': np.sum(model_df["COST_FP"])
                                        })
leaderboard_data = sorted(leaderboard_data, key=lambda x: x['cost_credit_ratio'])
for i, item in enumerate(leaderboard_data):
    item['rank'] = i+1


table_data = []
for d in leaderboard_data:
    row = {}
    for key, value in d.items():
        if key in ['cost_FN', 'cost_FP','cost_check','formal_name']:
            pass
        else:
            key = ' '.join(word.capitalize() for word in key.split('_'))

            if isinstance(value, str):
                row[key] = value
            elif isinstance(value, float) or isinstance(value, int):
                if abs(value) < 1:
                    row[key] = value
                else:
                    row[key] = round(value)
            if key=='Cost Credit Ratio':
                row[key] = f'{abs(value):.2f}%'
    table_data.append(row)




