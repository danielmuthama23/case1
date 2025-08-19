import pandas as pd
from io import StringIO
import json
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

retryable_reasons = {"Missing modifier", "Incorrect NPI", "Prior auth required"}
non_retryable = {"Authorization expired", "Incorrect provider type"}

def load_alpha_data(csv_str):
    try:
        df = pd.read_csv(StringIO(csv_str))
        df['source_system'] = 'alpha'
        df = df.rename(columns={
            'claim_id': 'claim_id', 'patient_id': 'patient_id', 'procedure_code': 'procedure_code',
            'denial_reason': 'denial_reason', 'submitted_at': 'submitted_at', 'status': 'status'
        })
        df = df.replace('None', pd.NA)
        df['patient_id'] = df['patient_id'].replace('', pd.NA)
        df['submitted_at'] = pd.to_datetime(df['submitted_at'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['status'] = df['status'].str.lower()
        return df
    except Exception as e:
        logging.error(f"Error loading alpha data: {e}")
        return pd.DataFrame()

def load_beta_data(json_str):
    try:
        data = json.loads(json_str)
        df = pd.DataFrame(data)
        df['source_system'] = 'beta'
        df = df.rename(columns={
            'id': 'claim_id', 'member': 'patient_id', 'code': 'procedure_code',
            'error_msg': 'denial_reason', 'date': 'submitted_at', 'status': 'status'
        })
        df['submitted_at'] = pd.to_datetime(df['submitted_at'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['status'] = df['status'].str.lower()
        return df
    except Exception as e:
        logging.error(f"Error loading beta data: {e}")
        return pd.DataFrame()

def mock_llm_classifier(reason):
    if pd.isna(reason) or reason is None:
        return False
    reason_str = str(reason).strip().lower()
    if any(word in reason_str for word in ['missing', 'incorrect', 'incomplete', 'prior']):
        return True
    return False

def is_retryable(reason):
    if pd.isna(reason) or reason is None:
        return False
    reason_str = str(reason).strip().lower()
    if reason_str in {r.lower() for r in retryable_reasons}:
        return True
    if reason_str in {r.lower() for r in non_retryable}:
        return False
    return mock_llm_classifier(reason)

def get_eligible_claims(df, today):
    eligible = df[
        (df['status'] == 'denied') &
        (df['patient_id'].notna()) &
        ((today - pd.to_datetime(df['submitted_at'])) > timedelta(days=7))
    ].copy()
    eligible.loc[:, 'is_retry'] = eligible['denial_reason'].apply(is_retryable)
    eligible = eligible[eligible['is_retry']]
    return eligible

def generate_output(eligible):
    output = []
    for _, row in eligible.iterrows():
        resub_reason = row['denial_reason']
        rec_changes_map = {
            "Incorrect NPI": "Review NPI number and resubmit",
            "Missing modifier": "Add missing modifier and resubmit",
            "Prior auth required": "Obtain prior authorization and resubmit",
            "incorrect procedure": "Correct procedure code and resubmit",
        }
        rec_changes = rec_changes_map.get(resub_reason, f"Review and correct the issue: {resub_reason}")
        out = {
            "claim_id": row['claim_id'],
            "resubmission_reason": resub_reason,
            "source_system": row['source_system'],
            "recommended_changes": rec_changes
        }
        output.append(out)
    return output

def log_metrics(df, eligible, today):
    total_claims = len(df)
    flagged = len(eligible)
    logging.info(f"Total claims processed: {total_claims}")
    logging.info(f"From alpha: {len(df[df['source_system'] == 'alpha'])}")
    logging.info(f"From beta: {len(df[df['source_system'] == 'beta'])}")
    logging.info(f"Flagged for resubmission: {flagged}")
    logging.info(f"Excluded because not denied: {len(df[df['status'] != 'denied'])}")
    logging.info(f"Excluded because patient_id null: {len(df[df['patient_id'].isna()])}")
    logging.info(f"Excluded because submitted <=7 days: {len(df[(today - pd.to_datetime(df['submitted_at'])) <= timedelta(days=7)])}")
    excluded_non_retry = len(df[
        (df['status'] == 'denied') &
        (df['patient_id'].notna()) &
        ((today - pd.to_datetime(df['submitted_at'])) > timedelta(days=7)) &
        ~df['denial_reason'].apply(is_retryable)
    ])
    logging.info(f"Excluded because non-retryable reason: {excluded_non_retry}")
    failed = df[~df.index.isin(eligible.index)]
    failed.to_csv('rejection_log.csv', index=False)
    logging.info("Failed records exported to rejection_log.csv")

def run_pipeline(csv_str, json_str):
    df_alpha = load_alpha_data(csv_str)
    df_beta = load_beta_data(json_str)
    df = pd.concat([df_alpha, df_beta], ignore_index=True)
    today = datetime(2025, 7, 30)
    eligible = get_eligible_claims(df, today)
    output = generate_output(eligible)
    log_metrics(df, eligible, today)
    with open('resubmission_candidates.json', 'w') as f:
        json.dump(output, f, indent=4)
    logging.info("Output saved to resubmission_candidates.json")
    return output

csv_data = """claim_id,patient_id,procedure_code,denial_reason,submitted_at,status
A123,P001,99213,Missing modifier,2025-07-01,denied
A124,P002,99214,Incorrect NPI,2025-07-10,denied
A125,,99215,Authorization expired,2025-07-05,denied
A126,P003,99381,None,2025-07-15,approved
A127,P004,99401,Prior auth required,2025-07-20,denied
"""
json_data = '''
[
  {"id": "B987", "member": "P010", "code": "99213", "error_msg": "Incorrect provider type", "date": "2025-07-03T00:00:00", "status": "denied"},
  {"id": "B988", "member": "P011", "code": "99214", "error_msg": "Missing modifier", "date": "2025-07-09T00:00:00", "status": "denied"},
  {"id": "B989", "member": "P012", "code": "99215", "error_msg": null, "date": "2025-07-10T00:00:00", "status": "approved"},
  {"id": "B990", "member": null, "code": "99401", "error_msg": "incorrect procedure", "date": "2025-07-01T00:00:00", "status": "denied"}
]
'''

if __name__ == "__main__":
    run_pipeline(csv_data, json_data)