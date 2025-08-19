import sys
import os
sys.path.append(os.path.abspath('../'))  # Add parent directory to path for notebook

import pandas as pd
import json
from claim_resubmission_pipeline import load_alpha_data, load_beta_data, get_eligible_claims, is_retryable

def test_load_alpha_data():
    csv_str = "claim_id,patient_id,procedure_code,denial_reason,submitted_at,status\nA123,P001,99213,Missing modifier,2025-07-01,denied"
    df = load_alpha_data(csv_str)
    assert not df.empty
    assert df['source_system'].iloc[0] == 'alpha'

def test_load_beta_data():
    json_str = '[{"id": "B987", "member": "P010", "code": "99213", "error_msg": "Incorrect provider type", "date": "2025-07-03T00:00:00", "status": "denied"}]'
    df = load_beta_data(json_str)
    assert not df.empty
    assert df['source_system'].iloc[0] == 'beta'

def test_is_retryable():
    assert is_retryable("Missing modifier") == True
    assert is_retryable("Authorization expired") == False
    assert is_retryable("incorrect procedure") == True

def test_get_eligible_claims():
    data = {"claim_id": ["A123"], "patient_id": ["P001"], "procedure_code": ["99213"], "denial_reason": ["Missing modifier"], "submitted_at": ["2025-07-01"], "status": ["denied"], "source_system": ["alpha"]}
    df = pd.DataFrame(data)
    today = pd.to_datetime("2025-07-30")
    eligible = get_eligible_claims(df, today)
    assert not eligible.empty

# Run tests
if __name__ == "__main__":
    test_load_alpha_data()
    test_load_beta_data()
    test_is_retryable()
    test_get_eligible_claims()
    print("All tests passed!")