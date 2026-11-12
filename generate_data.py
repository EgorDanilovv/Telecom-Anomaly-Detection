import pandas as pd
import numpy as np

NUM_ROWS = 2_000_000
np.random.seed(42)

subscriber_ids = np.arange(1, NUM_ROWS + 1)
zone_codes = np.random.choice(['Z01', 'Z02', 'Z03', 'Z04', 'Z05', np.nan], size=NUM_ROWS, p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02])
tower_ids = np.random.choice(['TWR-A', 'TWR-B', 'TWR-C', 'TWR-D'], size=NUM_ROWS)
status = np.random.choice(['Active', 'Disconnected'], size=NUM_ROWS, p=[0.85, 0.15])

usage_gb = np.random.lognormal(mean=2.5, sigma=0.8, size=NUM_ROWS).round(2)
topup_amount = np.random.choice([0, 100, 250, 500, 1000], size=NUM_ROWS, p=[0.4, 0.3, 0.15, 0.1, 0.05])
complaints_count = np.random.poisson(lam=0.5, size=NUM_ROWS)
tariff_changes = np.random.poisson(lam=0.2, size=NUM_ROWS)

df = pd.DataFrame({
    'subscriber_id': subscriber_ids,
    'zone_code': zone_codes,
    'tower_id': tower_ids,
    'status': status,
    'usage_gb': usage_gb,
    'topup_amount': topup_amount,
    'complaints_count': complaints_count,
    'tariff_changes': tariff_changes
})

anomaly_indices = np.random.choice(NUM_ROWS, size=int(NUM_ROWS * 0.05), replace=False)

mask_disconnected_active = np.random.choice(anomaly_indices, size=len(anomaly_indices)//3, replace=False)
df.loc[mask_disconnected_active, 'status'] = 'Disconnected'
df.loc[mask_disconnected_active, 'usage_gb'] = np.random.lognormal(mean=4.0, sigma=1.0, size=len(mask_disconnected_active)).round(2)
df.loc[mask_disconnected_active, 'topup_amount'] = np.random.choice([1000, 2500, 5000], size=len(mask_disconnected_active))

mask_mismatch = np.random.choice(anomaly_indices, size=len(anomaly_indices)//3, replace=False)
df.loc[mask_mismatch, 'tower_id'] = 'TWR-UNKNOWN'
df.loc[mask_mismatch, 'zone_code'] = np.random.choice(['Z99', 'INVALID', np.nan], size=len(mask_mismatch))

mask_complaints = np.setdiff1d(anomaly_indices, np.union1d(mask_disconnected_active, mask_mismatch))
df.loc[mask_complaints, 'complaints_count'] = np.random.randint(10, 50, size=len(mask_complaints))
df.loc[mask_complaints, 'tariff_changes'] = np.random.randint(3, 10, size=len(mask_complaints))

df['is_anomaly'] = 0
df.loc[anomaly_indices, 'is_anomaly'] = 1

df.to_csv('telecom_data_2M.csv', index=False)

anom_sample = df[df['is_anomaly'] == 1].sample(150, random_state=42)
norm_sample = df[df['is_anomaly'] == 0].sample(150, random_state=42)

val_df = pd.concat([anom_sample, norm_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df[['subscriber_id', 'is_anomaly', 'usage_gb', 'topup_amount', 'complaints_count', 'status']]
val_df.to_csv('validation_set_300.csv', index=False)
