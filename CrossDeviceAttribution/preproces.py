import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import timedelta

# Add synthetic column for 'time_since_last_click'
synthetic_data['time_since_last_click'] = synthetic_data.groupby('session_id')['timestamp'].diff().dt.total_seconds()
synthetic_data['time_since_last_click'] = synthetic_data['time_since_last_click'].fillna(0)

# ------------------- Step 1: Data Cleaning -------------------
# Handle missing values
synthetic_data.fillna({'time_since_last_click': 0}, inplace=True)  # Fill missing time differences

# Remove duplicates
synthetic_data.drop_duplicates(inplace=True)

# ------------------- Step 2: Data Transformation -------------------
# Normalize numeric columns
scaler = MinMaxScaler()
synthetic_data[['cost', 'time_since_last_click']] = scaler.fit_transform(synthetic_data[['cost', 'time_since_last_click']])

# Encode categorical variables
label_encoder = LabelEncoder()
synthetic_data['channel_encoded'] = label_encoder.fit_transform(synthetic_data['channel'])
synthetic_data['device_encoded'] = label_encoder.fit_transform(synthetic_data['device'])

# ------------------- Step 3: Feature Engineering -------------------
# Aggregated metrics
aggregated_metrics = synthetic_data.groupby('uid').agg({
    'click': 'sum',
    'conversion': 'sum',
    'cost': 'mean',
    'time_since_last_click': 'mean'
}).reset_index().rename(columns={
    'click': 'total_clicks',
    'conversion': 'total_conversions',
    'cost': 'avg_cost',
    'time_since_last_click': 'avg_time_since_click'
})

# Identify multi-device users
multi_device_users = synthetic_data.groupby('uid')['device'].nunique().reset_index()
multi_device_users.columns = ['uid', 'unique_devices']
multi_device_users['is_multi_device'] = multi_device_users['unique_devices'] > 1

# User journey sequences
user_journeys = synthetic_data.groupby('uid').agg({
    'channel': lambda x: ' > '.join(x.unique()),
    'device': lambda x: ' > '.join(x.unique())
}).reset_index().rename(columns={
    'channel': 'channel_sequence',
    'device': 'device_sequence'
})

# Merge all engineered features
processed_data = pd.merge(aggregated_metrics, multi_device_users, on='uid')
processed_data = pd.merge(processed_data, user_journeys, on='uid')

# ------------------- Step 4: Merging Datasets -------------------
# Assume a second dataset exists (e.g., Criteo dataset)
# Here, we use a placeholder to demonstrate merging
criteo_data_placeholder = synthetic_data.copy()
criteo_data_placeholder['source'] = 'Criteo'

synthetic_data['source'] = 'Synthetic'
merged_data = pd.concat([synthetic_data, criteo_data_placeholder], ignore_index=True)

# ------------------- Step 5: Final Validation -------------------
# Check for consistency
merged_data.sort_values(by=['uid', 'timestamp'], inplace=True)
merged_data.reset_index(drop=True, inplace=True)

# Display preprocessed data
import ace_tools as tools; tools.display_dataframe_to_user(name="Preprocessed Dataset", dataframe=merged_data)

# Outputs the processed data for further use
processed_data.head()
