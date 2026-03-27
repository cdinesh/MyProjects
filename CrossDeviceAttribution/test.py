import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
file_path = 'data/processed_synthetic_data_with_features.csv'
data = pd.read_csv(file_path)


# 1. Data preprocessing

import pandas as pd

# Load the dataset
file_path = 'data/processed_synthetic_data_with_features.csv'
data = pd.read_csv(file_path)

# Check for duplicates
print("Duplicates:", data.duplicated().sum())

# Drop duplicates
data = data.drop_duplicates()

# Handle missing values
print("Missing values per column:")
print(data.isnull().sum())

# Fill missing channels with 'Unknown'
data['channel'] = data['channel'].fillna('Unknown')

# Ensure channel names are consistent
data['channel'] = data['channel'].str.title().str.strip()

# Align time zones (if required)
data['timestamp'] = pd.to_datetime(data['daytime'])  # Assuming 'daytime' is the timestamp column

# Sort data by user and timestamp
data = data.sort_values(by=['uid', 'timestamp'])

# Time between interactions
data['time_between_interactions'] = data.groupby('uid')['timestamp'].diff().dt.total_seconds()

# Click position
data['click_position'] = data.groupby('uid').cumcount() + 1

# Device ratios
device_counts = data.groupby(['uid', 'device']).size().unstack(fill_value=0)
device_counts['mobile_to_desktop_ratio'] = device_counts.get('Mobile', 0) / (device_counts.get('Desktop', 1) + 1)
data = data.merge(device_counts[['mobile_to_desktop_ratio']], left_on='uid', right_index=True, how='left')

# Fill NaNs in new features
data = data.fillna(0)

# Train Markov Chain

def build_transition_matrix(data, channel_col='channel', conversion_col='conversion'):
    states = list(data[channel_col].unique()) + ['conversion']
    transition_matrix = pd.DataFrame(0, index=states, columns=states)

    for uid, user_data in data.groupby('uid'):
        channels = list(user_data[channel_col]) + (['conversion'] if user_data[conversion_col].iloc[-1] == 1 else [])
        for i in range(len(channels) - 1):
            current_state = channels[i]
            next_state = channels[i + 1]
            transition_matrix.loc[current_state, next_state] += 1

    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
    return transition_matrix

# Build the transition matrix
transition_matrix = build_transition_matrix(data)

# Attribution using Markov Chains
def calculate_attribution(transition_matrix):
    base_conversion_prob = simulate_conversion_probability(transition_matrix, start_states)
    attribution = {}
    for channel in transition_matrix.index:
        if channel == 'conversion':
            continue
        removed_prob = calculate_removal_effect(transition_matrix, channel, start_states)
        attribution[channel] = base_conversion_prob - removed_prob
    return pd.Series(attribution) / sum(attribution.values())
