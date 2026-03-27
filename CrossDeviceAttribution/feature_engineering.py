#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.optimize import minimize
# Import required libraries
import pandas as pd
import random
import numpy as np
from tabulate import tabulate
from datetime import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

file_path = 'data/synthetic_cross_device_data.csv'
data = pd.read_csv(file_path)

# Ensure timestamp is in datetime format
data['timestamp'] = pd.to_datetime(data['daytime'])

# Sort the data by user and timestamp
data = data.sort_values(by=['uid', 'timestamp'])

# 1. Time between interactions
data['time_between_interactions'] = data.groupby('uid')['timestamp'].diff().dt.total_seconds()

# 2. Clicks before conversion
data['clicks_before_conversion'] = data.groupby('uid')['click'].cumsum()


# 3. Mobile vs laptop ratio
device_counts = data.groupby(['uid', 'device']).size().unstack(fill_value=0)
device_counts['mobile_laptop_ratio'] = device_counts.get('Mobile', 0) / device_counts.get('Desktop', 1)
data = data.merge(device_counts[['mobile_laptop_ratio']], left_on='uid', right_index=True, how='left')

# 4. Number of touchpoints
data['number_of_touchpoints'] = data.groupby('uid')['uid'].transform('count')

# 5. Attribution lift
data['attributed_conversions'] = data.groupby('uid')['attribution'].cumsum()
data['attribution_lift'] = data['attributed_conversions'] - data['conversion']

# Summarize features for visualization
feature_means = data[
    ['time_between_interactions', 'clicks_before_conversion',
     'mobile_laptop_ratio', 'number_of_touchpoints', 'attribution_lift']
].count()

# Prepare data for visualization
features = feature_means.index.tolist()
values = feature_means.values
# Create a bar graph to visualize the engineered features

plt.figure(figsize=(8, 4))
plt.bar(features, values, alpha=0.8, color='skyblue', edgecolor='black')
plt.title("Feature Engineering", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("output/features.png")
plt.show()
