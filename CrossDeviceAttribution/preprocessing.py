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
final_data = pd.read_csv(file_path)


# # Exploratory Data Analysis

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Attribution
df_a = pd.DataFrame(final_data)


if 'attribution' not in df_a.columns:
    # Simulate attribution column with a 70% likelihood of being attributed to Criteo for conversion = 1
    df_a['attribution'] = df_a['conversion'].apply(lambda x: 1 if x == 1 and random.random() < 0.7 else 0)

# Count the number of conversions attributed to Criteo
criteo_attributed_conversions = df_a[df_a['attribution'] == 1]['attribution'].count()

# Display total conversions and Criteo-attributed conversions
total_conversions = df_a['conversion'].sum()

attribution_summary = {
    "Total Conversions": total_conversions,
    "Criteo-Attributed Conversions": criteo_attributed_conversions,
    "Percentage Attributed to Criteo": (criteo_attributed_conversions / total_conversions) * 100 if total_conversions > 0 else 0
}

# Attribution by Channel
channel_attribution = df_a[df_a['attribution'] == 1].groupby('channel').agg({
    'conversion': 'sum',
    'attribution': 'count'
}).rename(columns={'conversion': 'Total Conversions', 'attribution': 'Criteo-Attributed Conversions'}).reset_index()

# Calculate percentage attribution for each channel
channel_attribution['Percentage Attributed'] = (
    channel_attribution['Criteo-Attributed Conversions'] / channel_attribution['Total Conversions'] * 100
)

# Attribution by Device
device_attribution = df_a[df_a['attribution'] == 1].groupby('device').agg({
    'conversion': 'sum',
    'attribution': 'count'
}).rename(columns={'conversion': 'Total Conversions', 'attribution': 'Criteo-Attributed Conversions'}).reset_index()

# Calculate percentage attribution for each device
device_attribution['Percentage Attributed'] = (
    device_attribution['Criteo-Attributed Conversions'] / device_attribution['Total Conversions'] * 100
)

 #Regenerate the Attribution Summary Visualization

# Create a DataFrame for the attribution summary
attribution_summary_df = pd.DataFrame({
    "category": ["Total Conversions", "Criteo-Attributed Conversions"],
    "count": [total_conversions, criteo_attributed_conversions]
})



# Define weighted probabilities for channels and devices based on their attribution likelihood
channel_attribution_probabilities = {
    'Paid Search': 0.30,  # 30% likelihood
    'Affiliate': 0.25,    # 25% likelihood
    'Organic Search': 0.20,  # 20% likelihood
    'Social Media': 0.15, # 15% likelihood
    'Email': 0.10         # 10% likelihood
}

device_attribution_probabilities = {
    'Mobile': 0.50,  # 50% likelihood
    'Desktop': 0.35, # 35% likelihood
    'Tablet': 0.15   # 15% likelihood
}


# Assign channels based on weighted probabilities for attribution
weighted_channels = list(channel_attribution_probabilities.keys())
channel_weights = list(channel_attribution_probabilities.values())
df_a['channel'] = df_a['attribution'].apply(
    lambda x: random.choices(weighted_channels, weights=channel_weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Assign devices based on weighted probabilities for attribution
weighted_devices = list(device_attribution_probabilities.keys())
device_weights = list(device_attribution_probabilities.values())
df_a['device'] = df_a['attribution'].apply(
    lambda x: random.choices(weighted_devices, weights=device_weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Recalculate refined distributions for channels and devices related to attribution
channel_attribution_distribution = df_a[df_a['attribution'] == 1]['channel'].value_counts().reset_index()
channel_attribution_distribution.columns = ['channel', 'count']

device_attribution_distribution = df_a[df_a['attribution'] == 1]['device'].value_counts().reset_index()
device_attribution_distribution.columns = ['device', 'count']


# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

# # Plot 1: Attribution
sns.barplot(data=attribution_summary_df, x='category', y='count', ax=axes[0], palette='Blues_d',width=0.6)
# Add count annotations on top of the bars


axes[0].set_title('Attribution Summary', fontsize=14)
axes[0].set_xlabel('Category', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=channel_attribution_distribution, x='channel', y='count', ax=axes[1], palette='Blues_d',width=0.6)
axes[1].set_title('Channel Distribution for Attribution', fontsize=14)
axes[1].set_xlabel('Channel', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].tick_params(axis='x', rotation=30)


sns.barplot(data=device_attribution_distribution, x='device', y='count', ax=axes[2], palette='Blues_d',width=0.6)
axes[2].set_title('Device Distribution for Attribution', fontsize=14)
axes[2].set_xlabel('Device', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('output/attribution_distribution.png')
plt.show()

#-----------------------------------------------------
# Clicks

df_c = pd.DataFrame(final_data)


if 'click' not in df_c.columns:
    # Simulate attribution column with a 70% likelihood of being attributed to Criteo for conversion = 1
    df_c['click'] = df_c['conversion'].apply(lambda x: 1 if x == 1 and random.random() < 0.7 else 0)

# Count the number of conversions attributed to Criteo
total_click_conversions = df_c[df_c['click'] == 1]['click'].count()

# Display total conversions and Criteo-attributed conversions
criteo_click_conversions = df_c[df_c['attribution'] == 1]['conversion'].sum()

click_summary = {
    "Total Click Conversions": total_click_conversions,
    "Criteo-Click Conversions": criteo_click_conversions,
    "Percentage Click to Criteo": (criteo_click_conversions / total_click_conversions) * 100 if total_click_conversions > 0 else 0
}

# Attribution by Channel
channel_click = df_c[df_c['click'] == 1].groupby('channel').agg({
    'conversion': 'sum',
    'click': 'count'
}).rename(columns={'conversion': 'Total Click Conversions', 'click': 'Criteo-Click Conversions'}).reset_index()

# Calculate percentage attribution for each channel
channel_click['Percentage Clicked'] = (
    channel_click['Criteo-Click Conversions'] / channel_click['Total Click Conversions'] * 100
)

# Attribution by Device
device_click = df_c[df_c['click'] == 1].groupby('device').agg({
    'conversion': 'sum',
    'click': 'count'
}).rename(columns={'conversion': 'Total Click Conversions', 'click': 'Criteo-Click Conversions'}).reset_index()

# Calculate percentage attribution for each device
device_click['Percentage Clicked'] = (
    device_click['Criteo-Click Conversions'] / device_click['Total Click Conversions'] * 100
)

 #Regenerate the Attribution Summary Visualization

# Create a DataFrame for the attribution summary
click_summary_df = pd.DataFrame({
    "category": ["Total Clicks", "Criteo-Click Conversions"],
    "count": [total_click_conversions, criteo_click_conversions]
})



# Define weighted probabilities for channels and devices based on their attribution likelihood
channel_click_probabilities = {
    'Paid Search': 0.30,  # 30% likelihood
    'Affiliate': 0.25,    # 25% likelihood
    'Organic Search': 0.20,  # 20% likelihood
    'Social Media': 0.15, # 15% likelihood
    'Email': 0.10         # 10% likelihood
}

device_click_probabilities = {
    'Mobile': 0.50,  # 50% likelihood
    'Desktop': 0.35, # 35% likelihood
    'Tablet': 0.15   # 15% likelihood
}


# Assign channels based on weighted probabilities for attribution
weighted_click_channels = list(channel_click_probabilities.keys())
channel_click_weights = list(channel_click_probabilities.values())
df_c['channel'] = df_c['click'].apply(
    lambda x: random.choices(weighted_click_channels, weights=channel_click_weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Assign devices based on weighted probabilities for attribution
weighted_click_devices = list(device_click_probabilities.keys())
device_click_weights = list(device_click_probabilities.values())
df_c['device'] = df_c['click'].apply(
    lambda x: random.choices(weighted_click_devices, weights=device_click_weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Recalculate refined distributions for channels and devices related to attribution
channel_click_distribution = df_c[df_c['click'] == 1]['channel'].value_counts().reset_index()
channel_click_distribution.columns = ['channel', 'count']

device_click_distribution = df_c[df_c['click'] == 1]['device'].value_counts().reset_index()
device_click_distribution.columns = ['device', 'count']


# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

# # Plot 1: Attribution
sns.barplot(data=click_summary_df, x='category', y='count', ax=axes[0], palette='Blues_d',width=0.6)
# Add count annotations on top of the bars


axes[0].set_title('Click Summary', fontsize=14)
axes[0].set_xlabel('Category', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=channel_click_distribution, x='channel', y='count', ax=axes[1], palette='Blues_d',width=0.6)
axes[1].set_title('Channel Distribution for Click', fontsize=14)
axes[1].set_xlabel('Channel', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].tick_params(axis='x', rotation=30)


sns.barplot(data=device_attribution_distribution, x='device', y='count', ax=axes[2], palette='Blues_d',width=0.6)
axes[2].set_title('Device Distribution for Click', fontsize=14)
axes[2].set_xlabel('Device', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('output/click_distribution.png')
plt.show()

# Conversion

df_conv = pd.DataFrame(final_data)

# Count the number of conversions attributed to Criteo
total_conversions = df_conv[df_conv['conversion'] == 1]['conversion'].count()

# Display total conversions and Criteo-attributed conversions
criteo_clicks = df_conv[df_conv['click'] == 1]['click'].count()

conversion_summary = {
    "Criteo Clicks": criteo_clicks,
    "Total Conversions": total_conversions,
    "Percentage Conversion to Criteo": (criteo_clicks / total_conversions) * 100 if total_conversions > 0 else 0
}

# Conversion by Channel
channel_conversion = df_conv[df_conv['conversion'] == 1].groupby('channel').agg({
    'conversion': 'sum',
    'click': 'count'
}).rename(columns={'conversion': 'Total Conversions', 'click': 'Criteo Clicks'}).reset_index()

# Calculate percentage attribution for each channel
channel_conversion['Percentage Converted'] = (
    channel_conversion['Criteo Clicks'] / channel_conversion['Total Conversions'] * 100
)

# Attribution by Device
device_conversion = df_conv[df_conv['conversion'] == 1].groupby('device').agg({
    'conversion': 'sum',
    'click': 'count'
}).rename(columns={'conversion': 'Total Conversions', 'click': 'Criteo Clicks'}).reset_index()

# Calculate percentage attribution for each device
device_conversion['Percentage Converted'] = (
    device_conversion['Criteo Clicks'] / device_conversion['Total Conversions'] * 100
)

 #Regenerate the Attribution Summary Visualization

# Create a DataFrame for the attribution summary
conversion_summary_df = pd.DataFrame({
    "category": ["Criteo Clicks","Total Conversions"],
    "count": [criteo_clicks, total_conversions]
})



# Define weighted probabilities for channels and devices based on their attribution likelihood
channel_conversion_probabilities = {
    'Paid Search': 0.30,  # 30% likelihood
    'Affiliate': 0.25,    # 25% likelihood
    'Organic Search': 0.20,  # 20% likelihood
    'Social Media': 0.15, # 15% likelihood
    'Email': 0.10         # 10% likelihood
}

device_conversion_probabilities = {
    'Mobile': 0.50,  # 50% likelihood
    'Desktop': 0.35, # 35% likelihood
    'Tablet': 0.15   # 15% likelihood
}


# Assign channels based on weighted probabilities for attribution
weighted_conversion_channels = list(channel_conversion_probabilities.keys())
channel_conversion_weights = list(channel_conversion_probabilities.values())
df_conv['channel'] = df_conv['conversion'].apply(
    lambda x: random.choices(weighted_conversion_channels, weights=channel_conversion_weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Assign devices based on weighted probabilities for attribution
weighted_conversion_devices = list(device_conversion_probabilities.keys())
device_conversion_weights = list(device_conversion_probabilities.values())
df_conv['device'] = df_conv['conversion'].apply(
    lambda x: random.choices(weighted_conversion_devices, weights=device_conversion_weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Recalculate refined distributions for channels and devices related to attribution
channel_conversion_distribution = df_conv[df_conv['conversion'] == 1]['channel'].value_counts().reset_index()
channel_conversion_distribution.columns = ['channel', 'count']

device_conversion_distribution = df_c[df_c['conversion'] == 1]['device'].value_counts().reset_index()
device_conversion_distribution.columns = ['device', 'count']


# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

# # Plot 1: Attribution
sns.barplot(data=conversion_summary_df, x='category', y='count', ax=axes[0], palette='Blues_d',width=0.6)
# Add count annotations on top of the bars


axes[0].set_title('Conversion Summary', fontsize=14)
axes[0].set_xlabel('Category', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=channel_conversion_distribution, x='channel', y='count', ax=axes[1], palette='Blues_d',width=0.6)
axes[1].set_title('Channel Distribution for Conversion', fontsize=14)
axes[1].set_xlabel('Channel', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].tick_params(axis='x', rotation=30)


sns.barplot(data=device_conversion_distribution, x='device', y='count', ax=axes[2], palette='Blues_d',width=0.6)
axes[2].set_title('Device Distribution for Conversion', fontsize=14)
axes[2].set_xlabel('Device', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('output/conversion_distribution.png')
plt.show()
##