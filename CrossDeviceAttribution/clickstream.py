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

DATA_FILE= 'data/criteo_attribution_dataset.tsv.gz'
data = pd.read_csv(DATA_FILE, sep='\t', compression='gzip')


# Create Synthetic dataset
def convert_seconds_to_date(seconds):
    # Convert seconds to a datetime object
    date = datetime.utcfromtimestamp(seconds)
    # Replace the year with 2024
    date = date.replace(year=2024)
    return date.strftime('%m/%d/%Y %H:%M:%S')


def create_action(row):
    actions = []

    if row['conversion'] == 1:
        actions.append('conversion')
    if row['attribution'] == 1:
        actions.append('attribution')
    if row['click'] == 1:
        actions.append('click')

    return ', '.join(actions)


criteo_data = data
# Select 100 random unique user IDs
random_user_ids = np.random.choice(criteo_data['uid'].unique(), size=50000, replace=False)
criteo_data = criteo_data[criteo_data['uid'].isin(random_user_ids)]

# Flatten the timestamp list for merging user_id and timestamp
criteo_flattened = criteo_data.explode('timestamp').reset_index(drop=True)
random_seed = 42
# Parameters for synthetic data generation
channels = ['Paid Search', 'Affiliate', 'Organic Search', 'Social Media', 'Email']
devices = ['Mobile', 'Desktop', 'Tablet']
actions = ['Click', 'View', 'Conversion']
conversion_probability = {'Paid Search': 0.15, 'Affiliate': 0.12, 'Organic Search': 0.10, 'Social Media': 0.08,
                          'Email': 0.05}

# Assign channels randomly
# data['channel'] = random.choices(channels, k=len(data))

# Simulate conversions based on channel probabilities
# data['conversion'] = data['channel'].apply(lambda x: 1 if random.random() < conversion_probabilities[x] else 0)

# Recalculate conversion rates by channel
# channel_conversion = data.groupby('channel')['conversion'].mean().reset_index()
# channel_conversion.rename(columns={'conversion': 'conversion_rate'}, inplace=True)

# Generate synthetic data based on Criteo user_id and timestamp
synthetic_data = []
for _, row in criteo_flattened.iterrows():
    uid = row['uid'].astype(int)
    timestamp = row['timestamp'].astype(int)
    campaign = row['campaign'].astype(int)
    conversion = row['conversion'].astype(int)
    conversion_timestamp = convert_seconds_to_date(row['conversion_timestamp'].astype(int))
    conversion_id = row['conversion_id'].astype(int)
    attribution = row['attribution'].astype(int)
    click = row['click'].astype(int)
    click_pos = row['click_pos'].astype(int)
    click_nb = row['click_nb'].astype(int)
    cost = row['cost'].astype(float).round(8)
    time_since_last_click = convert_seconds_to_date(row['time_since_last_click'].astype(int))
    session_id = f"S-{uid}-{timestamp.astype(int)}"
    # channel = random.choices(channels, weights=conversion_probability)[10]
    channel = random.choice(channels)
    device = random.choice(devices)
    action = create_action(row)
    revenue = round(random.uniform(50, 300), 2) if conversion else 0
    utm_source = random.choice(['google', 'facebook', 'newsletter', 'linkedin', 'bing'])
    utm_medium = random.choice(['cpc', 'social', 'email', 'organic', 'affiliate'])
    utm_campaign = random.choice(['winter_sale', 'retargeting', 'newsletter_promo'])

    synthetic_data.append({
        'uid': uid,
        'daytime': convert_seconds_to_date(timestamp),
        'campaign': campaign,
        'conversion': conversion,
        'conversion_timestamp': conversion_timestamp,
        'conversion_id': conversion_id,
        'attribution': attribution,
        'click': click,
        'click_pos': click_pos,
        'click_nb': click_nb,
        'cost': cost,
        'time_since_last_click': time_since_last_click,
        'session_id': session_id,
        'channel': channel,
        'device': device,
        'action': action,
        'revenue': revenue,
        'utm_source': utm_source,
        'utm_medium': utm_medium,
        'utm_campaign': utm_campaign
    })

# Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data)
# Assign channels based on weighted probabilities for rows with conversion = 1
weighted_channels = list(conversion_probability.keys())
weights = list(conversion_probability.values())

synthetic_df['channel'] = synthetic_df['conversion'].apply(
    lambda x: random.choices(weighted_channels, weights=weights, k=1)[0] if x == 1 else 'Unassigned'
)

# Save synthetic data

synthetic_df.to_csv('data/synthetic_cross_device_data.csv', index=False)