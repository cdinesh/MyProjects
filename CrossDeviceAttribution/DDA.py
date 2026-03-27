import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import markovify
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Criteo Attribution Dataset
DATA_FILE = 'data/criteo_attribution_dataset.tsv.gz'
criteo_data = pd.read_csv(DATA_FILE, sep='\t', compression='gzip')

# Load the Synthetic Dataset
cross_device_data = pd.read_csv('data/synthetic_cross_device_data.csv')
print(cross_device_data.head())
# Display dataset shapes
print(f"Criteo Data Shape: {criteo_data.shape}")
print(f"Synthetic Data Shape: {cross_device_data.shape}")
