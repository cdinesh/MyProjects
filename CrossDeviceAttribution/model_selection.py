import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'data/processed_synthetic_data_with_features.csv'
data = pd.read_csv(file_path)

# Append "conversion" as a final absorbing state in each user journey
data['channel'] = data['channel'].fillna('conversion')  # Fill missing with 'conversion'
#
# Build Transition Matrix
def build_transition_matrix(data, channel_col='channel', conversion_col='conversion'):
    states = list(data[channel_col].unique()) + ['conversion']
    transition_matrix = pd.DataFrame(0, index=states, columns=states)

    for uid, user_data in data.groupby('uid'):
        channels = list(user_data[channel_col]) + (['conversion'] if user_data[conversion_col].iloc[-1] == 1 else [])
        for i in range(len(channels) - 1):
            current_state = channels[i]
            next_state = channels[i + 1]
            transition_matrix.loc[current_state, next_state] += 1

    # Normalize rows to probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
    for row in transition_matrix.index:
        if transition_matrix.loc[row].sum() == 0:
            transition_matrix.loc[row, row] = 1  # Self-loop for isolated states
    return transition_matrix

# Generate transition matrix
transition_matrix = build_transition_matrix(data)

# Optimize Simulate Conversion Probability
def simulate_conversion_probability_vectorized(transition_matrix, start_states, end_state='conversion'):
    state_probabilities = np.zeros(len(transition_matrix))
    state_probabilities[[transition_matrix.index.get_loc(s) for s in start_states]] = 1 / len(start_states)

    for _ in range(100):  # Fewer iterations for optimization
        state_probabilities = state_probabilities @ transition_matrix.values
        if np.allclose(state_probabilities[transition_matrix.columns.get_loc(end_state)], 1.0):
            break

    return state_probabilities[transition_matrix.columns.get_loc(end_state)]

# Calculate Removal Effect (Vectorized)
def calculate_removal_effect_vectorized(transition_matrix, channel_to_remove, start_states):
    temp_matrix = transition_matrix.copy()
    temp_matrix.loc[channel_to_remove] = 0
    temp_matrix = temp_matrix.div(temp_matrix.sum(axis=1), axis=0).fillna(0)
    for row in temp_matrix.index:
        if temp_matrix.loc[row].sum() == 0:
            temp_matrix.loc[row, row] = 1
    return simulate_conversion_probability_vectorized(temp_matrix, start_states)

# Perform Markov Chain Attribution (Vectorized)
def calculate_attribution_vectorized(transition_matrix, start_states):
    base_conversion_prob = simulate_conversion_probability_vectorized(transition_matrix, start_states)
    attribution = {}
    for channel in transition_matrix.index:
        if channel == 'conversion':
            continue
        removed_prob = calculate_removal_effect_vectorized(transition_matrix, channel, start_states)
        attribution[channel] = base_conversion_prob - removed_prob
    return pd.Series(attribution) / sum(attribution.values())

# Prepare start states (first channels)
start_states = data.groupby('uid')['channel'].first().unique()

# Calculate attribution
attribution_results = calculate_attribution_vectorized(transition_matrix, start_states)

# # Assuming 'transition_matrix' is already generated
# plt.figure(figsize=(10, 8))
# sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# plt.title("Markov Chain Transition Matrix", fontsize=16)
# plt.xlabel("Next State", fontsize=12)
# plt.ylabel("Current State", fontsize=12)
# plt.xticks(rotation=45, fontsize=10)
# plt.yticks(rotation=0, fontsize=10)
# plt.tight_layout()
# plt.savefig("output/transition_matrix_heatmap.png")  # Save the heatmap as an image
# plt.show()
#
# # Visualize the attribution results with further customization
#
# plt.figure(figsize=(12, 6))
# attribution_results.sort_values(ascending=False).plot(kind='bar', color='dodgerblue', edgecolor='black', alpha=0.8)
# plt.title("Channel Attribution Using Markov Chain Model", fontsize=16, fontweight='bold')
# plt.xlabel("Channel", fontsize=14)
# plt.ylabel("Attribution Score (%)", fontsize=14)
# plt.xticks(rotation=45, fontsize=12)
# plt.grid(alpha=0.4, linestyle='--')
# plt.tight_layout()
# plt.savefig("output/markov_chain_attribution.png")  # Save the visualization as an image
# plt.show()

# Logistic Regression

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score

# Sort data by user and timestamp
data = data.sort_values(by=['uid', 'timestamp'])

# Ensure 'timestamp' is in datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Drop rows with invalid datetime values
data = data.dropna(subset=['timestamp'])

# Sort data by user and timestamp
data = data.sort_values(by=['uid', 'timestamp'])

# Calculate time between interactions in seconds
data['time_between_interactions'] = data.groupby('uid')['timestamp'].diff().dt.total_seconds()

# Fill missing values in 'time_between_interactions' with 0
data['time_between_interactions'] = data['time_between_interactions'].fillna(0)
#
#
#
# Click position
data['click_position'] = data.groupby('uid').cumcount() + 1

# Device ratios
device_counts = data.groupby(['uid', 'device']).size().unstack(fill_value=0)
device_counts['mobile_to_desktop_ratio'] = device_counts.get('Mobile', 0) / (device_counts.get('Desktop', 1) + 1)
data = data.merge(device_counts[['mobile_to_desktop_ratio']], left_on='uid', right_index=True, how='left')

# Fill NaNs in new features
data = data.fillna(0)


# Features and target
features = ['time_between_interactions', 'click_position', 'mobile_to_desktop_ratio']
X = data[features]
y = data['conversion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# Train model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = logistic_model.predict(X_test)
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Assume y_test and y_pred_prob are already defined in the context of the Logistic Regression

# Sample values (replace these with actual values from your dataset)
# y_test = true labels
# y_pred_prob = predicted probabilities

# Step 1: Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])

# Visualize confusion matrix
# fig, ax = plt.subplots(figsize=(6, 6))
# disp.plot(cmap="Blues", ax=ax, colorbar=False)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("output/Logistic Regression Confusion Matrix.png")  # Save the heatmap as an image
# plt.show()


# Step 2: ROC Curve
# fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# roc_auc = auc(fpr, tpr)

# Visualize ROC Curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Random Model")
# plt.title("ROC Curve", fontsize=16)
# plt.xlabel("False Positive Rate", fontsize=12)
# plt.ylabel("True Positive Rate", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("output/ROC Curve.png")  # Save the heatmap as an image
# plt.show()

# K-means clustering.

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
#
# plt.figure(figsize=(8, 4))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
# plt.title("PCS Clustering Results", fontsize=16)
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.colorbar(label="Cluster")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("output/K-Clustering.png")  # Save the heatmap as an image
# plt.show()


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
#
# # Define the model and parameter grid for tuning
# model = LogisticRegression(max_iter=500, random_state=42)
# param_grid = {
#     'penalty': ['l1', 'l2', 'elasticnet', None],
#     'C': [0.1, 1, 10],  # Regularization strength
#     'solver': ['liblinear', 'saga'],  # Solvers for L1, L2, and elasticnet
# }
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=5,  # 5-fold cross-validation
#     verbose=1,
#     n_jobs=-1  # Use all available processors
# )
#
# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
#
# # Extract the best parameters and score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# # Perform cross-validation with the best model
# best_model = grid_search.best_estimator_
# cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
#
# # Output results
# best_params, best_score, cv_scores.mean(), cv_scores.std()

from sklearn.metrics import silhouette_score

# Function to perform "grid search" for optimal number of clusters in K-Means
def tune_kmeans(X, cluster_range):
    scores = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)  # Evaluate clustering using Silhouette Score
        scores.append((k, score))
    return scores

# Define a range of cluster numbers to evaluate
cluster_range = range(2, 11)  # Testing for 2 to 10 clusters
X_Scaled = StandardScaler().fit_transform(X)
# Perform tuning
scores = tune_kmeans(X_Scaled, cluster_range)

# Extract optimal number of clusters
best_k = max(scores, key=lambda x: x[1])[0]

# Visualize Silhouette Scores
k_values, silhouette_scores = zip(*scores)
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
plt.title("Silhouette Scores for Different Numbers of Clusters", fontsize=16)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_tuning_silhouette_scores.png")
plt.show()

# Output best k and silhouette score
best_k, max(silhouette_scores)

