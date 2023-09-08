import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Load the original CSV file
df = pd.read_csv('normalized_data.csv')

# Assuming 'target' is your target variable
X = df.drop(columns=['target', 'id'])  # Drop 'id' and 'target' columns
y = df['target']

# Normalize continuous features (using StandardScaler in this example)

# Train a Random Forest model to calculate feature importances
rf = RandomForestRegressor()
rf.fit(X, y)

# Access feature importances
feature_importances = rf.feature_importances_

# Calculate the median of feature importances
median_importance = pd.Series(feature_importances).median()

# Use the median as the threshold for feature selection
selector = SelectFromModel(rf, threshold=median_importance)
selector.fit(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support()

# Create a copy of the DataFrame
df_modified = df.copy()

# Zero out the values in the columns for non-selected features
zero_columns = X.columns[~selected_feature_indices]
df_modified[zero_columns] = 0

# Save the modified data to a new CSV file
df_modified.to_csv('modified_data.csv', index=False)
