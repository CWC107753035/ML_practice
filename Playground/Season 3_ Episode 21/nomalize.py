import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # You can use MinMaxScaler as well

# Load the original CSV file
df = pd.read_csv('sample_submission.csv')

# Extract 'id' and 'target' columns
id_column = df['id']
target_column = df['target']

# Drop 'id' and 'target' columns from the DataFrame
df = df.drop(columns=['id', 'target'])

# Normalize the remaining continuous features (using StandardScaler in this example)
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

# Create a DataFrame with normalized features
df_normalized = pd.DataFrame(df_normalized, columns=df.columns)

# Combine 'id' and 'target' columns with normalized features
df_normalized.insert(0, 'target', target_column)
df_normalized.insert(0, 'id', id_column)

# Save the modified data to a new CSV file
df_normalized.to_csv('normalized_data.csv', index=False)
