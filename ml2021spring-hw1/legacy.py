import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("train.csv")
train_data.head()
test_data = pd.read_csv("test.csv")
test_data.head()

#-------------------legacy learning
from sklearn.ensemble import RandomForestRegressor

# Extract the target variable
y = train_data["tested_positive3"]

# Define the features
feat = [75,57,42,60,78,43,61,79,40,58,76,41,59,77]
features = train_data.columns[ feat ] 

# Extract the features and handle missing data
X = train_data[features].copy()
X_test = test_data[features].copy()

# Train the model
model = RandomForestRegressor(n_estimators=2000, max_depth=10, random_state=1)
model.fit(X, y)

# Make predictions on the test data
predictions = model.predict(X_test)

# Prepare the submission file
output = pd.DataFrame({'id': test_data.id, 'tested_positive': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
