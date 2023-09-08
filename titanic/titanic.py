import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("train.csv")
train_data.head()
test_data = pd.read_csv("test.csv")
test_data.head()

# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)

# print("% of women who survived:", rate_women)

# man = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_man = sum(man)/len(man)

# print("% of men who survived:", rate_man)



# def checkAllSurvived(train_data, category):
#     elements = train_data[category].unique()

#     for element in elements:
#         item = train_data.loc[train_data[category] == element]["Survived"]
#         if len(item) > 0:
#             rate_item = sum(item) / len(item)
#             print(f"Survival rate for {element} in {category}: {rate_item}")

# checkAllSurvived(train_data , "Pclass" )
# checkAllSurvived(train_data , "Sex")
# checkAllSurvived(train_data , "Embarked")
# checkAllSurvived(train_data , "SibSp")
# checkAllSurvived(train_data , "Parch")




#-------------------legacy learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Extract the target variable
y = train_data["Survived"]

# Define the features
features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]

# Extract the features and handle missing data
X = train_data[features].copy()
X_test = test_data[features].copy()

# Define imputers for numerical and categorical features
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values in numerical features
X['Age'] = numerical_imputer.fit_transform(X[['Age']])
X_test['Age'] = numerical_imputer.transform(X_test[['Age']])

# One-hot encode categorical features
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=2000, max_depth=10, random_state=1)
model.fit(X, y)

# Make predictions on the test data
predictions = model.predict(X_test)

# Prepare the submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
