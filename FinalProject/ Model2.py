import numpy as np
import shap
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('model.csv')
unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
data.drop(unnamed_cols, axis=1, inplace=True)

# Initialize empty DataFrames for train and test sets
train = pd.DataFrame()
test = pd.DataFrame()

# Split data
for equipment in data['Equipment'].unique():
    equipment_data = data[data['Equipment'] == equipment]
    if len(equipment_data) > 3:
        test_subset = equipment_data.iloc[-3:]  # Take the last 3 rows for the test set
        train_subset = equipment_data.iloc[:-3]  # Take the rest for the train set
    else:
        train_subset = equipment_data  # If there aren't enough rows, keep all in train

    train = pd.concat([train, train_subset])
    test = pd.concat([test, test_subset])

# Ensure model-related operations are outside the loop
X_train = train.drop(['Label', 'Equipment', 'Start'], axis=1)  # Assuming 'Equipment' and 'Start' are not needed for modeling
y_train = train['Label']
X_test = test.drop(['Label', 'Equipment', 'Start'], axis=1)
y_test = test['Label']

# Calculate class weights
class_counts = y_train.value_counts()
class_weights = len(y_train) / (2 * class_counts)

# Initialize and fit the model
model = XGBClassifier(scale_pos_weight=class_weights[1], enable_categorical=True)
model.fit(X_train, y_train)

# Create the SHAP Explainer object
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Get the index of the feature with the highest SHAP value for each prediction
max_shap_indices = np.argmax(shap_values.values, axis=1)

# Create a DataFrame to show the result
results_df = pd.DataFrame({
    'Identifier': test['Equipment'].astype(str) + '_' + test['Start'].astype(str),
    'Predicted Probability': model.predict_proba(X_test)[:, 1],
    'Most Influential Feature': X_test.columns[max_shap_indices]
})

print(results_df)

# Predict on the test set and get probabilities for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Adding probabilities back to the test DataFrame
test['Damage Probability'] = y_pred_proba

# Optionally, include other identifiers from the test data:
test['Identifiers'] = test['Equipment'].astype(str) + '_' + test['Start'].astype(str)

# Display the result
print(test[['Identifiers', 'Damage Probability']])
