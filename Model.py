import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv('model.csv')
unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
data.drop(unnamed_cols, axis=1, inplace=True)

# Split data into features and labels
X = data.drop(['Label'], axis=1)
y = data['Label']

# Define binary features explicitly
binary_features = [
    'Area_Attachments/Implements', 'Area_Boom', 'Area_Brake System',
    'Area_Cabin', 'Area_Electrical Systems', 'Area_Engine',
    'Area_Frame/Chassis', 'Area_General', 'Area_Hydraulic System',
    'Area_Lubrication System', 'Area_OBJECT PART', 'Area_Pnuematic System',
    'Area_Power Train', 'Area_Steering System', 'Area_Suspension System',
    'Area_Wheels'
]


# Calculate class weights
class_counts = y.value_counts()
class_weights = len(y) / (2 * class_counts)

# Initialize metrics
total_cm = np.zeros((2, 2))
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0


# Initialize a list to store misclassified samples
misclassified_samples = []


# Initialize a dictionary to store cumulative results for each feature
feature_results_dict = {feature: {'Total': 0, 'Correct': 0, 'Incorrect': 0} for feature in binary_features}

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(data['Equipment'].unique()):
    # Split data into train and test sets
    train_equipments = data['Equipment'].unique()[train_index]
    test_equipments = data['Equipment'].unique()[test_index]
    train = data[data['Equipment'].isin(train_equipments)]
    test = data[data['Equipment'].isin(test_equipments)]

    # Prepare training and testing data
    X_train = train.drop(columns=['Label', 'Equipment', 'Start'])
    y_train = train['Label']
    X_test = test.drop(columns=['Label', 'Equipment', 'Start'])
    y_test = test['Label']


    # Train XGBoost model with class weights
    model = XGBClassifier(scale_pos_weight=class_weights[1])
    model.fit(X_train, y_train)
    # Define a custom threshold
    custom_threshold = 0.8  # You can adjust this value based on your requirements

    # Replace the prediction line with predict_proba and apply threshold
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability of the positive class
    y_pred = (y_pred_proba > custom_threshold).astype(int)

    # Identify misclassified samples
    misclassified_idx = test.index[y_test != y_pred]
    misclassified_samples.extend(misclassified_idx)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    total_cm += cm

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tp = cm[1, 1]
    fp = cm[0, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]

    total_tp += tp
    total_fp += fp
    total_tn += tn
    total_fn += fn

    print("Confusion Matrix:")
    print(cm)

    # Analyze predictions for each binary feature where the value is 1
    for feature in binary_features:
        total = (X_test[feature] == 1).sum()
        correct = sum((y_pred == y_test) & (X_test[feature] == 1))
        incorrect = total - correct
        feature_results_dict[feature]['Total'] += total
        feature_results_dict[feature]['Correct'] += correct
        feature_results_dict[feature]['Incorrect'] += incorrect

# Convert feature results dictionary to DataFrame
feature_results = pd.DataFrame([
    {'Feature': feature, 'Total': result['Total'], 'Correct': result['Correct'], 'Incorrect': result['Incorrect'], 'Incorrect_Ratio(%)': f"{(result['Incorrect'] / result['Total']) * 100:.2f}%" if result['Total'] > 0 else "0%"
    }
    for feature, result in feature_results_dict.items()
])

# Calculate average confusion matrix
avg_cm = total_cm / 5
print("\nAverage Confusion Matrix:")
print(avg_cm)

# Calculate average True Positive Rate (TPR) and False Positive Rate (FPR)
avg_tpr = total_tp / (total_tp + total_fn)
avg_fpr = total_fp / (total_fp + total_tn)
print("\nAverage TPR:", avg_tpr)
print("Average FPR:", avg_fpr)
print(classification_report(y_test, y_pred))
# Retrieve feature importance
feature_importance = model.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# Display misclassified samples
misclassified_df = data.loc[misclassified_samples]
print("\nMisclassified Samples:")
print(misclassified_df)

# Display prediction results for each feature
print("\nFeature Prediction Results:")
print(feature_results)


# Count the number of 1's in the Area_General feature
area_general_count = data['Area_Wheels'].sum()
print("Number of 1's in Area_General feature:", area_general_count)


# # Define parameter grid
# param_grid = {
#     'max_depth': [10, 20, 30, 40],
#     'min_child_weight': [1, 2],
#     'gamma': [0, 0.1, 0.2],
#     'learning_rate': [0.01, 0.1, 0.2, 0.3]
# }
#
# # Initialize the XGBClassifier
# model = XGBClassifier(scale_pos_weight=class_weights[1])
#
# # Set up GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='precision', cv=3)
#
# # Fit grid search
# best_model = grid_search.fit(X_train, y_train)
#
# # Print best parameters
# print("Best parameters:", best_model.best_params_)
#
# # Predict using the best model
# y_pred = best_model.predict(X_test)
#
# # Print new classification report
# print(classification_report(y_test, y_pred))