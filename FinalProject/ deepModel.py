import pandas as pd
import numpy as np
from keras.src.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data = pd.read_csv('model.csv')
unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
data.drop(unnamed_cols, axis=1, inplace=True)

# Split data into features and labels
X = data.drop(['Label', 'Equipment', 'Start'], axis=1)
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

# # Calculate class weights
# class_counts = y.value_counts()
# class_weights = {0: len(y) / (2 * class_counts[0]), 1: len(y) / (2 * class_counts[1])}

# Initialize metrics
total_cm = np.zeros((2, 2))
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0

misclassified_samples = []

# Initialize a dictionary to store cumulative results for each feature
feature_results_dict = {feature: {'Total': 0, 'Correct': 0, 'Incorrect': 0} for feature in binary_features}
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

    # Normalize the features for train and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the deep learning model
    model1 = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # model2 = Sequential([
    #     Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    # # Build the deep learning model
    # model3 = Sequential([
    #     Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # Compile the model
    model1.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    model1.fit(X_train_scaled, y_train, epochs=40, batch_size=10)

    # Define a custom threshold
    custom_threshold = 0.6  # Adjust based on your requirements

    # Predict on test set with custom threshold
    y_pred_proba = model1.predict(X_test_scaled)[:, 0]
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
    {'Feature': feature, 'Total': result['Total'], 'Correct': result['Correct'], 'Incorrect': result['Incorrect'], 'Incorrect_to_Total_Ratio(%)': f"{(result['Incorrect'] / result['Total']) * 100:.2f}%" if result['Total'] > 0 else "0%"
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

# Print misclassified samples
misclassified_df = data.loc[misclassified_samples]
print("\nMisclassified Samples:")
print(misclassified_df)

# Display prediction results for each feature
print("\nFeature Prediction Results:")
print(feature_results)
