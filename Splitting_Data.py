#Code to split the Dataset (80% Train and 20% Test set)

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file with dtype='str'
data = pd.read_csv('Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', dtype='str')

# Separate the features (X) and target (y)
X = data.iloc[:, :-1]  # Exclude the last column (Label)
y = data.iloc[:, -1]   # Last column (Label) as the target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Concatenate X and y back together
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save train and test data as CSV files
train_data.to_csv('train80_data.csv', index=False)
test_data.to_csv('test20_data.csv', index=False)