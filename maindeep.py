import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import mlflow
from mlflow.models import infer_signature
import mlflow.tensorflow
import warnings
import bentoml

warnings.filterwarnings('ignore')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train models with MLflow')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for the neural network')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the neural network')
args = parser.parse_args()

# Load your data
df = pd.read_csv('train.csv')

# EDA
high_nan_value = []
for i, j in df.isnull().sum().items():
    if j > 0:
        if j > 50:
            high_nan_value.append(i)

df.drop(high_nan_value, axis=1, inplace=True)

# Missing value imputation
for i, j in df.isnull().sum().items():
    if j > 0:
        if df[i].dtype == object:
            df[i].fillna(df[i].mode()[0], inplace=True)
        else:
            df[i].fillna(df[i].median(), inplace=True)

# Outlier removal
num_columns = [i for i in df.columns if df[i].dtype != object]

for i in num_columns:
    iqr = df[i].quantile(0.75) - df[i].quantile(0.25)
    lower_boundary = df[i].quantile(0.25) - (iqr * 1.5)
    upper_boundary = df[i].quantile(0.75) + (iqr * 1.5)
    df[i] = df[i].clip(lower=lower_boundary, upper=upper_boundary)

# Encoding
label_encoder = LabelEncoder()
for i, j in df.isnull().sum().items():
    if df[i].dtype == object:
        df[i] = label_encoder.fit_transform(df[i])

# Feature selection
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

mutual_info = mutual_info_regression(X, y)
mutual_data = pd.Series(mutual_info, index=X.columns)
mutual_data.sort_values(ascending=False, inplace=True)

count = 1
main_feature = []
for i, j in mutual_data.items():
    if count != 10:
        main_feature.append(i)
        count += 1
    else:
        break

X_df = df[main_feature]
y = df['SalePrice']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regression task, so linear activation

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)

# bentoml.keras.save_model("keras_resnet50", model)

# Log metrics and model in MLflow
with mlflow.start_run():
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)

    mlflow.log_metric("loss", loss)
    mlflow.log_metric("mae", mae)

    # Log the Keras model
    mlflow.tensorflow.log_model(model, "model")
    
    