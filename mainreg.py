import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
import seaborn as sns
import statsmodels.api as sm

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

#EDA
high_nan_value = []
for i,j in df.isnull().sum().items():
    if j>0:
        # print(f"{i} -------> {round((j/df.shape[0])*100,2)}")
        if j>50:
            high_nan_value.append(i)

df.drop(high_nan_value,axis=1,inplace=True)

# Missing value         
for i,j in df.isnull().sum().items():
    if j>0:
        if df[i].dtype == object:
            df[i].fillna(df[i].mode()[0],inplace=True)
        else:
            df[i].fillna(df[i].median(),inplace=True)
            
#outlier
# Plot boxplots for numerical columns
num_columns = [i for i in df.columns if df[i].dtype != object]

# Apply outlier removal using IQR for columns in 'yo'
for i in num_columns:
    iqr = df[i].quantile(0.75) - df[i].quantile(0.25)
    lower_boundary = df[i].quantile(0.25) - (iqr * 1.5)
    upper_boundary = df[i].quantile(0.75) + (iqr * 1.5)
        
        # Clip outliers
    df[i] = df[i].clip(lower=lower_boundary, upper=upper_boundary)
    
#Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for i,j in df.isnull().sum().items():
    if df[i].dtype == object:
        df[i] = label_encoder.fit_transform(df[i])
        
X = df.drop('SalePrice', axis = 1)
y = df['SalePrice']

from sklearn.feature_selection import mutual_info_regression
import pandas as pd

# Assuming X is your feature matrix and y is your target variable for regression

# Perform mutual information feature selection
mutual_info = mutual_info_regression(X, y)
mutual_data = pd.Series(mutual_info, index=X.columns)
mutual_data.sort_values(ascending=False, inplace=True)


count = 1
main_feature = []
for i,j in mutual_data.items():
    if count != 10:
        main_feature.append(i)
        count+=1
    else:
        break
    
X_df = df[main_feature]
y = df['SalePrice']


from sklearn.model_selection import train_test_split

# Assuming X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import make_scorer,r2_score

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train, y_train)

    predicted_qualities = lr.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
 
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    predictions = lr.predict(X_train)
    signature = infer_signature(X_train, predictions)

    ## For Remote server only(DAGShub)

    remote_server_uri="https://dagshub.com/shivamrana12321/adsad.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            lr, "model", registered_model_name="ElasticnetWineModel"
        )
    else:
        mlflow.sklearn.log_model(lr, "model")
