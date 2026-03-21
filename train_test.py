import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("creditcard.csv",na_values="?")


# Train_test_model
x=df.drop("Class",axis=1)
y=df["Class"]

# Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)

# Fit smote to LogisticRegression
model_smote=LogisticRegression(max_iter=1000)
model_smote.fit(x_train_smote,y_train_smote)
y_pred_smote=model_smote.predict(x_test)

# Export pkl
import joblib
joblib.dump(model_smote,"fraud_model.pkl")
column_list=x.columns.tolist()
joblib.dump(column_list,"fraud_feature_columns.pkl")


# Comments

# print(df.shape)
# print(df["Class"].value_counts())
# sns.countplot(x="Class",data=df)
# sns.boxplot(x="Class", y="Amount", data=df)
# plt.show()
# print(y_train.value_counts())
# print(y_test.value_counts())
# print(y_train_smote.value_counts())
# from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_test,y_pred_smote))
# print(classification_report(y_test,y_pred_smote))