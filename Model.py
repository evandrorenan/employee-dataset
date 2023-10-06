# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# COMMAND ----------

df_pd = pd.read_parquet("/Workspace/Users/renan3006@hotmail.com/corr_data1.parquet")

# COMMAND ----------

display(df_pd)

# COMMAND ----------

X = df_pd.drop("LeaveOrNot", axis=1)
y = df_pd["LeaveOrNot"]

# COMMAND ----------

display(X)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

model = RandomForestClassifier()
model.fit(X_train, y_train)

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

accuracy = accuracy_score(y_test, y_pred)
display(accuracy)

# COMMAND ----------

report = classification_report(y_test, y_pred)
print(report)

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)

# COMMAND ----------

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predito pelo Modelo')
plt.ylabel('Real')
plt.title('Matriz de confusão')
plt.show()

# COMMAND ----------

!pip install shap

# COMMAND ----------


import shap

# COMMAND ----------

#explainer SHAP
explainer = shap.Explainer(model, X_train)

# Calcular os valores SHAP para um único ponto de dados (por exemplo, o primeiro ponto de dados no conjunto de testes)
shap_values = explainer.shap_values(X_test)

# Plotar os valores SHAP
shap.summary_plot(shap_values, X_test)


# COMMAND ----------

#CONFIA
