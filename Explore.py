# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pyspark.sql.functions import when, col
import seaborn as sns


# COMMAND ----------

df = spark.read.table("default.employee")
display(df)

# COMMAND ----------

conditions = [
    (col("Education") == "Bachelors", 1),
    (col("Education") == "Masters", 2),
    (col("Education") == "PHD", 3)
]

df = df.withColumn("Education", when(conditions[0][0], conditions[0][1])
                                  .when(conditions[1][0], conditions[1][1])
                                  .when(conditions[2][0], conditions[2][1])
                                  .otherwise(0))
display(df)

# COMMAND ----------

conditions = [
    (col("City") == "Bangalore", 1),
    (col("City") == "Pune", 2),
    (col("City") == "New Delhi", 3)
]

df = df.withColumn("City", when(conditions[0][0], conditions[0][1])
                                  .when(conditions[1][0], conditions[1][1])
                                  .when(conditions[2][0], conditions[2][1])
                                  .otherwise(0))
display(df)

# COMMAND ----------

conditions = [
    (col("Gender") == "Male", 1),
]

df = df.withColumn("Gender", when(conditions[0][0], conditions[0][1])
                                  .otherwise(0))
display(df)

# COMMAND ----------

conditions = [
    (col("EverBenched") == "Yes", 1),
]

df = df.withColumn("EverBenched", when(conditions[0][0], conditions[0][1])
                                  .otherwise(0))
display(df)

# COMMAND ----------

df_pd_file = df.toPandas()
df_pd_file.to_parquet("./corr_data1.parquet")

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

df_pd.describe()

# COMMAND ----------

df_pd.groupby(["LeaveOrNot", "Gender"])["EverBenched"].value_counts(normalize=True)

# COMMAND ----------

## Criar o gráfico
#KDE
sns.kdeplot(
    data=df_pd,
    x="EverBenched",  # Substitua "variavel1" pelo nome da primeira variável
    y="Education",  # Substitua "variavel2" pelo nome da segunda variável
    hue="LeaveOrNot",  # Colorir por categoria de output
    fill=True,
    alpha=0.5,  # Ajustar a transparência
    thresh=0.1  # Ajustar o limiar de contorno (opcional)
)

# COMMAND ----------

df_pd["Gender"].value_counts()

# COMMAND ----------

df_pd["LeaveOrNot"].value_counts(normalize=True)
# 1 = Leave

# COMMAND ----------

df_pd.groupby("LeaveOrNot")["Gender"].value_counts(normalize=True)

# COMMAND ----------

display(df_pd)

# COMMAND ----------

df_pd.corr(method="spearman")

# COMMAND ----------

pd.read_parquet("/Workspace/Users/renan3006@hotmail.com/corr_data1.parquet")

# COMMAND ----------


