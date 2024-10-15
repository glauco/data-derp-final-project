# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project - Gold Layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip uninstall -y databricks_helpers
# MAGIC %pip install git+https://github.com/data-derp/databricks_helpers#egg=databricks_helpers

# COMMAND ----------

# Dependencies setup
from pyspark.sql import DataFrame

from databricks_helpers.databricks_helpers import DataDerpDatabricksHelpers

# COMMAND ----------

# Create working directory
helpers = DataDerpDatabricksHelpers(dbutils, 'final_project')

current_user = helpers.current_user()
working_directory = helpers.working_directory()

print(f"Your current working directory is: {working_directory}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data from Silver Layer
# MAGIC Let's read the parquet files that we created in the Silver layer!

# COMMAND ----------

def read_parquet(filepath: str) -> DataFrame:
    df = spark.read.parquet(filepath)
    return df
    
barcelona_avg_monthly_rental_prices = read_parquet(f"{working_directory}/output/barcelona_avg_monthly_rental_prices/")
barcelona_homes_started = read_parquet(f"{working_directory}/output/barcelona_homes_started/")
barcelona_homes_finished = read_parquet(f"{working_directory}/output/barcelona_homes_finished/")

display(barcelona_avg_monthly_rental_prices)
display(barcelona_homes_started)
display(barcelona_homes_finished)
