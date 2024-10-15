# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project - Silver Layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip uninstall -y databricks_helpers
# MAGIC %pip install git+https://github.com/data-derp/databricks_helpers#egg=databricks_helpers

# COMMAND ----------

# Dependencies setup
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from databricks_helpers.databricks_helpers import DataDerpDatabricksHelpers

# COMMAND ----------

# Create working directory
helpers = DataDerpDatabricksHelpers(dbutils, 'final_project')

current_user = helpers.current_user()
working_directory = helpers.working_directory()

print(f"Your current working directory is: {working_directory}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data from Bronze Layer
# MAGIC Let's read the parquet files that we created in the Bronze layer!

# COMMAND ----------

def read_parquet(filepath: str) -> DataFrame:
    df = spark.read.parquet(filepath)
    return df
    
average_monthly_rental_prices = read_parquet(f"{working_directory}/output/average_monthly_rental_prices/")
barcelona_neighborhoods = read_parquet(f"{working_directory}/output/barcelona_neighborhoods/")
homes_started = read_parquet(f"{working_directory}/output/homes_started/")
homes_finished = read_parquet(f"{working_directory}/output/homes_finished/")

display(average_monthly_rental_prices)
display(homes_started)
display(homes_finished)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Average Monthly Rental Prices

# COMMAND ----------

# "107" is the code for Barcelona neighborhoods
# "-1" is the annual average price
def create_barcelona_avg_monthly_rental_prices(average_monthly_rental_prices):
    return average_monthly_rental_prices\
        .filter(col('outer_scope') == '107')\
        .filter(col('trimester') == '-1')\
        .withColumn('year', col('year').cast('int'))\
        .withColumn('trimester', col('trimester').cast('int'))\
        .withColumn('amount', col('amount').cast('double'))\
        .join(barcelona_neighborhoods, barcelona_neighborhoods.code == average_monthly_rental_prices.inner_scope)\
        .withColumnRenamed('name', 'neighborhood')\
        .drop('outer_scope', 'inner_scope', 'code')

barcelona_avg_monthly_rental_prices = create_barcelona_avg_monthly_rental_prices(average_monthly_rental_prices)
display(barcelona_avg_monthly_rental_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Barcelona Homes Started

# COMMAND ----------

# "105" is the code for cities in Catalonia
# "8019" is the code for Barcelona
# TODO: Some years are pending the annual aggregated value, we have to calculate it ourselves
def create_barcelona_homes_started(homes_started):
    return homes_started\
        .filter(col('outer_scope') == '105')\
        .filter(col('inner_scope') == '8019')

barcelona_homes_started = create_barcelona_homes_started(homes_started)
display(barcelona_homes_started)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Barcelona Homes Finished

# COMMAND ----------

# "105" is the code for cities in Catalonia
# "8019" is the code for Barcelona
# TODO: Some years are pending the annual aggregated value, we have to calculate it ourselves
def create_barcelona_homes_finished(homes_finished):
    return homes_finished\
        .filter(col('outer_scope') == '105')\
        .filter(col('inner_scope') == '8019')

barcelona_homes_finished = create_barcelona_homes_finished(homes_finished)
display(barcelona_homes_finished)
