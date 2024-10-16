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
from pyspark.sql.functions import col, sum, count, avg

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
spain_consumer_index = read_parquet(f"{working_directory}/output/spain_consumer_index")

display(average_monthly_rental_prices)
display(homes_started)
display(homes_finished)
display(spain_consumer_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Average Monthly Rental Prices

# COMMAND ----------

CATALONIA_CITIES_CODE = '105'
BARCELONA_CITY_CODE = '8019'
ANNUAL_INTERVAL = '-1'

def create_barcelona_avg_monthly_rental_prices(average_monthly_rental_prices):
    return average_monthly_rental_prices\
        .filter(col('area') == CATALONIA_CITIES_CODE)\
        .filter(col('sub_area') == BARCELONA_CITY_CODE)\
        .filter(col('quarter') == ANNUAL_INTERVAL)\
        .withColumn('year', col('year').cast('int'))\
        .withColumn('amount', col('amount').cast('double'))\
        .drop('area', 'sub_area', 'quarter')\
        .sort('year')

barcelona_avg_monthly_rental_prices = create_barcelona_avg_monthly_rental_prices(average_monthly_rental_prices)
display(barcelona_avg_monthly_rental_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Barcelona Homes Started

# COMMAND ----------


def create_barcelona_homes_dataset(dataset):
    homes_in_barcelona = dataset\
        .filter(col('area') == CATALONIA_CITIES_CODE)\
        .filter(col('sub_area') == BARCELONA_CITY_CODE)\
        .withColumn('year', col('year').cast('int'))\
        .withColumn('quantity', col('quantity').cast('int'))

    existing_yearly_agg_entries = homes_in_barcelona\
        .filter(col('quarter') == ANNUAL_INTERVAL)\
        .drop('area', 'sub_area', 'quarter')

    # There are some Years in the dataset that doesnt have a record with ANNUAL_INTERVAL (-1), so we calculate that record based on the quarters
    quarter_agg_entries = homes_in_barcelona\
        .filter(col('quarter') != ANNUAL_INTERVAL)
    
    computed_yearly_agg_entries = quarter_agg_entries\
        .groupBy('year')\
        .agg(sum('quantity').alias('quantity'))

    return existing_yearly_agg_entries\
        .union(computed_yearly_agg_entries)\
        .dropDuplicates()\
        .sort('year')

# COMMAND ----------


barcelona_homes_started = create_barcelona_homes_dataset(homes_started)
display(barcelona_homes_started)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Barcelona Homes Finished

# COMMAND ----------


barcelona_homes_finished = create_barcelona_homes_dataset(homes_finished)
display(barcelona_homes_finished)

# COMMAND ----------

# MAGIC %md
# MAGIC # Process Spain Consumer Index

# COMMAND ----------

def create_spain_consumer_index(dataset):
    return dataset\
        .withColumn('year', col('year').cast('int'))\
        .withColumn('month', col('month').cast('int'))\
        .withColumn('value', col('value').cast('double'))\
        .sort('year', 'month')

spain_consumer_index = create_spain_consumer_index(spain_consumer_index)
display(spain_consumer_index)

# COMMAND ----------

# MAGIC %md
# MAGIC # Write to Parquet

# COMMAND ----------

def write(name: str, input_df: DataFrame):
    out_dir = f"{working_directory}/output/{name}"
    mode_name = 'overwrite'
    input_df\
        .write\
        .mode(mode_name)\
        .parquet(out_dir)

write('barcelona_avg_monthly_rental_prices', barcelona_avg_monthly_rental_prices)
write('barcelona_homes_started', barcelona_homes_started)
write('barcelona_homes_finished', barcelona_homes_finished)
write('spain_consumer_index', spain_consumer_index)

# COMMAND ----------

dbutils.fs.ls(f"{working_directory}/output/")
