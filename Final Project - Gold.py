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
from pyspark.sql.functions import col, sum, avg

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

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
spain_consumer_index = read_parquet(f"{working_directory}/output/spain_consumer_index/")

display(barcelona_avg_monthly_rental_prices)
display(barcelona_homes_started)
display(barcelona_homes_finished)
display(spain_consumer_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plots

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 1: Is Barcelona building more or less houses?
# MAGIC In order to answer this question, we are going to use the `barcelona_started` and `barcelona_finished` datasets and compare them using a Bar Plot where we will be able to identify the gaps throughout the years

# COMMAND ----------

homes_started = barcelona_homes_started.toPandas()
homes_finished = barcelona_homes_finished.toPandas()
width = 0.4
average_home_started_before_crisis = barcelona_homes_started.filter(col('year') < 2008).agg(avg(col('quantity'))).head()[0]
average_home_started_after_crisis = barcelona_homes_started.filter(col('year') > 2008).agg(avg(col('quantity'))).head()[0]

plt.figure(figsize=(15, 5))  # width:20, height:3
plt.bar(homes_started.year, homes_started.quantity, width, label='Started', color='#003f5c')
plt.bar(homes_finished.year + width, homes_finished.quantity, width, label='Finished', color='#ff6361')
plt.axhline(y=average_home_started_before_crisis, color='black', linestyle='dotted', label='Average Homes Started Before 2008')
plt.axhline(y=average_home_started_after_crisis, color='black', linestyle='dashed', label='Average Homes Started After 2008')

plt.xlabel("Year") 
plt.ylabel("Quantity of homes built") 
plt.title("Number of homes built per year") 
plt.legend()
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 2: Are there any relationship between the amount of houses built and the rent price?
# MAGIC In order to answer this question, we first plot a visualization to see the evolution of rental price throughout the years.

# COMMAND ----------

rental_prices = barcelona_avg_monthly_rental_prices.groupBy('year')\
        .agg(avg('amount').alias('amount'))\
        .sort(col('year'))\
        .toPandas()

plt.figure(figsize=(15, 5))
plt.plot(rental_prices.year, rental_prices.amount, label='Rental Price')
plt.xticks(range(2005,2025, 3))

plt.xlabel("Year")
plt.ylabel("Price (€)") 
plt.title("Average Monthly Price (€)")
plt.ylim(ymin=0)
plt.legend()
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC To be able to do a correlation between rental_prices and homes_finished we should first filter the dataframes by the same time interval or the correlation would not be realistic.

# COMMAND ----------

homes_finished_filtered_by_common_years = barcelona_homes_finished.filter(col('year') >= 2005).toPandas()

plt.figure(figsize=(15, 5))
plt.plot(rental_prices.year, rental_prices.amount, label='Rental Price')
plt.plot(homes_finished_filtered_by_common_years.year, homes_finished_filtered_by_common_years.quantity, label='Homes Finished')
plt.xticks(range(2005,2025, 3))

plt.xlabel("Year")
plt.ylabel("Price (€)") 
plt.title("Average Monthly Price (€)")
plt.ylim(ymin=0)
plt.legend()
plt.show() 


# COMMAND ----------

consumer_index = spain_consumer_index\
    .filter(col('month') == 12)\
    .filter(col('year') >= 2005)\
    .sort(col('year'))\
    .toPandas()

display(consumer_index)

plt.figure(figsize=(15, 5))
plt.plot(consumer_index.year, consumer_index.value, label='Consumer Index')
plt.xlabel("Year") 
plt.ylabel("Consumer Index") 
plt.title("Spain Consumer Index") 
plt.legend()
plt.show() 

# COMMAND ----------

rental_prices.amount.corr(consumer_index.value)
