# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project - Bronze Layer
# MAGIC
# MAGIC In this notebook we ingest data from [AMB Open Data Catalogue](https://www.amb.cat/en/web/area-metropolitana/dades-obertes/cataleg-opendata).
# MAGIC
# MAGIC The datasets used are:
# MAGIC - [Preu Mitjá de lloguer](https://www.amb.cat/en/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/preu-mitja-de-lloguer/6737819/11692?_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_pageNum=1&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_categoria=habitatge&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_detailBackURL=https%3A%2F%2Fwww.amb.cat%2Fen%2Fweb%2Farea-metropolitana%2Fdades-obertes%2Fcataleg%2Fllistat): Annual evolution of the average monthly rental price in the municipalities of the AMB
# MAGIC - [Habitatges iniciats](https://www.amb.cat/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/habitatges-iniciats/6701983/11692?_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_pageNum=1&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_queryText=obra&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_categoria=habitatge&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_detailBackURL=https%3A%2F%2Fwww.amb.cat%2Fen%2Fweb%2Farea-metropolitana%2Fdades-obertes%2Fcataleg%2Fllistat): Homes with work visas from the College of Apparatus and Technical Architects of Catalonia.
# MAGIC - [Habitatges acabats](https://www.amb.cat/en/web/area-metropolitana/dades-obertes/cataleg/detall/-/dataset/habitatges-acabats/1061026/11692?_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_pageNum=1&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_queryText=obra&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_categoria=habitatge&_DatasetSearchListPortlet_WAR_AMBSearchPortletportlet_detailBackURL=https%3A%2F%2Fwww.amb.cat%2Fen%2Fweb%2Farea-metropolitana%2Fdades-obertes%2Fcataleg%2Fllistat): Data on the evolution of the number of homes completed in the AMB.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip uninstall -y databricks_helpers
# MAGIC %pip install git+https://github.com/data-derp/databricks_helpers#egg=databricks_helpers

# COMMAND ----------

# Dependencies setup
import json
import requests as req
import pandas as pd    

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, IntegerType
from pyspark.sql.functions import arrays_zip, col, explode, map_keys, map_values
from databricks_helpers.databricks_helpers import DataDerpDatabricksHelpers

# COMMAND ----------

# Create working directory
helpers = DataDerpDatabricksHelpers(dbutils, 'final_project')

current_user = helpers.current_user()
working_directory = helpers.working_directory()

print(f"Your current working directory is: {working_directory}")

# COMMAND ----------

## This function CLEARS your current working directory. Only run this if you want a fresh start or if it is the first time you're doing this exercise.
helpers.clean_working_directory()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Average Monthly Rental Price

# COMMAND ----------

def load_json(url):
    response = req.get(url, verify=False)
    average_rent_prices = response.text
    return json.loads(average_rent_prices)

def create_rent_values_dataframe(data):
    schema = StructType([
        StructField('outer_scope', StringType(), True),
        StructField('inner_scope', StringType(), True),
        StructField('year', StringType(), True),
        StructField('trimester', StringType(), True),
        StructField('amount', StringType(), True),
    ])
    return spark.createDataFrame(data['valors'], schema=schema)

average_monthly_rental_prices = create_rent_values_dataframe(
    load_json(
        'https://iermbdb.uab.cat/datasets2/index.php?token=AGEF894MGIE0220GOLLEOF&id_ind=1660&type=json'
    )
)

display(average_monthly_rental_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Barcelona Neighborhoods

# COMMAND ----------

def create_barcelona_neighborhoods_dataframe(data):
    return spark.createDataFrame([(data['Àmbits'])])\
        .select(map_keys('107').alias('key'), map_values('107').alias('value'))\
        .withColumn('zip', arrays_zip('key', 'value'))\
        .withColumn('zip', explode('zip'))\
        .withColumn('code', col('zip.key'))\
        .withColumn('name', col('zip.value'))\
        .drop('key', 'value', 'zip')

barcelona_neighborhoods = create_barcelona_neighborhoods_dataframe(
    load_json(
        'https://iermbdb.uab.cat/datasets2/index.php?token=AGEF894MGIE0220GOLLEOF&id_ind=1660&type=json'
    )
)
display(barcelona_neighborhoods)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Homes Started

# COMMAND ----------

def create_homes_started_dataframe(data):
    schema = StructType([
        StructField('outer_scope', StringType(), True),
        StructField('inner_scope', StringType(), True),
        StructField('year', StringType(), True),
        StructField('trimester', StringType(), True),
        StructField('quantity', StringType(), True),
    ])
    return spark.createDataFrame(data['valors'], schema=schema)

homes_started = create_homes_started_dataframe(
    load_json('https://iermbdb.uab.cat/datasets2/index.php?token=AGEF894MGIE0220GOLLEOF&id_ind=1330&type=json')
)

display(homes_started)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Homes `Finished`

# COMMAND ----------

def create_homes_finished_dataframe(data):
    schema = StructType([
        StructField('outer_scope', StringType(), True),
        StructField('inner_scope', StringType(), True),
        StructField('year', StringType(), True),
        StructField('trimester', StringType(), True),
        StructField('quantity', StringType(), True),
    ])
    return spark.createDataFrame(data['valors'], schema=schema)

homes_finished = create_homes_finished_dataframe(
    load_json('https://iermbdb.uab.cat/datasets2/index.php?token=AGEF894MGIE0220GOLLEOF&id_ind=1331&type=json')
)

display(homes_finished)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Parquet

# COMMAND ----------

def write(name: str, input_df: DataFrame):
    out_dir = f"{working_directory}/output/{name}"
    mode_name = 'overwrite'
    input_df\
        .write\
        .mode(mode_name)\
        .parquet(out_dir)
    
write('average_monthly_rental_prices', average_monthly_rental_prices)
write('barcelona_neighborhoods', barcelona_neighborhoods)
write('homes_started', homes_started)
write('homes_finished', homes_finished)

# COMMAND ----------

dbutils.fs.ls(f"{working_directory}/output/")
