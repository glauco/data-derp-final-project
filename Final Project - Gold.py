# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project - Gold Layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Barcelona Rent Price Evolution (2005 - 2023)
# MAGIC
# MAGIC Since the 1992 Olympic Games, Barcelona has transformed into one of the most visited cities in the world. The event reshaped the city's dynamics, attracting an influx of tourists and immigrants. Alongside this growth the city saw an economic boost, improved infrastructure, and cultural recognition, but it also led to a sharp rise in the cost of living. 
# MAGIC
# MAGIC Today, one of the most pressing concerns for residents is the skyrocketing rent prices.
# MAGIC While tourism is often cited as the primary factor driving up rent costs, our analysis aims to explore the correlation between the number of new households and the rise in rental prices. As residents of the city, we are particularly interested in understanding the local dynamics that shape the current rent market.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip uninstall -y databricks_helpers
# MAGIC %pip install git+https://github.com/data-derp/databricks_helpers#egg=databricks_helpers

# COMMAND ----------

# Dependencies setup
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum, avg, lag
from pyspark.sql.window import Window

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
# MAGIC ## Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 1: Is Barcelona building more or less houses?
# MAGIC In order to answer this question, we are going to use the `barcelona_started` and `barcelona_finished` datasets and compare them using a Bar Plot where we will be able to identify the gaps throughout the years.

# COMMAND ----------

homes_started = barcelona_homes_started.toPandas()
homes_finished = barcelona_homes_finished.toPandas()

avg_homes_finished_before_crisis = barcelona_homes_finished.filter(col('year') < 2007).agg(avg(col('quantity'))).head()[0]
avg_homes_finished_after_crisis = barcelona_homes_finished.filter(col('year') > 2007).agg(avg(col('quantity'))).head()[0]

# General settings
plt.figure(figsize=(15, 5))
plt.xticks(range(1990, 2025, 2))
plt.xlim(xmin=1989, xmax=2024)
plt.ylim(ymin=0, ymax=7000)
plt.xlabel("Year") 
plt.ylabel("Number of homes built") 
plt.title("Number of homes built per year") 

# Data
width = 0.4
plt.bar(homes_started.year, homes_started.quantity, width, label='Number of house being built', color='#003f5c')
plt.bar(homes_finished.year + width, homes_finished.quantity, width, label='Number of houses built', color='#ff6361')

# Annotations
plt.annotate('2007-2008 Financial Crisis',
            xy=(2007, 5000), xycoords='data',
            xytext=(0.60, 0.90), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.annotate('2020-2022 Pandemic',
            xy=(2020, 2000), xycoords='data',
            xytext=(0.85, 0.65), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.axhline(y=avg_homes_finished_before_crisis, label='Average number of homes built before 2007', color='#000000', linestyle='dotted')
plt.axhline(y=avg_homes_finished_after_crisis, label='Average number of homes built after 2007', color='#000000', linestyle='dashed')
plt.legend()

# Display
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 2: Are there any relationship between the amount of houses built and the rent price?
# MAGIC In order to answer this question, we first plot a visualization to see the evolution of rental price throughout the years.

# COMMAND ----------

rental_prices = barcelona_avg_monthly_rental_prices.toPandas()

# General settings
plt.figure(figsize=(15, 5))
plt.xticks(range(2004, 2024, 2))
plt.xlim(xmin=2004, xmax=2024)
plt.ylim(ymin=0, ymax=1200)
plt.xlabel("Year")
plt.ylabel("Price (€)") 
plt.title("Average Monthly Price (€)")

# Data
plt.plot(rental_prices.year, rental_prices.amount, label='Average Monthly Rental Price', color="#003f5c")

# Annotations
plt.annotate('2007-2008 Financial Crisis',
            xy=(2007, 800), xycoords='data',
            xytext=(0.35, 0.80), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.annotate('2020-2022 Pandemic',
            xy=(2020, 900), xycoords='data',
            xytext=(0.85, 0.55), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.legend()

# Show
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC To be able to do a correlation between `rental_prices` and `homes_finished` we should first filter the dataframes by the same time interval or the correlation would not be realistic.

# COMMAND ----------

homes_finished_filtered_by_common_years = barcelona_homes_finished.filter(col('year') >= 2005).toPandas()
homes_finished_filtered_by_common_years.quantity.corr(rental_prices.amount)

# COMMAND ----------

# MAGIC %md
# MAGIC The correlation value of `-0.1615391701122174` indicates a weak negative linear relationship between the number of homes built in Barcelona (filtered for years 2005 and later) and the average monthly rental prices in Barcelona.
# MAGIC
# MAGIC Here's a brief explanation:
# MAGIC
# MAGIC Correlation Coefficient: The correlation coefficient ranges from `-1` to `1`.
# MAGIC
# MAGIC A value of `1` indicates a perfect positive linear relationship.
# MAGIC A value of `-1` indicates a perfect negative linear relationship.
# MAGIC A value of `0` indicates no linear relationship.
# MAGIC
# MAGIC Negative Correlation: A negative correlation means that as one variable increases, the other variable tends to decrease. **In this case, the weak negative correlation suggests that there is a slight tendency for rental prices to decrease as the number of homes built increases, but this relationship is not strong.**
# MAGIC
# MAGIC Magnitude: The magnitude of `-0.1615391701122174` is close to `0`, indicating that the relationship is weak. This means that the number of homes built and the rental prices are not strongly linearly related.
# MAGIC
# MAGIC In summary, the weak negative correlation suggests that there is a slight inverse relationship between the number of homes built and rental prices, but this relationship is not strong enough to be considered significant.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 3: Are rent values following inflation rate?

# COMMAND ----------

consumer_index = spain_consumer_index\
    .filter(col('month') == 12)\
    .filter(col('year') >= 2005)\
    .sort(col('year'))\
    .toPandas()

# General settings
plt.figure(figsize=(15, 5))
plt.xticks(range(2004, 2024, 2))
plt.xlim(xmin=2004, xmax=2024)
plt.ylim(ymin=-2, ymax=8)
plt.xlabel("Year") 
plt.ylabel("Consumer Index (in %)") 
plt.title("Spain Consumer Index") 

# Data
plt.plot(consumer_index.year, consumer_index.value, label='Consumer Index', color="#003f5c")

# Annotations
plt.annotate('2007-2008 Financial Crisis',
            xy=(2007, 5), xycoords='data',
            xytext=(0.35, 0.80), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.annotate('2020-2022 Pandemic',
            xy=(2020, 3), xycoords='data',
            xytext=(0.75, 0.75), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.legend()

# Show
plt.show()

# COMMAND ----------

# Define a window specification
window = Window.partitionBy().orderBy("x.year", "y.month")

# Join the barcelona_avg_monthly_rental_prices with spain_consumer_index on year and month
adjusted_rental_prices = barcelona_avg_monthly_rental_prices.alias("x")\
    .join(spain_consumer_index.alias("y"),
          (barcelona_avg_monthly_rental_prices.year == spain_consumer_index.year) & (12 == spain_consumer_index.month))\
    .withColumn('adjusted_price',
                lag(col('amount'), 1).over(window) * (1 + (spain_consumer_index.value/100)))\
    .select(barcelona_avg_monthly_rental_prices['*'], 'adjusted_price')

display(adjusted_rental_prices)

# COMMAND ----------

year_column = adjusted_rental_prices.toPandas().year
rental_prices_column = adjusted_rental_prices.toPandas().amount
inflated_prices_column = adjusted_rental_prices.toPandas().adjusted_price

# General settings
plt.figure(figsize=(15, 5))
plt.title("Average Monthly Price (€)")

plt.xticks(range(2005, 2025, 1))
plt.xlabel("Year")
plt.ylabel("Price (€)")

# Data
plt.plot(year_column, rental_prices_column, label='Rental Price', color="#003f5c")
plt.plot(year_column, inflated_prices_column, label='Inflated Rental Price', color="#ff6361")

plt.ylim(ymin=0, ymax=1200)

# Annotations
plt.annotate('Government Intervention',
            xy=(2020, 800), xycoords='data',
            xytext=(0.75, 0.35), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy",
                            facecolor='#000000',
                            fc="0.6",
                            ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"),
            horizontalalignment='right', verticalalignment='top')
plt.legend()

# Show
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Our original hypothesis was invalidated: the amount of homes built does not direct correlate with the rental price.
# MAGIC
# MAGIC Several factors can possibly contribute to the rental prices increase but require further investigation:
# MAGIC
# MAGIC - **Tourism and Short-term Rentals:** The rise of platforms like Airbnb and other short-term rental services may significantly affect the housing market.
# MAGIC - U**rban Development and Gentrification:** Several neighborhoods, especially in central areas like El Raval, Poble Sec, and Poblenou, have gone through processes of gentrification. As these areas became more desirable, rents increased to match the rising demand for housing in these newly "trendy" districts.
# MAGIC - **Demand from Foreign Investors and Expats:** Barcelona's international appeal, both as a tourist and a work destination, has attracted foreign investors and expatriates. This influx may have added to the demand for rental properties, particularly in central and well-connected neighborhoods, raising prices.
# MAGIC - **Government Regulation and Rent Control:** In recent years, local governments have introduced rent control measures, especially with the 2020 law aimed at capping rental prices in certain areas. These regulations had mixed effects, slowing the rise of rent in some places but also leading to a reduction in the supply of available rentals as some owners opted to sell or withdraw properties from the rental market.
# MAGIC - **Population Growth and Demographics:** Barcelona has experienced a growing population, with a younger demographic, including students and young professionals, increasing the demand for rental housing.
# MAGIC - **Public Transportation and Infrastructure Improvements:** Improvements to Barcelona’s public transport and infrastructure, particularly in neighborhoods further from the center, have made them more accessible, increasing demand for rentals in those areas and subsequently pushing up rents.
# MAGIC
# MAGIC These factors, combined, may have led to a steady increase in rental prices over time, making Barcelona one of the most expensive cities in Spain for housing.
