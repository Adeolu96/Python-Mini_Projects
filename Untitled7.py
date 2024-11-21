#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Read the data
df = pd.read_csv('Carbon_(CO2)_Emissions_by_Country.csv')

# Convert the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Display basic information about the dataset
print("Dataset Overview:")
df.head(100)
print("\
Basic Statistics:")
df.describe()


# In[5]:


# Plotting the CO2 emissions over time to visualize trends
plt.figure(figsize=(12, 6))

# Grouping data by year and summing the emissions
annual_emissions = df.groupby(df['Date'].dt.year)['Kilotons of Co2'].sum()

# Plotting
sns.lineplot(x=annual_emissions.index, y=annual_emissions.values)
plt.title('Annual CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('Kilotons of CO2')
plt.grid(True)
plt.show()


# In[6]:


# Analyzing emissions by region
# Grouping the data by region and summing the emissions
region_emissions = df.groupby('Region')['Kilotons of Co2'].sum().sort_values(ascending=False)

# Plotting the emissions by region
plt.figure(figsize=(12, 6))
sns.barplot(x=region_emissions.index, y=region_emissions.values)
plt.title('Total CO2 Emissions by Region')
plt.xlabel('Region')
plt.ylabel('Kilotons of CO2')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[7]:


# Identifying top emitting countries
# Grouping the data by country and summing the emissions
country_emissions = df.groupby('Country')['Kilotons of Co2'].sum().sort_values(ascending=False).head(10)

# Plotting the top emitting countries
plt.figure(figsize=(12, 6))
sns.barplot(x=country_emissions.index, y=country_emissions.values)
plt.title('Top 10 CO2 Emitting Countries')
plt.xlabel('Country')
plt.ylabel('Kilotons of CO2')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[8]:


# Analyzing per capita emissions
# Get the most recent year's data for per capita comparison
recent_per_capita = df.sort_values('Date').groupby('Country')['Metric Tons Per Capita'].last().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=recent_per_capita.index, y=recent_per_capita.values)
plt.title('Top 10 Countries by CO2 Emissions Per Capita')
plt.xlabel('Country')
plt.ylabel('Metric Tons Per Capita')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Time series prediction for global emissions
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data for prediction
yearly_emissions = df.groupby(df['Date'].dt.year)['Kilotons of Co2'].sum().reset_index()
X = yearly_emissions['Date'].values.reshape(-1, 1)
y = yearly_emissions['Kilotons of Co2'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions for next 5 years
future_years = np.array(range(X[-1][0] + 1, X[-1][0] + 6)).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Plot historical data and predictions
plt.figure(figsize=(12, 6))
plt.plot(X, y, label='Historical Data')
plt.plot(future_years, future_predictions, 'r--', label='Predictions')
plt.title('Global CO2 Emissions Trend and Predictions')
plt.xlabel('Year')
plt.ylabel('Kilotons of CO2')
plt.legend()
plt.grid(True)
plt.show()

# Print predictions
print("\
Predicted global CO2 emissions (Kilotons):")
for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"Year {year}: {pred:,.0f}")


# In[9]:


# Calculate emission growth rates by country
# Group by country and calculate the percentage change in emissions
country_growth = df.groupby('Country')['Kilotons of Co2'].sum().pct_change().sort_values(ascending=False)

# Display the top countries with the highest growth rates
print("Top countries by emission growth rates:")
print(country_growth.head(10))


# In[10]:


# Analyze seasonal patterns and GDP relationship
import seaborn as sns

# Calculate average emissions by month for seasonal patterns
df['Month'] = df['Date'].dt.month
monthly_emissions = df.groupby('Month')['Kilotons of Co2'].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_emissions.index, y=monthly_emissions.values)
plt.title('Average Monthly CO2 Emissions')
plt.xlabel('Month')
plt.ylabel('Average Kilotons of CO2')
plt.grid(True)
plt.show()

# Regional focus - let's analyze Asia's emissions trend
asia_data = df[df['Region'] == 'Asia'].groupby('Date')['Kilotons of Co2'].sum()

plt.figure(figsize=(12, 6))
sns.lineplot(data=asia_data)
plt.title('CO2 Emissions Trend in Asia')
plt.xlabel('Year')
plt.ylabel('Kilotons of CO2')
plt.grid(True)
plt.show()

print("\
Key findings:")
print("1. Highest emitting region: Asia")
print("2. Fastest growing major emitter: India")
print("3. Seasonal variation shows higher emissions in winter months")
print("4. Asia shows consistent upward trend in emissions")


# In[11]:


# Detailed analysis of specific countries
# Let's focus on the top 3 emitting countries: China, USA, and India
countries_of_interest = ['China', 'United States', 'India']

# Filter data for these countries
df_countries = df[df['Country'].isin(countries_of_interest)]

# Plotting emissions over time for these countries
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_countries, x='Date', y='Kilotons of Co2', hue='Country')
plt.title('CO2 Emissions Over Time for China, USA, and India')
plt.xlabel('Year')
plt.ylabel('Kilotons of CO2')
plt.grid(True)
plt.legend(title='Country')
plt.show()


# In[12]:


# Compare emission patterns between regions
regions = df['Region'].unique()
region_data = df.groupby(['Region', df['Date'].dt.year])['Kilotons of Co2'].sum().unstack()

# Calculate growth rates
growth_rates = region_data.pct_change().mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.heatmap(region_data.pct_change(), cmap='RdYlGn_r', center=0)
plt.title('Regional Emission Growth Patterns')
plt.xlabel('Year')
plt.ylabel('Region')
plt.show()

# Calculate emission intensity (CO2 per GDP)
recent_year = df['Date'].max().year
recent_data = df[df['Date'].dt.year == recent_year]
top_10_intensity = recent_data.nlargest(10, 'Metric Tons Per Capita')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_intensity, x='Country', y='Metric Tons Per Capita')
plt.title('Top 10 Countries by Emission Intensity')
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Metric Tons Per Capita')
plt.show()

print("\
Average Annual Growth Rates by Region:")
print(growth_rates)


# In[16]:


# Analyze emission trends and per capita metrics
df['Year'] = pd.to_datetime(df['Date']).dt.year
recent_year = df['Year'].max()

# Plot per capita emissions by region
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['Year'] == recent_year], x='Region', y='Metric Tons Per Capita')
plt.xticks(rotation=45)
plt.title('Per Capita Emissions by Region (Most Recent Year)')
plt.show()

# Calculate emission trends
yearly_emissions = df.groupby(['Year', 'Region'])['Kilotons of Co2'].sum().unstack()
yearly_change = yearly_emissions.pct_change()

plt.figure(figsize=(12, 6))
sns.heatmap(yearly_change, cmap='RdYlGn_r', center=0)
plt.title('Year-over-Year Emission Changes by Region')
plt.show()

print("\
Top 5 Regions by Average Per Capita Emissions:")
print(df.groupby('Region')['Metric Tons Per Capita'].mean().sort_values(ascending=False).head())


# In[17]:


# Analyze emission trends for major countries
major_countries = ['United States', 'China', 'India', 'Russia', 'Japan']

# Filter and plot emissions over time
plt.figure(figsize=(12, 6))
for country in major_countries:
    country_data = df[df['Country'] == country]
    plt.plot(country_data['Year'], country_data['Kilotons of Co2'], label=country)

plt.title('CO2 Emissions Trends for Major Countries')
plt.xlabel('Year')
plt.ylabel('Kilotons of CO2')
plt.legend()
plt.grid(True)
plt.show()

# Calculate year-over-year change
print("\
Average Annual Change in Emissions (%):")
for country in major_countries:
    country_data = df[df['Country'] == country].sort_values('Year')
    yearly_change = country_data['Kilotons of Co2'].pct_change().mean() * 100
    print(f"{country}: {yearly_change:.2f}%")


# In[19]:


# Inspect the dataframe to understand its structure and available columns
print(df.columns)
print(df.head())


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a sample DataFrame
np.random.seed(42)  # For reproducibility

countries = ['United States', 'China', 'India', 'Germany', 'Brazil']
years = range(2000, 2021)
data = []

for country in countries:
    for year in years:
        data.append({
            'Country': country,
            'Region': 'Sample Region',  # Placeholder
            'Date': f'{year}-01-01',
            'Kilotons of Co2': np.random.randint(5000, 50000),  # Simulated actual emissions
            'Metric Tons Per Capita': np.random.uniform(1, 10),  # Placeholder
            'Month': 1,
            'Year': year,
            'Target Emissions': np.random.randint(4000, 45000),  # Simulated target emissions
        })

emission_targets_df = pd.DataFrame(data)

# Step 2: Calculate the difference between actual and target emissions
emission_targets_df['Difference'] = emission_targets_df['Kilotons of Co2'] - emission_targets_df['Target Emissions']

# Step 3: Plot differences for selected countries
countries_to_plot = ['United States', 'China', 'India']
plt.figure(figsize=(14, 7))
for country in countries_to_plot:
    country_data = emission_targets_df[emission_targets_df['Country'] == country]
    plt.plot(country_data['Year'], country_data['Difference'], label=country, marker='o')

plt.title('Emission Reduction Progress for Selected Countries')
plt.xlabel('Year')
plt.ylabel('Difference in Emissions (Kilotons of CO₂)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame (you would replace this with your actual data)
data = {
    'Country': ['United States', 'China', 'India', 'United States', 'China', 'India'],
    'Year': [2020, 2020, 2020, 2021, 2021, 2021],
    'Target Emissions': [100, 150, 120, 95, 145, 115],
    'Actual Emissions': [110, 140, 125, 90, 150, 110]
}

# Create a pandas DataFrame
emission_targets_df = pd.DataFrame(data)

# Calculate the difference between actual and target emissions
emission_targets_df['Difference'] = emission_targets_df['Actual Emissions'] - emission_targets_df['Target Emissions']

# Plot the differences for a few countries
countries_to_plot = ['United States', 'China', 'India']
plt.figure(figsize=(14, 7))
for country in countries_to_plot:
    country_data = emission_targets_df[emission_targets_df['Country'] == country]
    plt.plot(country_data['Year'], country_data['Difference'], label=country)

plt.title('Emission Reduction Progress for Selected Countries')
plt.xlabel('Year')
plt.ylabel('Difference in Emissions (Kilotons)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




