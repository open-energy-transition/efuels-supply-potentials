import pandas as pd

# Load the CSV files
airports = pd.read_csv("airports.csv")
market_data = pd.read_csv("T100_Domestic_Market_and_Segment_Data_-3591723781169319541.csv")
fuel_data = pd.read_csv("fuel_jf.csv")  # Load the fuel consumption data file

# Filter the airports file to keep only US airports
airports_us = airports[airports["iso_country"] == "US"].copy()

# Remove the 'US-' prefix from 'iso_region'
airports_us['iso_region'] = airports_us['iso_region'].str.replace('US-', '', regex=False)

# Merge `market_data` with `airports_us` to get the ISO region for each airport
merged_check = pd.merge(
    market_data,
    airports_us[['iata_code', 'iso_region']],
    left_on='origin',
    right_on='iata_code',
    how='left',
    indicator=True
)

# Identify unmatched origins and calculate the total passengers excluded
unmatched_origins = merged_check[merged_check['_merge'] == 'left_only']
excluded_passengers_total = unmatched_origins['passengers'].sum()
total_passengers = market_data['passengers'].sum()
excluded_percentage = (excluded_passengers_total / total_passengers) * 100

print("Unmatched airports:")
print(unmatched_origins['origin'].unique())
print(f"\nTotal passengers from unmatched airports: {excluded_passengers_total}")
print(f"% of passengers from unmatched airports: {excluded_percentage:.2f}%")

# Calculate the total passengers for matched airports
matched_passengers_total = merged_check[merged_check['_merge'] == 'both']['passengers'].sum()
print(f"Total passengers from matched airports: {matched_passengers_total}")

# Filter only the rows that have a match
matched_data = merged_check[merged_check['_merge'] == 'both']

# Group by 'iso_region' and sum the passengers for each state
passengers_per_state = matched_data.groupby('iso_region')['passengers'].sum().reset_index()

# Rename columns for clarity
passengers_per_state.columns = ['State', 'Passengers (-)']

# Filter the fuel data to keep only rows where 'MSN' equals 'JFACP'
fuel_data_filtered = fuel_data[fuel_data['MSN'] == 'JFACP'].copy()

# Rename the column '2023' to 'Consumption (kbarrel)' for clarity
fuel_data_filtered = fuel_data_filtered[['State', '2023']]
fuel_data_filtered.columns = ['State', 'Consumption (kbarrel)']

# Merge passenger and consumption data by state
final_data = pd.merge(
    passengers_per_state,
    fuel_data_filtered,
    on='State',
    how='left'
)

# Calculate the total passengers and total consumption for percentage calculations
total_passengers_all_states = final_data['Passengers (-)'].sum()
total_consumption_all_states = final_data['Consumption (kbarrel)'].sum()

# Calculate the percentage of consumption and passengers per state, rounding to two decimal places
final_data['Consumption over total (%)'] = (
    (final_data['Consumption (kbarrel)'] / total_consumption_all_states) * 100
).round(2)

final_data['Passenger over total (%)'] = (
    (final_data['Passengers (-)'] / total_passengers_all_states) * 100
).round(2)

# Calculate the difference in percentage between passengers and consumption, rounding to two decimal places
final_data['Consumption-passenger mismatch (%)'] = (
    final_data['Passenger over total (%)'] - final_data['Consumption over total (%)']
).round(2)

# Save the result to a new CSV file
final_data.to_csv("passengers_vs_consumption.csv", index=False)

print("The file 'passengers_vs_consumption.csv' has been created with the required columns and formatted percentages.")
