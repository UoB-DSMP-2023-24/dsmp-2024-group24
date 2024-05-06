import pandas as pd

# This should be the path to your actual data CSV file
csv_file_path = 'E:/mini/f2/baoshen_predictions1.csv'

# Read the CSV file into a DataFrame, selecting only the specified columns
df = pd.read_csv(csv_file_path, usecols=['Open_predicted', 'Close_predicted', 'Max_predicted', 'Min_predicted'])


# Define the function to aggregate the data
def aggregate_data(df):
    # Group by every 60 rows for each day, starting with 1
    df['Day'] = (df.index // 60) + 1

    # Perform the aggregation
    aggregated_df = df.groupby('Day').agg(
        Opening_Price=('Open_predicted', 'first'),
        Closing_Price=('Close_predicted', 'last'),
        Max_Value=('Max_predicted', 'max'),
        Min_Value=('Min_predicted', 'min')
    ).reset_index()

    return aggregated_df


# Apply the function to get the aggregated data
aggregated_df = aggregate_data(df)

# This will be the path where the new aggregated data will be saved
# You should change this to the path where you want to save the new CSV
new_csv_file_path = 'E:/mini/f2/combine1.csv'

# Save the aggregated data to a new CSV file
aggregated_df.to_csv(new_csv_file_path, index=False)
