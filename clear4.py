import os
import csv
from tqdm import tqdm

Exch0 = "some_value"

# Function to process a single file
def process_file(file_path, output_folder):
    # Initialize storage structure
    time_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            data = eval(line.strip())
            time = data[0]

            # Initialize data structure for the current time
            if time not in time_data:
                time_data[time] = {'bids': [], 'asks': [], 'best_bid': 0, 'best_ask': float('inf'),
                                   'total_bid_quantity': 0, 'total_ask_quantity': 0,
                                   'best_bid_quantity': 0, 'best_ask_quantity': 0}

            # Process bid data
            bid_data = data[2][0][1]
            for value_1, value_2 in bid_data:
                if value_1 >= 100:
                    time_data[time]['bids'].append((value_1, value_2))
                    time_data[time]['total_bid_quantity'] += value_2
                    if value_1 > time_data[time]['best_bid']:
                        time_data[time]['best_bid'] = value_1
                        time_data[time]['best_bid_quantity'] = value_2
                    elif value_1 == time_data[time]['best_bid']:
                        time_data[time]['best_bid_quantity'] += value_2

            # Process ask data
            ask_data = data[2][1][1]
            for value_1, value_2 in ask_data:
                if value_1 <= 500:
                    time_data[time]['asks'].append((value_1, value_2))
                    time_data[time]['total_ask_quantity'] += value_2
                    if value_1 < time_data[time]['best_ask']:
                        time_data[time]['best_ask'] = value_1
                        time_data[time]['best_ask_quantity'] = value_2
                    elif value_1 == time_data[time]['best_ask']:
                        time_data[time]['best_ask_quantity'] += value_2

    result_list = []
    for time, data in time_data.items():
        spread = None if data['best_bid'] == 0 or data['best_ask'] == float('inf') else data['best_ask'] - data['best_bid']
        mid_price = None if spread is None else (data['best_ask'] + data['best_bid']) / 2
        result_list.append({'Time': time, 'Best Bid Price': data['best_bid'], 'Best Ask Price': data['best_ask'],
                            'Total Bid Quantity': data['total_bid_quantity'],
                            'Total Ask Quantity': data['total_ask_quantity'],
                            'Best Bid Quantity': data['best_bid_quantity'],
                            'Best Ask Quantity': data['best_ask_quantity'],
                            'Spread': spread, 'Mid Price': mid_price})

    # Extract date from the filename
    date_from_filename = os.path.basename(file_path).split('_')[2].split('LOBs')[0]

    # Create a new CSV file with the extracted date in the output folder
    new_csv_file_path = os.path.join(output_folder, f'result_{date_from_filename}.csv')
    with open(new_csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ['Time', 'Best Bid Price', 'Best Ask Price', 'Total Bid Quantity', 'Total Ask Quantity',
                      'Best Bid Quantity', 'Best Ask Quantity', 'Spread', 'Mid Price']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in tqdm(result_list, desc=f"Saving to CSV ({date_from_filename})"):
            writer.writerow(item)

    print(f"Results have been stored in {new_csv_file_path}")

# Folder containing input files
input_folder = '/Users/huashenglong/Desktop/input'

# Folder for storing output CSV files
output_folder = '/Users/huashenglong/Desktop/Result'

# Iterate over files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('LOBs.txt'):
        file_path = os.path.join(input_folder, filename)
        process_file(file_path, output_folder)
