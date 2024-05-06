import csv
from tqdm import tqdm

Exch0 = "some_value"

# Initialising the storage structure
time_data = {}

with open('E:/mini/lobs/UoB_Set01_2025-01-02LOBs.txt', 'r') as file:
    for line in file:
        data = eval(line.strip())
        time = data[0]

        # Initialise the data structure for that point in time
        if time not in time_data:
            time_data[time] = {'bids': [], 'asks': [], 'best_bid': 0, 'best_ask': float('inf'),
                               'total_bid_quantity': 0, 'total_ask_quantity': 0,
                               'best_bid_quantity': 0, 'best_ask_quantity': 0}  # Initialisation of order quantities to increase the best bid/offer price

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

csv_file_path = 'E:/mini/lobsz/result2.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ['Time', 'Best Bid Price', 'Best Ask Price', 'Total Bid Quantity', 'Total Ask Quantity',
                  'Best Bid Quantity', 'Best Ask Quantity', 'Spread', 'Mid Price'] 
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for item in tqdm(result_list, desc="Saving to CSV"):
        writer.writerow(item)

print(f"save in  {csv_file_path}")
