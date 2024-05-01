import csv
from tqdm import tqdm
import time

#lobs read v1

Exch0 = "some_value"

class Bid:
    def __init__(self, time, price, quantity):
        self.time = time
        self.price = price
        self.quantity = quantity

    def __repr__(self):
        return f"[{self.time}, {self.price}, {self.quantity}]"


class Ask:
    def __init__(self, time, price, quantity):
        self.time = time
        self.price = price
        self.quantity = quantity

    def __repr__(self):
        return f"[{self.time}, {self.price}, {self.quantity}]"


with open('E:/mini/lobs/UoB_Set01_2025-01-02LOBs.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    data = eval(line.strip())


    bid_data = data[2][0][1]
    for sub_element in bid_data:

        value_1 = sub_element[0]
        value_2 = sub_element[1]

        if value_1 < 200 :
            continue

        bid_instance = Bid(time, value_1, value_2)
        bid_list.append(bid_instance)

    ask_data = data[2][1][1]
    for sub_element in ask_data:

        value_1 = sub_element[0]
        value_2 = sub_element[1]

        if value_1 > 400:
            continue

        ask_instance = Ask(time, value_1, value_2)
        ask_list.append(ask_instance)

result_list = []

for time in time_list:

    filtered_bid_list = [bid for bid in bid_list if bid.time == time]
    filtered_ask_list = [ask for ask in ask_list if ask.time == time]


    sorted_bid_list = sorted(filtered_bid_list, key=lambda x: x.price, reverse=True)
    sorted_ask_list = sorted(filtered_ask_list, key=lambda x: x.price)


    best_bid_price = sorted_bid_list[0].price if sorted_bid_list else None
    best_ask_price = sorted_ask_list[0].price if sorted_ask_list else None

    spread = None if best_bid_price is None or best_ask_price is None else best_ask_price - best_bid_price
    mid_mrice = None if best_bid_price is None or best_ask_price is None else (best_ask_price + best_bid_price)/2

    total_bid_quantity = sum(bid.quantity for bid in filtered_bid_list)
    total_ask_quantity = sum(ask.quantity for ask in filtered_ask_list)

    result_list.append({'Time': time, 'Best Bid Price': best_bid_price, 'Best Ask Price': best_ask_price,\
                        'Total Bid Quantity': total_bid_quantity, 'Total Ask Quantity': total_ask_quantity,\
                        'Spread': spread, 'Mid Price': mid_mrice})

csv_file_path = 'E:/mini/lobsz/result1.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    fieldnames = ['Time', 'Best Bid Price', 'Best Ask Price', 'Total Bid Quantity', 'Total Ask Quantity', 'Spread', 'Mid Price']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


    writer.writeheader()


    for item in tqdm(result_list, desc="Saving to CSV"):
        writer.writerow(item)

print(f"结果已存储到 {csv_file_path}")