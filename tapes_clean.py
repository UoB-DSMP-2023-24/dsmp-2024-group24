import csv
import os



def extract_date_from_filename(file_name):
    parts = file_name.split('_')
    if len(parts) >= 3:
        date_part = parts[2].replace('tapes.csv', '')
        return date_part
    else:
        return "Unknown"


def read_csv_for_time_segments(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Find the smallest timestamp in the entire dataset
    start_time = float('inf')
    for row in rows:
        time = float(row[0])
        start_time = min(start_time, time)

    # Initialise a dictionary of 60 time slots, with the opening and closing data for each slot initially being None
    segments = {f'segment_{i+1}': {'open': None, 'close': None, 'max': None, 'min': None} for i in range(60)}

    for row in rows:
        time = float(row[0])
        # Calculate which time period the current time belongs to (1-60)
        segment_index = int((time - start_time) // 500) + 1

        # Ensure that time slots are numbered between 1 and 60
        if 1 <= segment_index <= 60:
            segment_key = f'segment_{segment_index}'
            if not segments[segment_key]['open']:
                segments[segment_key]['open'] = row
            segments[segment_key]['close'] = row
            # Update maximum and minimum values
            if segments[segment_key]['max'] is None or float(row[1]) > segments[segment_key]['max']:
                segments[segment_key]['max'] = float(row[1])
            if segments[segment_key]['min'] is None or float(row[1]) < segments[segment_key]['min']:
                segments[segment_key]['min'] = float(row[1])

     # Print the contents of the segments dictionary
    for key, value in segments.items():
        print(key, value)

    return segments


def process_multiple_csv(input_folder, output_file):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    date = 0

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "Segment", "Open", "Close", "Max", "Min"])

        for file_name in csv_files:
            file_path = os.path.join(input_folder, file_name)
            segments = read_csv_for_time_segments(file_path)

            for segment_name, segment_data in segments.items():
                date = date + 1
                open_price = segment_data.get('open', [None])
                close_price = segment_data.get('close', [None])
                max_price = segment_data.get('max', [None])
                min_price = segment_data.get('min', [None])

                print(open_price, close_price, max_price, min_price)

                if segment_data['open'] and segment_data['close'] and segment_data['max'] and segment_data['min']:
                    writer.writerow([
                        date,
                        segment_name,
                        open_price[1],  
                        close_price[1],  
                        max_price,  
                        min_price  
                    ])


input_folder = 'E:/mini/tapes' 
output_file = 'E:/mini/output2.csv'  
process_multiple_csv(input_folder, output_file)
