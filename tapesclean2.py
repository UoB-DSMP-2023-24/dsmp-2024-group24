import csv
import os

#tapes read v2

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


    segments = {f'segment_{i+1}': {'open': None, 'close': None} for i in range(60)}

    start_time = float(rows[0][0])
    for row in rows:
        time = float(row[0])

        segment_index = int((time - start_time) // 500) + 1


        if 1 <= segment_index <= 60:
            segment_key = f'segment_{segment_index}'
            if not segments[segment_key]['open']:
                segments[segment_key]['open'] = row
            segments[segment_key]['close'] = row

    return segments

def process_multiple_csv(input_folder, output_file):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "Segment", "Open", "Close"])

        for file_name in csv_files:
            file_path = os.path.join(input_folder, file_name)
            date = extract_date_from_filename(file_name)
            segments = read_csv_for_time_segments(file_path)

            for segment_name, segment_data in segments.items():
                if segment_data['open'] and segment_data['close']:
                    writer.writerow([
                        date,
                        segment_name,
                        segment_data['open'][1],
                        segment_data['close'][1]
                    ])



input_folder = 'E:/mini/tapes'
output_file = 'E:/mini/output1.csv'
process_multiple_csv(input_folder, output_file)