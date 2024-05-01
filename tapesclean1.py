import csv
import os

#tapes read v1

def extract_date_from_filename(file_name):

    parts = file_name.split('_')
    if len(parts) >= 3:
        date_part = parts[2]
        date_part = date_part.replace('tapes.csv', '')
        return date_part
    else:
        return "Unknown"

def read_csv_second_column(file_path):

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        first_row_second_column = int(next(reader)[1])

        for row in reader:
            last_row_second_column = int(row[1])

    first_row_second_column = int(first_row_second_column)
    last_row_second_column = int(last_row_second_column)

    return first_row_second_column, last_row_second_column


def process_multiple_csv(input_folder, output_file):

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]


    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)


        writer.writerow(["File", "Open", "Close"])


        for file_name in csv_files:

            file_path = os.path.join(input_folder, file_name)
            date = extract_date_from_filename(file_name)


            first_row, last_row = read_csv_second_column(file_path)


            writer.writerow([date, first_row, last_row])


input_folder = 'E:/mini/tapes'
output_file = 'E:/mini/output.csv'
process_multiple_csv(input_folder, output_file)