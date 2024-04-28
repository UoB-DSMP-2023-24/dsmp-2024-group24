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

    # 找到整个数据集中的最小时间戳
    start_time = float('inf')
    for row in rows:
        time = float(row[0])
        start_time = min(start_time, time)

    # 初始化60个时段的字典，每个时段的开盘和闭市数据初始为None
    segments = {f'segment_{i+1}': {'open': None, 'close': None, 'max': None, 'min': None} for i in range(60)}

    for row in rows:
        time = float(row[0])
        # 计算当前时间属于哪个时段（1-60）
        segment_index = int((time - start_time) // 500) + 1

        # 确保时段编号在1到60之间
        if 1 <= segment_index <= 60:
            segment_key = f'segment_{segment_index}'
            if not segments[segment_key]['open']:
                segments[segment_key]['open'] = row
            segments[segment_key]['close'] = row
            # 更新最大值和最小值
            if segments[segment_key]['max'] is None or float(row[1]) > segments[segment_key]['max']:
                segments[segment_key]['max'] = float(row[1])
            if segments[segment_key]['min'] is None or float(row[1]) < segments[segment_key]['min']:
                segments[segment_key]['min'] = float(row[1])

     # 打印 segments 字典的内容
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
                        open_price[1],  # 开盘价（第二列）
                        close_price[1],  # 闭市价（第二列）
                        max_price,  # 最大值（第二列）
                        min_price  # 最小值（第二列）
                    ])



# 使用示例
input_folder = 'E:/mini/tapes'  # 更新为你的输入文件夹路径
output_file = 'E:/mini/output2.csv'  # 更新为你的输出文件路径
process_multiple_csv(input_folder, output_file)