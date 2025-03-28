# utils.py
import os
import csv
from constants import PACKAGES_CSV_PATH, COLOR_LIST, COLORS_2D

def load_packages_from_csv(csv_path=PACKAGES_CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"Không tìm thấy file {csv_path}, sử dụng dữ liệu mặc định")
        return {
            'small': {'dimensions': (2, 2, 1), 'volume': 4, 'count': 5},
            'medium': {'dimensions': (4, 3, 3), 'volume': 36, 'count': 5},
            'large': {'dimensions': (6, 4, 3), 'volume': 72, 'count': 5}
        }
    packages = {}
    try:
        with open(csv_path, 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            if 'count' in header or 'Count' in header:
                count_index = header.index('count') if 'count' in header else header.index('Count')
                for row in reader:
                    if len(row) > count_index:
                        name = row[0].strip()
                        length = int(row[1])
                        width = int(row[2])
                        height = int(row[3])
                        count = int(row[count_index])
                        volume = length * width * height
                        packages[name] = {
                            'dimensions': (length, width, height),
                            'volume': volume,
                            'count': count
                        }
            else:
                package_groups = {}
                for row in reader:
                    if len(row) >= 4:
                        name = row[0].strip()
                        length = int(row[1])
                        width = int(row[2])
                        height = int(row[3])
                        prefix = name.split('_')[0] if '_' in name else name
                        if prefix not in package_groups:
                            package_groups[prefix] = []
                        volume = length * width * height
                        package_groups[prefix].append({
                            'name': name,
                            'dimensions': (length, width, height),
                            'volume': volume
                        })
                for prefix, items in package_groups.items():
                    avg_length = sum(item['dimensions'][0] for item in items) // len(items)
                    avg_width = sum(item['dimensions'][1] for item in items) // len(items)
                    avg_height = sum(item['dimensions'][2] for item in items) // len(items)
                    avg_volume = avg_length * avg_width * avg_height
                    packages[prefix.lower()] = {
                        'dimensions': (avg_length, avg_width, avg_height),
                        'volume': avg_volume,
                        'count': len(items)
                    }
        if not packages:
            raise ValueError("File CSV không chứa dữ liệu hợp lệ")
        print(f"Đã tải {len(packages)} loại kiện hàng từ file {csv_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        print("Sử dụng dữ liệu mặc định")
        return {
            'small': {'dimensions': (2, 2, 1), 'volume': 4, 'count': 5},
            'medium': {'dimensions': (4, 3, 3), 'volume': 36, 'count': 5},
            'large': {'dimensions': (6, 4, 3), 'volume': 72, 'count': 5}
        }
    return packages

def create_color_map(package_types):
    color_map = {}
    for i, key in enumerate(package_types.keys()):
        color_index = i % len(COLOR_LIST)
        color_map[key] = COLOR_LIST[color_index]
        COLORS_2D[key] = COLOR_LIST[color_index]
    return color_map

def print_packing_stats(container, package_types, is_best=False):
    prefix = "=== KỊCH BẢN TỐT NHẤT ===" if is_best else "=== KẾT QUẢ ĐÓNG GÓI ==="
    print(f"\n{prefix}")
    placed_counts = {}
    placed_volumes = {}
    for package_key, _, dimensions in container.placements:
        placed_counts[package_key] = placed_counts.get(package_key, 0) + 1
        l, w, h = dimensions
        placed_volumes[package_key] = placed_volumes.get(package_key, 0) + (l * w * h)
    total_placed = sum(placed_counts.values())
    total_available = sum(info['count'] for info in package_types.values())
    total_volume_used = sum(placed_volumes.values())
    container_volume = container.total_volume
    print(f"\nTổng quan:")
    print(f"- Tỷ lệ lấp đầy: {container.get_fill_ratio():.2f}%")
    print(f"- Không gian trống: {container.get_empty_ratio():.2f}%")
    print(f"- Tổng số kiện hàng đã xếp: {total_placed}/{total_available} ({total_placed/total_available*100:.2f}%)")
    print(f"- Thể tích sử dụng: {total_volume_used}/{container_volume} cm³")
    print("\nChi tiết theo loại:")
    print(f"{'Loại':^10} | {'Đã xếp':^10} | {'Tổng':^10} | {'Tỷ lệ':^10} | {'Kích thước':^15} | {'Thể tích':^12}")
    print("-" * 75)
    for key in package_types.keys():
        placed = placed_counts.get(key, 0)
        total = package_types[key]['count']
        dimensions = package_types[key]['dimensions']
        volume = placed_volumes.get(key, 0)
        print(f"{key:^10} | {placed:^10} | {total:^10} | {placed/total*100:^10.2f}% | {str(dimensions):^15} | {volume:^12}")
