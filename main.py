import gym
from gym import spaces
import numpy as np
import pygame
from pygame.locals import *
import os
import pickle
import time
import traceback
import argparse
import csv
import copy

# Kiểm tra và import OpenGL
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *

    OPENGL_AVAILABLE = True
except ImportError:
    print("OpenGL library not available. Please install PyOpenGL.")
    OPENGL_AVAILABLE = False

# Định nghĩa hằng số
CONTAINER_LENGTH = 200  # 100cm
CONTAINER_WIDTH = 100  # 50cm
CONTAINER_HEIGHT = 150  # 50cm

# Đường dẫn đến file CSV chứa thông tin kiện hàng
PACKAGES_CSV_PATH = "packages.csv"

# Tối ưu hóa: Bật chế độ đặt nhanh
FAST_PLACEMENT = True


# Hàm đọc dữ liệu từ file CSV
def load_packages_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found, using default data")
        return {
            'small': {'dimensions': (2, 2, 1), 'volume': 4, 'count': 5},
            'medium': {'dimensions': (4, 3, 3), 'volume': 36, 'count': 5},
            'large': {'dimensions': (6, 4, 3), 'volume': 72, 'count': 5}
        }

    packages = {}
    try:
        with open(csv_path, 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Bỏ qua header

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
            raise ValueError("CSV file does not contain valid data")

        print(f"Loaded {len(packages)} types of packages from {csv_path}")

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Using default data")
        return {
            'small': {'dimensions': (2, 2, 1), 'volume': 4, 'count': 5},
            'medium': {'dimensions': (4, 3, 3), 'volume': 36, 'count': 5},
            'large': {'dimensions': (6, 4, 3), 'volume': 72, 'count': 5}
        }

    return packages


# Tải dữ liệu kiện hàng
PACKAGE_TYPES = load_packages_from_csv(PACKAGES_CSV_PATH)

# Danh sách màu cố định cho các loại kiện hàng
COLOR_LIST = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 128, 0), (128, 255, 0),
    (255, 0, 128), (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128)
]


# Tạo ánh xạ màu cho các loại kiện hàng
def create_color_map(package_types):
    color_map = {}
    for i, key in enumerate(package_types.keys()):
        color_index = i % len(COLOR_LIST)
        color_map[key] = COLOR_LIST[color_index]
    return color_map


PACKAGE_COLORS = create_color_map(PACKAGE_TYPES)

# Đường dẫn lưu mô hình tốt nhất
BEST_MODEL_PATH = "best_q_model.pkl"


# Lớp quản lý container
class Container:
    def __init__(self, length, width, height):
        self.grid = np.zeros((length, width, height), dtype=int)
        self.placements = []
        self.occupied_volume = 0
        self.total_volume = length * width * height
        self.layer_fill_threshold = 0.8  # Ngưỡng lấp đầy tầng
        self.edge_bonus = 3  # Điểm thưởng cho việc đặt sát cạnh
        self.bottom_bonus = 5  # Điểm thưởng cho việc đặt ở đáy
        self.adjacent_bonus = 2  # Điểm thưởng cho việc đặt sát khối khác
        self.cache_file = "container_state.pkl"
        self.last_save_time = time.time()
        self.save_interval = 5  # Lưu cache mỗi 5 giây

    def can_place(self, dimensions, x, y, z):
        l, w, h = dimensions

        # Kiểm tra nhanh xem có vượt quá kích thước container không
        if (x + l > self.grid.shape[0] or
                y + w > self.grid.shape[1] or
                z + h > self.grid.shape[2]):
            return False

        # Kiểm tra nhanh với bước lớn hơn
        if FAST_PLACEMENT:
            step = 4
            for dx in range(0, l, step):
                for dy in range(0, w, step):
                    for dz in range(0, h, step):
                        if self.grid[x + dx, y + dy, z + dz] == 1:
                            return False
            # Kiểm tra các góc
            if (self.grid[x, y, z] == 1 or
                    self.grid[x + l - 1, y, z] == 1 or
                    self.grid[x, y + w - 1, z] == 1 or
                    self.grid[x + l - 1, y + w - 1, z] == 1 or
                    self.grid[x, y, z + h - 1] == 1 or
                    self.grid[x + l - 1, y, z + h - 1] == 1 or
                    self.grid[x, y + w - 1, z + h - 1] == 1 or
                    self.grid[x + l - 1, y + w - 1, z + h - 1] == 1):
                return False
            return True
        else:
            for dx in range(l):
                for dy in range(w):
                    for dz in range(h):
                        if self.grid[x + dx, y + dy, z + dz] == 1:
                            return False
            return True

    def place_package(self, dimensions, x, y, z, package_key):
        """Đặt kiện hàng vào container với các ràng buộc vật lý"""
        l, w, h = dimensions

        # Kiểm tra điều kiện đặt
        if not self.can_place(dimensions, x, y, z):
            return False

        # Kiểm tra điều kiện hỗ trợ
        if z > 0 and not self.has_support(x, y, z, l, w):
            return False

        # Cập nhật grid
        self.grid[x:x+l, y:y+w, z:z+h] = 1

        # Lưu thông tin về kiện hàng đã đặt
        self.placements.append((package_key, (x, y, z), dimensions))

        # Cập nhật thể tích đã sử dụng
        self.occupied_volume += l * w * h

        return True

    def find_optimal_position(self, dimensions):
        """Tìm vị trí tối ưu nhất để đặt kiện hàng
        Ưu tiên các vị trí sát đáy và có điểm tiếp xúc tối đa"""
        l, w, h = dimensions
        best_position = None
        best_score = float('-inf')

        # Tối ưu: Chỉ tìm kiếm từ dưới lên trên
        for z in range(CONTAINER_HEIGHT - h + 1):
            for y in range(CONTAINER_WIDTH - w + 1):
                for x in range(CONTAINER_LENGTH - l + 1):
                    if self.can_place(dimensions, x, y, z):
                        # Kiểm tra điều kiện đặt
                        if z == 0 or self.has_support(x, y, z, l, w):
                            score = self._calculate_position_score(x, y, z, l, w, h)
                            if score > best_score:
                                best_score = score
                                best_position = (x, y, z)

                                # Nếu tìm được vị trí tốt ở tầng thấp, ưu tiên chọn ngay
                                if z == 0 and score > 0:
                                    return best_position

        return best_position

    def has_support(self, x, y, z, l, w):
        """Kiểm tra xem khối có được đỡ từ bên dưới không"""
        support_count = 0
        support_needed = (l * w) // 2  # Yêu cầu ít nhất 50% diện tích được đỡ

        # Kiểm tra các điểm tiếp xúc với mặt dưới
        for dx in range(l):
            for dy in range(w):
                if self.grid[x + dx, y + dy, z - 1] == 1:
                    support_count += 1
                    if support_count >= support_needed:
                        return True

        return False

    def _calculate_position_score(self, x, y, z, l, w, h):
        """Tính điểm cho một vị trí dựa trên các yếu tố:
        1. Ưu tiên vị trí thấp
        2. Ưu tiên tiếp xúc nhiều mặt
        3. Ưu tiên vị trí gần góc và cạnh
        """
        score = 0

        # Điểm cơ bản cho độ cao (ưu tiên vị trí thấp)
        height_score = (CONTAINER_HEIGHT - z) / CONTAINER_HEIGHT * 100
        score += height_score

        # Điểm cho tiếp xúc với đáy
        if z == 0:
            score += 200  # Ưu tiên cao cho việc đặt trên đáy

        # Điểm cho tiếp xúc với các mặt
        contact_score = 0

        # Kiểm tra tiếp xúc mặt dưới
        if z > 0:
            bottom_contacts = sum(1 for dx in range(l) for dy in range(w)
                                if self.grid[x + dx, y + dy, z - 1] == 1)
            contact_score += bottom_contacts * 10

        # Kiểm tra tiếp xúc mặt trái
        if x > 0:
            left_contacts = sum(1 for dy in range(w) for dz in range(h)
                              if self.grid[x - 1, y + dy, z + dz] == 1)
            contact_score += left_contacts * 5

        # Kiểm tra tiếp xúc mặt phải
        if x + l < CONTAINER_LENGTH:
            right_contacts = sum(1 for dy in range(w) for dz in range(h)
                               if self.grid[x + l, y + dy, z + dz] == 1)
            contact_score += right_contacts * 5

        # Kiểm tra tiếp xúc mặt trước
        if y > 0:
            front_contacts = sum(1 for dx in range(l) for dz in range(h)
                               if self.grid[x + dx, y - 1, z + dz] == 1)
            contact_score += front_contacts * 5

        # Kiểm tra tiếp xúc mặt sau
        if y + w < CONTAINER_WIDTH:
            back_contacts = sum(1 for dx in range(l) for dz in range(h)
                              if self.grid[x + dx, y + w, z + dz] == 1)
            contact_score += back_contacts * 5

        score += contact_score

        # Điểm cho vị trí gần góc và cạnh
        if x == 0 or x + l == CONTAINER_LENGTH:
            score += 50
        if y == 0 or y + w == CONTAINER_WIDTH:
            score += 50

        return score

    def get_fill_ratio(self):
        """Tính tỷ lệ lấp đầy của container."""
        total_volume = CONTAINER_LENGTH * CONTAINER_WIDTH * CONTAINER_HEIGHT
        filled_volume = 0

        for _, _, dimensions in self.placements:
            l, w, h = dimensions
            filled_volume += l * w * h

        return (filled_volume / total_volume) * 100

    def get_empty_ratio(self):
        """Tính tỷ lệ trống của container."""
        return 100 - self.get_fill_ratio()

    def get_package_count(self):
        return len(self.placements)

    def apply_gravity(self):
        """Di chuyển tất cả các block xuống vị trí thấp nhất có thể"""
        moved = True
        while moved:
            moved = False
            new_placements = []

            # Sắp xếp theo chiều z tăng dần để xử lý các block thấp trước
            sorted_placements = sorted(self.placements, key=lambda x: x[1][2])

            # Reset grid
            self.grid.fill(0)

            for package_key, (x, y, z), dimensions in sorted_placements:
                # Tìm vị trí thấp nhất có thể cho block này
                new_z = z
                while new_z > 0:
                    if self.can_place(dimensions, x, y, new_z - 1):
                        new_z -= 1
                        moved = True
                    else:
                        break

                # Đặt block vào vị trí mới
                l, w, h = dimensions
                self.grid[x:x + l, y:y + w, new_z:new_z + h] = 1
                new_placements.append((package_key, (x, y, new_z), dimensions))

            if moved:
                self.placements = new_placements

    def save_state(self, force=False):
        """Lưu trạng thái container vào file cache"""
        current_time = time.time()
        if force or (current_time - self.last_save_time) >= self.save_interval:
            try:
                state = {
                    'grid': self.grid,
                    'placements': self.placements,
                    'occupied_volume': self.occupied_volume,
                    'timestamp': current_time
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(state, f)
                self.last_save_time = current_time
            except Exception as e:
                print(f"Lỗi khi lưu cache: {e}")

    def load_state(self):
        """Tải trạng thái container từ file cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    state = pickle.load(f)
                    self.grid = state['grid']
                    self.placements = state['placements']
                    self.occupied_volume = state['occupied_volume']
                    return True
            return False
        except Exception as e:
            print(f"Lỗi khi đọc cache: {e}")
            return False


# Lớp môi trường Gym
class PackingEnv(gym.Env):
    def __init__(self, package_types):
        super(PackingEnv, self).__init__()
        self.package_types = package_types
        self.action_space = spaces.Discrete(len(package_types))
        self.observation_space = spaces.MultiDiscrete([package_types[key]['count'] + 1 for key in package_types])

        self.container = Container(CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        self.placement_history = set()

        # Thử tải trạng thái từ cache
        if self.container.load_state():
            print("Đã tải trạng thái từ cache")
            self.remaining = {key: self.package_types[key]['count'] for key in self.package_types}
            # Cập nhật remaining dựa trên placements đã có
            for package_key, _, _ in self.container.placements:
                self.remaining[package_key] -= 1
        else:
            self.reset()

        # Biến cho render
        self.display = None
        self.screen = None
        self.initialized = False
        self.last_time = time.time()

        # Màu sắc cho các kiện hàng
        self.colors_3d = {}
        for key in package_types:
            r, g, b = PACKAGE_COLORS.get(key, (255, 255, 255))
            self.colors_3d[key] = (r / 255, g / 255, b / 255, 0.7)

        # Pre-allocate numpy arrays for better performance
        self.grid = np.zeros((CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT), dtype=np.int8)  # Use int8 instead of int
        self.vertex_buffer = np.zeros((8, 3), dtype=np.float32)  # Pre-allocate for cube vertices

        # Cache for display lists
        self.display_lists = {}

    def calculate_empty_space_penalty(self):
        empty_volume = self.container.total_volume - self.container.occupied_volume
        empty_ratio = empty_volume / self.container.total_volume
        return 5 * empty_ratio

    def calculate_placement_novelty_reward(self):
        current_placement = tuple(sorted((p[0], tuple(p[1]), tuple(p[2])) for p in self.container.placements))
        if current_placement not in self.placement_history:
            self.placement_history.add(current_placement)
            return 5
        return 0

    def reset(self):
        self.container = Container(CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        self.remaining = {key: self.package_types[key]['count'] for key in self.package_types}
        self.placement_history = set()
        # Xóa cache khi reset
        if os.path.exists(self.container.cache_file):
            os.remove(self.container.cache_file)
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.remaining[key] for key in self.package_types])

    def can_place_package(self, package_key, position):
        """Kiểm tra xem có thể đặt kiện hàng tại vị trí cụ thể không."""
        if package_key not in self.package_types:
            return False

        # Lấy kích thước kiện hàng
        dimensions = self.package_types[package_key]['dimensions']
        l, w, h = dimensions
        x, y, z = position

        # Kiểm tra nhanh xem kiện hàng có nằm trong container không
        if x < 0 or y < 0 or z < 0 or \
           x + l > CONTAINER_LENGTH or \
           y + w > CONTAINER_WIDTH or \
           z + h > CONTAINER_HEIGHT:
            return False

        # Kiểm tra va chạm với các kiện hàng đã đặt
        for placed_key, placed_pos, placed_dim in self.container.placements:
            px, py, pz = placed_pos
            pl, pw, ph = placed_dim

            # Kiểm tra va chạm giữa hai hình hộp
            if (x < px + pl and x + l > px and
                y < py + pw and y + w > py and
                z < pz + ph and z + h > pz):
                return False

        # Kiểm tra xem kiện hàng có được đặt trên bề mặt nào đó không
        if z > 0:
            has_support = False

            # Kiểm tra từng kiện hàng đã đặt
            for placed_key, placed_pos, placed_dim in self.container.placements:
                px, py, pz = placed_pos
                pl, pw, ph = placed_dim

                # Kiểm tra xem kiện hàng có được đặt trên kiện hàng này không
                if abs(pz + ph - z) < 0.001:  # Sử dụng epsilon để tránh lỗi làm tròn
                    # Tính diện tích giao nhau
                    overlap_x = max(0, min(x + l, px + pl) - max(x, px))
                    overlap_y = max(0, min(y + w, py + pw) - max(y, py))

                    if overlap_x > 0 and overlap_y > 0:
                        has_support = True
                        break

            if not has_support:
                return False

        return True

    def place_package(self, package_key):
        """Đặt kiện hàng vào container theo thuật toán Bottom-Left-Front."""
        if package_key not in self.package_types or self.remaining[package_key] <= 0:
            return False

        # Lấy kích thước kiện hàng
        dimensions = self.package_types[package_key]['dimensions']
        l, w, h = dimensions

        # Tạo danh sách các điểm có thể đặt
        potential_positions = []

        # Nếu container trống, chỉ có một vị trí có thể đặt (0,0,0)
        if not self.container.placements:
            if l <= CONTAINER_LENGTH and w <= CONTAINER_WIDTH and h <= CONTAINER_HEIGHT:
                self.container.placements.append((package_key, (0, 0, 0), dimensions))
                self.remaining[package_key] -= 1
                return True
            return False

        # Tạo danh sách các điểm có thể đặt từ các kiện hàng đã đặt
        for placed_key, placed_pos, placed_dim in self.container.placements:
            px, py, pz = placed_pos
            pl, pw, ph = placed_dim

            # Điểm trên đỉnh kiện hàng đã đặt
            potential_positions.append((px, py, pz + ph))

            # Điểm bên phải kiện hàng đã đặt
            potential_positions.append((px + pl, py, pz))

            # Điểm phía trước kiện hàng đã đặt
            potential_positions.append((px, py + pw, pz))

        # Loại bỏ các vị trí trùng lặp
        potential_positions = list(set(potential_positions))

        # Sắp xếp các vị trí theo thứ tự ưu tiên (z thấp nhất, rồi đến y, rồi đến x)
        potential_positions.sort(key=lambda pos: (pos[2], pos[1], pos[0]))

        # Thử đặt kiện hàng tại các vị trí tiềm năng
        for pos in potential_positions:
            if self.can_place_package(package_key, pos):
                self.container.placements.append((package_key, pos, dimensions))
                self.remaining[package_key] -= 1
                return True

        # Không thể đặt kiện hàng
        return False

    def step(self, action):
        """Thực hiện một bước trong môi trường."""
        # Lấy loại kiện hàng từ action
        package_keys = list(self.package_types.keys())
        if action >= len(package_keys):
            return self.state, -1, True, {}

        package_key = package_keys[action]

        # Kiểm tra xem còn kiện hàng loại này không
        if self.remaining[package_key] <= 0:
            return self.state, -1, False, {}  # Không kết thúc, chỉ báo lỗi

        # Thử đặt kiện hàng
        success = self.place_package(package_key)

        # Cập nhật trạng thái
        self.state = [self.remaining[key] for key in package_keys]

        # Tính toán phần thưởng
        if success:
            # Phần thưởng dựa trên thể tích kiện hàng
            l, w, h = self.package_types[package_key]['dimensions']
            volume = l * w * h
            reward = volume / (CONTAINER_LENGTH * CONTAINER_WIDTH * CONTAINER_HEIGHT) * 100
        else:
            reward = -1

        # Kiểm tra xem đã hết kiện hàng
        done = sum(self.remaining.values()) == 0

        # Nếu không thể đặt thêm kiện hàng nào nữa, kết thúc
        if not success and not any(self.can_place_any_package()):
            done = True

        return self.state, reward, done, {}

    def can_place_any_package(self):
        """Kiểm tra xem có thể đặt bất kỳ kiện hàng nào không."""
        result = []
        for key in self.package_types.keys():
            if self.remaining[key] <= 0:
                result.append(False)
                continue

            # Tạo danh sách các điểm có thể đặt
            potential_positions = [(0, 0, 0)] if not self.container.placements else []

            for placed_key, placed_pos, placed_dim in self.container.placements:
                px, py, pz = placed_pos
                pl, pw, ph = placed_dim

                # Điểm trên đỉnh kiện hàng đã đặt
                potential_positions.append((px, py, pz + ph))

                # Điểm bên phải kiện hàng đã đặt
                potential_positions.append((px + pl, py, pz))

                # Điểm phía trước kiện hàng đã đặt
                potential_positions.append((px, py + pw, pz))

            # Loại bỏ các vị trí trùng lặp
            potential_positions = list(set(potential_positions))

            # Kiểm tra từng vị trí
            can_place = False
            for pos in potential_positions:
                if self.can_place_package(key, pos):
                    can_place = True
                    break

            result.append(can_place)

        return result

    def render(self, mode='human'):
        global OPENGL_AVAILABLE

        if not OPENGL_AVAILABLE:
            return True

        # Đảm bảo pygame đã được khởi tạo
        if not pygame.get_init():
            pygame.init()

        # Đảm bảo thuộc tính auto_step luôn tồn tại
        if not hasattr(self, 'auto_step'):
            self.auto_step = False

        # Nếu chưa khởi tạo cửa sổ, thực hiện khởi tạo
        if not hasattr(self, 'initialized') or not self.initialized:
            try:
                # Lấy thông tin màn hình
                info = pygame.display.Info()
                screen_width, screen_height = info.current_w, info.current_h

                # Tạo cửa sổ với kích thước hợp lý
                window_width = min(1024, int(screen_width * 0.8))
                window_height = min(768, int(screen_height * 0.8))
                self.display = (window_width, window_height)

                pygame.display.set_caption("Mo phong 3D bin packing - Container")

                # Tạo cửa sổ
                self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)

                # Khởi tạo font
                pygame.font.init()
                self.font = pygame.font.SysFont('Arial', 18, bold=True)

                # Khởi tạo các biến điều khiển với góc nhìn mặc định
                # Thiết lập góc nhìn sao cho sàn nằm ở dưới
                self.rotation_x = 45  # Nhìn từ trên xuống
                self.rotation_y = 45  # Góc xoay để nhìn thấy 3 mặt
                self.zoom = -CONTAINER_LENGTH * 2  # Đủ xa để nhìn toàn cảnh

                # Thiết lập OpenGL
                glViewport(0, 0, self.display[0], self.display[1])
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 1000.0)

                # Bật depth test
                glEnable(GL_DEPTH_TEST)

                # Bật lighting
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                glEnable(GL_COLOR_MATERIAL)

                # Thiết lập ánh sáng
                glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))  # Ánh sáng từ góc trên
                glLightfv(GL_LIGHT0, GL_AMBIENT, (0.6, 0.6, 0.6, 1))  # Tăng ánh sáng môi trường
                glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1))  # Tăng ánh sáng khuếch tán

                # Thiết lập màu nền
                glClearColor(0.2, 0.2, 0.2, 1)

                self.initialized = True
                print("OpenGL đã được khởi tạo thành công!")
                print("Sử dụng phím mũi tên để điều chỉnh góc nhìn")
                print("SPACE: tiếp tục bước tiếp theo")
                print("A: bật/tắt chế độ tự động")
                print("ESC: thoát")

            except Exception as e:
                print(f"Lỗi khởi tạo OpenGL: {e}")
                OPENGL_AVAILABLE = False
                return False

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
                elif event.key == pygame.K_SPACE:
                    print("SPACE: Bước tiếp theo")
                    return True  # Cho phép bước tiếp theo
                elif event.key == pygame.K_a:
                    self.auto_step = not self.auto_step
                    print(f"Chế độ tự động: {'BẬT' if self.auto_step else 'TẮT'}")
                elif event.key == pygame.K_UP:
                    self.rotation_x += 5
                    print(f"UP: Góc X = {self.rotation_x}")
                elif event.key == pygame.K_DOWN:
                    self.rotation_x -= 5
                    print(f"DOWN: Góc X = {self.rotation_x}")
                elif event.key == pygame.K_LEFT:
                    self.rotation_y += 5
                    print(f"LEFT: Góc Y = {self.rotation_y}")
                elif event.key == pygame.K_RIGHT:
                    self.rotation_y -= 5
                    print(f"RIGHT: Góc Y = {self.rotation_y}")
                elif event.key == pygame.K_PAGEUP:
                    self.zoom += 2.0
                    print(f"PAGEUP: Zoom = {self.zoom}")
                elif event.key == pygame.K_PAGEDOWN:
                    self.zoom -= 2.0
                    print(f"PAGEDOWN: Zoom = {self.zoom}")

        # Xóa màn hình và depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Thiết lập ma trận modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Di chuyển camera lùi để nhìn thấy container
        glTranslatef(0, 0, self.zoom)

        # Xoay theo góc nhìn người dùng
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        # Di chuyển container để trung tâm nằm ở gốc tọa độ
        glTranslatef(-CONTAINER_LENGTH/2, -CONTAINER_WIDTH/2, -CONTAINER_HEIGHT/2)

        # Vẽ container
        glColor4f(1.0, 1.0, 1.0, 0.3)  # Màu trắng trong suốt
        glLineWidth(1.0)
        glBegin(GL_LINES)
        # Các cạnh đáy
        glVertex3f(0, 0, 0); glVertex3f(CONTAINER_LENGTH, 0, 0)
        glVertex3f(CONTAINER_LENGTH, 0, 0); glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, 0)
        glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, 0); glVertex3f(0, CONTAINER_WIDTH, 0)
        glVertex3f(0, CONTAINER_WIDTH, 0); glVertex3f(0, 0, 0)
        # Các cạnh đỉnh
        glVertex3f(0, 0, CONTAINER_HEIGHT); glVertex3f(CONTAINER_LENGTH, 0, CONTAINER_HEIGHT)
        glVertex3f(CONTAINER_LENGTH, 0, CONTAINER_HEIGHT); glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT); glVertex3f(0, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        glVertex3f(0, CONTAINER_WIDTH, CONTAINER_HEIGHT); glVertex3f(0, 0, CONTAINER_HEIGHT)
        # Các cạnh nối
        glVertex3f(0, 0, 0); glVertex3f(0, 0, CONTAINER_HEIGHT)
        glVertex3f(CONTAINER_LENGTH, 0, 0); glVertex3f(CONTAINER_LENGTH, 0, CONTAINER_HEIGHT)
        glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, 0); glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        glVertex3f(0, CONTAINER_WIDTH, 0); glVertex3f(0, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        glEnd()

        # Vẽ sàn container với màu xám đậm hơn và độ sáng cao hơn
        glColor4f(0.7, 0.7, 0.7, 1.0)  # Màu xám sáng
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(CONTAINER_LENGTH, 0, 0)
        glVertex3f(CONTAINER_LENGTH, CONTAINER_WIDTH, 0)
        glVertex3f(0, CONTAINER_WIDTH, 0)
        glEnd()

        # Vẽ các kiện hàng đã đặt
        for package_key, position, dimensions in self.container.placements:
            x, y, z = position
            l, w, h = dimensions

            # Lấy màu cho kiện hàng
            r, g, b = PACKAGE_COLORS.get(package_key, (255, 255, 255))
            r, g, b = r/255.0, g/255.0, b/255.0

            # Vẽ khối với màu đặc
            glColor4f(r, g, b, 0.9)  # Tăng độ đậm

            # Vẽ các mặt của khối
            glBegin(GL_QUADS)
            # Mặt dưới
            glVertex3f(x, y, z)
            glVertex3f(x+l, y, z)
            glVertex3f(x+l, y+w, z)
            glVertex3f(x, y+w, z)

            # Mặt trên
            glVertex3f(x, y, z+h)
            glVertex3f(x+l, y, z+h)
            glVertex3f(x+l, y+w, z+h)
            glVertex3f(x, y+w, z+h)

            # Mặt trước
            glVertex3f(x, y, z)
            glVertex3f(x+l, y, z)
            glVertex3f(x+l, y, z+h)
            glVertex3f(x, y, z+h)

            # Mặt sau
            glVertex3f(x, y+w, z)
            glVertex3f(x+l, y+w, z)
            glVertex3f(x+l, y+w, z+h)
            glVertex3f(x, y+w, z+h)

            # Mặt trái
            glVertex3f(x, y, z)
            glVertex3f(x, y+w, z)
            glVertex3f(x, y+w, z+h)
            glVertex3f(x, y, z+h)

            # Mặt phải
            glVertex3f(x+l, y, z)
            glVertex3f(x+l, y+w, z)
            glVertex3f(x+l, y+w, z+h)
            glVertex3f(x+l, y, z+h)
            glEnd()

            # Vẽ viền đen đậm cho khối
            glColor4f(0.0, 0.0, 0.0, 1.0)  # Màu đen
            glLineWidth(2.0)  # Tăng độ dày viền
            glBegin(GL_LINES)
            # Viền mặt dưới
            glVertex3f(x, y, z); glVertex3f(x+l, y, z)
            glVertex3f(x+l, y, z); glVertex3f(x+l, y+w, z)
            glVertex3f(x+l, y+w, z); glVertex3f(x, y+w, z)
            glVertex3f(x, y+w, z); glVertex3f(x, y, z)

            # Viền mặt trên
            glVertex3f(x, y, z+h); glVertex3f(x+l, y, z+h)
            glVertex3f(x+l, y, z+h); glVertex3f(x+l, y+w, z+h)
            glVertex3f(x+l, y+w, z+h); glVertex3f(x, y+w, z+h)
            glVertex3f(x, y+w, z+h); glVertex3f(x, y, z+h)

            # Cạnh nối
            glVertex3f(x, y, z); glVertex3f(x, y, z+h)
            glVertex3f(x+l, y, z); glVertex3f(x+l, y, z+h)
            glVertex3f(x, y+w, z); glVertex3f(x, y+w, z+h)
            glVertex3f(x+l, y+w, z); glVertex3f(x+l, y+w, z+h)
            glEnd()

        # Hiển thị thông tin
        # Tính các thông số
        fill_ratio = self.container.get_fill_ratio()
        empty_ratio = 100 - fill_ratio
        package_count = len(self.container.placements)

        # Hiển thị thông tin dưới dạng text trên màn hình console
        print(f"Tỷ lệ lấp đầy: {fill_ratio:.2f}%, Số kiện hàng: {package_count}, Tự động: {'BẬT' if self.auto_step else 'TẮT'}")

        # Cập nhật màn hình
        pygame.display.flip()

        # Giới hạn FPS
        pygame.time.wait(30)

        # Nếu ở chế độ tự động, cho phép bước tiếp
        return not self.auto_step


# Q-Learning
def q_learning(env, num_episodes=50, alpha=0.1, gamma=0.9, epsilon_start=1.5, epsilon_end=0.3, epsilon_decay=0.95, max_state_size=10000):
    # Khởi tạo Q-table dưới dạng dictionary
    Q = {}

    # Tối ưu hóa: Cache cho state_to_index
    state_index_cache = {}

    # Khởi tạo các biến theo dõi
    epsilon = epsilon_start
    total_rewards = []
    fill_ratios = []
    best_fill_ratio = 0
    best_placements = None

    # Lấy số lượng actions từ không gian hành động
    action_space = env.action_space.n

    print(f"Bắt đầu huấn luyện với {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        state_tuple = tuple(state)

        # Chuyển đổi state thành index sử dụng cache
        if state_tuple in state_index_cache:
            state_idx = state_index_cache[state_tuple]
        else:
            state_idx = hash(state_tuple) % max_state_size
            state_index_cache[state_tuple] = state_idx

        total_reward = 0
        done = False

        step_count = 0
        while not done:
            step_count += 1

            # Render môi trường và kiểm tra kết quả
            render_result = env.render()
            if render_result == False:
                print("Người dùng đã dừng huấn luyện")
                training_info = {
                    "total_rewards": total_rewards,
                    "fill_ratios": fill_ratios,
                    "best_fill_ratio": best_fill_ratio,
                    "best_placements": best_placements,
                    "episodes_completed": episode + 1
                }
                return Q, training_info

            # Nếu render trả về True, nghĩa là người dùng muốn tiếp tục
            if render_result == True:
                # Lựa chọn hành động bằng epsilon-greedy
                if np.random.random() < epsilon:
                    action = np.random.randint(0, action_space)
                else:
                    if state_idx in Q:
                        action = np.argmax(Q[state_idx])
                    else:
                        Q[state_idx] = np.zeros(action_space)
                        action = np.random.randint(0, action_space)

                # Thực hiện hành động
                next_state, reward, done, _ = env.step(action)
                next_state_tuple = tuple(next_state)

                # Chuyển đổi next_state thành index
                if next_state_tuple in state_index_cache:
                    next_state_idx = state_index_cache[next_state_tuple]
                else:
                    next_state_idx = hash(next_state_tuple) % max_state_size
                    state_index_cache[next_state_tuple] = next_state_idx

                # Cập nhật Q-table
                if state_idx not in Q:
                    Q[state_idx] = np.zeros(action_space)

                if next_state_idx not in Q:
                    Q[next_state_idx] = np.zeros(action_space)

                # Cập nhật giá trị Q
                if not done:
                    Q[state_idx][action] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx][action])
                else:
                    Q[state_idx][action] += alpha * (reward - Q[state_idx][action])

                # Cập nhật state
                state = next_state
                state_tuple = next_state_tuple
                state_idx = next_state_idx

                total_reward += reward
            else:
                # Người dùng muốn dừng lại, đợi phím SPACE
                pygame.time.wait(50)  # Đợi một chút để không làm nghẽn CPU

        # Cập nhật epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Lưu tổng phần thưởng của episode này
        total_rewards.append(total_reward)

        # Tính tỷ lệ lấp đầy container
        fill_ratio = env.container.get_fill_ratio()
        fill_ratios.append(fill_ratio)

        # Lưu lại cách xếp tốt nhất
        if fill_ratio > best_fill_ratio:
            best_fill_ratio = fill_ratio
            best_placements = copy.deepcopy(env.container.placements)

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Fill Ratio: {fill_ratio:.2f}%, Epsilon: {epsilon:.4f}")

    print(f"Hoàn thành {num_episodes} episodes!")

    # Trả về Q-table và thông tin huấn luyện
    training_info = {
        "total_rewards": total_rewards,
        "fill_ratios": fill_ratios,
        "best_fill_ratio": best_fill_ratio,
        "best_placements": best_placements,
        "episodes_completed": num_episodes
    }

    return Q, training_info


def state_to_index(state, max_counts, max_state_size=10000):
    full_state_size = np.prod([count + 1 for count in max_counts])
    if full_state_size > max_state_size:
        state_tuple = tuple(state)
        hash_value = hash(state_tuple)
        return abs(hash_value) % max_state_size

    index = 0
    multiplier = 1
    for count, max_count in zip(state, max_counts):
        count = min(count, max_count)
        index += count * multiplier
        multiplier *= (max_count + 1)

    if index >= full_state_size:
        print(f"Cảnh báo: Chỉ số {index} vượt quá kích thước không gian trạng thái {full_state_size}")
        index = index % full_state_size
    return index


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
    container_volume = CONTAINER_LENGTH * CONTAINER_WIDTH * CONTAINER_HEIGHT

    print(f"\nTổng quan:")
    print(f"- Tỷ lệ lấp đầy: {container.get_fill_ratio():.2f}%")
    print(f"- Không gian trống: {container.get_empty_ratio():.2f}%")
    print(f"- Tổng số kiện hàng đã xếp: {total_placed}/{total_available} ({total_placed / total_available * 100:.2f}%)")
    print(f"- Thể tích sử dụng: {total_volume_used}/{container_volume} cm³")

    print("\nChi tiết theo loại:")
    print(f"{'Loại':^10} | {'Đã xếp':^10} | {'Tổng':^10} | {'Tỷ lệ':^10} | {'Kích thước':^15} | {'Thể tích':^12}")
    print("-" * 75)

    for key in package_types.keys():
        placed = placed_counts.get(key, 0)
        total = package_types[key]['count']
        dimensions = package_types[key]['dimensions']
        volume = placed_volumes.get(key, 0)
        print(
            f"{key:^10} | {placed:^10} | {total:^10} | {placed / total * 100:^10.2f}% | {str(dimensions):^15} | {volume:^12}")


def main():
    parser = argparse.ArgumentParser(description='Tối ưu hoá sắp xếp kiện hàng sử dụng Q-Learning')
    parser.add_argument('--episodes', type=int, default=50, help='Số lượng episodes trong quá trình huấn luyện')
    parser.add_argument('--alpha', type=float, default=0.1, help='Tỷ lệ học (learning rate)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Hệ số giảm giá (discount factor)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Giá trị epsilon ban đầu')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Giá trị epsilon cuối cùng')
    parser.add_argument('--epsilon-decay', type=float, default=0.95, help='Tỷ lệ giảm của epsilon')
    parser.add_argument('--no-graphics', action='store_true', help='Chạy ở chế độ không đồ họa')
    parser.add_argument('--fullscreen', action='store_true', help='Chạy ở chế độ toàn màn hình')
    parser.add_argument('--auto-train', action='store_true', help='Tự động huấn luyện không cần nhấn nút')
    parser.add_argument('--fast-mode', action='store_true', help='Chạy ở chế độ nhanh, giảm chất lượng đồ họa')
    parser.add_argument('--eval-only', action='store_true', help='Chỉ đánh giá mô hình mà không huấn luyện')
    parser.add_argument('--reset-model', action='store_true', help='Bỏ qua mô hình đã lưu và tạo mới')
    parser.add_argument('--packages-csv', type=str, default=PACKAGES_CSV_PATH,
                        help='Đường dẫn tới file CSV chứa thông tin kiện hàng')
    parser.add_argument('--max-state-size', type=int, default=100000, help='Kích thước tối đa của không gian trạng thái')
    args = parser.parse_args()

    global FAST_PLACEMENT
    FAST_PLACEMENT = args.fast_mode

    packages = load_packages_from_csv(args.packages_csv)

    if args.reset_model and os.path.exists(BEST_MODEL_PATH):
        try:
            os.remove(BEST_MODEL_PATH)
            print(f"Đã xóa mô hình cũ: {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Không thể xóa mô hình cũ: {e}")

    if args.no_graphics:
        global OPENGL_AVAILABLE
        OPENGL_AVAILABLE = False

    env = PackingEnv(packages)
    env.fullscreen = args.fullscreen
    env.auto_train = args.auto_train
    env.fast_mode = args.fast_mode

    print("=" * 50)
    print("Tối ưu hóa sắp xếp kiện hàng trong container")
    print("=" * 50)
    print(f"Kích thước container: {CONTAINER_LENGTH}cm x {CONTAINER_WIDTH}cm x {CONTAINER_HEIGHT}cm")
    print("Loại kiện hàng:")
    total_packages = 0
    total_volume = 0
    for key, info in packages.items():
        print(f"  - {key}: kích thước {info['dimensions']}cm, số lượng: {info['count']}")
        total_packages += info['count']
        total_volume += info['count'] * info['volume']

    container_volume = CONTAINER_LENGTH * CONTAINER_WIDTH * CONTAINER_HEIGHT
    theoretical_fill = (total_volume / container_volume) * 100

    print(f"Tổng số kiện hàng: {total_packages}")
    print(f"Tổng thể tích hàng: {total_volume}cm³ ({theoretical_fill:.2f}% thể tích container)")
    print("=" * 50)
    print("Thông số huấn luyện:")
    print(f"  - Số episodes: {args.episodes}")
    print(f"  - Learning rate (alpha): {args.alpha}")
    print(f"  - Discount factor (gamma): {args.gamma}")
    print(f"  - Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print(f"  - Kích thước tối đa không gian trạng thái: {args.max_state_size}")

    if args.no_graphics:
        print(f"  - Chế độ hiển thị: Không đồ họa")
    else:
        print(f"  - Chế độ hiển thị: OpenGL 3D")
        if args.fullscreen:
            print(f"  - Chế độ toàn màn hình: Bật")

    if args.auto_train:
        print(f"  - Chế độ tự động huấn luyện: Bật")

    if args.fast_mode:
        print(f"  - Chế độ nhanh: Bật (giảm chất lượng đồ họa, tăng tốc độ xử lý)")

    print("=" * 50)
    print("Hướng dẫn điều khiển:")
    print("  - Chuột trái: Xoay mô hình")
    print("  - Cuộn chuột: Phóng to/thu nhỏ")
    print("  - Phím R: Đặt lại góc nhìn")
    print("  - Phím A: Bật/tắt chế độ tự động")
    print("  - Phím Space/N: Bước tiếp theo")
    print("  - Phím ESC: Thoát")
    print("=" * 50)

    try:
        if args.no_graphics:
            print("Chạy chương trình ở chế độ không đồ họa")
            env.render = lambda mode='human': True
        else:
            print("Chạy chương trình ở chế độ hiển thị OpenGL 3D")

        if not args.no_graphics:
            print("\nĐang khởi tạo cửa sổ hiển thị...")
            env.render()
            print("Cửa sổ hiển thị đã được khởi tạo.")
            if not args.auto_train:
                print("Nhấn Space hoặc N để bắt đầu huấn luyện.")

        Q = None

        if args.eval_only:
            if os.path.exists(BEST_MODEL_PATH):
                try:
                    with open(BEST_MODEL_PATH, 'rb') as f:
                        saved_data = pickle.load(f)
                        Q = saved_data['q_dict']
                        best_stats = saved_data.get('stats', None)

                        if best_stats and best_stats['placements']:
                            print("\nĐã tìm thấy kịch bản tốt nhất:")
                            env.container.placements = best_stats['placements'].copy()
                            env.container.occupied_volume = sum(
                                l * w * h for _, _, (l, w, h) in best_stats['placements'])
                            env.container.grid = np.zeros((CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT),
                                                          dtype=int)
                            for _, (x, y, z), (l, w, h) in best_stats['placements']:
                                env.container.grid[x:x + l, y:y + w, z:z + h] = 1
                            print_packing_stats(env.container, env.package_types, True)
                            if not args.no_graphics:
                                env.render()
                            return
                except Exception as e:
                    print(f"Lỗi khi tải mô hình tốt nhất: {e}")
                    print("Tiếp tục với đánh giá thông thường...")
            if Q is None:
                print("Không thể sử dụng mô hình đã lưu, vui lòng chạy huấn luyện mới")
                args.eval_only = False

        if not args.eval_only:
            print("\n=== BẮT ĐẦU HUẤN LUYỆN Q-LEARNING ===")
            if not args.auto_train:
                print("Nhấn Space hoặc N để tiếp tục từng bước huấn luyện")
            else:
                print("Chế độ tự động huấn luyện đang bật")

            try:
                Q, training_info = q_learning(
                    env=env,
                    num_episodes=args.episodes,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_end=args.epsilon_end,
                    epsilon_decay=args.epsilon_decay,
                    max_state_size=args.max_state_size
                )

                if training_info and len(training_info['fill_ratios']) > 0:
                    print("\nSơ lược kết quả huấn luyện:")
                    print(f"  - Tỷ lệ lấp đầy trung bình: {np.mean(training_info['fill_ratios']):.2f}%")
                    print(f"  - Tỷ lệ lấp đầy cao nhất: {np.max(training_info['fill_ratios']):.2f}%")
                    print(f"  - Tỷ lệ lấp đầy thấp nhất: {np.min(training_info['fill_ratios']):.2f}%")
            except Exception as e:
                print(f"\nLỗi trong quá trình huấn luyện: {e}")
                print("Chi tiết lỗi:")
                traceback.print_exc()
                return

        print("\n=== ĐÁNH GIÁ MÔ HÌNH ===")
        if Q is None and os.path.exists(BEST_MODEL_PATH):
            try:
                with open(BEST_MODEL_PATH, 'rb') as f:
                    saved_data = pickle.load(f)
                    Q = saved_data['q_dict']
                    best_stats = saved_data.get('stats', None)

                    if best_stats and best_stats['placements']:
                        print("\nĐã tìm thấy kịch bản tốt nhất:")
                        env.container.placements = best_stats['placements'].copy()
                        env.container.occupied_volume = sum(l * w * h for _, _, (l, w, h) in best_stats['placements'])
                        env.container.grid = np.zeros((CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT), dtype=int)
                        for _, (x, y, z), (l, w, h) in best_stats['placements']:
                            env.container.grid[x:x + l, y:y + w, z:z + h] = 1
                        print_packing_stats(env.container, env.package_types, True)
                        if not args.no_graphics:
                            env.render()
                        return
            except Exception as e:
                print(f"Lỗi khi tải mô hình tốt nhất: {e}")
                print("Tiếp tục với đánh giá thông thường...")

    except KeyboardInterrupt:
        print("\nChương trình đã bị dừng bởi người dùng")
    except Exception as e:
        print(f"\nLỗi không mong đợi: {e}")
        print("Chi tiết lỗi:")
        traceback.print_exc()

    print("\nChương trình kết thúc")
    try:
        pygame.quit()
    except:
        pass


if __name__ == "__main__":
    main()