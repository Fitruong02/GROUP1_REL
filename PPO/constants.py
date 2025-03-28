# constants.py
import os

# Kích thước container
CONTAINER_LENGTH = 200  # 200cm
CONTAINER_WIDTH = 100   # 100cm
CONTAINER_HEIGHT = 150  # 150cm

# Đường dẫn đến file CSV chứa thông tin kiện hàng
PACKAGES_CSV_PATH = "data/packages.csv"

# Đường dẫn lưu mô hình DQN-learning tốt nhất
BEST_MODEL_PATH = "best_dqn_model.h5"
# Đường dẫn lưu mô hình Q-learning tốt nhất
BEST_MODEL_PATH_QLeaning = " best_q_model.pkl"

# Thông số tối ưu hóa và tìm kiếm
MAX_SEARCH_DEPTH = 10
FAST_PLACEMENT = True

# Định nghĩa màu sắc cho hiển thị 2D
COLORS_2D = {
    'background': (0, 0, 0),
    'container': (200, 200, 200),
    'text': (255, 255, 255),
    'button': (100, 100, 100),
    'button_hover': (150, 150, 150)
}

COLOR_LIST = [
    (255, 0, 0),     # Đỏ
    (0, 255, 0),     # Xanh lá
    (0, 0, 255),     # Xanh dương
    (255, 255, 0),   # Vàng
    (255, 0, 255),   # Tím
    (0, 255, 255),   # Xanh ngọc
    (128, 0, 0),     # Đỏ đậm
    (0, 128, 0),     # Xanh lá đậm
    (0, 0, 128),     # Xanh dương đậm
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Tím đậm
    (0, 128, 128),   # Teal
    (128, 128, 128), # Xám
    (255, 128, 0),   # Cam
    (128, 255, 0),   # Xanh chanh
    (255, 0, 128),   # Hồng
    (0, 255, 128),   # Xanh ngọc nhạt
    (128, 0, 255),   # Tím nhạt
    (0, 128, 255),   # Xanh da trời
    (255, 128, 128), # Hồng nhạt
]
