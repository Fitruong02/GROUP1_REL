# container.py
import numpy as np
from constants import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT, FAST_PLACEMENT, MAX_SEARCH_DEPTH

class Container:
    def __init__(self, length, width, height):
        self.grid = np.zeros((length, width, height), dtype=int)
        self.placements = []
        self.occupied_volume = 0
        self.total_volume = length * width * height

    def can_place(self, dimensions, x, y, z):
        l, w, h = dimensions
        # Kiểm tra kích thước vượt quá container
        if (x + l > self.grid.shape[0] or
            y + w > self.grid.shape[1] or
            z + h > self.grid.shape[2]):
            return False

        # Kiểm tra các góc của khối
        if (self.grid[x, y, z] == 1 or
            self.grid[x+l-1, y, z] == 1 or
            self.grid[x, y+w-1, z] == 1 or
            self.grid[x+l-1, y+w-1, z] == 1 or
            self.grid[x, y, z+h-1] == 1 or
            self.grid[x+l-1, y, z+h-1] == 1 or
            self.grid[x, y+w-1, z+h-1] == 1 or
            self.grid[x+l-1, y+w-1, z+h-1] == 1):
            return False

        # Sử dụng kiểm tra mẫu nếu FAST_PLACEMENT bật
        if FAST_PLACEMENT:
            for dx in range(0, l, 2):
                for dy in range(0, w, 2):
                    for dz in range(0, h, 2):
                        if self.grid[x+dx, y+dy, z+dz] == 1:
                            return False
            if l > 2 and w > 2 and h > 2:
                mid_x = x + l // 2
                mid_y = y + w // 2
                mid_z = z + h // 2
                if self.grid[mid_x, mid_y, mid_z] == 1:
                    return False
            return True
        else:
            for dx in range(l):
                for dy in range(w):
                    for dz in range(h):
                        if self.grid[x+dx, y+dy, z+dz] == 1:
                            return False
            return True

    def place_package(self, dimensions, x, y, z, package_key):
        l, w, h = dimensions
        self.grid[x:x+l, y:y+w, z:z+h] = 1
        self.placements.append((package_key, (x, y, z), dimensions))
        self.occupied_volume += l * w * h
        print(f"Đã đặt {package_key} tại vị trí ({x}, {y}, {z}) với kích thước {dimensions}")

    def find_first_position(self, dimensions):
        l, w, h = dimensions
        best_position = None
        min_z = float('inf')
        for z in range(CONTAINER_HEIGHT - h + 1):
            for y in range(CONTAINER_WIDTH - w + 1):
                for x in range(CONTAINER_LENGTH - l + 1):
                    if self.can_place(dimensions, x, y, z):
                        if z < min_z:
                            min_z = z
                            best_position = (x, y, z)
                        if z == 0:
                            return (x, y, z)
        return best_position

    def find_optimal_position(self, dimensions):
        l, w, h = dimensions
        best_position = None
        best_score = -1
        if FAST_PLACEMENT:
            for z in range(min(MAX_SEARCH_DEPTH, CONTAINER_HEIGHT - h + 1)):
                for y in range(0, CONTAINER_WIDTH - w + 1, 2):
                    for x in range(0, CONTAINER_LENGTH - l + 1, 2):
                        if self.can_place(dimensions, x, y, z):
                            if z == 0:
                                return (x, y, z)
                            elif best_position is None:
                                best_position = (x, y, z)
                                best_score = 1
            if best_position is not None:
                return best_position
        for z in range(min(MAX_SEARCH_DEPTH, CONTAINER_HEIGHT - h + 1)):
            for y in range(CONTAINER_WIDTH - w + 1):
                for x in range(CONTAINER_LENGTH - l + 1):
                    if self.can_place(dimensions, x, y, z):
                        if z == 0:
                            score = 1000 + self._calculate_position_score(x, y, z, l, w, h)
                        else:
                            score = self._calculate_position_score(x, y, z, l, w, h)
                        if score > best_score:
                            best_score = score
                            best_position = (x, y, z)
                            if score > 1000:
                                return best_position
        if best_position is None:
            for z in range(MAX_SEARCH_DEPTH, CONTAINER_HEIGHT - h + 1, 3):
                for y in range(0, CONTAINER_WIDTH - w + 1, 3):
                    for x in range(0, CONTAINER_LENGTH - l + 1, 3):
                        if self.can_place(dimensions, x, y, z):
                            return (x, y, z)
        return best_position

    def _calculate_position_score(self, x, y, z, l, w, h):
        score = 0
        if z == 0:
            score += 10 * l * w
        if z > 0:
            contact_area = 0
            for dx in range(l):
                for dy in range(w):
                    if self.grid[x+dx, y+dy, z-1] == 1:
                        contact_area += 1
            score += 5 * contact_area
        if x == 0:
            score += 3 * w * h
        else:
            contact_area = 0
            for dy in range(w):
                for dz in range(h):
                    if self.grid[x-1, y+dy, z+dz] == 1:
                        contact_area += 1
            score += 2 * contact_area
        if x + l == CONTAINER_LENGTH:
            score += 3 * w * h
        else:
            contact_area = 0
            for dy in range(w):
                for dz in range(h):
                    if self.grid[x+l, y+dy, z+dz] == 1:
                        contact_area += 1
            score += 2 * contact_area
        if y == 0:
            score += 3 * l * h
        else:
            contact_area = 0
            for dx in range(l):
                for dz in range(h):
                    if self.grid[x+dx, y-1, z+dz] == 1:
                        contact_area += 1
            score += 2 * contact_area
        if y + w == CONTAINER_WIDTH:
            score += 3 * l * h
        else:
            contact_area = 0
            for dx in range(l):
                for dz in range(h):
                    if self.grid[x+dx, y+w, z+dz] == 1:
                        contact_area += 1
            score += 2 * contact_area
        score -= z * 0.5
        return score

    def get_fill_ratio(self):
        return (self.occupied_volume / self.total_volume) * 100

    def get_empty_ratio(self):
        return 100 - self.get_fill_ratio()

    def get_package_count(self):
        return len(self.placements)
