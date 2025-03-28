# environment.py

import gym
from gym import spaces
import numpy as np
import pygame
import os
import time
import traceback

# Kiểm tra và nhập các thư viện OpenGL với xử lý ngoại lệ
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
    GLUT_AVAILABLE = False  # Không sử dụng GLUT để tránh lỗi
except ImportError:
    print("Thư viện OpenGL không khả dụng. Vui lòng cài đặt PyOpenGL.")
    OPENGL_AVAILABLE = False
    GLUT_AVAILABLE = False

from container import Container
from constants import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT, COLORS_2D
from utils import create_color_map

class PackingEnv(gym.Env):
    def __init__(self, package_types):
        super(PackingEnv, self).__init__()
        self.package_types = package_types
        self.action_space = spaces.Discrete(len(package_types))
        self.observation_space = spaces.MultiDiscrete([package_types[key]['count'] + 1 for key in package_types])
        
        # Lưu lịch sử sắp xếp và tỷ lệ lấp đầy trước đó
        self.placement_history = set()
        self.previous_fill_ratio = 0
        
        self.reset()
        
        # Các biến liên quan đến render
        self.display = None
        self.screen = None
        self.font = None
        self.initialized = False

        self.display_2d = None
        self.screen_2d = None
        self.font_2d = None
        self.initialized_2d = False

        self.last_time = time.time()

        # Tạo màu cho các loại kiện hàng (chuyển từ RGB 0-255 sang OpenGL 0-1)
        self.colors_3d = {}
        package_colors = create_color_map(package_types)
        for key in package_types:
            r, g, b = package_colors.get(key, (255, 255, 255))
            self.colors_3d[key] = (r / 255, g / 255, b / 255, 0.7)

    def calculate_tight_faces(self, x, y, z, l, w, h):
        """Tính số mặt tiếp xúc của kiện hàng với container hoặc các kiện khác."""
        tight_faces = 0

        # Mặt dưới
        if z == 0:
            tight_faces += 1
        else:
            contact = False
            for dx in range(l):
                for dy in range(w):
                    if self.container.grid[x+dx, y+dy, z-1] == 1:
                        contact = True
                        break
                if contact:
                    break
            if contact:
                tight_faces += 1

        # Mặt trên
        if z + h == CONTAINER_HEIGHT:
            tight_faces += 1
        else:
            contact = False
            for dx in range(l):
                for dy in range(w):
                    if z + h < CONTAINER_HEIGHT and self.container.grid[x+dx, y+dy, z+h] == 1:
                        contact = True
                        break
                if contact:
                    break
            if contact:
                tight_faces += 1

        # Mặt trái
        if x == 0:
            tight_faces += 1
        else:
            contact = False
            for dy in range(w):
                for dz in range(h):
                    if self.container.grid[x-1, y+dy, z+dz] == 1:
                        contact = True
                        break
                if contact:
                    break
            if contact:
                tight_faces += 1

        # Mặt phải
        if x + l == CONTAINER_LENGTH:
            tight_faces += 1
        else:
            contact = False
            for dy in range(w):
                for dz in range(h):
                    if x + l < CONTAINER_LENGTH and self.container.grid[x+l, y+dy, z+dz] == 1:
                        contact = True
                        break
                if contact:
                    break
            if contact:
                tight_faces += 1

        # Mặt trước
        if y == 0:
            tight_faces += 1
        else:
            contact = False
            for dx in range(l):
                for dz in range(h):
                    if self.container.grid[x+dx, y-1, z+dz] == 1:
                        contact = True
                        break
                if contact:
                    break
            if contact:
                tight_faces += 1

        # Mặt sau
        if y + w == CONTAINER_WIDTH:
            tight_faces += 1
        else:
            contact = False
            for dx in range(l):
                for dz in range(h):
                    if y + w < CONTAINER_WIDTH and self.container.grid[x+dx, y+w, z+dz] == 1:
                        contact = True
                        break
                if contact:
                    break
            if contact:
                tight_faces += 1

        return tight_faces

    def calculate_empty_space_penalty(self):
        empty_volume = self.container.total_volume - self.container.occupied_volume
        empty_ratio = empty_volume / self.container.total_volume
        empty_space_penalty = 5 * empty_ratio
        return empty_space_penalty

    def calculate_placement_novelty_reward(self):
        current_placement = tuple(sorted((p[0], tuple(p[1]), tuple(p[2])) for p in self.container.placements))
        if current_placement not in self.placement_history:
            self.placement_history.add(current_placement)
            return 5
        return 0

    def reset(self):
        self.container = Container(CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
        self.remaining = {key: self.package_types[key]['count'] for key in self.package_types}
        self.previous_fill_ratio = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.remaining[key] for key in self.package_types])

    def step(self, action):
        package_key = list(self.package_types.keys())[action]
        reward = -1
        done = False
        old_fill_ratio = self.container.get_fill_ratio()

        if self.remaining[package_key] > 0:
            package = self.package_types[package_key]
            position = self.container.find_optimal_position(package['dimensions'])
            if position:
                x, y, z = position
                l, w, h = package['dimensions']
                tight_faces = self.calculate_tight_faces(x, y, z, l, w, h)
                self.container.place_package(package['dimensions'], x, y, z, package_key)
                self.remaining[package_key] -= 1
                volume_reward = package['volume'] * 2
                contact_reward = tight_faces * 10
                new_fill_ratio = self.container.get_fill_ratio()
                fill_ratio_reward = (new_fill_ratio - old_fill_ratio) * 100
                empty_space_penalty = self.calculate_empty_space_penalty()
                novelty_reward = self.calculate_placement_novelty_reward()
                reward = volume_reward + contact_reward + fill_ratio_reward - empty_space_penalty + novelty_reward
                self.previous_fill_ratio = new_fill_ratio

        done = self._is_done()
        if done:
            fill_ratio = self.container.get_fill_ratio()
            package_count = self.container.get_package_count()
            final_reward = package_count * 10 + fill_ratio * 5
            reward += final_reward

        obs = self._get_obs()
        return obs, reward, done, {}

    def _is_done(self):
        if all(count == 0 for count in self.remaining.values()):
            return True
        for key, count in self.remaining.items():
            if count > 0:
                dimensions = self.package_types[key]['dimensions']
                for z in range(0, min(5, CONTAINER_HEIGHT - dimensions[2] + 1), 2):
                    for y in range(0, CONTAINER_WIDTH - dimensions[1] + 1, 4):
                        for x in range(0, CONTAINER_LENGTH - dimensions[0] + 1, 4):
                            if self.container.can_place(dimensions, x, y, z):
                                return False
                for z in range(5, CONTAINER_HEIGHT - dimensions[2] + 1, 5):
                    for y in range(0, CONTAINER_WIDTH - dimensions[1] + 1, 5):
                        for x in range(0, CONTAINER_LENGTH - dimensions[0] + 1, 5):
                            if self.container.can_place(dimensions, x, y, z):
                                return False
        return True

    # Các phương thức render: OpenGL, render 2D và Pure Pygame
    def render(self, mode='human'):
        if not OPENGL_AVAILABLE:
            print("Không thể hiển thị do thiếu thư viện OpenGL. Vui lòng cài đặt PyOpenGL.")
            return True
        try:
            fast_mode = hasattr(self, 'fast_mode') and self.fast_mode
            if not self.initialized:
                pygame.init()
                info = pygame.display.Info()
                screen_width, screen_height = info.current_w, info.current_h
                window_width = int(screen_width * 0.8)
                window_height = int(screen_height * 0.8)
                self.display = (window_width, window_height)
                pygame.display.set_caption("Mô hình 3D container - Q-Learning")
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{(screen_width - window_width) // 2},{(screen_height - window_height) // 2}"
                pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
                pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 16)
                if fast_mode:
                    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)
                    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
                if hasattr(self, 'fullscreen') and self.fullscreen:
                    self.screen = pygame.display.set_mode(self.display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.FULLSCREEN)
                else:
                    self.screen = pygame.display.set_mode(self.display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                glClearColor(0.05, 0.05, 0.05, 1.0)
                glEnable(GL_DEPTH_TEST)
                if fast_mode:
                    glDisable(GL_BLEND)
                    glDisable(GL_LINE_SMOOTH)
                    glDisable(GL_POLYGON_SMOOTH)
                    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST)
                    glShadeModel(GL_FLAT)
                else:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                glEnable(GL_COLOR_MATERIAL)
                light_ambient = [0.5, 0.5, 0.5, 1.0]
                light_diffuse = [1.0, 1.0, 1.0, 1.0]
                light_position = [10.0, 10.0, 10.0, 0.0]
                glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
                glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
                glLightfv(GL_LIGHT0, GL_POSITION, light_position)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 2000.0)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glTranslatef(-CONTAINER_LENGTH / 2, -CONTAINER_WIDTH / 2, -CONTAINER_HEIGHT * 3)
                self.container_list = glGenLists(1)
                glNewList(self.container_list, GL_COMPILE)
                self._draw_container_edges()
                glEndList()
                pygame.font.init()
                try:
                    self.font = pygame.font.SysFont('Arial', 18)
                except:
                    self.font = pygame.font.Font(None, 18)
                self.buttons = {
                    'auto': pygame.Rect(10, 10, 80, 30),
                    'pause': pygame.Rect(100, 10, 80, 30),
                    'next': pygame.Rect(190, 10, 80, 30)
                }
                self.camera_state = {
                    'rotate_x': 20,
                    'rotate_y': 30,
                    'zoom': 0.5,
                    'translate_x': 0,
                    'translate_y': 0
                }
                if not hasattr(self, 'auto_mode'):
                    self.auto_mode = False
                self.button_surface = pygame.Surface((300, 50), pygame.SRCALPHA)
                self.initialized = True
                print("Đã khởi tạo thành công giao diện OpenGL")
            running = True
            next_step = False
            clock = pygame.time.Clock()
            auto_train = hasattr(self, 'auto_train') and self.auto_train
            rotate_x = self.camera_state['rotate_x']
            rotate_y = self.camera_state['rotate_y']
            zoom = self.camera_state['zoom']
            translate_x = self.camera_state['translate_x']
            translate_y = self.camera_state['translate_y']
            while running and not next_step:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key in (pygame.K_SPACE, pygame.K_n):
                            next_step = True
                        elif event.key == pygame.K_r:
                            rotate_x, rotate_y = 20, 30
                            zoom = 0.5
                            translate_x, translate_y = 0, 0
                        elif event.key == pygame.K_a:
                            self.auto_mode = not self.auto_mode
                        elif event.key == pygame.K_f:
                            current_rotate_x, current_rotate_y = rotate_x, rotate_y
                            current_zoom = zoom
                            current_translate_x, current_translate_y = translate_x, translate_y
                            if pygame.display.get_surface().get_flags() & pygame.FULLSCREEN:
                                self.screen = pygame.display.set_mode(self.display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                            else:
                                self.screen = pygame.display.set_mode(self.display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.FULLSCREEN)
                            glClearColor(0.05, 0.05, 0.05, 1.0)
                            glEnable(GL_DEPTH_TEST)
                            if not fast_mode:
                                glEnable(GL_BLEND)
                                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glEnable(GL_LIGHTING)
                            glEnable(GL_LIGHT0)
                            glEnable(GL_COLOR_MATERIAL)
                            light_ambient = [0.5, 0.5, 0.5, 1.0]
                            light_diffuse = [1.0, 1.0, 1.0, 1.0]
                            light_position = [10.0, 10.0, 10.0, 0.0]
                            glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
                            glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
                            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
                            glMatrixMode(GL_PROJECTION)
                            glLoadIdentity()
                            gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 2000.0)
                            glMatrixMode(GL_MODELVIEW)
                            glLoadIdentity()
                            rotate_x, rotate_y = current_rotate_x, current_rotate_y
                            zoom = current_zoom
                            translate_x, translate_y = current_translate_x, current_translate_y
                    elif event.type == pygame.MOUSEMOTION:
                        if event.buttons[0]:
                            rotate_y += event.rel[0] * 0.5
                            rotate_x += event.rel[1] * 0.5
                        elif event.buttons[2]:
                            translate_x += event.rel[0] * 0.1
                            translate_y -= event.rel[1] * 0.1
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            mouse_pos = event.pos
                            for button_name, button_rect in self.buttons.items():
                                if button_rect.collidepoint(mouse_pos):
                                    if button_name == 'auto':
                                        self.auto_mode = True
                                    elif button_name == 'pause':
                                        self.auto_mode = False
                                    elif button_name == 'next':
                                        next_step = True
                        elif event.button == 4:
                            zoom *= 1.1
                        elif event.button == 5:
                            zoom *= 0.9
                    elif event.type == pygame.VIDEORESIZE:
                        self.display = (event.w, event.h)
                        self.screen = pygame.display.set_mode(self.display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                        glMatrixMode(GL_PROJECTION)
                        glLoadIdentity()
                        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 2000.0)
                        glMatrixMode(GL_MODELVIEW)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glTranslatef(translate_x, translate_y, -500)
                glRotatef(rotate_x, 1, 0, 0)
                glRotatef(rotate_y, 0, 1, 0)
                glScalef(zoom, zoom, zoom)
                light_position = [10.0, 10.0, 10.0, 0.0]
                glLightfv(GL_LIGHT0, GL_POSITION, light_position)
                glCallList(self.container_list)
                self._draw_packages(fast_mode)
                self._draw_buttons()
                pygame.display.flip()
                if fast_mode:
                    clock.tick(0)
                else:
                    clock.tick(60)
                if auto_train or self.auto_mode:
                    next_step = True
            self.camera_state = {
                'rotate_x': rotate_x,
                'rotate_y': rotate_y,
                'zoom': zoom,
                'translate_x': translate_x,
                'translate_y': translate_y
            }
            if not running:
                pygame.quit()
                self.initialized = False
            return next_step
        except Exception as e:
            print(f"Lỗi khi hiển thị: {e}")
            traceback.print_exc()
            try:
                pygame.quit()
            except:
                pass
            self.initialized = False
            return True

    def _draw_buttons(self):
        glPushMatrix()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        self.button_surface.fill((0, 0, 0, 0))
        for button_name, button_rect in self.buttons.items():
            if button_name == 'auto' and self.auto_mode:
                color = (0, 200, 0, 150)
            elif button_name == 'pause' and not self.auto_mode:
                color = (200, 0, 0, 150)
            else:
                color = (100, 100, 100, 150)
            button_surface = pygame.Surface((button_rect.width, button_rect.height), pygame.SRCALPHA)
            button_surface.fill(color)
            pygame.draw.rect(button_surface, (255, 255, 255, 150), pygame.Rect(0, 0, button_rect.width, button_rect.height), 1)
            text_surface = self.font.render(button_name.upper(), True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(button_rect.width/2, button_rect.height/2))
            button_surface.blit(text_surface, text_rect)
            self.button_surface.blit(button_surface, (button_rect.x, button_rect.y))
        pygame.display.get_surface().blit(self.button_surface, (0, 0))
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_container_edges(self):
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        edges = [
            ((0, 0, 0), (CONTAINER_LENGTH, 0, 0)),
            ((CONTAINER_LENGTH, 0, 0), (CONTAINER_LENGTH, CONTAINER_WIDTH, 0)),
            ((CONTAINER_LENGTH, CONTAINER_WIDTH, 0), (0, CONTAINER_WIDTH, 0)),
            ((0, CONTAINER_WIDTH, 0), (0, 0, 0)),
            ((0, 0, CONTAINER_HEIGHT), (CONTAINER_LENGTH, 0, CONTAINER_HEIGHT)),
            ((CONTAINER_LENGTH, 0, CONTAINER_HEIGHT), (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)),
            ((CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT), (0, CONTAINER_WIDTH, CONTAINER_HEIGHT)),
            ((0, CONTAINER_WIDTH, CONTAINER_HEIGHT), (0, 0, CONTAINER_HEIGHT)),
            ((0, 0, 0), (0, 0, CONTAINER_HEIGHT)),
            ((CONTAINER_LENGTH, 0, 0), (CONTAINER_LENGTH, 0, CONTAINER_HEIGHT)),
            ((CONTAINER_LENGTH, CONTAINER_WIDTH, 0), (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)),
            ((0, CONTAINER_WIDTH, 0), (0, CONTAINER_WIDTH, CONTAINER_HEIGHT))
        ]
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertex)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_packages(self, fast_mode=False):
        for package_key, (x, y, z), (l, w, h) in self.container.placements:
            if package_key in self.colors_3d:
                color = self.colors_3d[package_key]
            else:
                color = (1.0, 0.0, 0.0, 1.0)
            glColor3f(color[0], color[1], color[2])
            glBegin(GL_QUADS)
            # Mặt dưới
            glNormal3f(0, 0, -1)
            glVertex3f(x, y, z)
            glVertex3f(x + l, y, z)
            glVertex3f(x + l, y + w, z)
            glVertex3f(x, y + w, z)
            # Mặt trên
            glNormal3f(0, 0, 1)
            glVertex3f(x, y, z + h)
            glVertex3f(x + l, y, z + h)
            glVertex3f(x + l, y + w, z + h)
            glVertex3f(x, y + w, z + h)
            # Mặt trước
            glNormal3f(0, -1, 0)
            glVertex3f(x, y, z)
            glVertex3f(x + l, y, z)
            glVertex3f(x + l, y, z + h)
            glVertex3f(x, y, z + h)
            # Mặt sau
            glNormal3f(0, 1, 0)
            glVertex3f(x, y + w, z)
            glVertex3f(x + l, y + w, z)
            glVertex3f(x + l, y + w, z + h)
            glVertex3f(x, y + w, z + h)
            # Mặt trái
            glNormal3f(-1, 0, 0)
            glVertex3f(x, y, z)
            glVertex3f(x, y + w, z)
            glVertex3f(x, y + w, z + h)
            glVertex3f(x, y, z + h)
            # Mặt phải
            glNormal3f(1, 0, 0)
            glVertex3f(x + l, y, z)
            glVertex3f(x + l, y + w, z)
            glVertex3f(x + l, y + w, z + h)
            glVertex3f(x + l, y, z + h)
            glEnd()
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(x, y, z)
            glVertex3f(x + l, y, z)
            glVertex3f(x + l, y + w, z)
            glVertex3f(x, y + w, z)
            glEnd()
            glBegin(GL_LINE_LOOP)
            glVertex3f(x, y, z + h)
            glVertex3f(x + l, y, z + h)
            glVertex3f(x + l, y + w, z + h)
            glVertex3f(x, y + w, z + h)
            glEnd()
            glBegin(GL_LINES)
            glVertex3f(x, y, z)
            glVertex3f(x, y, z + h)
            glVertex3f(x + l, y, z)
            glVertex3f(x + l, y, z + h)
            glVertex3f(x + l, y + w, z)
            glVertex3f(x + l, y + w, z + h)
            glVertex3f(x, y + w, z)
            glVertex3f(x, y + w, z + h)
            glEnd()
            glEnable(GL_LIGHTING)

    def render_text_opengl(self, x, y, text):
        if not GLUT_AVAILABLE:
            return
        try:
            if not bool(glutBitmapCharacter):
                return
            glRasterPos2f(x, y)
            for character in text:
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(character))
        except Exception as e:
            print(f"Không thể hiển thị text trong OpenGL: {e}")
            pass

    def render_2d(self, mode='human'):
        if not self.initialized_2d:
            pygame.init()
            pygame.font.init()
            self.display_2d = (800, 600)
            self.screen_2d = pygame.display.set_mode(self.display_2d)
            pygame.display.set_caption("Tối ưu hóa đóng gói 3D - Q-Learning (Hiển thị 2D)")
            try:
                self.font_2d = pygame.font.SysFont('Arial', 18)
                self.font_large_2d = pygame.font.SysFont('Arial', 24)
            except:
                self.font_2d = pygame.font.Font(None, 18)
                self.font_large_2d = pygame.font.Font(None, 24)
            self.buttons_2d = {
                'Next': pygame.Rect(10, 10, 80, 30),
                'Quit': pygame.Rect(280, 10, 80, 30)
            }
            self.initialized_2d = True
        clock = pygame.time.Clock()
        running = True
        next_step = False
        scale_x = 350 / CONTAINER_LENGTH
        scale_y = 250 / CONTAINER_WIDTH
        scale_z = 250 / CONTAINER_HEIGHT
        container_origin = (400, 300)
        while running and not next_step:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_n):
                        next_step = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        for button_name, button_rect in self.buttons_2d.items():
                            if button_rect.collidepoint(mouse_pos):
                                if button_name == 'Next':
                                    next_step = True
                                elif button_name == 'Quit':
                                    running = False
            self.screen_2d.fill(COLORS_2D['background'])
            title = self.font_large_2d.render("Tối ưu hóa đóng gói 3D - Hiển thị 2D", True, COLORS_2D['text'])
            self.screen_2d.blit(title, (self.display_2d[0]//2 - title.get_width()//2, 50))
            pygame.draw.rect(
                self.screen_2d,
                COLORS_2D['container'],
                pygame.Rect(
                    container_origin[0] - CONTAINER_LENGTH * scale_x / 2,
                    container_origin[1] - CONTAINER_HEIGHT * scale_z / 2,
                    CONTAINER_LENGTH * scale_x,
                    CONTAINER_HEIGHT * scale_z
                ),
                1
            )
            for package_key, (x, y, z), (l, w, h) in self.container.placements:
                rect_x = container_origin[0] - CONTAINER_LENGTH * scale_x / 2 + x * scale_x
                rect_z = container_origin[1] - CONTAINER_HEIGHT * scale_z / 2 + z * scale_z
                rect_width = l * scale_x
                rect_height = h * scale_z
                # Sử dụng hàm create_color_map để lấy màu (hoặc dùng giá trị từ COLORS_2D)
                color = create_color_map(self.package_types).get(package_key, (255, 0, 0))
                pygame.draw.rect(
                    self.screen_2d,
                    color,
                    pygame.Rect(rect_x, rect_z, rect_width, rect_height)
                )
                pygame.draw.rect(
                    self.screen_2d,
                    (255, 255, 255),
                    pygame.Rect(rect_x, rect_z, rect_width, rect_height),
                    1
                )
                if l * scale_x > 20 and h * scale_z > 20:
                    label = self.font_2d.render(package_key[0].upper(), True, (255, 255, 255))
                    self.screen_2d.blit(
                        label,
                        (rect_x + rect_width/2 - label.get_width()/2, rect_z + rect_height/2 - label.get_height()/2)
                    )
            stats_bg = pygame.Rect(10, 100, 250, 150)
            pygame.draw.rect(self.screen_2d, (30, 30, 30), stats_bg)
            pygame.draw.rect(self.screen_2d, (100, 100, 100), stats_bg, 1)
            fill_ratio = self.container.get_fill_ratio()
            empty_ratio = self.container.get_empty_ratio()
            package_count = self.container.get_package_count()
            stats = [
                f"Ty le lap day: {fill_ratio:.2f}%",
                f"Ty le con trong: {empty_ratio:.2f}%",
                f"So luong kien hang: {package_count}"
            ]
            for i, stat in enumerate(stats):
                text_surface = self.font_2d.render(stat, True, COLORS_2D['text'])
                self.screen_2d.blit(text_surface, (20, 110 + i * 30))
            packages_bg = pygame.Rect(10, 260, 250, 150)
            pygame.draw.rect(self.screen_2d, (30, 30, 30), packages_bg)
            pygame.draw.rect(self.screen_2d, (100, 100, 100), packages_bg, 1)
            text_surface = self.font_2d.render("Kien hang con lai:", True, COLORS_2D['text'])
            self.screen_2d.blit(text_surface, (20, 270))
            y_offset = 300
            for i, (key, count) in enumerate(self.remaining.items()):
                if i >= 5:
                    more_text = self.font_2d.render(f"...va {len(self.remaining) - 5} loai khac", True, COLORS_2D['text'])
                    self.screen_2d.blit(more_text, (20, y_offset))
                    break
                dimensions = self.package_types[key]['dimensions']
                text = f"{key}: {count} ({dimensions[0]}x{dimensions[1]}x{dimensions[2]})"
                text_surface = self.font_2d.render(text, True, COLORS_2D['text'])
                self.screen_2d.blit(text_surface, (20, y_offset))
                y_offset += 25
            controls_bg = pygame.Rect(self.display_2d[0] - 250, 100, 240, 150)
            pygame.draw.rect(self.screen_2d, (30, 30, 30), controls_bg)
            pygame.draw.rect(self.screen_2d, (100, 100, 100), controls_bg, 1)
            controls = [
                "Dieu khien:",
                "Space hoac N: Buoc tiep theo",
                "ESC: Thoat"
            ]
            for i, control in enumerate(controls):
                text_surface = self.font_2d.render(control, True, COLORS_2D['text'])
                self.screen_2d.blit(text_surface, (self.display_2d[0] - 240, 110 + i * 25))
            view_text = self.font_2d.render("Goc nhin: Mat chinh dien (x, z)", True, COLORS_2D['text'])
            self.screen_2d.blit(view_text, (container_origin[0] - view_text.get_width()/2, container_origin[1] + CONTAINER_HEIGHT * scale_z / 2 + 20))
            mouse_pos = pygame.mouse.get_pos()
            for button_name, button_rect in self.buttons_2d.items():
                color = COLORS_2D['button']
                if button_rect.collidepoint(mouse_pos):
                    color = COLORS_2D['button_hover']
                pygame.draw.rect(self.screen_2d, color, button_rect)
                pygame.draw.rect(self.screen_2d, (255, 255, 255), button_rect, 1)
                text_surface = self.font_2d.render(button_name, True, COLORS_2D['text'])
                text_rect = text_surface.get_rect(center=button_rect.center)
                self.screen_2d.blit(text_surface, text_rect)
            pygame.display.flip()
            clock.tick(60)
        if not running:
            pygame.quit()
            self.initialized_2d = False
        return next_step

    def render_pure_pygame(self, mode='human'):
        if not hasattr(self, 'pygame_initialized') or not self.pygame_initialized:
            pygame.init()
            pygame.font.init()
            self.pygame_display = (800, 600)
            self.pygame_screen = pygame.display.set_mode(self.pygame_display)
            pygame.display.set_caption("Tối ưu hóa sắp xếp kiện hàng 3D - Q-Learning (Pygame)")
            try:
                self.pygame_font = pygame.font.SysFont('Arial', 18)
                self.pygame_font_title = pygame.font.SysFont('Arial', 24)
            except:
                self.pygame_font = pygame.font.Font(None, 18)
                self.pygame_font_title = pygame.font.Font(None, 24)
            self.pygame_buttons = {
                'Next': pygame.Rect(10, 10, 80, 30),
                'Quit': pygame.Rect(100, 10, 80, 30)
            }
            self.pygame_initialized = True
        clock = pygame.time.Clock()
        running = True
        next_step = False
        scale_x = 350 / CONTAINER_LENGTH
        scale_y = 250 / CONTAINER_WIDTH
        scale_z = 250 / CONTAINER_HEIGHT
        container_origin = (400, 300)
        while running and not next_step:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_n):
                        next_step = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        for button_name, button_rect in self.pygame_buttons.items():
                            if button_rect.collidepoint(mouse_pos):
                                if button_name == 'Next':
                                    next_step = True
                                elif button_name == 'Quit':
                                    running = False
            self.pygame_screen.fill(COLORS_2D['background'])
            title = self.pygame_font_title.render("Tối ưu hóa sắp xếp kiện hàng 3D - Q-Learning", True, COLORS_2D['text'])
            self.pygame_screen.blit(title, (self.pygame_display[0]//2 - title.get_width()//2, 50))
            pygame.draw.rect(
                self.pygame_screen,
                COLORS_2D['container'],
                pygame.Rect(
                    container_origin[0] - CONTAINER_LENGTH * scale_x / 2,
                    container_origin[1] - CONTAINER_HEIGHT * scale_z / 2,
                    CONTAINER_LENGTH * scale_x,
                    CONTAINER_HEIGHT * scale_z
                ),
                1
            )
            for package_key, (x, y, z), (l, w, h) in self.container.placements:
                rect_x = container_origin[0] - CONTAINER_LENGTH * scale_x / 2 + x * scale_x
                rect_z = container_origin[1] - CONTAINER_HEIGHT * scale_z / 2 + z * scale_z
                rect_width = l * scale_x
                rect_height = h * scale_z
                color = create_color_map(self.package_types).get(package_key, (255, 0, 0))
                pygame.draw.rect(
                    self.pygame_screen,
                    color,
                    pygame.Rect(rect_x, rect_z, rect_width, rect_height)
                )
                pygame.draw.rect(
                    self.pygame_screen,
                    (255, 255, 255),
                    pygame.Rect(rect_x, rect_z, rect_width, rect_height),
                    1
                )
                if l * scale_x > 20 and h * scale_z > 20:
                    label = self.pygame_font.render(package_key[0].upper(), True, (255, 255, 255))
                    self.pygame_screen.blit(
                        label,
                        (rect_x + rect_width/2 - label.get_width()/2, rect_z + rect_height/2 - label.get_height()/2)
                    )
            stats_bg = pygame.Rect(10, 100, 250, 150)
            pygame.draw.rect(self.pygame_screen, (30, 30, 30), stats_bg)
            pygame.draw.rect(self.pygame_screen, (100, 100, 100), stats_bg, 1)
            fill_ratio = self.container.get_fill_ratio()
            empty_ratio = self.container.get_empty_ratio()
            package_count = self.container.get_package_count()
            stats = [
                f"Ty le lap day: {fill_ratio:.2f}%",
                f"Ty le con trong: {empty_ratio:.2f}%",
                f"So luong kien hang: {package_count}"
            ]
            for i, stat in enumerate(stats):
                text_surface = self.pygame_font.render(stat, True, COLORS_2D['text'])
                self.pygame_screen.blit(text_surface, (20, 110 + i * 30))
            packages_bg = pygame.Rect(10, 260, 250, 150)
            pygame.draw.rect(self.pygame_screen, (30, 30, 30), packages_bg)
            pygame.draw.rect(self.pygame_screen, (100, 100, 100), packages_bg, 1)
            text_surface = self.pygame_font.render("Kien hang con lai:", True, COLORS_2D['text'])
            self.pygame_screen.blit(text_surface, (20, 270))
            y_offset = 300
            for i, (key, count) in enumerate(self.remaining.items()):
                if i >= 5:
                    more_text = self.pygame_font.render(f"...va {len(self.remaining) - 5} loai khac", True, COLORS_2D['text'])
                    self.pygame_screen.blit(more_text, (20, y_offset))
                    break
                dimensions = self.package_types[key]['dimensions']
                text = f"{key}: {count} ({dimensions[0]}x{dimensions[1]}x{dimensions[2]})"
                text_surface = self.pygame_font.render(text, True, COLORS_2D['text'])
                self.pygame_screen.blit(text_surface, (20, y_offset))
                y_offset += 25
            controls_bg = pygame.Rect(self.pygame_display[0] - 250, 100, 240, 150)
            pygame.draw.rect(self.pygame_screen, (30, 30, 30), controls_bg)
            pygame.draw.rect(self.pygame_screen, (100, 100, 100), controls_bg, 1)
            controls = [
                "Dieu khien:",
                "Space hoac N: Buoc tiep theo",
                "ESC: Thoat"
            ]
            for i, control in enumerate(controls):
                text_surface = self.pygame_font.render(control, True, COLORS_2D['text'])
                self.pygame_screen.blit(text_surface, (self.pygame_display[0] - 240, 110 + i * 25))
            pygame.display.flip()
            clock.tick(60)
        if not running:
            pygame.quit()
            self.pygame_initialized = False
        return next_step
