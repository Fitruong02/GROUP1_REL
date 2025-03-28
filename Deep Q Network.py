# main_dqn.py
import argparse
import os
import pickle
import time
import traceback
import numpy as np
import pygame
import matplotlib.pyplot as plt

from constants import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT
from utils import load_packages_from_csv, print_packing_stats
from environment import PackingEnv
from dqn import dqn_train

def main():
    parser = argparse.ArgumentParser(description='Tối ưu hoá sắp xếp kiện hàng sử dụng Deep Q-Network (DQN)')
    parser.add_argument('--episodes', type=int, default=20, help='Số lượng episodes huấn luyện')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate của DQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Giá trị epsilon ban đầu')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Giá trị epsilon nhỏ nhất')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Tỷ lệ giảm epsilon')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size cho huấn luyện')
    parser.add_argument('--memory-size', type=int, default=2000, help='Kích thước bộ nhớ replay')
    parser.add_argument('--target-update', type=int, default=5, help='Số episode cập nhật mạng mục tiêu')
    parser.add_argument('--no-graphics', action='store_true', help='Chạy ở chế độ không đồ họa')
    parser.add_argument('--use-pygame', action='store_true', help='Sử dụng Pygame thay vì OpenGL')
    parser.add_argument('--fullscreen', action='store_true', help='Chạy ở chế độ toàn màn hình')
    parser.add_argument('--auto-train', action='store_true', help='Tự động huấn luyện không cần nhấn nút')
    parser.add_argument('--fast-mode', action='store_true', help='Chạy ở chế độ nhanh, giảm chất lượng đồ họa')
    parser.add_argument('--eval-only', action='store_true', help='Chỉ đánh giá mô hình đã huấn luyện')
    parser.add_argument('--reset-model', action='store_true', help='Bỏ qua mô hình đã lưu và tạo mới')
    parser.add_argument('--packages-csv', type=str, default=None, help='Đường dẫn tới file CSV chứa thông tin kiện hàng')
    args = parser.parse_args()

    packages = load_packages_from_csv(args.packages_csv) if args.packages_csv else load_packages_from_csv()
    
    if args.reset_model and os.path.exists("best_dqn_model.h5"):
        try:
            os.remove("best_dqn_model.h5")
            print("Đã xóa mô hình cũ: best_dqn_model.h5")
        except Exception as e:
            print(f"Không thể xóa mô hình cũ: {e}")

    if args.no_graphics:
        print("Chạy chương trình ở chế độ không đồ họa")

    use_pygame = args.use_pygame
    from environment import OPENGL_AVAILABLE
    if not OPENGL_AVAILABLE:
        use_pygame = True
        print("OpenGL không khả dụng, chuyển sang sử dụng Pygame")

    env = PackingEnv(packages)
    env.fullscreen = args.fullscreen
    env.auto_train = args.auto_train
    env.fast_mode = args.fast_mode

    print("=" * 50)
    print("Tối ưu hóa sắp xếp kiện hàng trong container (DQN)")
    print("=" * 50)
    print(f"Kích thước container: {CONTAINER_LENGTH}cm x {CONTAINER_WIDTH}cm x {CONTAINER_HEIGHT}cm")
    for key, info in packages.items():
        print(f"  - {key}: kích thước {info['dimensions']}cm, số lượng: {info['count']}")
    print("=" * 50)
    print("Thông số huấn luyện:")
    print(f"  - Số episodes: {args.episodes}")
    print(f"  - Learning rate (alpha): {args.learning_rate}")
    print(f"  - Discount factor (gamma): {args.gamma}")
    print(f"  - Epsilon: {args.epsilon_start} -> {args.epsilon_min} (decay: {args.epsilon_decay})")
    print("=" * 50)
    try:
        if args.no_graphics:
            env.render = lambda mode='human': True
        elif use_pygame:
            env.render = env.render_pure_pygame
        else:
            pass

        if not args.no_graphics:
            print("\nĐang khởi tạo cửa sổ hiển thị...")
            env.render()
            print("Cửa sổ hiển thị đã được khởi tạo.")

        if args.eval_only:
            if os.path.exists("best_dqn_model.h5"):
                try:
                    from tensorflow.keras.models import load_model
                    model = load_model("best_dqn_model.h5")
                    print("Đã tải mô hình DQN tốt nhất!")
                    env.render()
                    return
                except Exception as e:
                    print(f"Lỗi khi tải mô hình: {e}")
                    print("Tiếp tục huấn luyện mới...")
            else:
                print("Không tìm thấy mô hình đã huấn luyện, bắt đầu huấn luyện mới.")
        
        print("\n=== BẮT ĐẦU HUẤN LUYỆN DQN ===")
        model, training_info = dqn_train(
            env,
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            target_update_freq=args.target_update
        )
        
        # Sau khi huấn luyện, in ra thống kê và vẽ đồ thị kết quả
        rewards = training_info.get('rewards', [])
        fill_ratios = training_info.get('fill_ratios', [])
        episodes_range = range(1, len(rewards)+1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episodes_range, rewards, marker='o', label='Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward theo Episode')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(episodes_range, fill_ratios, marker='o', color='orange', label='Fill Ratio (%)')
        plt.xlabel('Episode')
        plt.ylabel('Fill Ratio (%)')
        plt.title('Tỷ lệ lấp đầy theo Episode')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("\nChương trình đã bị dừng bởi người dùng")
    except Exception as e:
        print(f"\nLỗi không mong đợi: {e}")
        traceback.print_exc()
    
    print("\nChương trình kết thúc")
    try:
        pygame.quit()
    except:
        pass

if __name__ == "__main__":
    main()
