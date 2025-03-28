import argparse
import os
import traceback
import numpy as np
import pygame
import matplotlib.pyplot as plt

from constants import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT
from utils import load_packages_from_csv, print_packing_stats
from environment import PackingEnv
from ppo import ppo_train  # Gọi hàm huấn luyện PPO

def main():
    parser = argparse.ArgumentParser(description='Tối ưu hoá sắp xếp kiện hàng sử dụng PPO (Proximal Policy Optimization)')
    parser.add_argument('--episodes', type=int, default=20, help='Số lượng episodes huấn luyện')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate của PPO')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--clip-ratio', type=float, default=0.2, help='Tỉ lệ clip trong PPO')
    parser.add_argument('--update-epochs', type=int, default=10, help='Số lần cập nhật cho mỗi episode')
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
    
    # Nếu sử dụng auto-train hoặc no-graphics, override render để không chờ tương tác
    if args.auto_train or args.no_graphics:
        print("Sử dụng chế độ tự động huấn luyện, không yêu cầu tương tác từ người dùng.")
        render_func = lambda mode='human': True
    else:
        render_func = None

    use_pygame = args.use_pygame
    from environment import OPENGL_AVAILABLE
    if not OPENGL_AVAILABLE:
        use_pygame = True
        print("OpenGL không khả dụng, chuyển sang sử dụng Pygame")

    env = PackingEnv(packages)
    env.fullscreen = args.fullscreen
    env.auto_train = args.auto_train
    env.fast_mode = args.fast_mode
    
    # Gán hàm render nếu cần
    if render_func is not None:
        env.render = render_func
    else:
        if use_pygame:
            env.render = env.render_pure_pygame

    print("=" * 50)
    print("Tối ưu hóa sắp xếp kiện hàng trong container (PPO)")
    print("=" * 50)
    print(f"Kích thước container: {CONTAINER_LENGTH}cm x {CONTAINER_WIDTH}cm x {CONTAINER_HEIGHT}cm")
    for key, info in packages.items():
        print(f"  - {key}: kích thước {info['dimensions']}cm, số lượng: {info['count']}")
    print("=" * 50)
    print("Thông số huấn luyện:")
    print(f"  - Số episodes: {args.episodes}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Discount factor (gamma): {args.gamma}")
    print(f"  - Clip ratio: {args.clip_ratio}")
    print(f"  - Update epochs: {args.update_epochs}")
    print("=" * 50)
    try:
        if not args.no_graphics and render_func is None:
            print("\nĐang khởi tạo cửa sổ hiển thị...")
            env.render()
            print("Cửa sổ hiển thị đã được khởi tạo.")
        
        if args.eval_only:
            print("Chế độ đánh giá cho PPO chưa được triển khai. Vui lòng chạy huấn luyện mới.")
            return
        
        print("\n=== BẮT ĐẦU HUẤN LUYỆN PPO ===")
        actor, critic, training_info = ppo_train(
            env,
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            update_epochs=args.update_epochs
        )
        
        # Lưu model sau khi huấn luyện
        actor.save("best_ppo_actor.h5")
        critic.save("best_ppo_critic.h5")
        print("Model PPO đã được lưu: best_ppo_actor.h5 và best_ppo_critic.h5")
        
        rewards = training_info.get('rewards', [])
        fill_ratios = training_info.get('fill_ratios', [])
        episodes_range = range(1, len(rewards) + 1)
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
