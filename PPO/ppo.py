import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as kutils

def build_actor_critic(state_size, action_size, learning_rate):
    # Actor: dự đoán phân phối xác suất cho các hành động (sử dụng softmax)
    state_input = Input(shape=(state_size,))
    common = Dense(64, activation='relu')(state_input)
    common = Dense(64, activation='relu')(common)
    action_probs = Dense(action_size, activation='softmax')(common)
    actor = Model(inputs=state_input, outputs=action_probs)
    actor.compile(optimizer=Adam(learning_rate=learning_rate))
    
    # Critic: ước lượng giá trị của trạng thái (sử dụng Input layer để loại bỏ cảnh báo)
    critic_input = Input(shape=(state_size,))
    x = Dense(64, activation='relu')(critic_input)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    critic = Model(inputs=critic_input, outputs=output)
    critic.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return actor, critic

def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns

def ppo_train(env,
              episodes=20,
              learning_rate=0.001,
              gamma=0.99,
              clip_ratio=0.2,
              update_epochs=10,
              max_steps=400):
    """
    Huấn luyện PPO cho bài toán đóng gói.
    Các bước:
      - Xây dựng mô hình actor-critic.
      - Thu thập trajectory (trạng thái, hành động, phần thưởng) cho mỗi episode.
      - Tính toán giá trị trả về (returns) và lợi thế (advantages).
      - Cập nhật mạng actor với hàm mục tiêu "clipped" và cập nhật critic.
    Thêm: Nếu số bước đạt max_steps (mặc định 400), episode sẽ dừng và chuyển qua episode mới.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    actor, critic = build_actor_critic(state_size, action_size, learning_rate)
    
    total_rewards = []
    fill_ratios = []
    
    for ep in range(episodes):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        old_probs = []
        ep_reward = 0
        steps = 0
        done = False
        
        while not done:
            state_input = state.reshape(1, -1)
            probs = actor.predict(state_input, verbose=0)[0]
            # Chọn hành động theo phân phối xác suất
            action = np.random.choice(action_size, p=probs)
            
            states.append(state)
            actions.append(action)
            old_probs.append(probs[action])
            
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = next_state
            ep_reward += reward
            steps += 1
            
            if steps >= max_steps:
                print("Đã đạt số bước tối đa ({} bước), chuyển sang episode tiếp theo.".format(max_steps))
                done = True
        
        total_rewards.append(ep_reward)
        fill_ratios.append(env.container.get_fill_ratio())
        
        # Tính toán returns và advantage
        returns = compute_returns(np.array(rewards), gamma)
        values = critic.predict(np.array(states), verbose=0).flatten()
        advantages = returns - values
        
        # Chuyển đổi hành động sang one-hot encoding
        actions_onehot = kutils.to_categorical(actions, num_classes=action_size)
        states_np = np.array(states)
        old_probs_np = np.array(old_probs)
        
        # Cập nhật actor và critic qua nhiều epoch
        for _ in range(update_epochs):
            with tf.GradientTape() as tape:
                new_probs = actor(states_np, training=True)
                # Lấy xác suất của các hành động đã thực hiện
                new_probs_act = tf.reduce_sum(new_probs * actions_onehot, axis=1)
                ratio = new_probs_act / (old_probs_np + 1e-10)
                clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            
            # Cập nhật critic với MSE loss
            critic.train_on_batch(states_np, returns)
        
        print(f"Episode {ep+1}/{episodes} - Reward: {ep_reward:.2f}, Fill Ratio: {env.container.get_fill_ratio():.2f}%")
    
    print("Huấn luyện PPO hoàn tất!")
    print(f"Tỷ lệ lấp đầy trung bình: {np.mean(fill_ratios):.2f}%")
    print(f"Phần thưởng trung bình: {np.mean(total_rewards):.2f}")
    
    return actor, critic, {'rewards': total_rewards, 'fill_ratios': fill_ratios}
