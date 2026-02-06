import random
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# ===================== 图像观察包装器 =====================
class ImageObsWrapper(gym.ObservationWrapper):
    """
    把 CartPole 的低维状态换成 3x96x96 的 RGB 图片作为观察。
    """
    def __init__(self, env, img_size=96):
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, img_size, img_size),
            dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render(mode="rgb_array")  # ✅ gym方式
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
        return frame


def make_env(seed=0, img_size=96):
    env = gym.make("CartPole-v1")
    env.seed(seed)
    env = ImageObsWrapper(env, img_size)
    return env


# ===================== CNN Q 网络 =====================
class CNNQNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),  # → 32×23×23
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), # → 64×10×10
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), # → 64×8×8
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ===================== Replay Buffer =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s.copy(), a, r, s2.copy(), d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r), np.stack(s2), np.array(d)

    def __len__(self):
        return len(self.buffer)


# ===================== 训练主函数 =====================
def train_cartpole_image_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
    num_actions = env.action_space.n

    q_net = CNNQNetwork(num_actions).to(device)
    target_net = CNNQNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay = ReplayBuffer(50000)

    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    eps_min = 0.05
    eps_decay = 0.995
    target_update = 1000

    global_step = 0
    num_episodes = 500   # ✅ 总训练轮数

    for episode in range(1, num_episodes + 1):  # ✅ 从 1 开始方便取模
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            global_step += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                s = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    action = q_net(s).argmax(1).item()

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(replay) > 1000:
                s, a, r, s2, d = replay.sample(batch_size)

                s = torch.FloatTensor(s).to(device) / 255.0
                s2 = torch.FloatTensor(s2).to(device) / 255.0
                a = torch.LongTensor(a).to(device)
                r = torch.FloatTensor(r).to(device)
                d = torch.FloatTensor(d).to(device)

                q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                    q_target = r + gamma * (1 - d) * q_next

                loss = nn.MSELoss()(q, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % target_update == 0:
                    target_net.load_state_dict(q_net.state_dict())

        # ✅ epsilon 衰减
        epsilon = max(epsilon * eps_decay, eps_min)

        print(f"Episode {episode}, Reward: {ep_reward}, Epsilon: {epsilon:.3f}")

        # ✅ ✅ ✅ 每 50 轮自动保存一次模型 ✅ ✅ ✅
        if episode % 50 == 0:
            save_path = f"cartpole_image_dqn_ep{episode}.pth"
            torch.save(q_net.state_dict(), save_path)
            print(f"✅ 模型已保存: {save_path}")

    # ✅ 最终再保存一次
    torch.save(q_net.state_dict(), "cartpole_image_dqn_final.pth")
    env.close()
    print("✅ 训练完成，最终模型已保存")


if __name__ == "__main__":
    train_cartpole_image_dqn()
