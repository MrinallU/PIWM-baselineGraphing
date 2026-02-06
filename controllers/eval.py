import cv2
import torch
import torch.nn as nn
import gym
import numpy as np
import time

# ============ 和训练时完全一致的 Wrapper ============
class ImageObsWrapper(gym.ObservationWrapper):
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
        frame = self.env.render(mode="rgb_array")
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = np.transpose(frame, (2, 0, 1))
        return frame


def make_env():
    env = gym.make("CartPole-v1")
    env = ImageObsWrapper(env)
    return env


# ============ 和训练时完全一致的 CNN 网络 ============
class CNNQNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
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


# ===================== 测试主函数 =====================
def eval_model(model_path="cartpole_image_dqn_ep50.pth", num_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
    num_actions = env.action_space.n

    model = CNNQNetwork(num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✅ 已加载模型: {model_path}")

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                action = model(s).argmax(1).item()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward

            # ✅ 实时显示窗口 (OpenCV)
            frame = env.render(mode="rgb_array")
            cv2.imshow("CartPole Image DQN (Evaluation)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.02)  # 控制播放速度

        print(f"🎯 Episode {ep+1} Reward: {ep_reward}")

    env.close()
    cv2.destroyAllWindows()
    print("✅ 测试结束")


if __name__ == "__main__":
    eval_model(
        model_path="cartpole_image_dqn_ep50.pth",
        num_episodes=5
    )
