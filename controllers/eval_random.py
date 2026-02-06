import cv2
import gym
import numpy as np
import time

# ========= 和前面一致的图像 Wrapper =========
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


# ===================== Random Policy 测试 =====================
def eval_random_policy(num_episodes=5):
    env = make_env()

    print("🎲 正在使用 RANDOM POLICY 进行测试...")

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = env.action_space.sample()  # ✅ 随机动作
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward

            # ✅ 可视化窗口
            frame = env.render(mode="rgb_array")
            cv2.imshow("CartPole Random Policy", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.02)

        print(f"🎯 [RANDOM] Episode {ep+1} Reward: {ep_reward}")

    env.close()
    cv2.destroyAllWindows()
    print("✅ Random Policy 测试结束")


if __name__ == "__main__":
    eval_random_policy(num_episodes=5)
