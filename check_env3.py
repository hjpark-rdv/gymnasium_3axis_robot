import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.state = np.zeros(6, dtype=np.float32)

    def reset(self,seed=None, options=None):
        self.state = np.zeros(6, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.state += np.concatenate((action, action))
        reward = -np.sum(np.square(self.state))
        terminated = np.linalg.norm(self.state) > 10
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")

gym.envs.registration.register(
    id='CustomEnv-v0',
    entry_point='__main__:CustomEnv',
)

if __name__ == "__main__":
    env = gym.make('CustomEnv-v0')
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()