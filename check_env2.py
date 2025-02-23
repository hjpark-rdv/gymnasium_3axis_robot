import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 1. 환경 클래스 정의
class MyCustomEnv(gym.Env):
    def __init__(self):
        super(MyCustomEnv, self).__init__()
        # 행동 공간 정의: 예를 들어, 0 또는 1의 이산 공간
        self.action_space = spaces.Discrete(2)
        # 관찰 공간 정의: 예를 들어, -1.0에서 1.0 사이의 연속 공간
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 초기 상태 설정
        self.state = np.zeros(3, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # 상태 변환 로직 구현
        self.state = self.state + (action - 0.5) * 2  # 예시 로직
        # 보상 계산
        reward = -np.sum(np.square(self.state))
        # 에피소드 종료 조건
        done = np.abs(self.state).sum() > 10
        return self.state, reward, done, False, {}

    def render(self):
        # 렌더링 로직 구현 (필요한 경우)
        print(f"State: {self.state}")

# 2. 환경 등록
gym.envs.registration.register(
    id='MyCustomEnv-v0',
    entry_point='__main__:MyCustomEnv',
)

# 3. 환경 사용
if __name__ == '__main__':
    env = gym.make('MyCustomEnv-v0')
    observation, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # 랜덤한 행동 선택
        observation, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            observation, info = env.reset()