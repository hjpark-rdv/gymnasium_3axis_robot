# custom_env/my_custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyCustomEnv(gym.Env):
    def __init__(self):
        super(MyCustomEnv, self).__init__()
        # 행동 공간: 에이전트가 취할 수 있는 행동의 범위 정의
        self.action_space = spaces.Discrete(2)  # 예: 0 또는 1의 이산적 행동
        # 관찰 공간: 환경의 상태를 나타내는 관찰의 범위 정의
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 환경 초기화 및 초기 상태 반환
        self.state = np.random.rand(4)
        return self.state, {}

    def step(self, action):
        # action에 따라 환경의 다음 상태와 보상 계산
        self.state = np.random.rand(4)  # 임의의 다음 상태
        reward = 1.0  # 임의의 보상
        terminated = False  # 에피소드 종료 여부
        truncated = False  # 시간 제한 등으로 에피소드가 중단되었는지 여부
        info = {}  # 추가 정보
        return self.state, reward, terminated, truncated, info

    def render(self):
        # 환경의 현재 상태를 시각화 (필요에 따라 구현)
        print(f"Current state: {self.state}")

    def close(self):
        # 환경 종료 시 필요한 정리 작업 (필요에 따라 구현)
        pass
