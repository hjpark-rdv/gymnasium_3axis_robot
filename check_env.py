import numpy as np
from cartesian_robot_env import CartesianRobotEnv

def test_cartesian_robot_env():
    # 환경 초기화
    env = CartesianRobotEnv()
    
    # 환경 재설정 및 초기 관찰값 출력
    observation, info = env.reset()
    print(f"Initial observation: {observation}")
    
    # 랜덤한 행동 선택
    action = env.action_space.sample()
    print(f"Sampled action: {action}")
    
    # 선택한 행동을 환경에 적용
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 결과 출력
    print(f"Observation after action: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    
    # 환경 종료
    env.close()

if __name__ == "__main__":
    test_cartesian_robot_env()

