import gymnasium as gym
from stable_baselines3 import PPO
from cartesian_robot_env import CartesianRobotEnv

# 환경 생성
env = CartesianRobotEnv()

# 모델 생성
model = PPO("MlpPolicy", env, verbose=1)

# 모델 학습
model.learn(total_timesteps=10000)

# 모델 저장
model.save("ppo_cartesian_robot")

# 학습된 모델 로드 및 평가
model = PPO.load("ppo_cartesian_robot")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
