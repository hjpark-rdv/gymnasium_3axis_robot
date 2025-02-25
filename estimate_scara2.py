from stable_baselines3 import PPO
import numpy as np
import pybullet as p
import time
from train_scara import ScaraEnv


# 평가용 환경 생성 (GUI 모드로 시각화)
env = ScaraEnv(render_mode=True)
obs, _ = env.reset()

x_min, x_max = 0.0, 0.87
y_min, y_max = -0.87, 0.87
z_min, z_max = 0.21, 0.51
new_target = np.array([
    np.random.uniform(x_min, x_max),
    np.random.uniform(y_min, y_max),
    np.random.uniform(z_min, z_max)
])
desired_target = new_target
# 원하는 목표 위치로 설정 (예: [0.7, 0.2, 0.3])
# desired_target = np.array([0.7, 0.2, 0.3])
env.target_pos = desired_target
# target_body는 reset() 시 생성한 빨간 구의 오브젝트 ID
p.resetBasePositionAndOrientation(env.target_body, desired_target.tolist(), [0, 0, 0, 1])
print("Set target to:", desired_target)

# 학습 완료된 모델 불러오기
model = PPO.load("scara_model", env=env)

done = False
while not done:
    # 결정론적 행동 선택 (deterministic=True)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    # 현재 end_effector의 위치와 목표와의 거리를 확인
    ee_index = env.get_link_index_by_name("end_effector")
    ee_state = p.getLinkState(env.robot, ee_index)
    ee_pos = np.array(ee_state[0])
    distance = np.linalg.norm(ee_pos - env.target_pos)
    print("Reward:", reward, "Distance:", distance)
    time.sleep(1.0/240.0)  # GUI 모드에서 시각적 확인

env.close()
