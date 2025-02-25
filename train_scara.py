import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time
import os
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import csv
import argparse
from datetime import datetime


from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import csv
import argparse

DBG_GRAPH_DISPLAY = True

# SCARA 로봇 환경 정의 (PyBullet 기반)
class ScaraEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=False):
        super(ScaraEnv, self).__init__()
        self.render_mode = render_mode
        # 4개의 관절 제어 (2 revolute, 1 prismatic, 1 revolute)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(4,), dtype=np.float32)
        # 기존 관절 상태 (위치 4 + 속도 4 = 8) + target_pos (3) → 총 11차원
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        # 초기 목표 엔드 이펙터 위치 (나중에 reset에서 변경됨)
        self.target_pos = np.array([0.5, 0.0, 0.2])
        self.physics_client = None
        self.robot = None
        self.joint_indices = []   # 제어 가능한 관절 인덱스
        self.joint_limits = []    # 관절 한계 (lower, upper)
        # 학습 도중 목표 위치를 변경하기 위한 스텝 카운터 및 업데이트 주기
        self.step_count = 0
        self.target_update_interval = 1000  # simulation 스텝 기준, 필요에 따라 조절
        self.consecutive_close_steps = 0  # 0.2m 이내로 근접한 상태를 몇 스텝 연속 유지했는지 카운팅

        self._connect()
        self.reset()

    def _connect(self):
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # 로봇이 도달 가능한 범위를 고려한 랜덤 목표 위치 설정 
        x_min, x_max = 0.0, 0.87
        y_min, y_max = -0.87, 0.87
        z_min, z_max = 0.21, 0.51
        self.target_pos = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ])
        # 목표 위치에 빨간 구 오브젝트 추가 (목표 시각화)
        target_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, 
            radius=0.05, 
            rgbaColor=[1, 0, 0, 1]
        )
        # 생성한 오브젝트의 ID를 저장합니다.
        self.target_body = p.createMultiBody(
            baseVisualShapeIndex=target_visual_shape_id, 
            basePosition=self.target_pos.tolist()
        )
        
        # 로봇 URDF 로드 (경로 수정 필요)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "robot_description/scara.urdf")
        self.robot = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
        
        # 관절 인덱스 및 한계 초기화
        self.joint_indices = []
        self.joint_limits = []
        num_joints = p.getNumJoints(self.robot)
        for j in range(num_joints):
            info = p.getJointInfo(self.robot, j)
            joint_type = info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(j)
                lower_limit = info[8]
                upper_limit = info[9]
                if lower_limit == 0 and upper_limit == 0 and joint_type == p.JOINT_REVOLUTE:
                    lower_limit = -np.pi
                    upper_limit = np.pi
                self.joint_limits.append((lower_limit, upper_limit))
                p.resetJointState(self.robot, j, 0)
        p.stepSimulation()
        # 에피소드 시작 시 스텝 카운터 초기화
        self.step_count = 0
        self.consecutive_close_steps = 0  # reset 시 카운터 초기화
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        # 관절 상태와 함께 현재 목표 위치(self.target_pos)를 observation에 포함합니다.
        return np.concatenate((
            np.array(joint_positions + joint_velocities, dtype=np.float32),
            self.target_pos
        ))
    
    def get_link_index_by_name(self, name):
        num_joints = p.getNumJoints(self.robot)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            if joint_info[12].decode("utf-8") == name:
                return i
        return -1

    def step(self, action):
        for idx, joint_index in enumerate(self.joint_indices):
            lower, upper = self.joint_limits[idx]
            target = np.clip(action[idx], lower, upper)
            p.setJointMotorControl2(
                self.robot, joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=1000
            )
        # 10 simulation step 진행
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(1.0/240.0)
        # 누적 스텝 수 업데이트 (10 step마다 호출되므로)
        self.step_count += 1

        # 주기적으로 목표 위치 업데이트
        if self.step_count % self.target_update_interval == 0:
            x_min, x_max = 0.0, 0.87
            y_min, y_max = -0.87, 0.87
            z_min, z_max = 0.21, 0.51
            new_target = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                np.random.uniform(z_min, z_max)
            ])
            self.step_count = 0
            self.target_pos = new_target
            # 목표 오브젝트의 위치를 업데이트
            p.resetBasePositionAndOrientation(self.target_body, self.target_pos.tolist(), [0, 0, 0, 1])
            print("Updated target position to:", self.target_pos)

        obs = self._get_obs()
        # 실제 end_effector 링크 위치 가져오기
        ee_index = self.get_link_index_by_name("end_effector")
        ee_state = p.getLinkState(self.robot, ee_index)
        ee_pos = np.array(ee_state[0])
        # 목표와의 거리 계산 (단위: m)
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance
        # 3cm 이내면 에피소드 종료
                
        # -----------------------------
        # done 조건 1: distance < 0.1 (10cm 이내면 즉시 종료)
        if distance < 0.1:
            self.consecutive_close_steps = 0
            done = True
        else:
            # done 조건 2: distance < 0.2 상태가 100번(=100스텝) 연속 발생
            if distance < 0.2:
                self.consecutive_close_steps += 1
            else:
                self.consecutive_close_steps = 0
            # 카운터가 100 이상이면 done 처리
            done = (self.consecutive_close_steps >= 100)
        # -----------------------------

        info = {}

        # print("ee_pos=", ee_pos, "/ target_pos=", self.target_pos, "/ distance", distance, "/ reward=", reward)
        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        p.disconnect()


class LivePlotCallback(BaseCallback):
    def __init__(self, update_freq=1000, log_dir="./training_logs"):
        super(LivePlotCallback, self).__init__()
        self.update_freq = update_freq
        self.rewards = []
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_rewards.csv")
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Mean Reward"])
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.line, = self.ax.plot([], [], label="Episode Reward")
        self.ax.set_xlabel("Training Iterations")
        self.ax.set_ylabel("Mean Reward")
        self.ax.set_title("Live Training Progress")
        self.ax.legend()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            episode_rewards = self.locals['rewards'] if 'rewards' in self.locals else 0
            self.rewards.append(episode_rewards)
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.n_calls, episode_rewards])
            self.line.set_xdata(range(len(self.rewards)))
            self.line.set_ydata(self.rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            if DBG_GRAPH_DISPLAY:
                plt.draw()
                plt.pause(0.1)
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--continue_training", action="store_true",
                        help="Continue training from a saved model")
    args = parser.parse_args()

    env = ScaraEnv(render_mode=False)
    callback = LivePlotCallback(update_freq=100)
    # if args.continue_training:
    #     if os.path.exists("scara_model.zip"):
    #         model = PPO.load("scara_model", env=env)
    #         print("Loaded saved model, continuing training...")
    #         model.learn(total_timesteps=10000000, reset_num_timesteps=False, callback=callback)
    #         model.save("scara_model_continue")
    #         env.close()
    #     else:
    #         print("No saved model found. Starting new training.")
    #         model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    #         model.learn(total_timesteps=10000000, callback=callback)
    #         model.save("scara_model")
    #         env.close()
    # else:

    print("Starting new training...")
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", device="cuda")
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,       # 학습률. 스케줄 함수로 지정할 수도 있음.
        n_steps=2048,             # 한 업데이트 당 수집하는 타임스텝 수. 벡터화된 환경과 함께 사용하면 더 많은 샘플을 얻을 수 있습니다.
        batch_size=64,            # 업데이트 시 한 번에 처리하는 배치 크기.
        n_epochs=10,              # 각 업데이트 당 데이터 반복 횟수.
        gamma=0.99,               # 할인 계수.
        gae_lambda=0.95,          # GAE(lambda) 계수로, advantage 추정에 사용됩니다.
        clip_range=0.2,           # PPO 클리핑 범위.
        ent_coef=0.01,            # 탐사(엔트로피) 보상의 계수. 값이 높을수록 탐사성이 증가합니다.
        vf_coef=0.5,              # 가치 함수 손실에 대한 계수.
        max_grad_norm=0.5,        # 최대 기울기 정규화 값으로, 기울기 폭주 방지에 도움을 줍니다.
        device="cuda"             # GPU 사용. GPU가 없다면 "cpu"로 변경하세요.
    )
    model.learn(total_timesteps=2000000, callback=callback)
    model.save("scara_model")
    env.close()

if __name__ == "__main__":
    main()
