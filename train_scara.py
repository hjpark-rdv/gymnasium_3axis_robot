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
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import csv

DBG_GRAPH_DISPLAY=True

# SCARA 로봇 환경 정의 (PyBullet 기반)
class ScaraEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=False):
        super(ScaraEnv, self).__init__()
        self.render_mode = render_mode
        # 여기서는 4개의 관절(2 revolute, 1 prismatic, 1 revolute)을 제어한다고 가정
        # 행동: 각 관절의 목표 위치 (연속값)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(4,), dtype=np.float32)
        # 관찰: 각 관절의 위치와 속도 (4 + 4 = 8차원)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # 목표 엔드 이펙터 위치 (x, y, z)
        self.target_pos = np.array([0.5, 0.0, 0.2])
        self.physics_client = None
        self.robot = None
        self.joint_indices = []  # 제어 가능한 관절 인덱스
        self.joint_limits = []   # 관절 한계 (lower, upper)
        self._connect()
        self.reset()

    def _connect(self):
        # 렌더링 여부에 따라 GUI 또는 DIRECT 모드 연결
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
        
        # 로봇이 도달 가능한 범위를 고려한 랜덤 목표 위치 설정 (예시)
        x_min, x_max = 0.4, 0.87
        y_min, y_max = -0.87, 0.87
        z_min, z_max = 0.21, 0.51
        # x_min, x_max = 0.0, 0.87
        # y_min, y_max = -0.87, 0.87
        # z_min, z_max = 0.41, 0.71
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
        p.createMultiBody(
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
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # 관절 상태: 위치와 속도
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def step(self, action):
        # 행동은 각 관절에 대한 목표 위치로 사용 (position control)
        for idx, joint_index in enumerate(self.joint_indices):
            lower, upper = self.joint_limits[idx]
            target = np.clip(action[idx], lower, upper)
            p.setJointMotorControl2(
                self.robot, joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=1000
            )
        # 시뮬레이션을 몇 단계 진행 (10 step)
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(1.0/240.0)
        obs = self._get_obs()
        # controllable 관절 중 마지막 관절의 링크 위치를 엔드 이펙터 위치로 사용
        ee_state = p.getLinkState(self.robot, self.joint_indices[-1])
        ee_pos = np.array(ee_state[0])
        # 목표와의 거리 계산 (단위: m)
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance
        # 만약 end_effector가 목표로부터 3cm (0.03m) 이하이면 done을 True로 설정
        done = distance < 0.03
        info = {}

        print("ee_pos=",ee_pos," / self.target_pos=",self.target_pos,"/ distance",distance," / reward=",reward)


        return obs, reward, done, False, info


    def render(self):
        # GUI 모드에서는 PyBullet 창이 기본적으로 표시되므로 별도 구현 생략
        pass

    def close(self):
        p.disconnect()


class LivePlotCallback(BaseCallback):
    """
    학습 중 실시간 보상 그래프를 업데이트하고 보상을 CSV 파일로 저장하는 콜백 클래스
    """
    def __init__(self, update_freq=1000, log_dir="./training_logs"):
        super(LivePlotCallback, self).__init__()
        self.update_freq = update_freq
        self.rewards = []
        self.log_dir = log_dir

        # ✅ 로그 저장 폴더 생성
        os.makedirs(self.log_dir, exist_ok=True)

        # ✅ 현재 시간 기반 파일명 생성 (예: "training_logs/2025_02_23_11_22_33_rewards.csv")
        self.log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_rewards.csv")

        # ✅ CSV 파일 생성 (헤더 추가)
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Mean Reward"])  # 컬럼 헤더 추가

        # Matplotlib 인터랙티브 모드 활성화
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.line, = self.ax.plot([], [], label="Episode Reward")

        self.ax.set_xlabel("Training Iterations")
        self.ax.set_ylabel("Mean Reward")
        self.ax.set_title("Live Training Progress")
        self.ax.legend()
        # plt.show()

    def _on_step(self) -> bool:
        # 일정 step마다 보상을 그래프에 업데이트 & CSV에 저장
        if self.n_calls % self.update_freq == 0:
            # episode_rewards = np.mean(self.training_env.get_attr("last_reward"))  # ✅ 올바른 보상값 가져오기
            episode_rewards = self.locals['rewards'] if 'rewards' in self.locals else 0
            self.rewards.append(episode_rewards)

            
            # ✅ CSV에 보상 데이터 저장
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.n_calls, episode_rewards])

            # 그래프 업데이트
            self.line.set_xdata(range(len(self.rewards)))
            self.line.set_ydata(self.rewards)
            self.ax.relim()
            self.ax.autoscale_view()

            if DBG_GRAPH_DISPLAY == True:
                plt.draw()
                plt.pause(0.1)  # 실시간 업데이트

        return True

# Stable-Baselines3를 사용한 학습 예제
if __name__ == '__main__':
    from stable_baselines3 import PPO

    # 환경 생성 (GUI 모드 활성화를 위해 render_mode=True)
    env = ScaraEnv(render_mode=False)
    

    # PPO 모델 생성 (MLP 정책)
    # model = PPO("MlpPolicy", env, verbose=1)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    
    
    callback = LivePlotCallback(update_freq=1000)
    # 학습 (예: 10,000 타임스텝)
    model.learn(total_timesteps=100000, callback=callback)
    
    # 학습된 모델 저장
    model.save("scara_model")
    
    # 환경 종료
    env.close()
