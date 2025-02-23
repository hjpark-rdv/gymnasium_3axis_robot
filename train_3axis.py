import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import csv
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
        plt.show()

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

            plt.draw()
            plt.pause(0.1)  # 실시간 업데이트

        return True

class CartesianRobotEnv(gym.Env):
    def __init__(self):
        super(CartesianRobotEnv, self).__init__()

        # 관절의 위치 한계를 정의합니다.
        self.joint_limits = {
            'joint_x': (-1.0, 1.0),
            'joint_y': (-1.0, 1.0),
            'joint_z': (-1.0, 1.0)
        }

        # 관절의 초기 위치를 설정합니다.
        self.joint_positions = {
            'joint_x': 0.0,
            'joint_y': 0.0,
            'joint_z': 0.0
        }

        # 관절의 이름을 리스트로 저장합니다.
        self.joint_names = list(self.joint_positions.keys())

        # 관절의 개수를 저장합니다.
        self.num_joints = len(self.joint_names)

        # 관절의 위치와 속도를 관찰값으로 사용합니다.
        self.observation_space = spaces.Box(
            low=np.array([self.joint_limits[j][0] for j in self.joint_names] + [-np.inf] * self.num_joints),
            high=np.array([self.joint_limits[j][1] for j in self.joint_names] + [np.inf] * self.num_joints),
            dtype=np.float32
        )

        # 각 관절에 대한 힘을 액션으로 사용합니다.
        self.action_space = spaces.Box(
            low=np.array([-1.0] * self.num_joints),
            high=np.array([1.0] * self.num_joints),
            dtype=np.float32
        )

        # PyBullet 물리 시뮬레이터를 초기화합니다.
        # self.physics_client = p.connect(p.GUI)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

     
        # p.connect(p.GUI)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())


        # plane_id = p.loadURDF("plane.urdf")

        # 로봇 URDF 파일을 로드합니다.
 # self.robot_id = p.loadURDF("robot_description/3dof_cartesian_robot.urdf")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "robot_description/3dof_cartesian_robot.urdf")

        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        # 각 관절의 인덱스를 저장합니다.
        self.joint_indices = {self.joint_names[i]: i for i in range(self.num_joints)}

        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./1000.)
        
        self.target_position = np.random.uniform(low=[0.5, 0.3, 0.2], high=[0.9, 0.7, 0.6])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 초기 목표 좌표 설정
        # self.target_position = np.random.uniform(low=[0.5, 0.3, 0.2], high=[0.9, 0.7, 0.6])

        # 기존 목표 오브젝트 삭제 후 새로 생성
        if hasattr(self, "target_visual_id"):
            p.removeBody(self.target_visual_id)
        
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        self.target_visual_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)

        # 조인트 초기화 (한계를 넘지 않도록 설정)
        for joint_name in self.joint_names:
            joint_index = self.joint_indices[joint_name]
            initial_position = (self.joint_limits[joint_name][0] + self.joint_limits[joint_name][1]) / 2

            p.resetJointState(self.robot_id, joint_index, targetValue=initial_position)

        return self._get_observation(), {}

    def step(self, action):
        """
        X, Y, Z가 부모 링크에서 분리되지 않도록, 부모 링크의 위치를 강제 반영하여 설정.
        """
        # 현재 각 조인트의 상태(위치)를 가져옴
        joint_states = p.getJointStates(self.robot_id, list(self.joint_indices.values()))
        current_positions = {name: joint_states[i][0] for i, name in enumerate(self.joint_names)}

        # 부모 링크 위치 반영하여 새로운 목표 위치 계산
        new_z = max(min(current_positions["joint_z"] + action[2] * 0.05, self.joint_limits["joint_z"][1]), self.joint_limits["joint_z"][0])
        new_y = max(min(current_positions["joint_y"] + action[1] * 0.05, self.joint_limits["joint_y"][1]), self.joint_limits["joint_y"][0]) + new_z
        new_x = max(min(current_positions["joint_x"] + action[0] * 0.05, self.joint_limits["joint_x"][1]), self.joint_limits["joint_x"][0]) + new_y

        # 부모 위치를 반영한 조인트 위치 설정
        p.setJointMotorControl2(self.robot_id, self.joint_indices["joint_z"], controlMode=p.POSITION_CONTROL, targetPosition=new_z, force=100)
        p.setJointMotorControl2(self.robot_id, self.joint_indices["joint_y"], controlMode=p.POSITION_CONTROL, targetPosition=new_y, force=50)
        p.setJointMotorControl2(self.robot_id, self.joint_indices["joint_x"], controlMode=p.POSITION_CONTROL, targetPosition=new_x, force=50)

        # PyBullet 시뮬레이션 한 스텝 진행
        p.stepSimulation()

        # 디버깅용 출력 (각 조인트의 위치 확인)
        print(f"🔄 Step Debug - Z: {new_z:.3f}, Y: {new_y:.3f}, X: {new_x:.3f}")

        # 관찰값, 보상, 종료 여부 반환
        observation = self._get_observation()
        reward = self._compute_reward(observation)
        terminated = self._is_terminated(observation)
        truncated = False

        return observation, reward, terminated, truncated, {}

        # 현재 조인트 상태 출력 (디버깅용)
        for i, joint_name in enumerate(self.joint_names):
            joint_index = self.joint_indices[joint_name]
            joint_state = p.getJointState(self.robot_id, joint_index)
            print(f"Joint {joint_name} - Position: {joint_state[0]:.2f}, Velocity: {joint_state[1]:.2f}")

        return observation, reward, terminated, truncated, {}


    def _get_observation(self):
        # 각 관절의 위치와 속도를 관찰값으로 반환합니다.
        joint_states = p.getJointStates(self.robot_id, list(self.joint_indices.values()))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return np.array(joint_positions + joint_velocities, dtype=np.float32)
    def _get_end_effector_position(self):
        """
        현재 End-Effector(로봇 팔 끝) 위치를 가져옴.
        """
        link_state = p.getLinkState(self.robot_id, self.joint_indices["joint_x"])  # X축 끝 위치
        end_effector_position = np.array(link_state[0])  # (x, y, z) 좌표 가져오기
        return end_effector_position
    def _compute_reward(self, observation):
        """
        현재 End-Effector 위치와 목표 위치의 거리 차이를 기반으로 보상을 계산.
        목표에 가까울수록 높은 보상을 주고, 멀어질수록 보상을 낮춘다.
        """
        end_effector_position = self._get_end_effector_position()
        
        # 목표 좌표와 현재 좌표의 유클리드 거리 계산
        distance_to_target = np.linalg.norm(self.target_position - end_effector_position)

        # 보상 계산 (목표에 가까울수록 보상이 커짐)
        reward = -distance_to_target  # 거리 자체를 보상으로 사용 (작을수록 보상이 높음)
        
        return reward

    def _is_terminated(self, observation):
        # 관절이 한계를 벗어나면 에피소드를 종료합니다.
        joint_positions = observation[:self.num_joints]
        for i, joint_name in enumerate(self.joint_names):
            if not (self.joint_limits[joint_name][0] <= joint_positions[i] <= self.joint_limits[joint_name][1]):
                return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    # 환경을 생성합니다.
    env = CartesianRobotEnv()

    # ✅ 현재 시간을 기반으로 파일명 생성 (예: "2025_02_23_11_22_33.csv")
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = "./training_logs"
    os.makedirs(log_dir, exist_ok=True)  # 폴더 생성

    # ✅ CSV 파일 경로 설정
    log_file = os.path.join(log_dir, f"{current_time}_")

    # ✅ Monitor에 파일 경로 적용
    env = Monitor(env, log_file)

    # 환경이 올바르게 정의되었는지 확인합니다.
    check_env(env, warn=True)

    # PPO 에이전트를 초기화합니다.
    model = PPO("MlpPolicy", env, verbose=1)
    # 콜백을 사용하여 학습 중 실시간 그래프 표시
    callback = LivePlotCallback(update_freq=1)
    # 에이전트를 학습시킵니다.
    model.learn(total_timesteps=100000, callback=callback)

    # 학습된 에이전트를 저장합니다.
    model.save("ppo_cartesian_robot")

    # 환경을 종료합니다.
    env.close()
