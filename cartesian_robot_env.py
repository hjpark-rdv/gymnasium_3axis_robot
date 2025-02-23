import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os

class CartesianRobotEnv(gym.Env):
    def __init__(self):
        super(CartesianRobotEnv, self).__init__()

        # PyBullet 초기화
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 로봇 로드
        # self.robot_id = p.loadURDF("robot_description/3dof_cartesian_robot.urdf")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "robot_description/3dof_cartesian_robot.urdf")

        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        # 관절 정보
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))

        # 관절 위치 및 속도 제한
        self.joint_limits = {
            "lower": np.array([0, 0, 0]),
            "upper": np.array([1, 1, 1]),
            "velocity": np.array([0.5, 0.5, 0.5])
        }

        # Gymnasium 공간 정의
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -0.5, -0.5, -0.5]),
            high=np.array([1, 1, 1, 0.5, 0.5, 0.5]),
            dtype=np.float32
        )

    def reset(self):
        # 로봇 초기화
        for i in self.joint_indices:
            p.resetJointState(self.robot_id, i, 0)
        observation = self._get_observation()
        return observation

    def step(self, action):
        # 액션 적용
        for i, act in enumerate(action):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=act * self.joint_limits["velocity"][i]
            )
        p.stepSimulation()

        # 관찰값, 보상, 종료 여부, 정보 반환
        observation = self._get_observation()
        reward = self._compute_reward(observation)
        done = self._is_done(observation)
        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        observation = np.array(joint_positions + joint_velocities, dtype=np.float32)
        return observation

    def _compute_reward(self, observation):
        # 목표 위치와의 거리로 보상 계산
        target_position = np.array([0.5, 0.5, 0.5])
        current_position = observation[:3]
        distance = np.linalg.norm(target_position - current_position)
        reward = -distance  # 거리의 음수 값이 보상
        return reward

    def _is_done(self, observation):
        # 목표 위치에 도달하면 종료
        target_position = np.array([0.5, 0.5, 0.5])
        current_position = observation[:3]
        distance = np.linalg.norm(target_position - current_position)
        return distance < 0.05  # 5cm 이내면 종료

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
