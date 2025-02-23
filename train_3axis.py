import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
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
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 로봇 URDF 파일을 로드합니다.
 # self.robot_id = p.loadURDF("robot_description/3dof_cartesian_robot.urdf")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "robot_description/3dof_cartesian_robot.urdf")

        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        # 각 관절의 인덱스를 저장합니다.
        self.joint_indices = {self.joint_names[i]: i for i in range(self.num_joints)}

    # def reset(self):
    #     # 모든 관절을 초기 위치로 재설정합니다.
    #     for joint_name, joint_index in self.joint_indices.items():
    #         p.resetJointState(self.robot_id, joint_index, self.joint_positions[joint_name])

    #     # 초기 관찰값을 반환합니다.
    #     return self._get_observation(), {}
    
    def reset(self, seed=None, options=None):
        # 상위 클래스의 reset 메서드 호출하여 시드 설정
        super().reset(seed=seed)
        
        # 초기 상태 설정
        self.state = np.zeros(6, dtype=np.float32)
        
        # 초기 관찰 반환
        return self.state, {}

    def step(self, action):
        # 각 관절에 대해 주어진 액션에 따라 힘을 적용합니다.
        for i, joint_name in enumerate(self.joint_names):
            joint_index = self.joint_indices[joint_name]
            force = action[i]
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=force
            )

        # 시뮬레이션을 한 스텝 진행합니다.
        p.stepSimulation()

        # 새로운 관찰값을 얻습니다.
        observation = self._get_observation()

        # 보상과 에피소드 종료 여부를 결정합니다.
        reward = self._compute_reward(observation)
        terminated = self._is_terminated(observation)
        truncated = False  # 시간 제한 등의 이유로 에피소드가 중단되지 않는다고 가정합니다.

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        # 각 관절의 위치와 속도를 관찰값으로 반환합니다.
        joint_states = p.getJointStates(self.robot_id, list(self.joint_indices.values()))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def _compute_reward(self, observation):
        # 목표 위치와의 거리를 기반으로 보상을 계산합니다.
        target_position = np.array([0.5, 0.5, 0.5])
        current_position = observation[:self.num_joints]
        distance = np.linalg.norm(target_position - current_position)
        reward = -distance  # 거리가 가까울수록 보상이 높습니다.
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

    # 환경이 올바르게 정의되었는지 확인합니다.
    check_env(env, warn=True)

    # PPO 에이전트를 초기화합니다.
    model = PPO("MlpPolicy", env, verbose=1)

    # 에이전트를 학습시킵니다.
    model.learn(total_timesteps=10000)

    # 학습된 에이전트를 저장합니다.
    model.save("ppo_cartesian_robot")

    # 환경을 종료합니다.
    env.close()
