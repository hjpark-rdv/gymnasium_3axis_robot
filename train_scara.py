import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time
import os

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
        # 바닥면 로드
        p.loadURDF("plane.urdf")
        
        # 목표 위치에 빨간 구 형태의 오브젝트 추가 (목표 시각화)
        target_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, 
            radius=0.05, 
            rgbaColor=[1, 0, 0, 1]  # 빨간색
        )
        # p.createMultiBody를 통해 시각적 오브젝트를 생성
        p.createMultiBody(
            baseVisualShapeIndex=target_visual_shape_id, 
            basePosition=self.target_pos.tolist()
        )
        
        # 스카라 로봇 URDF 파일 로드 (현재 파일 기준 상대 경로)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "robot_description/scara.urdf")
        self.robot = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
        
        self.joint_indices = []
        self.joint_limits = []
        num_joints = p.getNumJoints(self.robot)
        # controllable한 관절(REVOLUTE, PRISMATIC)만 선택
        for j in range(num_joints):
            info = p.getJointInfo(self.robot, j)
            joint_type = info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(j)
                lower_limit = info[8]
                upper_limit = info[9]
                # 관절 제한이 (0, 0)인 경우, 기본적으로 -pi ~ pi 설정 (특히 revolute인 경우)
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
        # 예제에서는 controllable 관절 중 마지막 관절의 링크 위치를 엔드 이펙터 위치로 사용
        ee_state = p.getLinkState(self.robot, self.joint_indices[-1])
        ee_pos = np.array(ee_state[0])
        # 목표와의 거리로 보상 (거리가 작을수록 높은 보상)
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance
        done = False  # 필요에 따라 도달 시 종료 조건을 추가할 수 있음
        info = {}
        return obs, reward, done, False, info

    def render(self):
        # GUI 모드에서는 PyBullet 창이 기본적으로 표시되므로 별도 구현 생략
        pass

    def close(self):
        p.disconnect()


# Stable-Baselines3를 사용한 학습 예제
if __name__ == '__main__':
    from stable_baselines3 import PPO

    # 환경 생성 (GUI 모드 활성화를 위해 render_mode=True)
    env = ScaraEnv(render_mode=True)
    
    # PPO 모델 생성 (MLP 정책)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 학습 (예: 10,000 타임스텝)
    model.learn(total_timesteps=10000)
    
    # 학습된 모델 저장
    model.save("scara_model")
    
    # 환경 종료
    env.close()
