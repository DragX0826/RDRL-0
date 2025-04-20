import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any
import pybullet as p
from terrain_generator import TerrainGenerator
from robot_model import RobotModel

class RobotDogEnv(gym.Env):
    def render(self, mode='human'):
        if mode == 'rgb_array':
            width, height = 320, 240
            # Camera setup: adjust as needed for your scene
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1, 1, 1],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics_client_id
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0,
                physicsClientId=self.physics_client_id
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=width, height=height,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                physicsClientId=self.physics_client_id
            )
            rgb_array = np.array(px)[:, :, :3]
            return rgb_array
        else:
            return super().render(mode=mode)

    def __init__(self, urdf_path: str, render: bool = False):
        print(f"[DEBUG] Entering RobotDogEnv.__init__ (urdf_path={urdf_path}, render={render})")
        super().__init__()
        print("[DEBUG] After super().__init__ in RobotDogEnv")
        
        # Initialize PyBullet
        if render:
            # Attempt GUI connection and error out if it fails
            self.physics_client_id = p.connect(p.GUI)
            print(f"[DEBUG] PyBullet GUI client id: {self.physics_client_id}")
            if self.physics_client_id < 0:
                raise RuntimeError(
                    "PyBullet GUI failed to initialize. Ensure a display is available."
                )
        else:
            self.physics_client_id = p.connect(p.DIRECT)
            print(f"[DEBUG] PyBullet DIRECT client id: {self.physics_client_id}")
            
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        p.setTimeStep(1/240, physicsClientId=self.physics_client_id)
        # Optimize simulation performance
        p.setPhysicsEngineParameter(numSolverIterations=4, numSubSteps=1, physicsClientId=self.physics_client_id)
        # Disable shadows and other visual effects
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.physics_client_id)
        print("[DEBUG] Physics and visualizer initialized")
        
        # Initialize components
        from terrain_generator import TerrainGenerator
        print("[DEBUG] Import TerrainGenerator")
        self.terrain_generator = TerrainGenerator()
        print("[DEBUG] TerrainGenerator created")
        from robot_model import RobotModel
        print("[DEBUG] Import RobotModel")
        print(f"[DEBUG] About to create RobotModel in __init__ (urdf_path={urdf_path}, physics_client_id={self.physics_client_id})")
        self.robot = RobotModel(urdf_path, self.physics_client_id)
        print(f"[DEBUG] RobotModel created in __init__, robot_id={getattr(self.robot, 'robot_id', None)}")
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.robot.num_joints,),
            dtype=np.float32
        )
        
        # Define observation space (only Box types)
        self.observation_space = gym.spaces.Dict({
            'joints': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot.num_joints * 2,),
                dtype=np.float32
            ),
            'imu': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(9,),  # 3 orientation + 3 angular vel + 3 linear acc
                dtype=np.float32
            ),
            'target': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            )
        })
        
        # Store binary contacts separately
        self.contact_space = gym.spaces.MultiBinary(2)
        
        # Initialize environment state
        self.terrain = None
        self.target_position = None
        self.current_position = None
        self.steps = 0
        self.max_steps = 1000
        self.difficulty = 0.0  # Initial difficulty level
        
    def _generate_terrain(self):
        print("[DEBUG] Entering _generate_terrain")
        """Generate a new terrain and set up the environment."""
        # Generate terrain data with current difficulty
        terrain_data = self.terrain_generator.generate_terrain(difficulty=self.difficulty)
        print("[DEBUG] Terrain data generated")
        # Create ground plane
        ground_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            heightfieldData=terrain_data['height_map'].flatten(),
            numHeightfieldRows=terrain_data['height_map'].shape[0],
            numHeightfieldColumns=terrain_data['height_map'].shape[1],
            physicsClientId=self.physics_client_id
        )
        print(f"[DEBUG] Collision shape created: {ground_shape}")
        # Create ground body without separate visual shape
        self.ground_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client_id
        )
        print(f"[DEBUG] Ground multi body created: {self.ground_id}")
        # Set a default visual color for the ground
        p.changeVisualShape(
            self.ground_id,
            -1,
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=self.physics_client_id
        )
        print("[DEBUG] Ground visual shape changed")
        # Set biome-specific properties
        biome_type = self.terrain_generator.get_biome_type(
            terrain_data['temperature'].mean(),
            terrain_data['humidity'].mean()
        )
        biome_props = self.terrain_generator.get_biome_properties(biome_type)
        p.changeDynamics(
            self.ground_id,
            -1,
            lateralFriction=biome_props['friction'],
            restitution=biome_props['bounce']
        )
        print(f"[DEBUG] Ground dynamics set: {biome_props}")
        # Set random target position
        size = terrain_data['height_map'].shape[0]
        self.target_position = np.array([
            np.random.uniform(-size/2, size/2),
            np.random.uniform(-size/2, size/2),
            terrain_data['height_map'][size//2, size//2]
        ])
        print(f"[DEBUG] Target position set: {self.target_position}")
        # Set camera to view the center of the terrain
        try:
            p.resetDebugVisualizerCamera(
                cameraDistance=size//2,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0.5],
                physicsClientId=self.physics_client_id
            )
        except Exception as e:
            print(f"[DEBUG] Camera set failed: {e}")
        print(f"[DEBUG] Terrain and camera set. Size: {size}")
        
    def increase_difficulty(self):
        print("[DEBUG] Entering increase_difficulty")
        """Increase the environment difficulty."""
        self.difficulty = min(1.0, self.difficulty + 0.1)
        print(f"[DEBUG] Difficulty increased to {self.difficulty}")
        
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        print("[DEBUG] Entering RobotDogEnv.reset")
        """Reset the environment to initial state. Automatically increases terrain difficulty."""
        super().reset(seed=seed)

        # Automatically increase difficulty every episode
        self.increase_difficulty()
        
        # Reset physics simulation
        p.resetSimulation(physicsClientId=self.physics_client_id)
        
        # Generate new terrain
        self._generate_terrain()

        # Reset robot
        print(f"[DEBUG] About to create RobotModel in reset (urdf_path={self.robot.urdf_path}, physics_client_id={self.physics_client_id})")
        self.robot = RobotModel(self.robot.urdf_path, self.physics_client_id)
        print(f"[DEBUG] RobotModel created in reset, robot_id={getattr(self.robot, 'robot_id', None)}")

        # Reset environment state
        self.steps = 0
        self.current_position = np.array([0, 0, 1])
        
        # Get initial observation
        print("[DEBUG] Calling _get_observation after robot creation in reset")
        obs, info = self._get_observation()
        print(f"[DEBUG] Reset returning obs keys: {list(obs.keys())}, info keys: {list(info.keys())}")
        return obs, info
    
    def _get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Get the current observation."""
        state = self.robot.get_state()
        
        # Get contacts for info dict
        contacts = np.array([
            float(state['foot_contacts']['left_foot']),
            float(state['foot_contacts']['right_foot'])
        ])
        
        return {
            'joints': np.concatenate([state['joint_positions'], state['joint_velocities']]),
            'imu': np.concatenate([
                state['imu_orientation'],
                state['imu_angular_velocity'],
                state['imu_linear_acceleration']
            ]),
            'target': self.target_position
        }, {'contacts': contacts}
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        # Get current position
        position, _ = p.getBasePositionAndOrientation(
            self.robot.robot_id,
            physicsClientId=self.physics_client_id
        )
        self.current_position = np.array(position)
        
        # Distance to target
        distance_to_target = np.linalg.norm(
            self.current_position[:2] - self.target_position[:2]
        )
        
        # Stability reward (based on IMU orientation)
        imu_data = self.robot.get_imu_measurements()
        stability_reward = -np.sum(np.abs(imu_data['orientation']))
        
        # Energy efficiency (based on joint torques)
        state = self.robot.get_state()
        energy_reward = -np.sum(np.square(state['joint_torques']))
        
        # Foot contact reward
        foot_contacts = self.robot.get_foot_contact()
        contact_reward = sum(foot_contacts.values())
        
        # Combine rewards with difficulty scaling
        reward = (
            -0.1 * distance_to_target +  # Distance penalty
            0.1 * stability_reward * (1 + self.difficulty) +  # Stability reward
            0.01 * energy_reward +  # Energy efficiency
            0.1 * contact_reward * (1 + self.difficulty)  # Foot contact reward
        )
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        # Apply action to robot
        self.robot.set_joint_commands(action)
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.physics_client_id)
        
        # Update environment state
        self.steps += 1
        
        # Get observation
        obs, info = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False
        truncated = False
        info['difficulty'] = self.difficulty
        
        # Check if target reached
        if np.linalg.norm(self.current_position[:2] - self.target_position[:2]) < 1.0:
            reward += 1000  # Large reward for reaching target
            terminated = True
            info['success'] = True
            
        # Check if robot fell
        if self.current_position[2] < 0.5:  # Fell below half height
            reward -= 500  # Large penalty for falling
            terminated = True
            info['success'] = False
            
        # Check if max steps reached
        if self.steps >= self.max_steps:
            truncated = True
            info['success'] = False
            
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up the environment."""
        p.disconnect(physicsClientId=self.physics_client_id) 
