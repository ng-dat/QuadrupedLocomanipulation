import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np

from box_robot import BoxRobot
from objects import EnvObject


class Config():
    def __init__(self):
        self.is_render = True
        if self.is_render:
            self.dt = 1
        else:
            self.dt = 1

        # self.num_observations = 3 * 4 + 3 + 3 + 3  # box1_position, box1_orientation, box1_lin_velocity, box1_ang_velocity, box1_to_box2_vector, box2_shape, box2_goal_vector
        self.num_observations = 3 * 4 + 3 + 3 # box1_position, box1_orientation, box1_lin_velocity, box1_ang_velocity, box2_position, goal_position
        self.control_mode = 'JointTorque'  # Modes: 'JointTorque','PDJoint'
        if self.control_mode == 'JointTorque':
            self.num_actions = 2 # Force in x-y directions to COM of the box
        if self.control_mode == 'PDJoint':
            self.num_actions = 6 # Targeted position & contact-force at that position
        self.action_clip = 1.0
        self.observation_clip = 50.0
        self.torque_limits = np.array([33.5]*self.num_actions)
        self.max_episode_length = 199


class BoxEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Load parameters configuration
        self.temp_config = Config()
        self.run_steps = 0

        # Connect to Pybullet
        if self.temp_config.is_render:
            self.pybullet_client = bc.BulletClient(connection_mode=p.GUI)
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        else:
            self.pybullet_client = bc.BulletClient()
        self.pybullet_client.setTimeStep(self.temp_config.dt)
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set up environment specifications
        self.box1 = BoxRobot(self.pybullet_client,
                             [-5,-3,0.5],
                             [0,0,0,0],
                             self.temp_config.torque_limits,
                            )
        self.box2 = EnvObject(self.pybullet_client,
                             [0,-3,0.5],
                             [0,0,0,0]
                            )
        self.goal = EnvObject(self.pybullet_client,
                             [0,0,1],
                             [0,0,0,0]
                            )
        
        self.action_space = spaces.Box(np.array([-self.temp_config.action_clip] * self.temp_config.num_actions),
                                       np.array([self.temp_config.action_clip] * self.temp_config.num_actions))
        self.observation_space = spaces.Box(np.array([-self.temp_config.observation_clip] * self.temp_config.num_observations),
                                            np.array([self.temp_config.observation_clip] * self.temp_config.num_observations))

        # Add viewer
        self.pybullet_client.resetDebugVisualizerCamera(cameraDistance=1.5,
                                                        cameraYaw=0,
                                                        cameraPitch=-40,
                                                        cameraTargetPosition=[0,-10, 3]
                                                        )

    def reset(self):
        self.run_steps = 0

        # Reset the simulation
        self.pybullet_client.resetSimulation()
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # turn off redering befor re-loading
        self.pybullet_client.setGravity(0, 0, -10)

        # Load objects
        self.plane_uid = self.pybullet_client.loadURDF("plane.urdf", basePosition=[0,0,0])
        if self.temp_config.is_render:
            self.pybullet_client.changeVisualShape(self.plane_uid, -1, rgbaColor=[1, 1, 1, 0.9])

        self.box1.load_urdf("cube.urdf")
        self.box2.load_urdf("cube.urdf")
        if self.temp_config.is_render:
            self.pybullet_client.changeVisualShape(self.box1.uid, -1, rgbaColor=[0, 0, 0, 1])
            self.pybullet_client.changeVisualShape(self.box2.uid, -1, rgbaColor=[255, 166, 1, 1])
        self.goal.load_urdf("sphere2red.urdf", fixed=True)

        # Getting the observation & info
        observation = self.get_observation()

        # Return
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) # turn on redering after loading
        return observation

    def step(self, action):
        # Setup
        self.run_steps += 1
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Step simulation
        self.apply_action(action)
        self.pybullet_client.stepSimulation()

        # Get the observation & info
        observation = self.get_observation()
        info = 'None'

        # Get reward & done
        reward, done = 0, True
        if self.run_steps <= self.temp_config.max_episode_length:
            reward, done = self.get_reward_and_done()

        # Return
        print(self.run_steps, self.box1.uid, self.box2.uid, reward)
        return observation, reward, done, info
    
    def get_observation(self):
        # box1_position, box1_orientation, box1_lin_velocity, box1_ang_velocity, box2_position, goal_position
        observation = []
        observation.extend(self.box1.get_body_position())
        observation.extend(self.box1.get_body_orientation())
        observation.extend(self.box1.get_body_linear_velocity())
        observation.extend(self.box1.get_body_angular_velocity())
        observation.extend(self.box2.get_body_position())
        observation.extend(self.goal.get_body_position())

        return observation

    def apply_action(self, action):
        if self.temp_config.control_mode == 'JointTorque':
            command = action * self.temp_config.torque_limits
            self.box1.apply_torques((command[0], command[1], 0))
        elif self.temp_config.control_mode == 'PDJoint':
            pass

    def get_reward_and_done(self):
        current_box2_position = self.box2.get_body_position()
        goal_position = self.goal.get_body_position()

        distance = np.sqrt((current_box2_position[0]-goal_position[0]) ** 2 + (current_box2_position[1]-goal_position[1]) ** 2 )

        done = self.run_steps > self.temp_config.max_episode_length
        done |= (distance < 0.1)

        reward = np.exp(-2*distance)

        return reward, done

    def render(self, mode='human'):
        view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
                            cameraTargetPosition=[0.7,0,0.05],
                            distance=.7,
                            yaw=90,
                            pitch=-70,
                            roll=0,
                            upAxisIndex=2)
        proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(
                            fov=60,
                            aspect=float(960) /720,
                            nearVal=0.1,
                            farVal=100.0)
        (_, _, px, _, _) = self.pybullet_client.getCameraImage(
                            width=960,
                            height=720,
                            viewMatrix=view_matrix,
                            projectionMatrix=proj_matrix,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        self.pybullet_client.disconnect()
