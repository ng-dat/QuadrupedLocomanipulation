import numpy as np
import pybullet as p


class BoxRobot(object):
    def __init__(self, pybullet_client, init_position, init_orientation, torque_limits):
        self.pybullet_client = pybullet_client

        self.init_position = init_position
        self.init_orientation = init_orientation
        self.torque_limits = torque_limits

    def load_urdf(self, urdf_file):
        self.uid = self.pybullet_client.loadURDF(urdf_file, useFixedBase=False, basePosition=self.init_position)#, baseOrientation=self.init_orientation)
    
    def get_body_position(self):
        body_position, _ = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_position

    def get_body_orientation(self):
        _, body_orientation = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_orientation

    def get_body_linear_velocity(self):
        body_linear_velocity, _ = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_linear_velocity)

    def get_body_angular_velocity(self):
        _, body_angular_velocity = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_angular_velocity)

    def apply_torques(self, command):
        self.pybullet_client.applyExternalForce(objectUniqueId=self.uid,
                                                linkIndex=-1,
                                                forceObj=command,
                                                posObj=(0,0,0),
                                                flags=1) #LINK_FRAME
