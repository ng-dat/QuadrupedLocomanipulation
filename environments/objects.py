import pybullet as p


class EnvObject(object):
    def __init__(self, pybullet_client, init_position, init_orientation):
        self.pybullet_client = pybullet_client

        self.init_position = init_position
        self.init_orientation = init_orientation

    def load_urdf(self, urdf_file, fixed=False):
        self.uid = self.pybullet_client.loadURDF(urdf_file, useFixedBase=fixed, basePosition=self.init_position)
    
    def get_body_position(self):
        body_position, _ = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_position

    def get_body_orientation(self):
        _, body_orientation = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_orientation