"""A prescribed-motion planar 3R robot and a rigid flange camera."""
import cv2
import numpy as np
import pybullet as p


PERIOD = 12.0
WIDTH, HEIGHT = 320, 240
LENGTHS = (0.45, 0.35, 0.18)


def joint_angles(t):
    phase = 2 * np.pi * t / PERIOD
    return [0.35 + 0.10*np.sin(phase),
            -0.65 + 0.12*np.sin(phase + 0.5),
            0.30 + 0.65*np.sin(phase)]


class TriangleScene:
    def __init__(self, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Cannot connect to PyBullet")
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        p.setTimeStep(1/240, physicsClientId=self.client)
        self.projection = p.computeProjectionMatrixFOV(75, WIDTH/HEIGHT, 0.02, 5)
        self.robot = self._robot()
        self.set_time(0)
        # Center the fixed board below the nominal flange position.
        eye, _ = self.camera_pose()
        self.marker_center = np.array([eye[0], eye[1], 0.012])
        board = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.5, 1.5, 0.005],
                                    rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
        p.createMultiBody(baseVisualShapeIndex=board,
                          basePosition=[eye[0], eye[1], 0], physicsClientId=self.client)
        # World order matches IDs 1, 2, 3 for the downward camera at yaw zero.
        self.marker_vertices = np.array([(0, 0.105, 0), (0.068, -0.065, 0),
                                         (-0.068, -0.065, 0)]) + self.marker_center
        mesh = p.createVisualShape(p.GEOM_MESH, vertices=self.marker_vertices.tolist(),
                                   indices=[0, 2, 1, 0, 1, 2],
                                   rgbaColor=[0.015, 0.015, 0.015, 1],
                                   physicsClientId=self.client)
        p.createMultiBody(baseVisualShapeIndex=mesh, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(1.65, 35, -48, [0.55, 0, 0.15],
                                     physicsClientId=self.client)

    def _robot(self):
        base = p.createVisualShape(p.GEOM_CYLINDER, radius=0.07, length=0.65,
                                   visualFramePosition=[0, 0, -0.325],
                                   rgbaColor=[0.25, 0.28, 0.3, 1], physicsClientId=self.client)
        visuals = []
        for length, color in zip(LENGTHS, ([0.15, 0.45, 0.8, 1],
                                          [0.9, 0.55, 0.12, 1], [0.2, 0.65, 0.5, 1])):
            visuals.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[length/2, 0.025, 0.022],
                visualFramePosition=[length/2, 0, 0], rgbaColor=color,
                physicsClientId=self.client))
        camera = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.035, 0.025, 0.025],
                                     rgbaColor=[0.15, 0.15, 0.18, 1], physicsClientId=self.client)
        # Three revolute joints about +Z and a fixed camera below the link tip.
        return p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=base, basePosition=[0, 0, 0.65],
            linkMasses=[1, 1, 0.5, 0.1], linkCollisionShapeIndices=[-1]*4,
            linkVisualShapeIndices=visuals + [camera],
            linkPositions=[[0, 0, 0], [LENGTHS[0], 0, 0],
                           [LENGTHS[1], 0, 0], [LENGTHS[2], 0, -0.08]],
            linkOrientations=[[0, 0, 0, 1]]*4,
            linkInertialFramePositions=[[0, 0, 0]]*4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]]*4,
            linkParentIndices=[0, 1, 2, 3],
            linkJointTypes=[p.JOINT_REVOLUTE]*3 + [p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]]*3 + [[0, 0, 0]], physicsClientId=self.client)

    def set_time(self, t):
        # Ideal joint tracking: prescribed kinematics, rather than a dynamics
        # controller. No camera look-at tracking or marker pose is used here.
        for index, angle in enumerate(joint_angles(t)):
            p.resetJointState(self.robot, index, float(angle), physicsClientId=self.client)

    def camera_pose(self):
        state = p.getLinkState(self.robot, 3, computeForwardKinematics=True,
                               physicsClientId=self.client)
        return np.array(state[4]), np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)

    def camera_view(self):
        eye, rotation = self.camera_pose()
        # Optical axis and up vector are rigid in the flange frame.
        return p.computeViewMatrix(eye.tolist(), (eye - rotation[:, 2]).tolist(),
                                   rotation[:, 1].tolist())

    def render(self, overview=False):
        view = (p.computeViewMatrixFromYawPitchRoll([0.55, 0, 0.2], 1.65, 35, -48, 0, 2)
                if overview else self.camera_view())
        result = p.getCameraImage(
            WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=self.projection,
            renderer=p.ER_TINY_RENDERER, shadow=0, lightAmbientCoeff=1,
            lightDiffuseCoeff=0, lightSpecularCoeff=0,
            flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.client)
        rgba = np.asarray(result[2], dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def projected_vertices(self):
        """Ground-truth projection for validation only; never used by detector."""
        matrix = np.array(self.projection).reshape(4, 4, order="F") @ np.array(
            self.camera_view()).reshape(4, 4, order="F")
        clip = np.column_stack((self.marker_vertices, np.ones(3))) @ matrix.T
        ndc = clip[:, :3] / clip[:, 3:4]
        return np.column_stack(((ndc[:, 0]+1)*WIDTH/2, (1-ndc[:, 1])*HEIGHT/2))

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
