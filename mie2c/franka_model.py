from mie2c.e2c import Encoder, Decoder, Transition, LinearTransition, PWATransition

import torch
from torch import nn

import pybullet as pb
import time
import math
from datetime import datetime

import pybullet_data
class Panda:
    # https://alexanderfabisch.github.io/pybullet.html
    # https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
    def __init__(self, use_gui=False, stepsize=1e-3):
        if use_gui:
            pb.connect(pb.GUI, options="--opengl2")
        else:
            pb.connect(pb.DIRECT)
        self.stepsize = stepsize
        pb.setRealTimeSimulation(0)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = pb.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)

        self.robot = pb.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
        pb.resetBasePositionAndOrientation(self.robot, [0,0,0,], [0,0,0,1])

        self.num_joints = pb.getNumJoints(self.robot)

        pb.setGravity(0, 0, -10.)
        self.t = 0.

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []

        for jj in range(self.num_joints):
            joint_info = pb.getJointInfo(self.robot, jj)
            self.joints.append(jj)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[jj] + self.q_max[jj])/2.0)
            self.target_torque.append(0.)

    def reset(self):
        self.t = 0.
        for jj in range(self.num_joints):
            self.target_pos[jj] = (self.q_min[jj] + self.q_max[jj])/2.0
            pb.resetJointState(self.robot, jj, targetValue=self.target_pos[jj])

    def step(self):
        self.t += self.stepsize
        pb.stepSimulation()

    def getJointStates(self):
        joint_states = pb.getJointStates(self.robot, self.joints)
        joint_positions = [x[0] for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]
        return joint_positions, joint_velocities

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        pb.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=pb.TORQUE_CONTROL,
                                    forces=self.target_torque)

if __name__ == "__main__":
    pd = Panda()
    while True:
       pd.step() 
