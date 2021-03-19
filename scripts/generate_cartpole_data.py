#!/usr/bin/env python3

import pybullet as pb
import pybullet_data
import os, glob, random, time, sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb


q_lo = np.array([-.6, -np.pi/4])
q_up = np.array([.6, np.pi/4])

v_lo = np.array([-.5, -1.])
v_up = np.array([.5, 1.])

force_mag = 10

sim_time_step = 1./240.
data_time_step = .1
restitution = .8

width = 128
height = 128

def update_progress(progress):
    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" *
                                             (bar_length - block),
                                             progress * 100)
    print(text, end="\r")

def show_samples(X, X_next, num_samples_to_show=4):
    samples_to_show = np.random.choice(X.shape[0], num_samples_to_show)
    fig = plt.figure(figsize=(10,10))
    for k in range(num_samples_to_show):
        fig.add_subplot(num_samples_to_show,3,k*3+1)
        plt.imshow(X[k,:3,:,:].transpose(1,2,0))
        fig.add_subplot(num_samples_to_show,3,k*3+2)
        plt.imshow(X[k,3:,:,:].transpose(1,2,0))
        fig.add_subplot(num_samples_to_show,3,k*3+3)
        plt.imshow(X_next[k,3:,:,:].transpose(1,2,0))
    plt.show()

def generate_cartpole_data(num_samples, torque_control=False):
    # physicsClient = pb.connect(pb.GUI)
    physicsClient = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    cartpole_id =  pb.loadURDF('hybrid_cartpole.urdf', flags=pb.URDF_USE_SELF_COLLISION)

    pb.setGravity(0, 0, -9.8)
    pb.changeDynamics(cartpole_id, 1, restitution=restitution)
    pb.changeDynamics(cartpole_id, 2, restitution=restitution)
    pb.changeDynamics(cartpole_id, 3, restitution=restitution)
    pb.setTimeStep(sim_time_step)
    
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[0, -3, .5],
        cameraTargetPosition=[0, 0, .5],
        cameraUpVector=[0, 0, 1])
    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3.1)

    num_step = int(data_time_step/sim_time_step)

    X = np.zeros((num_samples,6,width,height), dtype=np.uint8)
    X_next = np.zeros((num_samples,6,width,height), dtype=np.uint8)
    U = np.zeros((num_samples, 1), dtype=np.float64)

    k = 0
    while k < num_samples:
        q0 = np.random.rand(2) * (q_up - q_lo) + q_lo
        v0 = np.random.rand(2) * (v_up - v_lo) + v_lo

        u0 = 2*(np.random.rand()-1) * force_mag

        for i in range(len(q0)):
            pb.resetJointState(cartpole_id, i, q0[i], v0[i])

        if torque_control:
            pb.setJointMotorControl2(cartpole_id, 1, pb.TORQUE_CONTROL, force=u0)
        else:
            pb.setJointMotorControl2(cartpole_id, 1, pb.VELOCITY_CONTROL, force=0)

        # check don't start with overlap already
        aabb_pole_min, aabb_pole_max = pb.getAABB(cartpole_id, 1)
        overlaps = pb.getOverlappingObjects(aabb_pole_min, aabb_pole_max)
        if overlaps is not None and len(overlaps) > 0:
            if (0, 2) in overlaps or (0, 3) in overlaps:
                continue

        width0, height0, rgb0, depth0, seg0 = pb.getCameraImage(
            width=width, 
            height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        for i in range (num_step):
            pb.stepSimulation()

        width1, height1, rgb1, depth1, seg1 = pb.getCameraImage(
            width=width, 
            height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        for i in range (num_step):
            pb.stepSimulation()

        width2, height2, rgb2, depth2, seg2 = pb.getCameraImage(
            width=width, 
            height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        X[k,:3,:,:] = rgb0[:,:,:3].transpose(2,0,1)
        X[k,3:,:,:] = rgb1[:,:,:3].transpose(2,0,1)
        X_next[k,:3,:,:] = rgb1[:,:,:3].transpose(2,0,1)
        X_next[k,3:,:,:] = rgb2[:,:,:3].transpose(2,0,1)
        U[k,:] = u0

        k += 1
        update_progress(float(k) / num_samples)

    pb.disconnect()
    return X, X_next, U


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_samples", metavar="N", 
        type=int, nargs=1, help="number of samples to generate")
    parser.add_argument("--show_samples", "-s", 
        action='store_true', help="show some random samples after generation")
    parser.add_argument("--torque_control", "-t",
        action='store_true', help='use torque control')

    args = parser.parse_args()

    X, X_next, U = generate_cartpole_data(args.num_samples[0], torque_control=args.torque_control)

    np.save('cartpole_X.npy', X)
    np.save('cartpole_X_next.npy', X_next)
    print(args.torque_control)
    if args.torque_control:
        np.save('cartpole_U.npy', U)

    if args.show_samples:
        show_samples(X, X_next, min(args.num_samples[0], 4))
