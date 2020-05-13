import pdb
import numpy as np
from scipy import stats

import h5py

import time
import random
import string
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from datetime import datetime

def create_img(x, y, pos_bounds, radius=0.5, W=32):
  # Check if center of ball outside image frame
  if x < pos_bounds[0] or x > pos_bounds[1]:
    return None
  elif y < pos_bounds[0] or y > pos_bounds[1]:
    return None

  x_px = int(round(W * x / posbounds[1]))
  y_px = int(round(W * y / posbounds[1]))
  r_px = int(round(radius / pos_bounds[1] * W))

  # Check if perimeter of ball outside image frame
  if x_px+r_px > W or x_px-r_px < 0:
    return None
  elif y_px+r_px > W or y_px-r_px < 0:
    return None

  img = np.ones((3,W,W))
  xx,yy = np.mgrid[:W, :W]
  circle = (xx-x_px)**2 + (yy-y_px)**2
  img[:, circle < r_px**2] = 0.

  return img

def step(x0, add_noise=False):
  if x0[1] >= 0.5*posbounds[0]:
    Ak = Aks[0]
  else:
    Ak = Aks[1]
  update = Ak @ x0
  if add_noise:
    mn = np.array([0.1, 0.1])
    cov = np.diag([0.05, 0.05])
    frzn = stats.multivariate_normal(mn, cov)
    update += frzn.rvs(1)
  return update

n = 4 
dhs = [0.05, 0.1]

posbounds = np.array([0,4]) # 4x4m square
velmax = 0.10

Aks = []
for dh in dhs:
  Ak = np.eye(n)
  Ak[0:int(n/2), int(n/2):] = dh * np.eye(int(n/2))
  Aks.append(Ak)

np.random.seed(12)

W = 32
NUM_DATA = 50

X = np.zeros((NUM_DATA,3,W,W))
X_next = np.zeros((NUM_DATA,3,W,W))

count = 1 
while count < NUM_DATA:
  x0 = np.hstack((posbounds[1] * np.random.rand(2), velmax*np.random.rand(2)))

  img = create_img(x0[0], x0[1], posbounds)
  if img is None:
    continue

  x0_new = step(x0)
  img_new = create_img(x0_new[0], x0_new[1], posbounds)
  if img_new is None:
    continue

  X[count,:,:,:] = img
  X_next[count,:,:,:] = img_new

  count += 1

# dim_in, dim_out = 32, 32 
# dim_z = 6
