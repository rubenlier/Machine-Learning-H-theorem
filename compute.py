#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle

X, Y = 0, 1

class MDSimulation:
    def __init__(self, pos, vel, r1, m1, counter):  
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.n = self.pos.shape[0]
        self.radius = r1
        self.mass = m1
        self.nsteps = 0
        self.counter = counter

    def advance(self, dt):
        self.nsteps += 1
        self.pos += self.vel * dt
        self.pos %= 1.0  # periodic boundaries

        delta = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        delta -= np.round(delta)  # minimum image convention
        dist = np.linalg.norm(delta, axis=-1)
        radii_pair = 2 * self.radius

        iarr, jarr = np.where(dist < radii_pair)
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        for i, j in zip(iarr, jarr):
            pos_i, vel_i = self.pos[i], self.vel[i]
            pos_j, vel_j = self.pos[j], self.vel[j]
            rel_pos = pos_i - pos_j
            rel_pos -= np.round(rel_pos)
            r_rel = rel_pos @ rel_pos
            rel_vel = vel_i - vel_j
            nor = rel_pos / np.sqrt(r_rel)
            v_rel_n = np.dot(rel_vel, nor)
            if v_rel_n > 0:
                continue
            self.counter += 1
            delta_v = nor * v_rel_n
            self.vel[i] += -delta_v
            self.vel[j] += delta_v

# Parameters
n = 1000  
r1 = 3e-3
sbar = 35 * 0.005 / np.sqrt(2)
FPS = 60
dt = 1 / FPS
m1 = 1
frames = 500
n_runs = 500

# Store results from all runs
import os

os.makedirs("velocities", exist_ok=True)  # create folder to store output

for run in range(n_runs):
    print(f"this is run {run}")
    pos = np.random.random((n, 2))
    theta = np.random.random(n) * 2 * np.pi
    s0 = sbar
    vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
    sim = MDSimulation(pos, vel, r1, m1, counter=0)

    run_velocities = []
    for i in range(frames):
        sim.advance(dt)
        run_velocities.append(sim.vel.copy())

    # Save run-specific file
    filename = f"velocities/run_{run:03d}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(run_velocities, f)


