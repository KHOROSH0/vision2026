import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt

guiFlag = False

th0 = 0.0
dt = 1/240 # pybullet simulation step
th_d = 1.0  # desired position (radian)
kp = 1.0  # proportional coefficient

physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT) # or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)

# get rid of all the default damping forces
# think of it as imagined "air drag"
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5 # seconds
logTime = np.arange(0, 5, dt)
sz = len(logTime)
logThetaSim = np.zeros(sz)
logVelSim = np.zeros(sz)
idx = 0
for t in logTime:
    th = p.getJointState(boxId, 1)[0]
    logThetaSim[idx] = th    

    dth = -kp * (th - th_d)

    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=dth, controlMode=p.VELOCITY_CONTROL)
    p.stepSimulation()

    vel = p.getJointState(boxId, 1)[1]
    logVelSim[idx] = vel

    idx += 1
    if guiFlag:
        time.sleep(dt)
p.disconnect()

plt.subplot(2,1,1)
plt.plot(logTime, logThetaSim, 'b', label="Sim Pos")
plt.plot([logTime[0], logTime[-1]], [th_d, th_d], 'r--', label="Ref Pos")
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(logTime, logVelSim, 'b', label="Sim Vel")
plt.grid(True)
plt.legend()

plt.legend()
plt.show()
