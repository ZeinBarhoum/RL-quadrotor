import pybullet as p
import time
import pybullet_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple

DT = 1./240.
G = 9.81
SIM_TIME = 10.0
LOG_TIME = .5
LOG_STEPS = LOG_TIME//DT
KF = 3.16e-10
KM = 7.94e-12
M = 0.027
W = M*G
HOVER_RPM = np.sqrt(W/(4*KF))
EPS_TIME = 1.0
state_tuple = namedtuple('state_tuple', ['P', 'O', 'V', 'W', 'A'])
planeId = None
droneId = None

def time_range(t, dt = DT):
    return range(int(t/dt))

def print_state(x,y,z, phi,theta,psi, *, step = 0, print_interval = 10):
    
    with open('state.csv','a') as f:
        f.write(f'{step*DT:.6f},{x:.6f},{y:.6f},{z:.6f},{phi:.6f},{theta:.6f},{psi:.6f}\n')
    
    column_width = 12
    t = f't={step*DT:<12.2f}'
    x = f'{x=:<12.3f}'
    y = f'{y=:<12.3f}'
    z = f'{z=:<12.3f}'
    phi = f'\u03C6 ={phi:<12.3f}'
    theta = f'\u03B8= {theta:<12.3f}'
    psi = f'\u03C8= {psi:<12.3f}'
    if(step % print_interval == 0):
        print(f'{t}{x}{y}{z}{phi}{theta}{psi}')
        
def visualize_results(file='state.csv'):
    df = pd.read_csv(file)
    fig, ax = plt.subplots(3,2)
    ax[0][0].plot(df['t'], df['x'], label='x')
    ax[1][0].plot(df['t'], df['y'], label='y')
    ax[2][0].plot(df['t'], df['z'], label='z')
    ax[0][1].plot(df['t'], df['phi'], label='phi')
    ax[1][1].plot(df['t'], df['theta'], label='theta')
    ax[2][1].plot(df['t'], df['psi'], label='psi')
    [ax[i][j].legend() for i in range(3) for j in range(2)]
    plt.show()
    
    
    z = df['z'].values
    ddz = np.diff(z,2)/DT**2
    print(ddz)

def calculate_F_T(ws):
    ws = np.array(ws)
    F = np.array(ws**2)*KF
    T = np.array(ws**2)*KM
    Tz = (-T[0] + T[1] - T[2] + T[3])
    return F, np.array([0,0,Tz])

def apply_F_T(F, T, droneId):
    #applying forces on the propellers links in the body frame upwards
    for i in range(4):
        p.applyExternalForce(droneId, i, forceObj= [0,0,F[i]], posObj= [0,0,0], flags=p.LINK_FRAME)
        
    # applying Tz on the center of mass, the only one that depend on the drag and isn't simulated by the forces before
    p.applyExternalTorque(droneId, 4, torqueObj= T, flags=p.LINK_FRAME)

def calculate_rpm(action):
    action = np.array(action)
    return HOVER_RPM*(1+0.05*action)

def calculate_reward(state):
    Pos = state.P
    DesiredPos = np.array([0,0,1])
    return -np.linalg.norm(Pos-DesiredPos)**2
def calculate_done(state, step):
    t = step*DT
    if(t > EPS_TIME): return True
    return False

def perform_step(action, *, step= 0):
    F, T = calculate_F_T(calculate_rpm(action))
    apply_F_T(F, T, droneId)
    p.stepSimulation()
    dronePos, droneQuat = p.getBasePositionAndOrientation(droneId)
    #transform quat to euler
    droneEuler = p.getEulerFromQuaternion(droneQuat)
    droneV, droneW = p.getBaseVelocity(droneId)
    
    obs = state_tuple(dronePos, droneEuler, droneV, droneW, action)
    reward = calculate_reward(obs)
    done = calculate_done(obs, step)
    print_state(*dronePos,*droneEuler, step = step, print_interval = LOG_STEPS)

    return obs, reward, done, {}
    
def reset():
    p.resetSimulation()
    p.setGravity(0,0,-G)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    droneId = p.loadURDF("cf2x.urdf",startPos, startOrientation)

    return state_tuple(np.array([0,0,1]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0,0]))

with open('state.csv','w+') as f:
    f.write('t,x,y,z,phi,theta,psi\n')
  
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally



start_time = time.time()

obs = reset()

for i in time_range(SIM_TIME):
    obs, reward, done, info = perform_step([0,0,0,0], step = i)
    time.sleep(DT)


end_time = time.time()
total_time = end_time - start_time
print(f'Total simulation time: {total_time:.2f}s')   
p.disconnect()

visualize_results()
