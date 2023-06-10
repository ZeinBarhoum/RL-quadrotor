# Reinforcement Learning for Quadrotor trajectory control

In this project, multiple reinforcement learning algorithms were tested for learning quadrotor take off.

The environment used is a custom created environment that uses pybullet as a simulator.

The tested algorithms are DDPG, SAC, PPO, TRPO.

## Running

For best algorithm results `SAC`, run the following in the project root folder

```bash
python3 SAC.py
```

To run other algorithms, inside the root folder run the following in the project root folder

```bash
python3 <algo>.py
```

Where `<algo>` is one of the following: `pid`,`DDPG`, `SAC`, `PPO`, `TRPO`.

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `pybullet==3.2.5`
- `gym==0.21.0`
- `torch==1.13.0`
- `stable_baselines3==1.8.0`
- `sb3_contrib==1.8.0` for `TRPO` implementation


This project is implemented as a course project for the 'Machine Learning In Robotics' course.
