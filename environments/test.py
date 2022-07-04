import gym
import time

from box_env import BoxEnvironment


env = BoxEnvironment()
for ep in range(1):
    observation = env.reset()
    total_reward = 0
    time_step = 0
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        time_step += 1
        if done:
            print("Episode {} finished after {} timesteps with reward {}".format(ep + 1, time_step + 1, total_reward))
            observation = env.reset()
            total_reward = 0
            time_step = 0
env.close()