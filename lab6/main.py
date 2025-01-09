import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0", render_mode="human")

actions_size = env.action_space.n
states_size = env.observation_space.n


def main():
    epochs = 1000
    seed = np.random.seed


if __name__ == "__main__":
    main()
