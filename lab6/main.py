from Qlearning import Qlearning, EpsilonGreedy
from visualization import (
    plot_q_values_map,
    plot_states_actions_distribution,
    postprocess,
    qtable_directions_map,
)
import gymnasium as gym
import numpy as np
import pandas as pd


env = gym.make("CliffWalking-v0", render_mode="human")

ACTIONS_SIZE = env.action_space.n
STATES_SIZE = env.observation_space.n


def train(epochs, seed, learner, explorer):

    rewards_log = np.zeros(epochs)
    steps_log = np.zeros(epochs)
    states_log = []
    actions_log = []

    for epoch in range(epochs):
        state = env.reset(seed=seed)[0]  # static starting state
        step = 0
        endpoint = False
        total_rewards = 0

        while not endpoint:
            action = explorer.choose_action(
                actions_space=env.action_space, state=state, qtable=learner.qtable
            )  # with every epoch more of learner.qtable is being filled with env exploration and exploation values

            states_log.append(state)
            actions_log.append(action)

            new_state, reward, terminated, truncated, info = env.step(action)
            # reward for:
            #   every step -> -1
            #   fall off the cliff -> -100
            #   goal_state (win) -> 0

            endpoint = (
                terminated or truncated
            )  # if we reach goal_state (win) or fall off the cliff (lose)

            learner.update(
                state, action, reward, new_state
            )  # updates learner.qtable[state, action]

            total_rewards += reward
            step += 1

            # new state is state
            state = new_state

        # saves all rewards and steps
        rewards_log[epoch] = total_rewards
        steps_log[epoch] = step

    env.render()

    return rewards_log, steps_log, learner.qtable, states_log, actions_log


def main():
    epochs = 10
    seed = np.random.seed()
    learner = Qlearning(
        states_size=STATES_SIZE,
        actions_size=ACTIONS_SIZE,
        discount_factor=0.8,
        learning_rate=0.9,
    )
    explorer = EpsilonGreedy(epsilon=0.2)

    rewards, steps, qtable, states, actions = train(epochs, seed, learner, explorer)

    # Save the results in dataframes
    results, steps = postprocess(epochs, params, rewards, steps)
    # qtable = qtable.mean(axis=0)  # Average the Q-table between runs

    plot_states_actions_distribution(
        states=all_states, actions=all_actions, map_size=map_size
    )  # Sanity check
    plot_q_values_map(qtable, env, map_size)

    env.close()


if __name__ == "__main__":
    main()
