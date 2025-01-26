from Qlearning import Qlearning, EpsilonGreedy
from visualization import (
    postprocess,
    plot_states_actions_distribution,
    plot_q_values_map,
)
import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


env = gym.make("CliffWalking-v0", render_mode="rgb_array")

ACTIONS_SIZE = env.action_space.n
STATES_SIZE = env.observation_space.n


def train_n_runs(epochs, n_runs, seed, learner, explorer):

    rewards_log = np.zeros((epochs, n_runs))
    steps_log = np.zeros((epochs, n_runs))
    epochs = np.arange(epochs)
    qtables = np.zeros((n_runs, STATES_SIZE, ACTIONS_SIZE))
    states_log = []
    actions_log = []

    for run in range(n_runs):
        learner.reset_qtable()

        for epoch in tqdm(epochs, desc=f"Run {run}/{n_runs} - Episodes", leave=False):
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

                state = new_state

            # saves all rewards and steps
            rewards_log[epoch, run] = total_rewards
            steps_log[epoch, run] = step

        qtables[run, :, :] = learner.qtable

        # env.render()

    return rewards_log, steps_log, epochs, qtables, states_log, actions_log


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)

    args = parser.parse_args()

    seed = np.random.seed()
    learner = Qlearning(
        states_size=STATES_SIZE,
        actions_size=ACTIONS_SIZE,
        discount_factor=args.gamma,
        learning_rate=args.lr,
    )
    explorer = EpsilonGreedy(epsilon=args.epsilon)

    rewards, steps, epochs, qtables, states, actions = train_n_runs(
        args.epochs, args.n_runs, seed, learner, explorer
    )

    # only transforms data into df
    results, steps = postprocess(args.n_runs, epochs, rewards, steps)

    qtable = qtables.mean(axis=0)  # Average the Q-table between runs
    plot_states_actions_distribution(
        states=states, actions=actions, params=args
    )  # Sanity check
    plot_q_values_map(qtable=qtable, env=env, params=args)

    print(f"{qtable}\n\n\n")
    print(results)

    env.close()


if __name__ == "__main__":
    main()
