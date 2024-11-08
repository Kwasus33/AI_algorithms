import argparse
import pathlib
import copy

import numpy as np
import pandas as pd

from TSP import TSP
from solution_utils import generate_solution, decode_solution
from visualizer import Visualizer

MINI_CITIES_NUM = 5


def parse_args():

    def validate(value):
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("Invalid value. Cannot be negative")
        return value 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cities-path",
        type=pathlib.Path,
        required=True,
        help="Path to cities csv file",
    )
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="mini",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Łomża")
    parser.add_argument("--finish", type=str, default="Częstochowa")
    parser.add_argument(
        "--experiment",
        action='store_true',
        help="Run algorithm on experiment mode with best fitting hiperparameters",
    )
    parser.add_argument("--pop-size", type=validate)
    parser.add_argument("--generations", type=validate)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[
        ~((data.index == args.finish) | (data.index == args.start))
    ]
    if args.problem_size == "mini":
        city_names = (
            [args.start]
            + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist()
            + [args.finish]
        )
    else:
        city_names = (
            [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]
        )
        # .index gives data frame 'data' column indexes, .tolist() converts them to list

    return data[city_names].loc[city_names]


def produce_results(data, args):
    tsp = TSP(data)
    
    min_values = []
    max_values = []
    std_devs = []
    means = []

    generations = args.generations if args.generations else 100
    pop_size = args.pop_size if args.pop_size else 1000

    if args.experiment:
        all_best_individuals = []

        for i in range(100, pop_size+1, 100):
            best_individual, best_indivs_vec = tsp.TSP_run(generations, i)
            print(f"This population best individual {best_individual.evaluation}\n")
            all_best_individuals.append(best_individual)
            best_indivs_vec = [ind.evaluation for ind in best_indivs_vec]
            min_values.append(np.min(best_indivs_vec))
            max_values.append(np.max(best_indivs_vec))
            std_devs.append(np.std(best_indivs_vec))
            means.append(np.mean(best_indivs_vec))

        best_of_all = sorted(all_best_individuals, key=lambda ind: ind.evaluation)[0]
        print(
                f"Best individual in all generations is:\n {best_of_all.solution},\n {best_of_all.evaluation}"
            )

    else:
        best_individual, best_indivs_vec = tsp.TSP_run(generations, pop_size)
        all_best_individuals = copy.deep_copy(best_indivs_vec)
        print(f"This population best individual {best_individual.evaluation}\n")
        best_indivs_vec = [ind.evaluation for ind in best_indivs_vec]
        min_values = np.min(best_indivs_vec)
        max_values = np.max(best_indivs_vec)
        std_devs = np.std(best_indivs_vec)
        means = np.mean(best_indivs_vec)

    visualizer = Visualizer(data, best_of_all.solution)
    visualizer.draw_route_on_map()
    visualizer.generate_table("results_table.png", all_best_individuals, min_values, max_values, std_devs, means, pop_size)


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)
    produce_results(data, args)
    

if __name__ == "__main__":
    main()
