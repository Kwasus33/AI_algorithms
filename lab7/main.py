import kagglehub
import pandas as pd
import os
import bnlearn as bn


def load_data(file_path, columns):

    crimes_df = pd.read_csv(
        file_path, usecols=columns, low_memory=False
    )  # low_memory=false cause Dtypes are mixed

    crimes_df = crimes_df.replace("Unknown", pd.NA).dropna()

    return crimes_df


def net_learn_bn(crimes_df, states):
    net = bn.structure_learning.fit(crimes_df, methodtype="hc")
    net = bn.parameter_learning.fit(net, crimes_df)
    return net


def generate_data_bn(model, observations):
    predictions = bn.inference.fit(
        model, variables=list(observations.keys()), evidence=observations
    )
    return predictions


def main():
    test_bn = False

    dir_path = kagglehub.dataset_download("mrayushagrawal/us-crime-dataset")
    file_path = os.path.join(dir_path, "US_Crime_DataSet.csv")
    columns = [
        "Victim Sex",
        "Victim Age",
        "Victim Race",
        "Perpetrator Sex",
        "Perpetrator Age",
        "Perpetrator Race",
        "Relationship",
        "Weapon",
    ]
    crimes_df = load_data(file_path, columns)

    model = net_learn_bn(crimes_df, columns)
    print(f"Network structure: {model['adjmat']}\n")

    # it returns list of tuples of distribution indexes which are parents of given distribution
    # while having structure like ( (), (0, 2), (3), () ) - it means there's 4 nodes/distributions,
    # first(id=0) and fourth(id=3) are roots, second(id=1) - (0, 2) stores indexes of parents - is child of first and third, third - (3) is child of forth - ()

    bn.plot(model, interactive=True)

    for node in model["model"].states:
        print(f"Rozkład dla {node.name}:")
        print(node.distribution)

    observations = [
        {"Victim Sex": "male", "Victim Age": "30"},
        {"Perpetrator Race": "black", "Weapon": "knife"},
        {"Victim Sex": "female", "Perpetrator Age": "25"},
    ]

    for obs in observations:
        print(f"Niepełne obserwacje: {obs}\n")
        print(f"Otrzymane przewidywania: {generate_data_bn(model, obs)}\n")


if __name__ == "__main__":
    main()
