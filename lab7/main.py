import kagglehub
import os
import pandas as pd
import numpy as np
import bnlearn as bn


def load_data(file_path, columns):

    crimes_df = pd.read_csv(
        file_path, usecols=columns, low_memory=False
    )  # low_memory=false cause Dtypes are mixed

    crimes_df = crimes_df.replace("Unknown", pd.NA).dropna()

    return crimes_df


def net_learn_bn(crimes_df):
    net_struct = bn.structure_learning.fit(crimes_df, methodtype="hc")
    net = bn.parameter_learning.fit(net_struct, crimes_df)
    return net


def generate_data_bn(model, predictions, observations):
    query = bn.inference.fit(model, variables=predictions, evidence=observations)
    return query


def main():

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

    crimes_df["Victim Age"] = crimes_df["Victim Age"].astype(int)
    crimes_df["Perpetrator Age"] = crimes_df["Perpetrator Age"].astype(int)

    """ 
    # different ways to further discretize age distribution - using those alows 
    # not using age feature as evidence while prediciting tuple in generate_data_bn() 

    crimes_df = crimes_df.iloc[: (len(crimes_df) // 3)]

    crimes_df = crimes_df[
        (crimes_df["Victim Age"] < 51) & (crimes_df["Perpetrator Age"] < 51)
    ]
    """

    # print(crimes_df)

    model = net_learn_bn(crimes_df)
    print(f"Network structure: {model['adjmat']}\n")

    # model is pgmpy Bayesian Model obj (dict)
    for node in model["model"].nodes():
        print(f"Rozkład dla {node}: ")
        cpd = model["model"].get_cpds(node)
        print(cpd)
        cpd.to_csv(f"csv/output_{node}.csv")
        df = pd.read_csv(f"csv/output_{node}.csv")
        df.to_excel(f"sheets/output_{node}.xlsx", index=False)

    # way of describing distributions connections is presenting distributions in list of tuples
    # while having structure like ( (), (0, 2), (3), () ) - it means there's 4 nodes/distributions,
    # first(id=0) and fourth(id=3) are roots, second(id=1) - (0, 2) stores indexes of parents - is child of first and third, third - (3) is child of forth - ()

    observations = [
        {"Victim Sex": "Male", "Perpetrator Age": 30, "Relationship": "Wife"},
        {"Victim Age": 20, "Perpetrator Race": "Black", "Weapon": "Knife"},
        {"Victim Age": 50, "Victim Sex": "Female", "Perpetrator Race": "White"},
    ]
    predictions = [
        [column for column in columns if column not in obs.keys()]
        for obs in observations
    ]

    with open("predictions.txt", "w") as fh:
        for preds, obs in zip(predictions, observations):
            query = generate_data_bn(model, preds, obs)
            predicted_index = np.random.choice(query.df.index, p=query.df["p"])

            print(f"Niepełne obserwacje: {obs}\n")
            fh.write(f"Niepełne obserwacje: {obs}\n")

            print(f"Otrzymane przewidywania: {query.df.iloc[predicted_index]}\n")
            fh.write(f"Otrzymane przewidywania: {query.df.iloc[predicted_index]}\n\n")

    bn.plot(model)


if __name__ == "__main__":
    main()
