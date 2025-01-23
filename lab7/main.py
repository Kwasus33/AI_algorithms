import kagglehub
import pandas as pd
import os
import pomegranate

# import bnlearn


def load_data(file_path, columns):

    crimes_df = pd.read_csv(
        file_path, usecols=columns, low_memory=False
    )  # low_memory=false cause Dtypes are mixed

    # crimes_df = crimes_df[
    #     ~crimes_df.isin(["Unknown"]).any(
    #         axis=1
    #     )  # takes only lines where Uknown doesn't appear in any cell
    # ]
    # crimes_df.dropna(inplace=True)

    crimes_df = crimes_df.replace("Unknown", pd.NA).dropna()

    return crimes_df


def net_learn(crimes_df, states):
    net = pomegranate.BayesianNetwork.from_samples(
        crimes_df, state_names=states, algorithm="exact"
    )
    return net


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
    model = net_learn(crimes_df, columns)
    print(model.structure)
    for node in model.states:
        print(node.name)
        print(f"{node.distribution}\n")


if __name__ == "__main__":
    main()
