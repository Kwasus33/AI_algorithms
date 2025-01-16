import kagglehub
import pandas as pd
import os
import pomegranate


def load_data():
    dir_path = kagglehub.dataset_download("mrayushagrawal/us-crime-dataset")
    file_path = os.path.join(dir_path, "US_Crime_DataSet.csv")
    crimes_df = pd.read_csv(
        file_path, low_memory=False
    )  # low_memory=false cause Dtypes are mixed
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
    crimes_df = crimes_df.filter(columns, axis=1).dropna(inplace=True)
    # print(crimes_df)

    return crimes_df


def net():
    net = pomegranate.BayesianNetwork()


def main():
    crimes_df = load_data()


if __name__ == "__main__":
    main()
