from ucimlrepo import fetch_ucirepo
from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import argparse


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", "-lr", type=float, required=True)
    parser.add_argument("--iters", "-i", type=int, required=True)
    parser.add_argument("--threshold", "-th", type=float, required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_data():
    cancer_data = fetch_ucirepo(id=17)
    data_df = cancer_data.data.features
    targets_df = cancer_data.data.targets

    # any_missing_data = data_df.isnull().any()
    # any_missing_targets = target_df.isnull().any()
    # print(f"{any_missing_data}\n")
    # print(f"{any_missing_targets}\n")

    return (data_df, targets_df)


def prep_data(seed):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=seed
    )
    y_train = np.array([1 if y == "M" else 0 for y in y_train["Diagnosis"]])
    y_test = np.array([1 if y == "M" else 0 for y in y_test["Diagnosis"]])
    return (X_train, X_test, y_train, y_test)


def main():
    args = parseArgs()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1

    X_train, X_test, y_train, y_test = prep_data(seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = LogisticRegression(args.learning_rate, args.iters, args.threshold)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F-score: {f1:.3f}")


if __name__ == "__main__":
    main()
