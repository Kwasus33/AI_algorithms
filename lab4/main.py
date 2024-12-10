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


def minMax_normalization(df):
    for column in df.columns:
        Max = max(df[column])
        Min = min(df[column])
        df[column] = df[column].map(lambda x: (x - Min) / (Max - Min))
    return df


def get_results(args, data):
    X_train, X_test, y_train, y_test = data
    
    model = LogisticRegression(args.learning_rate, args.iters, args.threshold)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_prob(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_prob)

    return (accuracy, f1, auroc)


def main():
    args = parseArgs()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1

    X_train, X_test, y_train, y_test = prep_data(seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y_train)
    X_test_scaled = scaler.fit_transform(X_test, y_test)

    data =  X_train_scaled, X_test_scaled, y_train, y_test
    accuracy, f1, auroc = get_results(args, data)

    print("Test 1: ")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F-score: {f1:.3f}")
    print(f"Auroc: {auroc:.3f}")

    columns = X_train.columns
    drop_columns = np.random.choice(columns, size=10, replace=False)
    X_test_dropped = scaler.fit_transform(X_test.drop(columns=drop_columns))
    X_train_dropped = scaler.fit_transform(X_train.drop(columns=drop_columns))

    data =  X_train_dropped, X_test_dropped, y_train, y_test
    accuracy, f1, auroc = get_results(args, data)

    print("Test 2: ")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F-score: {f1:.3f}")
    print(f"Auroc: {auroc:.3f}")

    X_train_minMax = minMax_normalization(X_train)
    X_test_minMax = minMax_normalization(X_test)

    data =  X_train_minMax, X_test_minMax, y_train, y_test
    accuracy, f1, auroc = get_results(args, data)

    print("Test 3: ")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F-score: {f1:.3f}")
    print(f"Auroc: {auroc:.3f}")


if __name__ == "__main__":
    main()
