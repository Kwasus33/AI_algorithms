from ucimlrepo import fetch_ucirepo
from logistic_regression import LogisticRegression


def load_data():
    cancer_data = fetch_ucirepo(id=17)
    return cancer_data


def main():
    data = load_data()
    model = LogisticRegression(data)


if __name__ == "__main__":
    main()
