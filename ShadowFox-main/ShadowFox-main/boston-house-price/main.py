from src.preprocessing import load_and_preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate_model


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        "data/boston.csv"
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
