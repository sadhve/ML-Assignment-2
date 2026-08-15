from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']
RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR

def load_training_data():
    """Load the public Breast Cancer Wisconsin dataset and use 30 numeric features."""
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    X.columns = FEATURE_NAMES
    y = pd.Series(1-data.target, name="diagnosis")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

from sklearn.naive_bayes import GaussianNB

def main():
    X_train, _, y_train, _ = load_training_data()

    model = GaussianNB()
    model.fit(X_train, y_train)

    joblib.dump(model, OUTPUT_DIR / "naive_bayes.pkl")
    print("Saved naive_bayes.pkl")

if __name__ == "__main__":
    main()
