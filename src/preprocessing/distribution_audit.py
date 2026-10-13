import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


CLASS_NAMES = {
    0: "BROWSING",
    1: "CHAT",
    2: "FT",
    3: "MAIL",
    4: "P2P",
    5: "STREAMING",
    6: "VOIP"
}


def audit_distribution(context):
    data_dir = DATA_DIR / context

    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    print(f"\n===== {context.upper()} =====")

    print("\nTRAIN DISTRIBUTION")
    for label, count in train_counts.items():
        percentage = count / len(y_train) * 100
        print(
            f"{CLASS_NAMES[int(label)]:10s}: "
            f"{count:4d} ({percentage:6.2f}%)"
        )

    print("\nTEST DISTRIBUTION")
    for label, count in test_counts.items():
        percentage = count / len(y_test) * 100
        print(
            f"{CLASS_NAMES[int(label)]:10s}: "
            f"{count:4d} ({percentage:6.2f}%)"
        )


if __name__ == "__main__":
    audit_distribution("novpn")
    audit_distribution("vpn")