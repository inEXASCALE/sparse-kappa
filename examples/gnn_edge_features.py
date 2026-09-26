"""Train, validate, save, reload, and predict with optional edge features.

Run: python examples/gnn_edge_features.py --epochs 5
Use --no-edge-features for unweighted message passing; --strategy 1 for
inverse-norm learning. This tiny dataset demonstrates the API, not accuracy
on unseen matrix families.
"""

import argparse
import tempfile
from pathlib import Path

import torch

from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator
from sparse_kappa.gnn import GNNConditionEstimator


def sample(kappa, norm):
    matrix = torch.diag(torch.tensor([1.0, kappa**0.5, kappa], dtype=torch.float64))
    return {
        "matrix": matrix,
        "condition_number": float(torch.linalg.cond(matrix, p=norm)),
        "norm_A": float(torch.linalg.matrix_norm(matrix, ord=norm)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--norm", type=int, choices=(1, 2), default=2)
    parser.add_argument("--strategy", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-edge-features", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    train = [sample(k, args.norm) for k in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)]
    validation = [sample(k, args.norm) for k in (3.0, 12.0)]
    test = sample(6.0, args.norm)
    config = make_gnn_strategy_config(
        norm=args.norm,
        strategy=args.strategy,
        epochs=args.epochs,
        device="cpu",
        scheduler="none",
        use_edge_features=not args.no_edge_features,
    )
    # A temporary checkpoint keeps the demonstration self-contained.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "estimator.pt"
        estimator = train_gnn_strategy_estimator(
            train,
            norm=args.norm,
            strategy=args.strategy,
            val_data=validation,
            config=config,
            save_path=path,
        )
        loaded = GNNConditionEstimator.load(path)
        prediction = loaded.predict(test["matrix"], return_dict=True)
        print("Edge features:", loaded.config.use_edge_features)
        print("Prediction:", prediction)
        print("Reference:", test["condition_number"])
        print("Validation loss in log10 target space:", estimator.evaluate(validation))
        print("Batch:", loaded.predict([s["matrix"] for s in validation]))
        print(
            "Reload agreement:",
            abs(estimator.predict(test["matrix"]) - prediction["condition_number"]),
        )


if __name__ == "__main__":
    main()
