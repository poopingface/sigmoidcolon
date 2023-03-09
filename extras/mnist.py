from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sigmoidcolon import SigmoidColon

activations = {
    "sigmoidcolon": SigmoidColon(),
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
}


def main(args):
    torch.manual_seed(args.seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_ds = datasets.MNIST("./data", train=True, transform=transform, download=True)
    test_ds = datasets.MNIST("./data", train=False, transform=transform, download=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        activations[args.activation],
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=-1),
    ).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    test_accs = []

    for _ in range(args.epochs):
        model.train()
        for data, target in train_dl:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            optimizer.zero_grad()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        for data, target in test_dl:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            correct += (output.argmax(-1) == target).sum().item()
            total += output.shape[0]

        test_accs.append(correct / total)
        print(test_accs[-1])

    np.save(f"{args.activation}.npy", np.array(test_accs))


def smooth(xs, fact=0.5):
        y = [xs[0]]
        for x in xs[1:]:
            y.append(y[-1] * fact + x * (1 - fact))
        return y

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("activation", choices=activations.keys())
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot:
        colors = plt.get_cmap("tab10")
        for i, f in enumerate(Path("./").glob("*.npy")):
            accs = np.load(f)
            plt.plot(accs, alpha=0.1, color=colors(i))
            plt.plot(smooth(accs), label=f.stem, color=colors(i))
        plt.ylim(0.8, 1.0)
        plt.title("MNIST")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.savefig("mnist_accuracy.png", dpi=150, bbox_inches="tight")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main(args)
