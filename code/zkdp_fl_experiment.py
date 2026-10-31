from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.05
CLIP_NORM = 1.0

SIGMA_MIN = 0.005
SIGMA_MAX = 0.020

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRUST = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0}

def sigma_from_trust(t):
    return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * (1.0 - t)

SIGMA = {cid: sigma_from_trust(t) for cid, t in TRUST.items()}

class DigitsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

def get_parameters(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]

def set_parameters(model, parameters):
    state = model.state_dict()
    new_state = {}
    for (name, old), new in zip(state.items(), parameters):
        new_state[name] = torch.tensor(new, dtype=old.dtype)
    model.load_state_dict(new_state, strict=True)

def make_partitions(y, num_clients=5, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for cid in range(num_clients):
        labels = [2 * cid, 2 * cid + 1]
        idx = np.where(np.isin(y, labels))[0]
        rng.shuffle(idx)
        parts.append(idx)
    return parts

def load_data():
    d = load_digits()
    x = d.data.astype(np.float32) / 16.0
    y = d.target.astype(np.int64)
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y
    )
    return xtr, xte, ytr, yte

XTR, XTE, YTR, YTE = load_data()
TRAIN_PARTS = make_partitions(YTR, NUM_CLIENTS, SEED)
TEST_PARTS = make_partitions(YTE, NUM_CLIENTS, SEED)

def loaders(cid):
    tx = torch.tensor(XTR[TRAIN_PARTS[cid]], dtype=torch.float32)
    ty = torch.tensor(YTR[TRAIN_PARTS[cid]], dtype=torch.long)
    vx = torch.tensor(XTE[TEST_PARTS[cid]], dtype=torch.float32)
    vy = torch.tensor(YTE[TEST_PARTS[cid]], dtype=torch.long)
    return (
        DataLoader(TensorDataset(tx, ty), BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(vx, vy), 64, shuffle=False)
    )

def local_train(global_params, cid, seed):
    torch.manual_seed(seed)
    model = DigitsNet()
    set_parameters(model, global_params)
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, _ = loaders(cid)
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return get_parameters(model)

def noisy_update(global_params, local_params, sigma, seed):
    update = [
        local_param.astype(np.float64) - global_param.astype(np.float64)
        for global_param, local_param in zip(global_params, local_params)
    ]

    flat = np.concatenate([u.ravel() for u in update])
    norm = np.linalg.norm(flat)

    if norm > CLIP_NORM:
        scale = CLIP_NORM / (norm + 1e-12)
        update = [u * scale for u in update]

    rng = np.random.default_rng(seed)
    result = []

    for global_param, delta in zip(global_params, update):
        noise = rng.normal(0.0, sigma, delta.shape)
        result.append(
            (global_param.astype(np.float64) + delta + noise)
            .astype(global_param.dtype)
        )
    return result

def fedavg(parameter_sets, weights):
    total = sum(weights)
    return [
        sum(params[i] * w for params, w in zip(parameter_sets, weights)) / total
        for i in range(len(parameter_sets[0]))
    ]

@torch.no_grad()
def evaluate(global_params):
    model = DigitsNet()
    set_parameters(model, global_params)
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    for cid in range(NUM_CLIENTS):
        _, test_loader = loaders(cid)
        for xb, yb in test_loader:
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

    return total_loss / total, correct / total

def evaluate_clients(global_params):
    model = DigitsNet()
    set_parameters(model, global_params)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    rows = []
    for cid in range(NUM_CLIENTS):
        _, test_loader = loaders(cid)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                total_loss += loss_fn(logits, yb).item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)

        rows.append({
            "client": cid,
            "trust": TRUST[cid],
            "sigma": SIGMA[cid],
            "accuracy": correct / total
        })
    return pd.DataFrame(rows)

def run(mode, base_seed, save_client_results=False):
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    model = DigitsNet()
    global_params = get_parameters(model)
    history = []

    for rnd in range(1, ROUNDS + 1):
        client_params = []
        weights = []

        for cid in range(NUM_CLIENTS):
            local = local_train(
                global_params, cid,
                base_seed + 100 * rnd + cid
            )

            sigma = 0.011 if mode == "fixed" else SIGMA[cid]

            noisy = noisy_update(
                global_params, local, sigma,
                base_seed + 1000 * rnd + cid
            )

            client_params.append(noisy)
            weights.append(len(TRAIN_PARTS[cid]))

        global_params = fedavg(client_params, weights)
        loss, accuracy = evaluate(global_params)
        history.append([rnd, loss, accuracy])

    df = pd.DataFrame(history, columns=["round", "loss", "accuracy"])
    df["mode"] = mode
    df["seed"] = base_seed

    if save_client_results:
        evaluate_clients(global_params).to_csv(
            RESULTS_DIR / f"client_results_{mode}_seed_{base_seed}.csv", index=False
        )

    return df

if __name__ == "__main__":
    all_results = []

    for mode in ["fixed", "trust"]:
        for seed in [42, 52, 62]:
            print(f"Running {mode}, seed {seed}...")
            all_results.append(run(
                mode, seed,
                save_client_results=(mode == "trust" and seed == 42)
            ))

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(RESULTS_DIR / "zkdp_fl_results.csv", index=False)

    summary = (
        results[results["round"] == ROUNDS]
        .groupby("mode")[["loss", "accuracy"]]
        .agg(["mean", "std"])
    )
    print("\nFINAL RESULTS")
    print(summary)

    print("\nTRUST-TO-NOISE VALUES")
    for cid in range(NUM_CLIENTS):
        print(
            f"Client {cid}: trust={TRUST[cid]:.1f}, "
            f"sigma={SIGMA[cid]:.3f}"
        )
