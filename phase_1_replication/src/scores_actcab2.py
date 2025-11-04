# phase_1_replication/src/scores_actcab.py
import json, argparse, numpy as np, torch
import torch.nn as nn
from sklearn.model_selection import KFold
from pathlib import Path

class Head(nn.Module):
    def __init__(self, h): super().__init__(); self.lin = nn.Linear(h,1)
    def forward(self,x): return torch.sigmoid(self.lin(x)).squeeze(-1)

def build_dataset(path):
    feats, y, meta = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            feats.append(np.array(ex["hidden_last_mean"], dtype=np.float32))
            # correctness aus gold/pred_text ableiten → für MC: prüfe enthaltenen Buchstaben; hier placeholder:
            correct = 1 if ex.get("gold") and ex["gold"] in ex["pred_text"] else 0
            y.append(correct); meta.append({"id": ex["id"]})
    X = np.stack(feats); y = np.array(y, dtype=np.float32); return X,y,meta

def kfold_soft_labels(X, y, k=5, epochs=20, lr=1e-3, device="cuda"):
    N,H = X.shape; oof = np.zeros(N, dtype=np.float32)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for tr,va in kf.split(np.arange(N)):
        head = Head(H).to(device); opt = torch.optim.AdamW(head.parameters(), lr=lr)
        Xtr = torch.tensor(X[tr], device=device); ytr = torch.tensor(y[tr], device=device)
        for _ in range(epochs):
            pred = head(Xtr); loss = ((pred - ytr)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Xva = torch.tensor(X[va], device=device)
            oof[va] = head(Xva).detach().cpu().numpy()
    # Bin-Accuracy als Soft-Label
    bins = np.clip((oof*10).astype(int), 0, 9); soft = np.zeros_like(oof)
    for b in range(10):
        idx = np.where(bins==b)[0]
        if len(idx): soft[idx] = y[idx].mean()
    return soft

def train_final(X, soft, epochs=40, lr=5e-4, device="cuda"):
    H = X.shape[1]; head = Head(H).to(device); opt = torch.optim.AdamW(head.parameters(), lr=lr)
    X = torch.tensor(X, device=device); y = torch.tensor(soft, device=device)
    for _ in range(epochs):
        pred = head(X); loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head

def run(args):
    X,y,meta = build_dataset(args.inputs)
    soft = kfold_soft_labels(X,y, k=5, epochs=args.k_epochs, lr=args.k_lr, device=args.device)
    head = train_final(X,soft, epochs=args.epochs, lr=args.lr, device=args.device)
    with torch.no_grad():
        conf = head(torch.tensor(X, device=args.device)).cpu().numpy().tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as g:
        for m,c in zip(meta, conf):
            g.write(json.dumps({"id": m["id"], "score_actcab": float(max(1e-6,min(1-1e-6,c)))})+"\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k_epochs", type=int, default=20)
    ap.add_argument("--k_lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    run(ap.parse_args())
