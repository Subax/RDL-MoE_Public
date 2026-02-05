import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from sklearn.base import BaseEstimator
from .config import DEVICE, RATIO_LAMBDA
from .losses import cox_loss, ratio_loss


class LinearExpert(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.linear(x)


class GatingNetwork(nn.Module):
    def __init__(self, in_features, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class DeepRMoE(BaseEstimator):
    def __init__(self, in_features, lr=0.01, epochs=1000, patience=50):
        self.in_features = in_features
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.expert0 = None
        self.expert1 = None
        self.gate = None
        self.best_model_state = None

    def _to_tensor(self, X, y=None):
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        if y is not None:
            times = torch.tensor([t for e, t in y], dtype=torch.float32).to(DEVICE)
            events = torch.tensor([e for e, t in y], dtype=torch.float32).to(DEVICE)
            return X_t, times, events
        return X_t

    def fit(self, X_train, y_train, val_data=None):
        X_tr_t, T_tr_t, E_tr_t = self._to_tensor(X_train, y_train)

        has_val = val_data is not None
        if has_val:
            X_val_t, T_val_t, E_val_t = self._to_tensor(val_data[0], val_data[1])

        self.expert0 = LinearExpert(self.in_features).to(DEVICE)
        self.expert1 = LinearExpert(self.in_features).to(DEVICE)
        self.gate = GatingNetwork(self.in_features).to(DEVICE)

        optimizer = optim.Adam(
            list(self.expert0.parameters())
            + list(self.expert1.parameters())
            + list(self.gate.parameters()),
            lr=self.lr,
            weight_decay=1e-3,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_val_loss = float("inf")
        trigger_times = 0

        for epoch in range(self.epochs):
            self.expert0.train()
            self.expert1.train()
            self.gate.train()
            optimizer.zero_grad()

            risk0 = self.expert0(X_tr_t).squeeze()
            risk1 = self.expert1(X_tr_t).squeeze()
            gate_probs = self.gate(X_tr_t)

            combined_risk = (gate_probs[:, 0] * risk0) + (gate_probs[:, 1] * risk1)

            main_loss = cox_loss(combined_risk, T_tr_t, E_tr_t)
            r_loss = ratio_loss(gate_probs)

            total_loss = main_loss + (RATIO_LAMBDA * r_loss)

            total_loss.backward()
            optimizer.step()

            if has_val:
                self.expert0.eval()
                self.expert1.eval()
                self.gate.eval()
                with torch.no_grad():
                    v_risk0 = self.expert0(X_val_t).squeeze()
                    v_risk1 = self.expert1(X_val_t).squeeze()
                    v_gate = self.gate(X_val_t)
                    v_combined = (v_gate[:, 0] * v_risk0) + (v_gate[:, 1] * v_risk1)
                    val_loss = cox_loss(v_combined, T_val_t, E_val_t).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trigger_times = 0
                    self.best_model_state = {
                        "expert0": copy.deepcopy(self.expert0.state_dict()),
                        "expert1": copy.deepcopy(self.expert1.state_dict()),
                        "gate": copy.deepcopy(self.gate.state_dict()),
                    }
                else:
                    trigger_times += 1
                    if trigger_times >= self.patience:
                        break

        if self.best_model_state is not None:
            self.expert0.load_state_dict(self.best_model_state["expert0"])
            self.expert1.load_state_dict(self.best_model_state["expert1"])
            self.gate.load_state_dict(self.best_model_state["gate"])

        return self

    def predict(self, X, return_experts=False):
        self.expert0.eval()
        self.expert1.eval()
        self.gate.eval()
        X_tensor = self._to_tensor(X)

        with torch.no_grad():
            risk0 = self.expert0(X_tensor).cpu().numpy().flatten()
            risk1 = self.expert1(X_tensor).cpu().numpy().flatten()
            gate_probs = self.gate(X_tensor).cpu().numpy()

            expert_indices = np.argmax(gate_probs, axis=1)
            final_risk = np.zeros_like(risk0)

            mask0 = expert_indices == 0
            final_risk[mask0] = risk0[mask0]

            mask1 = expert_indices == 1
            final_risk[mask1] = risk1[mask1]

        if return_experts:
            return final_risk, expert_indices
        return final_risk
