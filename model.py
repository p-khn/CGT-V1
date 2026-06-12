import math
from collections import defaultdict

import torch
import torch.nn as nn


def gaussian_nll(mu, logv, y):
    """Gaussian negative log likelihood, supporting optional MC dimension."""
    if mu.dim() == 2:
        y = y.unsqueeze(1).expand_as(mu)
    var = torch.exp(logv)
    return 0.5 * (math.log(2 * math.pi) + logv) + (y - mu) ** 2 / (2 * var)

class TwoHeadCausalTFBlock(nn.Module):
    def __init__(self, pi, feature_dim, d_model, n_head, d_ff, n_layers, z_dim, sensor_ids):
        super().__init__()
        self.feature_dim = feature_dim
        self.z_dim = z_dim

        self.ln = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, n_layers)

        self.enc_z = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * self.z_dim),
        )
        self.prior_z = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * self.z_dim),
        )

        self.mean_c = nn.Linear(d_model + self.z_dim, 1)
        self.logv_c = nn.Linear(d_model + self.z_dim, 1)
        self.dmean_o = nn.Linear(d_model + self.z_dim, 1)
        self.dlogv_o = nn.Linear(d_model + self.z_dim, 1)

        self.logits = nn.Parameter(torch.empty(feature_dim))

        pi = torch.as_tensor(pi, dtype=torch.float32)
        self.register_buffer("pi", pi)
        self.register_buffer("idx_causal", pi > 0.5)
        self.register_buffer("idx_other", ~(pi > 0.5))

        grp = defaultdict(list)
        for idx, sid in enumerate(sensor_ids):
            grp[sid].append(idx)
        self.grp_idx = list(grp.values())

        with torch.no_grad():
            p_par, p_oth = 0.7, 0.3
            logit = lambda p: math.log(p / (1 - p))
            self.logits[self.idx_causal] = logit(p_par)
            self.logits[self.idx_other] = logit(p_oth)

    def _trunk(self, xg):
        t = self.proj(xg)
        t = self.encoder(t)
        return t[:, -1]

    def forward(self, x, y=None, mc_samples=1, use_prior_at_eval=True):
        x = self.ln(x)

        x_c = x * self.pi
        x_c = self.ln(x_c)
        h_c = self._trunk(x_c)

        B = h_c.size(0)
        mu_p, logv_p = self.prior_z(h_c).split(self.z_dim, dim=-1)
        logv_p = torch.clamp(logv_p, -6.0, 6.0)

        use_posterior = self.training and (y is not None) and (not use_prior_at_eval)
        if use_posterior:
            stats = self.enc_z(torch.cat([h_c, y.unsqueeze(-1)], dim=-1))
            mu_q, logv_q = stats.split(self.z_dim, dim=-1)
            logv_q = torch.clamp(logv_q, -6.0, 6.0)
            if mc_samples > 1:
                eps = torch.randn(B, mc_samples, self.z_dim, device=h_c.device)
                z = mu_q.unsqueeze(1) + torch.exp(0.5 * logv_q).unsqueeze(1) * eps
            else:
                eps = torch.randn_like(mu_q)
                z = mu_q + torch.exp(0.5 * logv_q) * eps
            diff = mu_q - mu_p
            kl_per_dim = (
                torch.exp(logv_q - logv_p)
                + diff.pow(2) * torch.exp(-logv_p)
                - 1.0
                + (logv_p - logv_q)
            )
            kl_z = 0.5 * kl_per_dim.sum(dim=1).mean()
        else:
            if mc_samples > 1:
                eps = torch.randn(B, mc_samples, self.z_dim, device=h_c.device)
                z = mu_p.unsqueeze(1) + torch.exp(0.5 * logv_p).unsqueeze(1) * eps
            else:
                eps = torch.randn_like(mu_p)
                z = mu_p + torch.exp(0.5 * logv_p) * eps
            kl_z = torch.zeros((), device=h_c.device)

        def _hz(h_base, z_ten):
            if z_ten.dim() == 3:
                h = h_base.unsqueeze(1).expand(-1, z_ten.size(1), -1)
                return torch.cat([h, z_ten], dim=-1)
            return torch.cat([h_base, z_ten], dim=-1)

        hcz = _hz(h_c, z)
        mu_c = self.mean_c(hcz).squeeze(-1)
        logv_c = torch.clamp(self.logv_c(hcz).squeeze(-1), -8, 2)

        alpha = torch.sigmoid(self.logits)
        alpha_sg = alpha.detach()
        x_o = x * alpha_sg
        with torch.no_grad():
            x_o = self.ln(x_o)
            h_o = self._trunk(x_o)
        hoz = _hz(h_o, z.detach())

        dmu = self.dmean_o(hoz).squeeze(-1)
        dlogv = self.dlogv_o(hoz).squeeze(-1)

        mu_o = mu_c + dmu
        logv_o = torch.clamp(logv_c + dlogv, -8, 2)

        if len(self.grp_idx) > 0:
            grp_pen = torch.stack([torch.linalg.norm(alpha[idxs], 2) for idxs in self.grp_idx]).mean()
        else:
            grp_pen = torch.zeros((), device=alpha.device)

        return (mu_c, logv_c), (mu_o, logv_o), alpha, grp_pen, dmu, dlogv, kl_z

