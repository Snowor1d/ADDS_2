# diffusion_policy.py  (새 파일)
import torch, torch.nn as nn, torch.nn.functional as F
from einops import repeat

class SinPosEmb(nn.Module):
    def __init__(self, dim): 
        super().__init__(); 
        self.dim = dim
    def forward(self, t):
        freqs = torch.arange(self.dim, device=t.device) / self.dim
        emb = torch.einsum('b, d -> bd', t, freqs) * 2 * torch.pi
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class EpsModel(nn.Module):                 # εθ(a_i, s, i)
    def __init__(self, a_dim, s_dim, hidden=256):
        super().__init__()
        self.t_emb = SinPosEmb(hidden//2)
        self.net = nn.Sequential(
            nn.Linear(a_dim + s_dim + hidden, hidden),
            nn.Mish(), nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, a_dim)
        )
    def forward(self, a_i, s, t_idx):
        emb = self.t_emb(t_idx.float())
        x = torch.cat([a_i, s, emb], dim=-1)
        return self.net(x)

class DiffusionPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, N=5, beta_min=.1, beta_max=10.):
        super().__init__()
        self.N, self.a_dim = N, a_dim
        self.betas = self._make_beta(N, beta_min, beta_max)
        self.eps_model = EpsModel(a_dim, s_dim)
    # ---- forward: training step i ----
    def forward(self, s):
        # sample (i, ε), construct a_i, predict ε̂, return L2
        B = s.size(0)
        i = torch.randint(1, self.N+1, (B,1), device=s.device)
        eps = torch.randn(B, self.a_dim, device=s.device)
        a0 = torch.randn_like(eps)  # use replay buffer action
        alpha_bar = torch.exp(-self.betas.cumsum(0))[i-1]
        a_i = (alpha_bar.sqrt()*a0 + (1-alpha_bar).sqrt()*eps)
        eps_hat = self.eps_model(a_i, s, i)
        return F.mse_loss(eps_hat, eps)
    # ---- inference (Alg.1 line 1) ----
    @torch.no_grad()
    def sample(self, s):
        B = s.size(0)
        a = torch.randn(B, self.a_dim, device=s.device)
        for i in reversed(range(1, self.N+1)):
            beta, alpha = self.betas[i-1], 1-self.betas[i-1]
            eps_hat = self.eps_model(a, s, torch.full((B,1), i, device=s.device))
            a = (1/alpha**.5)*(a - beta/((1-alpha).sqrt())*eps_hat)
            if i>1: a += beta.sqrt()*torch.randn_like(a)
        return torch.tanh(a)*2    # [-2,2]
    def _make_beta(self, N,bmin,bmax):
        idx = torch.arange(1, N+1)
        return 1 - torch.exp(-bmin/N -0.5*(bmax-bmin)*(2*idx-1)/N**2)
