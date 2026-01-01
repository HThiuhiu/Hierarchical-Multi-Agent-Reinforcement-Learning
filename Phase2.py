from abc import ABC 
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import math

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

def sample_gumbel(shape, device=None, eps=1e-20):
    """Sample Gumbel(0,1)"""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u+eps) + eps)

def gumbel_sigmoid(logits, tau=1.0, hard=False, threshold=0.5):
    """
    Binary Concrete (Gumbel-Sigmoid) for sampling Bernoulli in a differentiable way.
    logits: real-valued tensor
    """
    g1 = sample_gumbel(logits.shape, device=logits.device)
    g2 = sample_gumbel(logits.shape, device=logits.device)
    y = torch.sigmoid((logits + g1 - g2) / tau)
 
    if hard:
        y_hard = (y > threshold).float()
        return (y_hard - y).detach() + y 
 
    return y

class SkillPolicy(nn.Module, ABC):
    def __init__(self, n_agents, obs_dim, goal_dim, local_state_dim, num_skills, hidden_dim=256):
        super(SkillPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.num_skills = num_skills
        self.local_state_dim = local_state_dim
        
        self.q_proj = nn.Linear(self.goal_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.obs_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.obs_dim, self.hidden_dim)
        
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.skill_logits = nn.Linear(self.local_state_dim + self.goal_dim + self.hidden_dim, self.num_skills)

    def forward(self, 
                goals: torch.Tensor,
                agents: torch.Tensor,
                agents_local: torch.Tensor,
                tau: float = 1.0,
                hard: bool = False,
                threshold: float = 0.5,
                eps: float = 1e-8):
        B, N, _ = goals.shape
        device = goals.device
        
        #project
        Q = self.q_proj(goals)
        K = self.k_proj(agents)
        V = self.v_proj(agents)
        
        scale = math.sqrt(self.hidden_dim)
        logits = torch.matmul(Q, K.transpose(-2,-1))/(scale + 1e-12) #[B, N, N]
        mask = gumbel_sigmoid(logits, tau=tau, hard=hard, threshold=threshold)
        scores = logits + torch.log(mask + eps)
        V_masked = V.unsqueeze(1) * mask.unsqueeze(-1)
        
        attn = F.softmax(scores, dim=-1)
        attn_unsq = attn.unsqueeze(-1)  # [B, N, N, 1]
        info = (attn_unsq * V_masked).sum(dim=2)  # [B, N, d]
 
        # optional output projection
        out = self.out_proj(info)  # [B, N, d]
           
        concat = torch.cat([agents_local, goals, out], dim=-1)
        logit = self.skill_logits(concat)  
        
        return logit, mask, attn, out

