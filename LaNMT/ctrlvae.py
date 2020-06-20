import torch
import torch.nn as nn
from typing import Optional


class ctrlVAE(nn.Module):
    def __init__(
        self, 
        args, 
        init_I: Optional[float] = 0., 
        init_beta: Optional[float] = 0.,
    ):
        super().__init__()
        self.v_kl = args.v_kl
        self.Kp = getattr(args, "Kp", 1e-2)
        self.Ki = getattr(args, "Ki", 1e-4)
        self.beta_min = getattr(args, "beta_min", 0.)
        self.beta_max = getattr(args, "beta_max", 1.)

        self.register_buffer("I", torch.tensor(init_I))
        self.register_buffer("beta_prev", torch.tensor(init_beta))

    def forward(self, kl: torch.Tensor):        
        with torch.no_grad():
            e_t = self.v_kl - kl
            P_t = self.Kp / (1 + e_t.exp())
            if self.beta_min <= self.beta_prev <= self.beta_max:
                I_t = self.I - self.Ki*e_t
            else:
                I_t = self.I
            beta_t = (P_t + I_t + self.beta_min).clamp(
                min=self.beta_min, 
                max=self.beta_max
            )
            self.I.fill_(I_t)
            self.beta_prev.fill_(beta_t)
            return beta_t
