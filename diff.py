import math
import time as time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from Args import args
from Modules_ori import *


class diffusion:
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        if args.beta_sche == "linear":
            self.betas = self.linear_beta_schedule(
                timesteps=self.timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )
        elif args.beta_sche == "exp":
            self.betas = self.exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == "cosine":
            self.betas = self.cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == "sqrt":
            self.betas = torch.tensor(
                self.betas_for_alpha_bar(
                    self.timesteps,
                    lambda t: 1 - np.sqrt(t + 0.0001),
                )
            ).float()

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self, denoise_model, x_start, guide_info, t, noise=None, loss_type="l2", flag=0
    ):
        #
        # if noise is None:
        noise_init = torch.randn_like(x_start)
        golden_noise = denoise_model.npnet(noise_init, guide_info)
            # noise = torch.randn_like(x_start) / 100
        #
        # 
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=golden_noise)
        if(flag == 0):
            x_noisy = guide_info
            x_end = x_start + guide_info
            predicted_x = denoise_model(x_noisy, x_start, t)
        elif(flag == 1):
            x_noisy = guide_info 
            x_end = x_start
            predicted_x = denoise_model(x_noisy, x_start, t)
        elif(flag == 2):
            # x_noisy = x_start + guide_info
            x_end = x_start
            predicted_x = denoise_model.forward_uncon(x_noisy, t)
        elif(flag == 3):
            x_noisy = x_start
            x_end = x_start + guide_info
            predicted_x = denoise_model(x_noisy, guide_info, t)
            
        if loss_type == "l1":
            loss = F.l1_loss(x_end, predicted_x)
        elif loss_type == "l2":
            loss = F.mse_loss(x_end, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_end, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(
            x, t
        )
        x_t = x
        model_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, x, guide_info):
        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(
                model_forward,
                model_forward_uncon,
                x,
                guide_info,
                torch.full((guide_info.shape[0],), n, device=x.device, dtype=torch.long),
                n,
            )
        return x

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        x = torch.linspace(1, 2 * timesteps + 1, timesteps)
        betas = 1 - torch.exp(
            -beta_min / timesteps
            - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps)
        )
        return betas

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        beta_start = beta_start
        beta_end = beta_end
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Tenc(nn.Module):
    def __init__(
        self,
        hidden_size,
        feat_dim,
        item_num,
        state_size,
        dropout,
        diffuser_type,
        device,
        num_heads=1,
        with_text_emb=True,
    ):
        super(Tenc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.npnet = NPNet(item_num, hidden_size, with_text_emb)
        self.feat_trans = nn.Linear(feat_dim, hidden_size)
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size, embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        self.f = nn.Sigmoid()
        # self.ac_func = nn.ReLU()

        # self.step_embeddings = nn.Embedding(
        #     num_embeddings=50,
        #     embedding_dim=hidden_size
        # )

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size * 2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        if self.diffuser_type == "mlp1":
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size)
            )
        elif self.diffuser_type == "mlp2":
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
            )

    def forward(self, x, h, step):
        t = self.step_mlp(step)

        if self.diffuser_type == "mlp1":
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == "mlp2":
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0], dtype=torch.int).to(self.device))
        h = torch.cat([h.view(1, 64)] * x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == "mlp1":
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == "mlp2":
            res = self.diffuser(torch.cat((x, h, t), dim=1))

        return res

    def predict(self, x_start, guide_info, diff, flag=0):
        if (flag == 0 or flag == 1):
            x = guide_info
            item_emb = diff.sample(self.forward, self.forward_uncon, x, x_start)
        elif (flag == 2):
            x = x_start + guide_info
            item_emb = diff.sample(self.forward, self.forward_uncon, x, guide_info)
        elif (flag == 3):
            x = x_start
            item_emb = diff.sample(self.forward, self.forward_uncon, x, guide_info)
        return item_emb