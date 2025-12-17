import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res
        
class NoiseTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, residual: bool = True) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        elif x.dim() == 3:
            pass 
        else:
            raise ValueError(f"Expected input x with shape [B, D] or [B, 1, D], but got {x.shape}")

        out = self.transformer(x) 
        out = out.squeeze(1)      

        if residual:
            out = out + x.squeeze(1)

        return out



class SVDNoiseUnet(nn.Module):
    def __init__(self, dim, hidden_mult=2):
        super().__init__()
        self.dim = dim

        self.u_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.s_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 3, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim)
        )

        self.rescale = nn.Parameter(torch.ones(1, dim))  

    def forward(self, x):  
        B, D = x.shape
        U, s, Vh = torch.linalg.svd(x, full_matrices=False)  

        U_proj = self.u_proj(U)              
        Vh_proj = self.v_proj(Vh[0]).unsqueeze(0).expand(B, -1)  
        s_proj = self.s_proj(s).unsqueeze(0).expand(B, -1)       

        fused = torch.cat([U_proj, Vh_proj, s_proj], dim=-1)    
        fused = self.fusion_proj(fused)                          

        return x + fused * self.rescale  

    
class NPNet(nn.Module):
    def __init__(self, num_items: int, dim: int, with_text_emb: bool = True):
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.with_text_emb = with_text_emb

        if self.with_text_emb:
            self.text_embedding = nn.Linear(dim, dim)

        self.noise_transformer = NoiseTransformer(d_model=dim, nhead=4, num_layers=2)
        self.svd_noiset = SVDNoiseUnet(dim=dim)

        self._alpha = nn.Parameter(torch.ones(1))  
        self._beta = nn.Parameter(torch.ones(1))   

    def forward(self, initial_noise: torch.Tensor, prompt_embeds: torch.Tensor = None) -> torch.Tensor:
        # B, D = initial_noise.shape

        if self.with_text_emb and prompt_embeds is not None:
            text_emb = self.text_embedding(prompt_embeds)
            encoder_hidden_states_svd = initial_noise + text_emb
            encoder_hidden_states_embedding = initial_noise + text_emb
        else:
            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise

        info_embedding = self.noise_transformer(encoder_hidden_states_embedding)
        svd_noise = self.svd_noiset(encoder_hidden_states_svd)

        if self.with_text_emb and prompt_embeds is not None:
            info_noise = svd_noise + (2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * info_embedding
        else:
            info_noise = svd_noise * (2 * torch.sigmoid(self._alpha) - 1) + info_embedding * self._beta

        return info_noise
