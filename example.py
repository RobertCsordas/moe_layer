import torch
from moe_layer import MoE

bs = 64
seq_len = 512
d_model = 512

n_experts = 32
expert_size = 128
k = 4


x = torch.randn((bs, seq_len, d_model), device=torch.device("cuda"))

sigma_moe = MoE(d_model, n_experts, expert_size, k).cuda()
out, loss = sigma_moe(x)

# Note that the loss is negative. We are maximizing the entropy, not minimizing it.
print(f"Input shape: {x.shape}, output_shape {out.shape}, loss: {loss.item()}")
