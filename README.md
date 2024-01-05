# σ-MoE layer

A simple version of our $\sigma$-MoE layer from the paper "Approximating Two-Layer Feedforward Networks for Efficient Transformers" (https://arxiv.org/abs//2310.10837). It is intended to replace the feedforward block (linear-activation-linear stack) in the Transformer. No layernorm or residual connection is included internally.

For the full experimental setup, please visit https://github.com/robertcsordas/moe.

# Choosing the implementation

## Triton (recommended)

A fast Triton based implementation is avaliable under the directory `triton_src`. However it requrest PyTorch 2.1 and Triton 2.1. The minimal supported compute capability is 7.0 (Volta). Due to a known bug in Triton (https://github.com/openai/triton/issues/2377), it is ~1.5x slower on Volta than it should be. Even with this bug, it is faster than the CUDA implementation.

## CUDA

Works on any CUDA compatible GPU (them minimal tested version was Pascal) and most of the recent PyTorch releases. However it is significantly slower than the Triton implementation, and it lacks TensorCore support. It can be found in the `cuda_src` folder.

# torch.compile support

Currently, torch.compile is supported with the Triton implementation, but it requires at least PyTorch 2.2.0-dev (nighly). It also breaks the grap. The PyTorch team is actively working on resolving the remaining issues: https://github.com/pytorch/pytorch/issues/115344.

# Usage

```python
from moe_layer import MoE
```

The signature of the init function is as follows:
```python
def __init__(self, dmodel: int, n_experts: int, expert_size: int, k: int,
                dropout: float = 0, selection_mode: str = "sigmoid",
                activation_after_topk: bool = False,
                activation=F.relu,
                bias: bool = False, v_dim: Optional[int] = None,
                sinkhorn_n_iters: int = 3, expert_dropout: float = 0.0,
                weight_std_scale: float = 1.0):
```

The meaning of the arguments:
- `dmodel` - the number of channels for the input of the layer (same as the output if no v_dim is specified)
- `n_experts` - number of experts
- `expert_size` - the size of a single expert
- `k` - the number of experts active in a single forward pass
- `dropout` - standard dropout on the up-projection part
- `selection_mode` - `"softmax"`, `"sigmoid"` or `"sinkmoid"`. Sinkmoid is Sinkhorn during training and sigmoid during inference, like in S-BASE.
- `activation_after_topk` - set true to apply the activation function for the selection logic after the top-k. Needed e.g. for "Sparsely Gated Mixtures of Experts"-style MoE
- `activation` - the activation function
- `bias` - whether it has bias
- `v_dim` - the number of channels in the output. If None, the same as `dmodel`.
- `sinkhorn_n_iters` - number of Sinkhorn iterations for `sinkmoid` selection mode
- `expert_dropout` - the amount of expert dropout. 0.05 works well.
- `weight_std_scale` - the scaling of the initial weights. Use 1.0 for post-layernorm and $\sqrt{\frac{2}{n_{layers}}}$ for pre-layernorm. If used, use the same scaling for the attention layers as well.


The signature of the forward function:
```python
def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```
The forward pass returns a tuple: (output, entropy regularization loss). The regularization loss should be _added_ to the main loss after multiplying it by a positive factor (typically around 0.001 works well).


The CUDA kernel compiles automatically when the constructor of the layer is called.

# Example

```python
from moe_layer import MoE

sigma_moe = MoE(d_model, n_experts, expert_size, k).cuda()
out, loss = sigma_moe(x)
```

A simple example can be found in `example.py`.
