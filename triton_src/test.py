import torch
from moe_layer.cvmm import CVMMSel, cvmm, cvmm_prepare_sel, cvmm_prepare_sel2, CVMM
from moe_layer.cvmm import cvmm_triton_backward as cvmm_triton_backward_orig
from typing import Union
from packaging import version
import functools
import triton


cvmm_triton = functools.partial(torch.ops.mylib.cvmm_triton, out_index=torch.tensor(-1).cuda())
cvmm_triton_backward = functools.partial(cvmm_triton_backward_orig, out_index=torch.tensor(-1).cuda())


def cvmm_hack(x: torch.Tensor, sel: Union[torch.Tensor, CVMMSel], keys: torch.Tensor):
    if not isinstance(sel, CVMMSel):
        sel = cvmm_prepare_sel(sel, keys.shape[0])

    res = CVMM.apply(x, sel.sel_index, sel.sel, keys, sel.out_index, None)
    if sel.reduction_weight is not None:
        res = res.view(*sel.reduction_weight.shape, res.shape[-1])
        res = (sel.reduction_weight.unsqueeze(-2).type_as(res) @ res).squeeze(-2)
    return res


def test_wsum():
    n_experts = 2
    n_channels = 3
    expert_size = 3
    bs = 2

    # n_experts = 8
    # n_channels = 64
    # expert_size = 64
    # bs = 32

    # n_per_batch = 1

    n_per_batch = 2
    # reduction_factor = 2
    reduction_factor = 1

    device = torch.device("cuda")
    dtype = torch.float32
    atol_tresh = 1e-2

    keys = torch.nn.Parameter(torch.randn(n_experts, n_channels, expert_size, dtype=dtype, device=device))
    testvec = torch.randn(bs, n_channels, dtype=dtype, device=device)
    sel_raw = torch.randint(0, n_experts, (bs,n_per_batch), dtype=torch.int32, device=device)

    # w = torch.randn_like(sel, dtype=torch.float32)
    w = torch.randn((bs // reduction_factor, n_per_batch * reduction_factor), dtype=torch.float32, device=device)
    # sel = torch.tensor([[1,0]], dtype=torch.int32, device=device)

    sel = cvmm_prepare_sel2(sel_raw, w)
    out = cvmm(testvec, sel, keys)

    def cwmm_ref2(x: torch.Tensor, isel: Union[torch.Tensor, CVMMSel], keys: torch.Tensor):
        if isinstance(isel, CVMMSel):
            sel = isel.raw_sel
            getw = lambda b, c: (isel.reduction_weight[b, c] if isel.reduction_weight is not None else 1.0)
        else:
            sel = isel
            getw = lambda b, c: 1.0

        olist2 = []
        for c in range(sel.shape[-1]):
            olist = []
            for b in range(x.shape[0]):
                olist.append(x[b:b+1] @ keys[sel[b, c]] * getw(b, c))
            olist2.append(torch.cat(olist, dim=0))

        res = torch.stack(olist2, dim=-2)
        if isinstance(isel, CVMMSel) and isel.reduction_weight is not None:
            res = res.sum(-2)
        return res

    ref = cwmm_ref2(testvec, sel, keys)

    if torch.allclose(out, ref, atol=1e-2, rtol=0):
        print("✅ Multi-output: Triton and Torch match")
    else:
        print("❌ Multi-output: Triton and Torch differ")

    grad_out = torch.randn(*out.shape, dtype=dtype, device=device)

    keys_ref = keys.detach().clone().requires_grad_(True)
    testvec_ref = testvec.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    sel = cvmm_prepare_sel2(sel_raw, w_ref)

    print("CVMM hack")
    out_ref = cvmm_hack(testvec_ref, sel, keys_ref)
    out_ref.backward(grad_out)

    keys_full = keys.detach().clone().requires_grad_(True)
    testvec_full = testvec.detach().clone().requires_grad_(True)
    w_full = w.detach().clone().requires_grad_(True)

    sel = cvmm_prepare_sel2(sel_raw, w_full)

    print("CVMM full")
    out_full = cvmm(testvec_full, sel, keys_full)
    out_full.backward(grad_out)

    if torch.allclose(keys_ref.grad, keys_full.grad, atol=1e-2, rtol=0):
        print("✅  Multi-output: Triton weight grad ok")
    else:
        print("❌  Multi-output: Triton weight grad not ok")

    if torch.allclose(testvec_ref.grad, testvec_full.grad, atol=1e-2, rtol=0):
        print("✅  Multi-output: Triton input grad ok")
    else:
        print("❌  Multi-output: Triton input grad not ok")

    if torch.allclose(w_ref.grad, w_full.grad, atol=1e-2, rtol=0):
        print("✅  Multi-output: Triton reduction weight grad ok")
    else:
        print("❌  Multi-output: Triton reduction weight grad not ok")

    from torch.autograd import gradcheck
    assert gradcheck(cvmm, (testvec, sel, keys), eps=1e-2, atol=1e-4)
    print("✅ Gradcheck ok.")


def test_module():
    from torch.autograd import gradcheck

    n_experts = 4
    n_channels = 64
    expert_size = 64
    bs = 32


    device = torch.device("cuda")
    dtype = torch.float32
    atol_tresh = 1e-2

    keys = torch.nn.Parameter(torch.randn(n_experts, n_channels, expert_size, dtype=dtype, device=device))
    testvec = torch.randn(bs, n_channels, dtype=dtype, device=device)
    sel = torch.randint(0, n_experts, (bs,), dtype=torch.int32, device=device)
    test_grad = torch.randn(bs, expert_size, dtype=dtype, device=device)

    olist = []
    for b in range(bs):
        olist.append(testvec[b:b+1] @ keys[sel[b]])
    ref = torch.cat(olist, dim=0)

    out = cvmm(testvec, sel, keys)
    assert torch.allclose(ref, out, atol=atol_tresh, rtol=0)

    print("✅ Forward ok.")

    keys = keys.requires_grad_(True)
    testvec = testvec.requires_grad_(True)

    assert gradcheck(cvmm, (testvec, sel, keys), eps=1e-2, atol=atol_tresh, rtol=0)

    print("✅ Backward ok.")


test_wsum()
test_module()

def compile_test():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 8
            self.n_experts = 8
            self.expert_size = 64
            self.k_vec_dim = 128
            self.v_dim = 128
            self.keys = torch.nn.Parameter(
                torch.empty(self.n_experts, self.k_vec_dim, self.expert_size)
            )
            self.values = torch.nn.Parameter(
                torch.empty(self.n_experts, self.expert_size, self.v_dim)
            )
            self.expert_sel = torch.nn.Linear(self.k_vec_dim, self.n_experts, bias=False)
            self.sel_activation = torch.nn.Sigmoid()

        def compute_scores(self, input: torch.Tensor, index: CVMMSel) -> torch.Tensor:
            scores = cvmm(input, index, self.keys)
            return scores

        def forward(self, input: torch.Tensor):
            sel = sel_raw = self.expert_sel(input)
            sel = self.sel_activation(sel)
            sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)
            # Preprocess the selection indices. They will be needed for both layers and save some time
            sel_indices = cvmm_prepare_sel2(sel_index.int())
            # "Up-projection" layer for each head
            scores = self.compute_scores(input, sel_indices)
            # Down projection layer for each head
            sel_indices = sel_indices.clone()
            sel_indices.reduction_weight = sel_val
            sel_indices.sel_index = sel_indices.out_index
            sel_indices.out_index = None
            out = cvmm(scores, sel_indices, self.values)
            return out


    model = Model().to(torch.float16).cuda()
    model = torch.compile(model)

    torch.manual_seed(0)
    n_experts = 8
    n_channels = 128
    expert_size = 64
    bs = 64

    device = torch.device("cuda")
    dtype = torch.float16

    testvec = torch.randn(bs, n_channels, dtype=dtype, device=device)

    out = model(testvec)
    loss = out.sum()
    loss.backward()

    print(model.keys.grad.shape)
    print(out.shape)

    # If compile is not working, it should crash before this point
    print("✅  Compile test ok.")

if version.parse(torch.__version__) >= version.parse("2.2.0.dev"):
    compile_test()
else:
    print("⚠️ Compile test skipped. PyTorch version is too old. Needs 2.2.0.dev or newer.")
