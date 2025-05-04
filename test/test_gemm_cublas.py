# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch

from gemm_cublas import gemm, gemm_ref
from gemm_cublas import Linear as LinearCB


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_out", [False, True])
# @pytest.mark.parametrize("has_out", [True])
@pytest.mark.parametrize("has_c", [False, True])
# @pytest.mark.parametrize("has_c", [False])
@pytest.mark.parametrize("B_rowmajor", [False, True])
# @pytest.mark.parametrize("B_rowmajor", [True])
@pytest.mark.parametrize("A_rowmajor", [False, True])
# @pytest.mark.parametrize("A_rowmajor", [True])
@pytest.mark.parametrize("n", [1481, 4096])
@pytest.mark.parametrize("k", [732, 4096])
# @pytest.mark.parametrize("n", [4096])
# @pytest.mark.parametrize("k", [2048])
def test_gemm(k, n, A_rowmajor, B_rowmajor, has_c, has_out, input_dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    A = torch.randn((m, k) if A_rowmajor else (k, m), device=device, dtype=input_dtype) / math.sqrt(k)
    A = A.transpose(0, 1) if not A_rowmajor else A
    B = torch.randn((k, n) if B_rowmajor else (n, k), device=device, dtype=input_dtype)
    B = B.transpose(0, 1) if not B_rowmajor else B
    for out_dtype in [input_dtype, torch.float32] if (has_c or has_out) else [input_dtype]:
        C = torch.randn((m, n), device=device, dtype=out_dtype) if has_c else None
        if has_out:
            out_given = torch.empty((m, n), device=device, dtype=out_dtype)
        else:
            out_given = None
        torch.library.opcheck(torch.ops.gemm_cublas.gemm_impl, args=(A, B),
                              kwargs=dict(C=C, out=out_given))
        out = gemm(A, B, C=C, out=out_given)
        out_ref = gemm_ref(A.double(), B.double(), C=C.double() if C is not None else C)
        out_pt = gemm_ref(A, B, C=C, out=out_given.clone()if out_given is not None else None)
        assert out.dtype == out_pt.dtype
        assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6
        if has_out:
            assert torch.equal(out, out_given)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("fuse_grad_accum", [False, True])
# @pytest.mark.parametrize("fuse_grad_accum", [True])
@pytest.mark.parametrize("compiled", [False, True])
# @pytest.mark.parametrize("compiled", [True])
@pytest.mark.parametrize("amp", [False, True])
# @pytest.mark.parametrize("amp", [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("out_features", [1024, 4096])
@pytest.mark.parametrize("in_features", [1024, 4096])
# @pytest.mark.parametrize("out_features", [1024])
# @pytest.mark.parametrize("in_features", [1024])
def test_linear(in_features, out_features, has_bias, amp, compiled, fuse_grad_accum, dtype):
    if compiled:  # Don't fall back to eager just bc of recompilation
        torch._dynamo.config.recompile_limit = 2 ** 31
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    dtype_gen = dtype if not amp else torch.float32
    x_ref = torch.randn(
        batch_size, seqlen, in_features, device=device, dtype=dtype_gen
    ).to(torch.float32).requires_grad_()
    x_pt = x_ref.detach().to(dtype_gen).clone().requires_grad_()
    x = x_ref.detach().to(dtype_gen).clone().requires_grad_()
    model_ref = torch.nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=torch.float32)
    model_pt = torch.nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype_gen)
    model = LinearCB(
        in_features,
        out_features,
        bias=has_bias,
        device=device,
        dtype=dtype_gen,
        fuse_grad_accum=fuse_grad_accum,
    )
    model = torch.compile(model) if compiled else model
    with torch.no_grad():
        model.weight.copy_(model_ref.weight)
        model_pt.weight.copy_(model_ref.weight)
        if has_bias:
            model.bias.copy_(model_ref.bias)
            model_pt.bias.copy_(model_ref.bias)
    out_ref = model_ref(x_ref)
    with torch.autocast(device_type=device, dtype=dtype, enabled=amp):
        out_pt = model_pt(x_pt)
        out = model(x)
    is_close = lambda t, t_pt, t_ref: (t - t_ref).abs().max().item() <= 2 * (t_pt - t_ref).abs().max().item()
    assert is_close(out, out_pt, out_ref)

    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out_pt.backward(g)
    out.backward(g)

    # A second pass to check grad accum
    out_ref = model_ref(x_ref + 0.3)
    with torch.autocast(device_type=device, dtype=dtype, enabled=amp):
        out_pt = model_pt(x_pt + 0.3)
        out = model(x + 0.3)
    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out_pt.backward(g)
    out.backward(g)

    assert is_close(x.grad, x_pt.grad, x_ref.grad)
    assert is_close(model.weight.grad, model_pt.weight.grad, model_ref.weight.grad)
    if has_bias:
        assert is_close(model.bias.grad, model_pt.bias.grad, model_ref.bias.grad)
