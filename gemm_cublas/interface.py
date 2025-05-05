# Copyright (C) 2025, Tri Dao.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd

from gemm_cublas.autotuner import autotune, AutotuneConfig

NUM_AUTOTUNE_HEURISTICS = 8


@torch.library.register_fake("gemm_cublas::gemm_impl")
def gemm_impl_ref(
    A: Tensor, B: Tensor, C: Optional[Tensor] = None,
    out: Optional[Tensor] = None, out_dtype: Optional[torch.dtype] = None,
    heuristic: int = -1
) -> Tensor:
    """
    Perform matrix multiplication of A and B, with optional bias C.
    If C is provided, it will be added to the result of the matrix multiplication.
    """
    out_given = out
    torch._check(A.ndim == 2)
    torch._check(B.ndim == 2)
    m, k = A.shape
    _, n = B.shape
    torch._check(k == B.shape[0])
    torch._check(A.dtype == B.dtype)
    torch._check(A.dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    if out is None:
        out = torch.empty(
            m, n, device=A.device,
            dtype=C.dtype if C is not None else (out_dtype if out_dtype is not None else A.dtype),
        )
    else:
        torch._check(out.shape == (m, n))
        torch._check(out.device == A.device)
        torch._check(out.dtype in [A.dtype, torch.float])
    if C is not None:
        torch._check(C.shape == (m, n))
        torch._check(C.device == A.device)
        torch._check(C.dtype in [A.dtype, torch.float])
        C = C.to(A.dtype)
    if out.dtype != A.dtype:
        result = A @ B if C is None else torch.addmm(C, A, B)
    else:
        result = torch.mm(A, B, out=out) if C is None else torch.addmm(C, A, B, out=out)
    if out.dtype != A.dtype:
        out.copy_(result)
    return out if out_given is None else None


def gemm_out(
    A: Tensor, B: Tensor, C: Optional[Tensor] = None, out: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None, heuristic: int = -1,
) -> Tensor:
    """
    Perform matrix multiplication of A and B, with optional bias C.
    If C is provided, it will be added to the result of the matrix multiplication.
    The result will be stored in out if provided, otherwise A new tensor will be created.
    """
    # We need to wrap the call to gemm_impl since torch.library doesn't like it when
    # we return a tensor that aliases the input tensor.
    # When out is provided, gemm_impl will return None, so we need to return out.
    out_optional = torch.ops.gemm_cublas.gemm_impl.default(A, B, C, out, out_dtype, heuristic)
    return out if out is not None else out_optional


def gemm_out_ref(A: Tensor, B: Tensor, C: Optional[Tensor] = None, out: Optional[Tensor] = None) -> Tensor:
    out_optional = gemm_impl_ref(A, B, C, out)
    return out if out is not None else out_optional


@torch.library.custom_op("gemm_cublas::gemm", mutates_args={})
def gemm(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
    return gemm_out(A, B, out_dtype=out_dtype)


@torch.library.register_fake("gemm_cublas::gemm")
def gemm_ref(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
    return torch.mm(A, B).to(out_dtype if out_dtype is not None else A.dtype)


@autotune(configs=[AutotuneConfig(heuristic=i) for i in range(NUM_AUTOTUNE_HEURISTICS)])
def gemm_tuned_impl(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None,
                    heuristic: int = -1) -> Tensor:
    return gemm_out(A, B, out_dtype=out_dtype, heuristic=heuristic)


@torch.library.custom_op("gemm_cublas::gemm_tuned", mutates_args={})
def gemm_tuned(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
    return gemm_tuned_impl(A, B, out_dtype=out_dtype)


@torch.library.register_fake("gemm_cublas::gemm_tuned")
def gemm_tuned_ref(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
    return gemm_ref(A, B, out_dtype=out_dtype)


# from torch._inductor.lowering import add_layout_constraint
# add_layout_constraint(torch.ops.gemm_cublas.gemm_not.default, None)


@torch.library.custom_op("gemm_cublas::gemm_add", mutates_args={})
def gemm_add(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_out(A, B, C)


@torch.library.register_fake("gemm_cublas::gemm_add")
def gemm_add_ref(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return C + torch.mm(A, B).to(C.dtype)


@autotune(configs=[AutotuneConfig(heuristic=i) for i in range(NUM_AUTOTUNE_HEURISTICS)])
def gemm_add_tuned_impl(A: Tensor, B: Tensor, C: Tensor, heuristic: int = -1) -> Tensor:
    return gemm_out(A, B, C, heuristic=heuristic)


@torch.library.custom_op("gemm_cublas::gemm_add_tuned", mutates_args={})
def gemm_add_tuned(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_add_tuned_impl(A, B, C)


@torch.library.register_fake("gemm_cublas::gemm_add_tuned")
def gemm_add_tuned_ref(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_add_ref(A, B, C)


@torch.library.custom_op("gemm_cublas::gemm_add_", mutates_args={"C"},
                         schema="(Tensor A, Tensor B, Tensor(a!) C) -> ()")
def gemm_add_(A: Tensor, B: Tensor, C: Tensor) -> ():  # In-place, will modify C
    gemm_out(A, B, C, out=C)


@torch.library.register_fake("gemm_cublas::gemm_add_")
def gemm_add_inplace_ref(A: Tensor, B: Tensor, C: Tensor) -> ():
    C.add_(torch.mm(A, B).to(C.dtype))


@autotune(configs=[AutotuneConfig(heuristic=i) for i in range(NUM_AUTOTUNE_HEURISTICS)],
          restore_value="C")
def gemm_add_inplace_tuned_impl(A: Tensor, B: Tensor, C: Tensor, heuristic: int = -1) -> ():
    gemm_out(A, B, C, out=C, heuristic=heuristic)


@torch.library.custom_op("gemm_cublas::gemm_add_tuned_", mutates_args={"C"},
                         schema="(Tensor A, Tensor B, Tensor(a!) C) -> ()")
def gemm_add_inplace_tuned(A: Tensor, B: Tensor, C: Tensor) -> ():  # In-place, will modify C
    gemm_add_inplace_tuned_impl(A, B, C)


@torch.library.register_fake("gemm_cublas::gemm_add_tuned_")
def gemm_add_inplace_tuned_ref(A: Tensor, B: Tensor, C: Tensor) -> ():
    gemm_add_inplace_ref(A, B, C)


try:
    from torch._inductor.fx_passes.reinplace import InplaceableOp
    torch._inductor.fx_passes.reinplace.inplaceable_ops.update({
        torch.ops.gemm_cublas.gemm_add.default:
        InplaceableOp(torch.ops.gemm_cublas.gemm_add_.default, mutated_arg=2)
    })
except ImportError:
    pass


class LinearFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, fuse_grad_accum=False):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        out: (..., out_features)
        """
        needs_weight_grad = weight.requires_grad
        needs_input_grad = x.requires_grad
        ctx.weight_dtype = weight.dtype
        autocast_dtype = torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled():
            x = x.to(dtype=autocast_dtype)
        weight_og = weight
        if torch.is_autocast_enabled():
            weight = weight.to(dtype=autocast_dtype)
        out = F.linear(x, weight)
        if not needs_input_grad:
            weight, weight_og = None, None
        if not needs_weight_grad:
            x = None
        if not fuse_grad_accum:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, weight_og)
        ctx.fuse_grad_accum = fuse_grad_accum
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        if not ctx.fuse_grad_accum:
            x, weight = ctx.saved_tensors
        else:
            x, weight, weight_og = ctx.saved_tensors
        batch_shape = dout.shape[:-1]
        batch_dim = batch_shape.numel()
        dout = dout.reshape(batch_dim, dout.shape[-1])
        if ctx.needs_input_grad[0]:
            assert weight is not None
            dx = dout @ weight
            dx = dx.reshape(*batch_shape, dx.shape[-1])
        else:
            dx = None
        if ctx.needs_input_grad[1]:
            assert x is not None
            x = x.reshape(batch_dim, x.shape[-1])
            # fuse_grad_accum is not compatible with torch.compile
            if not ctx.fuse_grad_accum or weight_og.grad is None or torch.compiler.is_compiling():
                dweight = gemm(dout.T, x, out_dtype=ctx.weight_dtype)
            else:
                gemm_add_(dout.T, x, weight_og.grad)
                dweight = weight_og.grad
                weight_og.grad = None  # So that pytorch doesn't add dweight to weight_og.grad again
        else:
            dweight = None
        return dx, dweight, None


def linear_func(x, weight, fuse_grad_accum=False):
    return LinearFunc.apply(x, weight, fuse_grad_accum)


class Linear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fuse_grad_accum = fuse_grad_accum

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None and input.is_cuda:
            return linear_func(input, self.weight, fuse_grad_accum=self.fuse_grad_accum)
        else:
            return F.linear(input, self.weight, self.bias)
