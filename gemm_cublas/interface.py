# Copyright (C) 2025, Tri Dao.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd


@torch.library.register_fake("gemm_cublas::gemm_out")
def gemm_out_ref(A: Tensor, B: Tensor, out: Tensor, C: Optional[Tensor] = None) -> ():
    """
    Perform matrix multiplication of A and B, with optional bias C.
    If C is provided, it will be added to the result of the matrix multiplication.
    """
    torch._check(A.ndim == 2)
    torch._check(B.ndim == 2)
    m, k = A.shape
    _, n = B.shape
    torch._check(k == B.shape[0])
    torch._check(A.dtype == B.dtype)
    torch._check(A.dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    if C is not None:
        torch._check(C.shape == (m, n))
        torch._check(C.device == A.device)
        torch._check(C.dtype in [A.dtype, torch.float])
        C = C.to(A.dtype)
    torch._check(out.shape == (m, n))
    torch._check(out.device == A.device)
    torch._check(out.dtype in [A.dtype, torch.float])
    if out.dtype != A.dtype:
        result = A @ B if C is None else torch.addmm(C, A, B)
    else:
        result = torch.mm(A, B, out=out) if C is None else torch.addmm(C, A, B, out=out)
    if out.dtype != A.dtype:
        out.copy_(result)


def gemm_ref(A: Tensor, B: Tensor, C: Optional[Tensor] = None, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        out = torch.empty(A.shape[0], B.shape[1], dtype=A.dtype if C is None else C.dtype,
                          device=A.device)
    gemm_out_ref(A, B, out, C)
    return out


def gemm_out(A: Tensor, B: Tensor, out: Tensor, C: Optional[Tensor] = None) -> ():
    """
    Perform matrix multiplication of A and B, with optional bias C.
    If C is provided, it will be added to the result of the matrix multiplication.
    """
    torch.ops.gemm_cublas.gemm_out.default(A, B, out, C)


def gemm(A: Tensor, B: Tensor, C: Optional[Tensor] = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Perform matrix multiplication of A and B, with optional bias C.
    If C is provided, it will be added to the result of the matrix multiplication.
    The result will be stored in out if provided, otherwise A new tensor will be created.
    """
    if out is None:
        out = torch.empty(A.shape[0], B.shape[1], dtype=A.dtype if C is None else C.dtype,
                          device=A.device)
    gemm_out(A, B, out, C)
    return out


@torch.library.register_fake("gemm_cublas::gemm_t")
def gemm_t_ref(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
    return torch.mm(A.T, B).to(out_dtype if out_dtype is not None else A.dtype)


def gemm_t(A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
    return torch.ops.gemm_cublas.gemm_t.default(A, B, out_dtype)


@torch.library.register_fake("gemm_cublas::gemm_t_add")
def gemm_t_add_ref(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return C + torch.mm(A.T, B).to(C.dtype)


def gemm_t_add(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return torch.ops.gemm_cublas.gemm_t_add.default(A, B, C)


@torch.library.register_fake("gemm_cublas::gemm_t_add_")
def gemm_t_add_inplace_ref(A: Tensor, B: Tensor, C: Tensor) -> ():
    C.add_(torch.mm(A.T, B).to(C.dtype))


def gemm_t_add_(A: Tensor, B: Tensor, C: Tensor) -> ():  # In-place, will modify C
    torch.ops.gemm_cublas.gemm_t_add_.default(A, B, C)


try:
    from torch._inductor.fx_passes.reinplace import InplaceableOp
    torch._inductor.fx_passes.reinplace.inplaceable_ops.update({
        torch.ops.gemm_cublas.gemm_t_add.default:
        InplaceableOp(torch.ops.gemm_cublas.gemm_t_add_.default, mutated_arg=2)
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
        Caution: fuse_grad_accum is not compatible with torch.compile
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
            if not ctx.fuse_grad_accum or weight_og.grad is None:
                dweight = gemm_t(dout, x, out_dtype=ctx.weight_dtype)
            else:
                gemm_t_add_(dout, x, weight_og.grad)
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
        """ Caution: fuse_grad_accum is not compatible with torch.compile
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fuse_grad_accum = fuse_grad_accum

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None and input.is_cuda:
            return linear_func(input, self.weight, fuse_grad_accum=self.fuse_grad_accum)
        else:
            return F.linear(input, self.weight, self.bias)
