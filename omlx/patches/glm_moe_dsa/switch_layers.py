# Copyright © 2023-2024 Apple Inc.

import math
import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu


def _env_flag(name, default="0"):
    return os.environ.get(name, default).lower() in {"1", "true", "on", "yes"}


def use_fused_gate_up_switch():
    return _env_flag("MLX_LM_SWITCH_FUSED_GATE_UP")


def use_fused_swiglu_down_switch():
    return _env_flag("MLX_LM_GLM_MOE_SWIGLU_DOWN")


def use_fused_swiglu_gate_up_switch():
    return _env_flag("MLX_LM_GLM_MOE_SWIGLU_GATE_UP")


def _inverse_permutation(order, inverse_scatter=False):
    if inverse_scatter or _env_flag("MLX_LM_SWITCH_INVERSE_SCATTER"):
        return mx.put_along_axis(
            mx.zeros_like(order),
            order,
            mx.arange(order.size, dtype=order.dtype),
            axis=0,
        )
    return mx.argsort(order)


def _gather_sort(x, indices, inverse_scatter=False):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = _inverse_permutation(order, inverse_scatter)
    lhs_indices = order // M
    x = x.flatten(0, -3)
    return x[lhs_indices], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
        fused_gate_up: bool = False,
        fused_swiglu_down: bool = False,
        inverse_scatter: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation
        self.fused_swiglu_down = fused_swiglu_down
        self.inverse_scatter = inverse_scatter
        if fused_gate_up or use_fused_gate_up_switch():
            self.gate_up_proj = SwitchLinear(
                input_dims, hidden_dims * 2, num_experts, bias=bias
            )
            del self.gate_proj
            del self.up_proj

    def __call__(
        self,
        x,
        indices,
        scores: mx.array | None = None,
        weighted_sum: bool = False,
    ) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(
                x, indices, inverse_scatter=self.inverse_scatter
            )
        if self.training:
            idx = mx.stop_gradient(idx)
        if hasattr(self, "gate_up_proj"):
            if (
                use_fused_swiglu_gate_up_switch()
                and do_sort
                and isinstance(self.activation, SwiGLU)
                and isinstance(self.gate_up_proj, QuantizedSwitchLinear)
                and isinstance(self.down_proj, QuantizedSwitchLinear)
                and hasattr(mx.fast, "glm_moe_swiglu_gate_up")
            ):
                x = mx.fast.glm_moe_swiglu_gate_up(
                    x,
                    self.gate_up_proj["weight"],
                    self.gate_up_proj["scales"],
                    self.gate_up_proj.get("biases"),
                    idx,
                    self.gate_up_proj.group_size,
                    self.gate_up_proj.bits,
                    self.gate_up_proj.mode,
                )
                x = self.down_proj(
                    x,
                    idx,
                    sorted_indices=do_sort,
                )
            else:
                x_gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
                if (
                    (self.fused_swiglu_down or use_fused_swiglu_down_switch())
                    and do_sort
                    and isinstance(self.activation, SwiGLU)
                    and isinstance(self.down_proj, QuantizedSwitchLinear)
                    and hasattr(mx.fast, "glm_moe_swiglu_down")
                ):
                    x = mx.fast.glm_moe_swiglu_down(
                        x_gate_up,
                        self.down_proj["weight"],
                        self.down_proj["scales"],
                        self.down_proj.get("biases"),
                        idx,
                        self.down_proj.group_size,
                        self.down_proj.bits,
                        self.down_proj.mode,
                    )
                    if "bias" in self.down_proj:
                        x = x + mx.expand_dims(self.down_proj["bias"][idx], -2)
                else:
                    x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
                    x = self.down_proj(
                        self.activation(x_up, x_gate),
                        idx,
                        sorted_indices=do_sort,
                    )
        else:
            x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
            x = self.down_proj(
                self.activation(x_up, x_gate),
                idx,
                sorted_indices=do_sort,
            )

        if (
            weighted_sum
            and scores is not None
            and do_sort
            and hasattr(mx.fast, "glm_moe_weighted_sum")
        ):
            return mx.fast.glm_moe_weighted_sum(x, inv_order, scores)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
