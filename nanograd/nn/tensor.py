from __future__ import annotations
import dataclasses
import time, math, itertools, functools, struct, sys, inspect
from contextlib import ContextDecorator
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, DefaultDict, cast, get_args, Set
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor as TorchTensor
import os
from itertools import chain

# Helper functions and constants
def _from_np_dtype(npdtype:type) -> torch.dtype: return torch.dtype(np.dtype(npdtype).name)
def _to_np_dtype(dtype:torch.dtype) -> Optional[type]: return np.dtype(dtype).type if dtype is not None else None

def _fromnp(x: np.ndarray) -> TorchTensor:
    return torch.tensor(x, dtype=_from_np_dtype(x.dtype))

def _frompy(x: Union[List, Tuple, bytes], dtype: torch.dtype) -> TorchTensor:
    if isinstance(x, bytes):
        return torch.tensor(list(memoryview(x)), dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype)

def _get_winograd_matcols(mat, dims:int, shp:Tuple[int, ...], device:Union[str, Tuple[str, ...]]) -> List[List[TorchTensor]]:
    return [[torch.cat([torch.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device) for m in mat], dim=dim)
             for k in range(len(mat[0]))] for dim in range(dims)]

def _apply_winograd_matrix(mat, t:TorchTensor, dims:int) -> TorchTensor:
    t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])
    matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device)
    ret = sum(torch.prod(torch.stack([col[idx] for col, idx in zip(matcols, mat_is)]), dim=0) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
    assert isinstance(ret, TorchTensor), "sum didn't return a Tensor"
    return ret

def _pad_left(*shapes:Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    max_dim = max(len(shape) for shape in shapes)
    return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)

def _broadcast_shape(*shapes:Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(0 if 0 in nth_dim_sizes else max(nth_dim_sizes) for nth_dim_sizes in zip(*_pad_left(*shapes)))

class Function:
    def __init__(self, device:Union[str, Tuple[str, ...]], *tensors: TorchTensor, metadata:Optional[Dict[str, any]]=None):
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
        if self.requires_grad: self.parents = tensors
        self.metadata = metadata

    def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn:Type[Function], *x: TorchTensor, **kwargs) -> TorchTensor:
        ctx = fxn(x[0].device, *x, metadata=None)
        ret = TorchTensor.__new__(TorchTensor)
        ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*x, **kwargs), ctx.requires_grad, None
        ret._ctx = ctx if ctx.requires_grad and not TorchTensor.no_grad else None  # used by autograd engine
        return ret


def fully_flatten(lst):
    """Flattens a nested list or tuple."""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from fully_flatten(item)
        else:
            yield item

def all_int(lst):
    """Checks if all elements in the list are integers."""
    return all(isinstance(x, int) for x in lst)

def getenv(var_name, default=None):
    """Gets an environment variable with a default value."""
    return os.getenv(var_name, default)

def create_schedule_with_vars(*args, **kwargs):
    """Placeholder for creating a schedule with variables."""
    # Replace with actual implementation
    return [], {}

def memory_planner(schedule):
    """Placeholder for memory planning."""
    # Replace with actual implementation
    return []

def fully_flatten(lst):
    """Flattens a nested list or tuple."""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from fully_flatten(item)
        else:
            yield item

def all_int(lst):
    """Checks if all elements in the list are integers."""
    return all(isinstance(x, int) for x in lst)

def getenv(var_name, default=None):
    """Gets an environment variable with a default value."""
    return os.getenv(var_name, default)

def create_schedule_with_vars(*args, **kwargs):
    """Placeholder for creating a schedule with variables."""
    return [], {}

def memory_planner(schedule):
    """Placeholder for memory planning."""
    return []

class ScheduleItem:
    """A placeholder class for schedule items."""
    def __init__(self, operation, inputs, outputs):
        self.operation = operation
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return f"<ScheduleItem op={self.operation}, inputs={self.inputs}, outputs={self.outputs}>"

class Variable:
    """A placeholder class for Variable."""
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def unbind(self):
        return (self.tensor,)

    def __repr__(self):
        return f"<Variable tensor={self.tensor}>"

def fuzz_schedule(lbs):
    """A placeholder function for fuzz_schedule."""
    print(f"Fuzzing schedule with LazyBuffers: {lbs}")

class Tensor:
    def __init__(self, data: Union[None, torch.Tensor, List, Tuple, np.ndarray, bytes],
                 device: Optional[Union[str, Tuple[str, ...]]] = None, dtype: Optional[torch.dtype] = None, requires_grad: Optional[bool] = None):
        if dtype is not None:
            dtype = torch.dtype(dtype)
        assert dtype is None or isinstance(dtype, torch.dtype), f"invalid dtype {dtype}"
        device = torch.device(device) if device is not None else torch.device('cpu')

        if isinstance(data, torch.Tensor):
            if dtype is not None:
                data = data.to(dtype)
        elif isinstance(data, bytes):
            data = torch.tensor(np.frombuffer(data, dtype=torch.uint8), dtype=dtype)
        elif isinstance(data, (list, tuple)):
            if dtype is None:
                d = list(fully_flatten(data))
                dtype = torch.bool if d and all(isinstance(s, bool) for s in d) else (torch.int32 if d and all_int(d) else torch.float32)
            data = torch.tensor(data, dtype=dtype)
        elif data is None:
            data = torch.empty((0,), dtype=dtype or torch.float32, device=device)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=dtype)

        if not isinstance(data, torch.Tensor):
            raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")

        self.lazydata = data.to(device) if data.device != device else data
        self.requires_grad = requires_grad
        self.grad = None

    def realize(self, *lst: 'Tensor', do_update_stats=True) -> 'Tensor':
        """Triggers the computation needed to create these Tensor(s)."""
        # This would be a no-op in PyTorch as operations are eagerly evaluated
        return self

    def replace(self, x: 'Tensor') -> 'Tensor':
        """
        Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
        """
        assert not x.requires_grad and getattr(self, '_ctx', None) is None
        assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
        self.lazydata = x.lazydata
        return self

    def assign(self, x) -> 'Tensor':
        # For simplicity, we'll assume that `x` is always a Tensor here
        if isinstance(x, Tensor):
            x = x.lazydata
        else:
            x = Tensor(x, device=self.device, dtype=self.dtype)
        
        if self.lazydata is x:
            return self  # a self assign is a NOOP

        assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
        assert self.device == x.device, f"assign device mismatch {self.device} != {x.device}"
        assert self.dtype == x.dtype, f"assign dtype mismatch {self.dtype} != {x.dtype}"
        assert not x.requires_grad
        
        self.lazydata.copy_(x)  # Copy data directly in PyTorch
        return self

    def detach(self) -> 'Tensor':
        """
        Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
        """
        return Tensor(self.lazydata.detach(), device=self.device, dtype=self.dtype, requires_grad=False)

    def _data(self) -> memoryview:
        if 0 in self.shape:
            return memoryview(bytearray(0))
        # Convert to CPU and get raw data as a memoryview
        cpu = self.lazydata.contiguous().to('cpu')
        return memoryview(cpu.numpy().data)

    def data(self) -> memoryview:
        """
        Returns the data of this tensor as a memoryview.
        """
        assert self.dtype.is_floating_point or self.dtype.is_integer, f"no fmt dtype for {self.dtype}"
        assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
        return self._data()

    def item(self) -> Union[int, float]:
        """
        Returns the value of this tensor as a standard Python number.
        """
        assert self.numel() == 1, "must have one element for item"
        return self.lazydata.item()

    def tolist(self) -> Union[Sequence, int, float]:
        """
        Returns the value of this tensor as a nested list.
        """
        return self.lazydata.tolist()

    def numpy(self) -> np.ndarray:
        """
        Returns the value of this tensor as a `numpy.ndarray`.
        """
        assert self.dtype is not None, f"no np dtype for {self.dtype}"
        assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
        return self.lazydata.cpu().numpy()
