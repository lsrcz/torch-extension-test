from typing import Annotated, List
import torch

def forward_cpu(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    old_h: torch.Tensor,
    old_cell: torch.Tensor,
) -> Annotated[List[torch.Tensor], 7]:
    pass

def backward_cpu(
    grad_h: torch.Tensor,
    grad_cell: torch.Tensor,
    *saved_tensors: torch.Tensor,
) -> Annotated[List[torch.Tensor], 5]:
    pass

def forward_cuda(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    old_h: torch.Tensor,
    old_cell: torch.Tensor,
) -> Annotated[List[torch.Tensor], 7]:
    pass

def backward_cuda(
    grad_h: torch.Tensor,
    grad_cell: torch.Tensor,
    *saved_tensors: torch.Tensor,
) -> Annotated[List[torch.Tensor], 5]:
    pass
