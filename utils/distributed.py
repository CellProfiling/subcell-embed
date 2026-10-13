import torch
from torch import distributed as dist


def all_gather(tensor: torch.Tensor) -> list:
    """Gather tensors from all ranks; first dimension may differ across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return [tensor]
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]

    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]

    padded = tensor.new_zeros((max(sizes), *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return [g[:s] for g, s in zip(gathered, sizes)]
