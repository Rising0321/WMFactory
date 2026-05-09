# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 The lingbot-world Authors. Portions derived from:
# https://github.com/Robbyant/lingbot-world/blob/main/wan/distributed/util.py
#
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt at repo root).
#
# Modifications Copyright (c) 2026 SkyworkAI and contributors.
import torch
import torch.distributed as dist


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    if not _dist_ready():
        return 0
    return dist.get_rank()


def get_world_size():
    if not _dist_ready():
        return 1
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
