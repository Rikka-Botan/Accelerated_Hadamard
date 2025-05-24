# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"
# AHT modeling

import torch
from torch import nn

class BotanHadamardTransform(nn.Module):
    def __init__(
        self,
        hidden_size: int
    ):
        """
        ## Fast Hadamard Transform optimized for GPU

        Input shape: (..., n), n must be a power of 2.
        """
        super().__init__()
        if (hidden_size & (hidden_size - 1) == 0) and hidden_size != 0:
            ValueError("Last dimension must be a power of 2")
        self.register_buffer(
            "H",
            self.generate_hadamard(hidden_size).to(torch.float32) / hidden_size**0.5)
        
    def generate_hadamard(
            self,
            n: int
    ) -> torch.Tensor:
        """
        ## Hadamard matrix generation by Sylvester construction
        """
        if n == 1:
            return torch.tensor([[1]], dtype=torch.int8)
        else:
            H_n = self.generate_hadamard(n // 2)
            top = torch.cat((H_n, H_n), dim=1)
            bottom = torch.cat((H_n, -H_n), dim=1)
            return torch.cat((top, bottom), dim=0)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = torch.matmul(x, self.H)
        return x


class BotanAHT(nn.Module):
    def __init__(
        self,
        hadamard_size: int
    ):
        """
        ## AHT: Accelerated Hadamard Transform optimized for GPU

        Original implementation for BitNet v2
        paper: 'BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs'

        hadamard_size: int
        """
        super().__init__()
        self.hs = hadamard_size
        self.ht = BotanHadamardTransform(hidden_size=hadamard_size)

    def FH(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        ## FH: Flexisible Hadamard
        """
        bsz, seql, embs = hidden_states.size()
        if embs % self.hs != 0:
            ValueError('hidden size should be a multiple of hadamard size.')
        hidden_states = hidden_states.reshape(bsz, seql, -1, self.hs)
        hidden_states = self.ht(hidden_states)
        hidden_states = hidden_states.reshape(bsz, seql, -1)
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.FH(hidden_states)
        return hidden_states

