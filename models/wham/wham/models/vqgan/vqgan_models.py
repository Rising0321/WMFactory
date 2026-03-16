# MIT License
# Copyright (c) 2018 Zalando Research
#               2023 Microsoft Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from wham.models.nn.nanoGPT import GPTConfig, SelfAttentionBlock
from wham.models.nn.model_blocks import ConvNextBlock, ConvNextDownsample, ConvNextDownsampleBig

# Mainly following https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
"""
ViT-VQGAN is based on:
Yu, Jiahui, et al. "Vector-quantized image modeling with improved vqgan." 
ICLR 2022
"""


def _convert_encoding_indices_to_quantized_embeddings(encoding_indices, embedding_layer, vocab_size, embedding_dim):
    """
    Args:
        encoding_indices: tensor of integers (batch_size, bottleneck_size)
                            Each batch item represents a single image as a sequence of integers (indeces of codebook vectors)
    Output:
        quantized: tensor of floats (batch_size, bottleneck_size, embedding_dim)
    """
    batch_dim, bottleneck_size = encoding_indices.shape[:2]

    encoding_indices = encoding_indices.view(-1).unsqueeze(1)
    one_hot_encoding_indices = torch.zeros(encoding_indices.shape[0], vocab_size, device=encoding_indices.device)
    one_hot_encoding_indices.scatter_(1, encoding_indices, 1)

    quantized = torch.matmul(one_hot_encoding_indices, embedding_layer)
    quantized = quantized.view(batch_dim, bottleneck_size, embedding_dim).contiguous()
    return quantized


class ViTVectorQuantizer(nn.Module):
    """
    Vector Quantizer for a Vision Transformer based VQ model using normalised codebook embeddings as in https://arxiv.org/abs/2110.04627.
    """

    def __init__(self, vocab_size, embedding_dim, commitment_cost, epsilon=1e-5):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._epsilon = epsilon

        self._embedding = nn.Embedding(self._vocab_size, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._vocab_size, 1 / self._vocab_size)
        self._commitment_cost = commitment_cost

    @property
    def vocab_size(self):
        """Return the number of entries in the codebook."""
        return self._vocab_size

    def convert_encoding_indices_to_quantized_embeddings(self, encoding_indices):
        """
        Args:
            encoding_indices: tensor of integers (batch_size, bottleneck_size)
                              Each batch item represents a single image as a sequence of integers (indeces of codebook vectors)
        Output:
            quantized: tensor of floats (batch_size, self._embedding_dim, bottleneck_size)
        """
        return _convert_encoding_indices_to_quantized_embeddings(encoding_indices, F.normalize(self._embedding.weight), self._vocab_size, self._embedding_dim)

    def forward(self, inputs, only_return_encoding_indices=False):
        """
        If only_return_encoding_indices is True, then only return the indices of codebook vectors
        """
        input_shape = inputs.shape

        # Flatten input from Batch Tokens Embedding to B*T E
        flat_input = inputs.view(-1, self._embedding_dim)
        # Normalize inputs
        flat_input = F.normalize(flat_input)

        # Embeddings are always normalized
        embeddings_to_use = F.normalize(self._embedding.weight)

        # Calculate distances
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(embeddings_to_use**2, dim=1) - 2 * torch.matmul(flat_input, embeddings_to_use.t())

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if only_return_encoding_indices:
            # Add back batch dimension
            return encoding_indices.view(input_shape[0], -1)
        one_hot_encoding_indices = torch.zeros(encoding_indices.shape[0], self._vocab_size, device=inputs.device)
        one_hot_encoding_indices.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(one_hot_encoding_indices, embeddings_to_use).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(one_hot_encoding_indices, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        return quantized, loss, perplexity, encoding_indices.view(input_shape[0], -1)


class ViTEncoder(nn.Module):
    def __init__(self, patch_size, transf_dim, embedding_dim, image_size_x, image_size_y, num_layers, head_size):
        super().__init__()

        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        # We will pad the image to make it divisible by patch_size
        self.x_pad = (patch_size - (self.image_size_x % patch_size)) % patch_size
        self.y_pad = (patch_size - (self.image_size_y % patch_size)) % patch_size
        assert (self.image_size_x + self.x_pad) % patch_size == 0 and (
            self.image_size_y + self.y_pad
        ) % patch_size == 0, "image_size_x and image_size_y must be divisible by patch_size"

        self.vit_tokens = ((image_size_x + self.x_pad) // patch_size) * ((image_size_y + self.y_pad) // patch_size)
        self._bottleneck = self.vit_tokens
        print(f"Bottleneck is {self.bottleneck} for image size {image_size_x}x{image_size_y} with ViT Encoder and patch size {patch_size}")

        self.patch_size = patch_size
        self.transf_dim = transf_dim
        self.embedding_dim = embedding_dim

        self.proj1 = nn.Linear(3 * patch_size * patch_size, transf_dim)
        self.pos_embeds = nn.Embedding(self.vit_tokens, transf_dim)

        assert self.transf_dim % head_size == 0, "transf_dim must be divisible by head_size"
        n_heads = self.transf_dim // head_size
        transformer_config = GPTConfig(block_size=self.vit_tokens, n_layer=num_layers, n_head=n_heads, n_embd=transf_dim, bias=False, dropout=0)
        self.vit = nn.Sequential(*[SelfAttentionBlock(transformer_config) for _ in range(num_layers)])

        self.output_ln = nn.LayerNorm(transf_dim)
        self.output_proj = nn.Linear(transf_dim, embedding_dim)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / sqrt(2 * transformer_config.n_layer))

    @property
    def bottleneck(self):
        return self._bottleneck

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs):
        # inputs: (batch_size, 3, image_size_x, image_size_y)

        # Patch input images
        batch_size = inputs.shape[0]
        padded_inputs = F.pad(inputs, (0, self.x_pad, 0, self.y_pad), mode="constant", value=0)
        x = padded_inputs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        num_x_patches = (self.image_size_x + self.x_pad) // self.patch_size
        num_y_patches = (self.image_size_y + self.y_pad) // self.patch_size

        # inputs is of shape (batch_size, 3, num_x_patches, num_y_patches, patch_size, patch_size)
        # Turn it into (batch_size, patches, input_dim)
        patches = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, num_x_patches * num_y_patches, 3 * self.patch_size * self.patch_size)

        proj_patches = self.proj1(patches)

        pos_embeds = self.pos_embeds.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        vit_input = proj_patches + pos_embeds
        vit_output = self.vit(vit_input)

        vit_output = self.output_ln(vit_output)
        embeddings = self.output_proj(vit_output)
        normalised_embeddings = F.normalize(embeddings, dim=-1)

        return normalised_embeddings


class ViTDecoder(nn.Module):
    def __init__(self, patch_size, transf_dim, embedding_dim, image_size_x, image_size_y, num_layers, head_size, expected_bottleneck=None):
        super().__init__()

        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.x_pad = (patch_size - (self.image_size_x % patch_size)) % patch_size
        self.y_pad = (patch_size - (self.image_size_y % patch_size)) % patch_size

        assert (self.image_size_x + self.x_pad) % patch_size == 0 and (
            self.image_size_y + self.y_pad
        ) % patch_size == 0, "image_size_x and image_size_y must be divisible by patch_size"

        self.vit_tokens = ((image_size_x + self.x_pad) // patch_size) * ((image_size_y + self.y_pad) // patch_size)
        if expected_bottleneck is not None:
            assert (
                self.vit_tokens == expected_bottleneck
            ), f"Expected bottleneck of {expected_bottleneck} but got {self.vit_tokens} for image size {image_size_x}x{image_size_y} with ViT Decoder and patch size {patch_size}"

        self.patch_size = patch_size
        self.transf_dim = transf_dim
        self.embedding_dim = embedding_dim

        self.proj1 = nn.Linear(embedding_dim, transf_dim)
        self.pos_embeds = nn.Embedding(self.vit_tokens, transf_dim)

        assert self.transf_dim % head_size == 0, "transf_dim must be divisible by head_size"
        n_heads = self.transf_dim // head_size
        transformer_config = GPTConfig(block_size=self.vit_tokens, n_layer=num_layers, n_head=n_heads, n_embd=transf_dim, bias=False, dropout=0)
        self.vit = nn.Sequential(*[SelfAttentionBlock(transformer_config) for _ in range(num_layers)])

        self.output_ln = nn.LayerNorm(transf_dim)
        self.output_proj = nn.Linear(transf_dim, 3 * patch_size * patch_size)

        # Couldn't resist the name
        self.folder = nn.Fold(
            output_size=(self.image_size_y + self.y_pad, self.image_size_x + self.x_pad),
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / sqrt(2 * transformer_config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs):
        # Patch input images
        batch_size = inputs.shape[0]

        # Unproject the embeddings from the VQ embedding space to the transformer space
        proj_patches = self.proj1(inputs).reshape(batch_size, self.vit_tokens, self.transf_dim)

        pos_embeds = self.pos_embeds.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        vit_input = proj_patches + pos_embeds
        vit_output = self.vit(vit_input)

        vit_output = self.output_ln(vit_output)

        predictions = self.output_proj(vit_output)  # (batch, patches, 3 * patch_size * patch_size)

        # Reassemble the image into (batch, 3, image_size_x, image_size_y)
        fold_inputs = predictions.permute(0, 2, 1).contiguous()
        image_pred = self.folder(fold_inputs)

        unpadded_image_pred = image_pred[:, :, : self.image_size_y, : self.image_size_x]  # Remove padding in the same way it was applied in the encoder

        # Anything on the output?
        return unpadded_image_pred

    def get_last_layer(self):
        """
        Return the last layer weights of the model, to use for loss balancing.
        """
        return self.output_proj.weight


class PatchGan(nn.Module):
    def __init__(self, channel_start):
        super().__init__()
        x = channel_start
        self.downsample1 = ConvNextDownsampleBig(3, x)
        self.block1 = ConvNextBlock(x)
        self.downsample2 = ConvNextDownsampleBig(x, x)
        self.block2 = ConvNextBlock(x)
        self.last = nn.Conv2d(x, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.nn.functional.gelu(self.downsample1(x))
        y = self.block1(y)
        z = torch.nn.functional.gelu(self.downsample2(y))
        z = self.block2(z)
        return self.last(z).reshape(batch_size, -1)