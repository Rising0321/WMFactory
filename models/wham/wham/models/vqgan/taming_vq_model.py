# Wrapper for the VQ models from the taming-transformers repo
# https://github.com/CompVis/taming-transformers

from typing import Any, Mapping
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from wham.models.vqgan.taming.model import Encoder, Decoder
from wham.models.vqgan.taming.quantize import VectorQuantizer2 as VectorQuantizer

from wham.models.wham_base.tensor_spaces import TensorSpace
from wham.models.wham_base.encoder_decoder import EncoderDecoderBase


HARDCODED_IMAGE_SIZE = 128


def taming_vq_preprocess_images(imgs):
    """Normalize images (as pytorch tensor uint8s) as in taming-transformers"""
    return imgs.float() / 127.5 - 1.0


def taming_vq_revert_preprocess_images(imgs):
    """Revert preprocessing of images from taming to uint8 as in taming-transformers"""
    # Clamp first
    imgs = torch.clamp(imgs, -1.0, 1.0)
    return ((imgs + 1) * 127.5).byte()


class _VQModelFromTamingRepository(pl.LightningModule):
    """
    This aims to be the original VQ model from the taming-transformers repo with as little modifications as possible. This should not be used directly.
    Source: https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py

    MIT License
    Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
                  2023 Microsoft Research

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.
    """

    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # NOTE: Loss is disabled for this repo (we only want inference)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # Note: the '!= "None"' check is for checkpoints that mistakenly stored the None as a string
        if ckpt_path is not None and ckpt_path != "None":
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError("This copy of the model code does not support training")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("This copy of the model code does not support training")

    def configure_optimizers(self):
        raise NotImplementedError("This copy of the model code does not support training")

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class TamingVQModel(EncoderDecoderBase):

    __DEBUG_CREATION_KWARGS__ = {
        "ckpt_path": None,
        "model_spec": {
            "taming_n_embed": 16,
            "taming_embed_dim": 8,
            "taming_num_indices_per_axis": 8,
            "taming_ddconfig": {
                "double_z": False,
                "z_channels": 16,
                "resolution": 128,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 1, 1, 1, 1],
                "num_res_blocks": 1,
                "attn_resolutions": [16],
                "dropout": 0.0,
            },
        },
    }

    def __init__(self, model_spec, ckpt_path, **kwargs):
        super().__init__()
        self._vocab_size = model_spec["taming_n_embed"]
        self.num_indices_per_axis = model_spec["taming_num_indices_per_axis"]
        self.num_indices_total = self.num_indices_per_axis**2
        self.taming_embed_dim = model_spec["taming_embed_dim"]
        taming_ddconfig = model_spec.get("taming_ddconfig", None)
        if taming_ddconfig is None:
            raise ValueError("To run TamingVQModel, specify model_spec.taming_ddconfig, which should match the ddconfig used when training the model")

        self.vq_model = _VQModelFromTamingRepository(taming_ddconfig, self._vocab_size, self.taming_embed_dim, ckpt_path=ckpt_path)

        resolution = taming_ddconfig["resolution"]
        in_channels = taming_ddconfig["in_channels"]
        self.world_space = TensorSpace((in_channels, resolution, resolution), dtype=torch.uint8, low=0, high=255)
        self.encoder_space = TensorSpace((self.num_indices_total,), dtype=torch.long, low=0, high=self.vocab_size - 1)

    @property
    def vocab_size(self):
        """Return the number of entries in the codebook."""
        return self._vocab_size

    @property
    def encoded_bottleneck_dim(self):
        """Return the dimensionality of the latent vector encoded into codebook indices."""
        return self.num_indices_total

    def _preprocess_images(self, images):
        """Preprocess images (B, C, H, W)"""
        return taming_vq_preprocess_images(images)

    def _revert_image_preprocess(self, x_batch):
        """Revert the preprocessing done in _preprocess_images"""
        return taming_vq_revert_preprocess_images(x_batch)

    def decode_from_encoding_indices(self, encoding_indices, return_vq_embeddings=False):
        """Return decoded images (B, C, H, W) for a batch of encoding indices (B, self.encoded_bottleneck_dim)"""
        batch_size = encoding_indices.shape[0]
        z = self.vq_model.quantize.get_codebook_entry(encoding_indices, shape=(batch_size, self.num_indices_per_axis, self.num_indices_per_axis, self.taming_embed_dim))
        data_recon = self.vq_model.decode(z)
        # Denormalize and cast to uint8
        data_recon = self._revert_image_preprocess(data_recon)
        if return_vq_embeddings:
            return data_recon, z
        return data_recon

    def get_encoding_indices_for_images(self, images):
        """
        Return encoding indices (B, self.encoded_bottleneck_dim) for a batch of images (B, C, H, W).
        Useful auxiliary method for testing.
        """
        x_batch = self._preprocess_images(images)
        _, _, (_, _, encoding_indices) = self.vq_model.encode(x_batch)
        # Split back into (B, self.encoded_bottleneck_dim)
        encoding_indices = encoding_indices.view(images.shape[0], -1)
        return encoding_indices

    def forward_returning_action_and_embedding(self, states, actions_input, timesteps, attention_mask, images):
        seq_len_dim = 1
        assert images.shape[seq_len_dim] == 1, f"We require seq_len==1, but provided {images.shape[seq_len_dim]}."
        images = images.squeeze(dim=seq_len_dim)  # get rid of timestep dimension
        x_batch = self._preprocess_images(images)
        quant, _, (_, _, encoding_indices) = self.vq_model.encode(x_batch)
        # Split back into (B, self.encoded_bottleneck_dim)
        encoding_indices = encoding_indices.reshape(quant.shape[0], 1, quant.shape[2], quant.shape[3])
        quant = quant.unsqueeze(seq_len_dim)
        return None, {"quantized": quant, "encoding_indices": encoding_indices}

    def _encode(self, world_space_tensor: torch.tensor) -> torch.tensor:
        batch, time = world_space_tensor.shape[:2]
        world_space_tensor = world_space_tensor.view(batch * time, *world_space_tensor.shape[2:])
        encodings = self.get_encoding_indices_for_images(world_space_tensor)
        # Reshape back to (batch, time, ...)
        encodings = encodings.view(batch, time, -1)
        return encodings

    def _decode(self, encoder_space_tensor: torch.tensor) -> torch.tensor:
        batch, time = encoder_space_tensor.shape[:2]
        encoder_space_tensor = encoder_space_tensor.view(batch * time, *encoder_space_tensor.shape[2:])
        decoded = self.decode_from_encoding_indices(encoder_space_tensor)
        # Reshape back to (batch, time, ...)
        decoded = decoded.view(batch, time, *decoded.shape[1:])
        return decoded
