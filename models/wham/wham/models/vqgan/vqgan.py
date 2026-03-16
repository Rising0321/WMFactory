import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wham.models.wham_base.tensor_spaces import TensorSpace
from wham.models.wham_base.encoder_decoder import EncoderDecoderBase

from wham.models.vqgan import vqgan_models as vqgan
from wham.models.vqvae.vqvae_utils import make_grid, normalise_rgb, rev_normalise_rgb

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

TARGET_GAN_UPDATE = 5
GAN_DWEIGHT_MAX = 250
GAN_LOGIT_CAP = 5.0
MAX_PIXEL_WEIGHTING = 0.1

# The GAN parts are from Taming Transformers (https://github.com/CompVis/taming-transformers)
"""
ViT-VQGAN is based on:
Yu, Jiahui, et al. "Vector-quantized image modeling with improved vqgan." 
ICLR 2022
"""


def create_vqgan_model_for_training(variant):
    return VQGANModel(variant=variant)


class VQGANModel(EncoderDecoderBase):
    @classmethod
    def create_from_variant(cls, variant):
        return VQGANModel(variant=variant)

    def __init__(self, variant=None, ckpt_path=None, model_spec=None):
        super().__init__()
        self.save_hyperparameters()
        self.variant = variant
        if model_spec is not None:
            self.model_spec = model_spec
        else:
            self.model_spec = variant["model_spec"]

        # Batches of images we will use for logging
        self.reference_x_batch = None  # Same images used throughout training to see progress of the model
        self.random_batch = None  # Different images every iteration

        if variant is None and "image_size_per_y_axis" in self.model_spec:
            self.image_size_x = self.model_spec["image_size_per_x_axis"]
            self.image_size_y = self.model_spec["image_size_per_y_axis"]
        else:
            assert "image_size_per_x_axis" in variant and "image_size_per_y_axis" in variant, "Please provide the image size as separate x and y for the VQGAN model"
            self.image_size_x = variant["image_size_per_x_axis"]
            self.image_size_y = variant["image_size_per_y_axis"]

        self._embedding_dim = self.model_spec["embedding_dim"]
        self.encoder = vqgan.ViTEncoder(
            patch_size=self.model_spec["patch_size"],
            transf_dim=self.model_spec["transf_dim"],
            embedding_dim=self.model_spec["embedding_dim"],
            image_size_x=self.image_size_x,
            image_size_y=self.image_size_y,
            num_layers=self.model_spec["num_layers"],
            head_size=self.model_spec["head_size"],
        )
        self._bottleneck_size = self.encoder.bottleneck

        self.vq_vae = vqgan.ViTVectorQuantizer(
            self.model_spec["vocab_size"],
            self.model_spec["embedding_dim"],
            self.model_spec["commitment_cost"],
        )

        self.decoder = vqgan.ViTDecoder(
            patch_size=self.model_spec["patch_size"],
            transf_dim=self.model_spec["transf_dim"],
            embedding_dim=self.model_spec["embedding_dim"],
            image_size_x=self.image_size_x,
            image_size_y=self.image_size_y,
            num_layers=self.model_spec["num_layers"],
            head_size=self.model_spec["head_size"],
            expected_bottleneck=self._bottleneck_size,
        )

        self.is_perceptual = self.model_spec["is_perceptual"]
        assert self.is_perceptual  # This should be on

        # Keep track of the usage of the codebook indices
        self.codebook_index_usage = np.zeros(self.model_spec["vocab_size"], dtype=np.int64)

        self.gan = self.model_spec.get("use_gan", False)
        if self.gan:
            # Only make the patchgan if we are using it. This makes it easier to experiment with GAN settings after pretraining the VQ-VAE for instance
            self.patch_gan = vqgan.PatchGan(channel_start=self.model_spec["gan_channel_start"])
            # Make a copy of the patchgan since we are only using a single optimizer
            self.target_patchgan = vqgan.PatchGan(channel_start=self.model_spec["gan_channel_start"])
            self.target_patchgan.requires_grad_(False)
            self.target_patchgan.load_state_dict(self.patch_gan.state_dict())
            self.target_update = TARGET_GAN_UPDATE

            # At which iteration to start using the GAN loss
            self.gan_start = self.model_spec["gan_start"]
            # How much weight to give to the GAN loss gradients compared to the vq autoencoder loss
            self.gan_weight = self.model_spec["gan_weight"]
            # How many steps to train the discriminator before applying the gan loss.
            self.gan_discrim_pretrain = self.model_spec["gan_discrim_pretrain"]
            # How many steps to warmup the gan loss
            self.gan_discrim_warmup = self.model_spec["gan_discrim_warmup"]
            # Keeping track of the number of updates
            self.updates = 0
            print(f"Using GAN with weight {self.gan_weight} and target update {self.target_update} and gan start {self.gan_start} over {self.gan_discrim_warmup} steps")

        self.lpips_model = None
        # We don't need this for using the encoder/decoder
        # self.lpips_model = lpips.LPIPS(net=self.model_spec["lpips_model"]).eval()
        # for param in self.lpips_model.parameters():
            # param.requires_grad = False

        if ckpt_path is not None and ckpt_path != "None":
            print(f"Initing VQGAN model from {ckpt_path}")
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
            # Can ignore stuff here
            self.load_state_dict(loaded_ckpt["state_dict"], strict=False)

        self.world_space = TensorSpace((3, self.image_size_y, self.image_size_x), dtype=torch.uint8, low=0, high=255)
        self.encoder_space = TensorSpace((self._bottleneck_size,), dtype=torch.long, low=0, high=self.vocab_size - 1)

    @property
    def vocab_size(self):
        """Return the number of entries in the codebook."""
        return self.vq_vae._vocab_size

    @property
    def encoded_bottleneck_dim(self):
        """Return the dimensionality of the latent vector encoded into codebook indices."""
        return self._bottleneck_size

    @property
    def embedding_dim(self):
        """The dimensionality of quantized vectors (the dimension of codebook vectors)."""
        return self.vq_vae._embedding_dim

    def _get_last_layer(self):
        """
        The last layer used for generating the image.
        Used for balancing the gradients of the reconstruction and the GAN loss.
        """
        return self.decoder.get_last_layer()

    def _preprocess_images(self, images):
        """Preprocess images (B, C, H, W)"""
        x_batch = images.float() / 255
        x_batch = normalise_rgb(x_batch)
        return x_batch

    def _revert_image_preprocess(self, x_batch):
        """Revert the preprocessing done in _preprocess_images"""
        normalized_imgs = rev_normalise_rgb(x_batch.clone())
        x_batch = torch.clip(normalized_imgs, 0, 1)
        images = (x_batch * 255).byte()
        return images

    def _get_latent_continuous(self, batch):
        z = self.encoder(batch)
        return z

    def _get_latent_discretized(self, z):
        z_quantized, vq_loss, perplexity, indices = self.vq_vae(z)
        return z_quantized, vq_loss, perplexity, indices

    def _encode_decode(self, x_batch):
        z = self._get_latent_continuous(x_batch)
        z_quantized, vq_loss, perplexity, indices = self._get_latent_discretized(z)
        data_recon = self.decoder(z_quantized)
        return vq_loss, perplexity, data_recon, indices

    def _log_vars(self, log_vars):
        prefix = "train" if self.training else "val"
        for key, val in log_vars.items():
            self.log(f"{prefix}/{key}", val, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def decode_from_encoding_indices(self, encoding_indices):
        """Return decoded images (B, C, H, W) for a batch of encoding indices (B, self.encoded_bottleneck_dim)"""
        z = self.vq_vae.convert_encoding_indices_to_quantized_embeddings(encoding_indices)
        data_recon = self.decoder(z)
        # Denormalize and cast to uint8
        data_recon = self._revert_image_preprocess(data_recon)
        return data_recon

    def get_encoding_indices_for_images(self, images):
        """
        Return encoding indices (B, self.encoded_bottleneck_dim) for a batch of images (B, C, H, W).
        Useful auxiliary method for testing.
        """
        x_batch = self._preprocess_images(images)
        z = self._get_latent_continuous(x_batch)
        encoding_indices = self.vq_vae(z, only_return_encoding_indices=True)
        return encoding_indices

    def forward_returning_action_and_embedding(self, states, actions_input, timesteps, attention_mask, images):
        raise NotImplementedError

    def get_encoding_output(self, images):
        """
        Return outputs from the encoder for a batch of images (B, C, H, W).
        Returns:
            quantized_z: (B, self.encoded_bottleneck_dim, self.embedding_dim), quantized latent vectors with straight-through gradient estimator
            vq_loss: (B, ), VQ loss for each image
            perplexity: (B, ), perplexity for each image
            encoding_indices: (B, self.encoded_bottleneck_dim), encoding indices for each image
        """
        x_batch = self._preprocess_images(images)
        z = self._get_latent_continuous(x_batch)
        quantized_z, vq_loss, perplexity, encoding_indices = self.vq_vae(z)
        quantized_z = quantized_z.view(quantized_z.shape[0], self.encoded_bottleneck_dim, self.embedding_dim)
        return quantized_z, vq_loss, perplexity, encoding_indices

    def _encode(self, world_space_tensor: torch.tensor) -> torch.tensor:
        # Flatten time and batch dim into one
        batch, time = world_space_tensor.shape[:2]
        world_space_tensor = world_space_tensor.view(batch * time, *world_space_tensor.shape[2:])
        encodings = self.get_encoding_indices_for_images(world_space_tensor)
        # Reshape back to (batch, time, ...)
        encodings = encodings.view(batch, time, -1)
        return encodings

    def _decode(self, encoder_space_tensor: torch.tensor) -> torch.tensor:
        # Flatten time and batch dim into one
        batch, time = encoder_space_tensor.shape[:2]
        encoder_space_tensor = encoder_space_tensor.view(batch * time, *encoder_space_tensor.shape[2:])
        decoded = self.decode_from_encoding_indices(encoder_space_tensor)
        # Reshape back to (batch, time, ...)
        decoded = decoded.view(batch, time, *decoded.shape[1:])
        return decoded
