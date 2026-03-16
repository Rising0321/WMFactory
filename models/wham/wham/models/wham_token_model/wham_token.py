import torch
from wham.models.wham_base.encode_predict_decode_base import (
    EncodePredictDecodeModule,
    create_encoder_modules_from_args,
)
from wham.models.wham_token_model.gpt_token_transformer_predictor import GPTTokenPredictor
from wham.models.vqgan.taming_vq_model import TamingVQModel
from wham.models.wham_token_model.token_action_encoder import ActionTokenEncoder
from wham.models.pl.pl_base_model import BaseTrainingModel

LOSS_MASK_KEY = "loss_mask"


def class_name_to_model(class_name):
    if class_name == "GPTTokenPredictor":
        return GPTTokenPredictor
    if class_name == "TamingVQModel":
        return TamingVQModel
    if class_name == "ActionTokenEncoder":
        return ActionTokenEncoder
    raise NotImplementedError(f"Model type {class_name} not implemented.")


class WHAMTokenModule(BaseTrainingModel, EncodePredictDecodeModule):
    """A model that functions on a token level (e.g., combines all states and actions into one long sequence)"""

    def __init__(self, predictor_args, context_encoder_args, variant):
        self.save_hyperparameters()
        self.variant = variant

        context_encoders = create_encoder_modules_from_args(context_encoder_args)
        # Freeze the context encoders
        for context_encoder_param in context_encoders.parameters():
            context_encoder_param.requires_grad = False

        # Determine whether to use only the BC loss
        self.bc_loss_only = variant["model_spec"].get("bc_loss_only", False)

        super().__init__(predictor_args=predictor_args, context_encoders=context_encoders)

    def predict_next_step(self, world_space_context, **kwargs):
        """
        Predict the next step in the world space context.

        Args:
            world_space_context (TensorDict): A TensorDict containing the world space context.
            **kwargs: passed to predictor "predict_next_step"
        Returns:
            TensorDict: A TensorDict containing the predicted next step (batch, 1, ...)
        """
        context = self.encode_context(world_space_context)

        # If we have tokens for an image, lets override their tokens
        # Code is not great, but it gets the job done...
        tokens = kwargs.get("tokens", None)
        batch_size = context["images"].shape[0]
        if tokens is not None:
            for batch_idx in range(batch_size):
                for timestep in range(context["images"][batch_idx].shape[0]):
                    if tokens[batch_idx][timestep] is not None:
                        tensored_tokens = torch.tensor(tokens[batch_idx][timestep], device=context["images"].device)
                        context["images"][batch_idx][timestep] = tensored_tokens
        if "tokens" in kwargs:
            del kwargs["tokens"] # We've used this, so remove it
        predicted_next_step, _ = self.predictor.predict_next_step(context, **kwargs)
        image_tokens = predicted_next_step["images"]
        decoded_next_step = self.decode_context(predicted_next_step)
        return decoded_next_step, image_tokens
