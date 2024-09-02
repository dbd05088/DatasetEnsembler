from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPVisionConfig, CLIPVisionTransformer, add_start_docstrings_to_model_forward, replace_return_docstrings, BaseModelOutputWithPooling
from .nets_utils import EmbeddingRecorder
from .nets_utils.docstring import *
from typing import Optional, Union, Tuple
from torch import set_grad_enabled, flatten, Tensor
import torch
import torch.nn as nn

class CLIPViT(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig, record_embedding: bool = False,
                 no_grad: bool = False):
        super().__init__(config)
        # embedding recorder
        
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad
        self.visual_projection = nn.Linear(768, 512, bias=False)
        self.head = nn.Identity()
        
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        with set_grad_enabled(not self.no_grad):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            hidden_states = self.embeddings(pixel_values)
            hidden_states = self.pre_layrnorm(hidden_states)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.post_layernorm(pooled_output)

            vision_output = (last_hidden_state, pooled_output) + encoder_outputs[1:]
            image_embeds = self.visual_projection(vision_output[1])
            image_embeds = image_embeds / _get_vector_norm(image_embeds)
            
            image_embeds = self.embedding_recorder(image_embeds)
            output = self.head(image_embeds)
            return output
            # if not return_dict:
            #     return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            # return BaseModelOutputWithPooling(
            #     last_hidden_state=last_hidden_state,
            #     pooler_output=pooled_output,
            #     hidden_states=encoder_outputs.hidden_states,
            #     attentions=encoder_outputs.attentions,
            # )
    
    def get_last_layer(self):
        return self.head

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor