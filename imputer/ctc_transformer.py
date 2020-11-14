# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.models import (
    register_model,
    register_model_architecture
)
from .imputer import (
    ImputerModel,
    imputer_iwslt_16,
    imputer_base_architecture
)

@register_model("ctc_transformer")
class CTCTransformerModel(ImputerModel):
    """
    CTCTransformer is simply the Imputer, 
    albiet with a different forwarding path.    
    """

    def forward(
        self, src_tokens, src_lengths, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=None, # no prior alignment available
            encoder_out=encoder_out)

        return {
            "word_ins": {
                "out": word_ins_out,
                "src_lengths": src_lengths * self.encoder.upsample_scale
            }
        }

    # directly use argmax decoding (inherited ) 
    # def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):


@register_model_architecture("ctc_transformer", "ctc_transformer")
def ctc_transformer_base_architecture(args):
    imputer_base_architecture(args)

@register_model_architecture(
    "ctc_transformer", "ctc_transformer_iwslt16"
)
def ctc_transformer_iwslt_16(args):
    imputer_iwslt_16(args)
