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
            "logits": word_ins_out,
            "src_lengths": src_lengths * self.encoder.upsample_scale
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        raise NotImplementedError("TODO")
        # step = decoder_out.step
        # max_step = decoder_out.max_step

        # output_tokens = decoder_out.output_tokens
        # output_scores = decoder_out.output_scores
        # history = decoder_out.history

        # # execute the decoder
        # output_masks = output_tokens.eq(self.unk)
        # _scores, _tokens = self.decoder(
        #     normalize=True,
        #     prev_output_tokens=output_tokens,
        #     encoder_out=encoder_out,
        # ).max(-1)
        # output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        # output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # if history is not None:
        #     history.append(output_tokens.clone())

        # # skeptical decoding (depend on the maximum decoding steps.)
        # if (step + 1) < max_step:
        #     skeptical_mask = _skeptical_unmasking(
        #         output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
        #     )

        #     output_tokens.masked_fill_(skeptical_mask, self.unk)
        #     output_scores.masked_fill_(skeptical_mask, 0.0)

        #     if history is not None:
        #         history.append(output_tokens.clone())

        # return decoder_out._replace(
        #     output_tokens=output_tokens,
        #     output_scores=output_scores,
        #     attn=None,
        #     history=history
        # )

    


@register_model_architecture("ctc_transformer", "ctc_transformer")
def ctc_transformer_base_architecture(args):
    imputer_base_architecture(args)

@register_model_architecture(
    "ctc_transformer", "ctc_transformer_iwslt16"
)
def ctc_transformer_iwslt_16(args):
    imputer_iwslt_16(args)