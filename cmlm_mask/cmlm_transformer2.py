# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel, base_architecture
from fairseq.models.nat.cmlm_transformer import CMLMNATransformerModel, cmlm_base_architecture

@register_model("cmlm_mask")
class CMLMNATransformerModel2(CMLMNATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.unk = decoder.dictionary.index('<mask>')

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)
        word_ins_mask = prev_output_tokens.type_as(tgt_tokens).eq(self.unk)

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": word_ins_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }

@register_model("nat_mask")
class NATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.unk = decoder.dictionary.index('<mask>')

@register_model_architecture("cmlm_mask", "cmlm_mask")
def cmlm_mask(args):
    cmlm_base_architecture(args)

@register_model_architecture("nat_mask", "nat_mask")
def cmlm_mask(args):
    base_architecture(args)
