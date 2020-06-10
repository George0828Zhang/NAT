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
from fairseq.models.nat.cmlm_transformer import CMLMNATransformerModel, cmlm_base_architecture

import re
import logging

logger = logging.getLogger(__name__)

@register_model("cmlm_mask")
class CMLMNATransformerModel2(CMLMNATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.unk = decoder.dictionary.index('<mask>')
        self.predict_masked_only = getattr(args, "predict_masked_only", False)

    @staticmethod
    def add_args(parser):
        CMLMNATransformerModel.add_args(parser)
        parser.add_argument("--load-weight-level", default='all',
                            choices=['all', 'encoder_decoder', 'encoder'],
                            help="which components needs to load weights from checkpoint. all: load all. base: load encoder and decoder only. encoder: load encoder only.")
        parser.add_argument("--predict-masked-only", action="store_true",
                            help="When computing loss, whether to ignore the unmasked tokens.")

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        """Overrides fairseq_model.py

        """
        # pdb.set_trace()
        if args.load_weight_level == "encoder":
            logger.warning("Will only load encoder weights!")
            cur = self.state_dict()
            for param in state_dict:
                if re.match(r"^encoder\.", param) is not None:
                    cur[param] = state_dict[param]
            state_dict = cur
        elif args.load_weight_level == "encoder_decoder":
            logger.warning("Will only load encoder and decoder weights!")
            cur = self.state_dict()
            for param in state_dict:
                if re.match(r"^(encoder|decoder)\.", param) is not None:
                    cur[param] = state_dict[param]
            state_dict = cur
        return super().load_state_dict(state_dict, strict=strict, args=args)

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

        if self.predict_masked_only:
            word_ins_mask = prev_output_tokens.type_as(tgt_tokens).eq(self.unk)
        else:
            word_ins_mask = tgt_tokens.ne(self.pad)

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

@register_model_architecture("cmlm_mask", "cmlm_mask")
def cmlm_mask(args):
    cmlm_base_architecture(args)
    args.predict_masked_only = True

@register_model_architecture("cmlm_mask", "nat_mask")
def nat_mask(args):
    cmlm_base_architecture(args)
    args.predict_masked_only = getattr(args, "predict_masked_only", False)
