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
from fairseq.utils import new_arange
import re
import logging

logger = logging.getLogger(__name__)

def _skeptical_unmasking(output_scores, output_masks, p):
    """ Non-consecutive Decoding """
    if p >= 0.5:
        sorted_index = output_scores.sort(-1)[1]
        boundary_len = (
            (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
        ).long()
        skeptical_mask = new_arange(output_masks) < boundary_len
        return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

    mask_ratio = p

    # large number gauranteed to be sorted to last
    _large_num = output_scores.shape[1] * 10
    # we don't want to mask pad or eos
    target_masks = output_masks
    # create random sampled scores, which is later used for selecting mask locations
    target_score = output_scores
    # get length for each example in the batch
    target_length = target_masks.sum(1).float()

    # get num of masks & unmasks for each example in the batch
    mask_nums = (target_length * mask_ratio).long() + 1
    unmask_nums = target_length - mask_nums + 1
    # create a binary mask where 1 means unmasks
    unmask_cutoff = new_arange(target_score) < unmask_nums[:, None]

    # make indices larger than B be sorted last -> we will only sample from 0~B
    target_score.masked_fill_(~unmask_cutoff, _large_num)

    # sorting for the top locations
    _, target_rank = target_score.sort(1)

    # create a binary mask where 1 means masks
    mask_cutoff = new_arange(target_score) < mask_nums[:, None]
    
    # sort the desired indices while discarding the rest
    mask_pre,_ = target_rank.masked_fill(~mask_cutoff, _large_num).sort(1)
    # add the cumsum to indices to transform to sequence indices
    mask_mid = mask_pre + new_arange(mask_pre)
    # replace the discarded part with duplicated first column -> ensures correctness.
    duped = mask_mid[:,:1].expand(*mask_mid.shape)
    mask_fin = mask_mid * mask_cutoff + duped * (~mask_cutoff)

    # import pdb
    # pdb.set_trace()

    # scatter 1 to locations indicated by mask_fin, then fill
    skeptical_mask = mask_cutoff.new_zeros(mask_cutoff.size())
    return skeptical_mask.scatter(1, mask_fin, 1)



@register_model("cmlm_noncon")
class CMLMNonConsecutiveModel(CMLMNATransformerModel):
    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )


@register_model_architecture("cmlm_noncon", "cmlm_noncon")
def cmlm_noncon(args):
    cmlm_base_architecture(args)

@register_model_architecture("cmlm_noncon", "cmlm_noncon_iwslt16")
def cmlm_noncon_iwslt16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    cmlm_base_architecture(args)
