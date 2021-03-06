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
from fairseq.models.nat import (
    NATransformerModel,
    base_architecture as nonautoregressive_transformer_base_architecture,
    CMLMNATransformerModel,
    cmlm_base_architecture
)
from fairseq.models.roberta import XLMRModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import pdb

logger = logging.getLogger(__name__)

def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False

def load_teacher(
    model_name_or_path,
    checkpoint_file="model.pt",
    data_name_or_path=".",
    bpe="sentencepiece",
    **kwargs
):
    from fairseq import hub_utils
    x = hub_utils.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,
        archive_map=XLMRModel.hub_models(),
        bpe=bpe,
        load_checkpoint_heads=False,
        **kwargs,
    )
    return x["models"][0].encoder

# class BERT2NATransformerModel(CMLMNATransformerModel):
@register_model("bert2nat")
class BERT2NATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.hint_loss_factor = args.hint_loss_factor
        self.hint_from_layer = args.hint_from_layer
        self.teacher = load_teacher(
            args.teacher_dir, 
            checkpoint_file='model.pt',
        )
        # if args.max_source_positions + args.max_target_positions > self.teacher.max_positions():
        #     logger.warning("maximum possible source+target tokens exceeds maximum length: {} > {}. might lead to error.".format(
        #                 args.max_source_positions + args.max_target_positions, self.teacher.max_positions()
        #             ))
        logger.info(f"Vocab size: teacher {self.teacher.sentence_encoder.vocab_size} nat {len(decoder.dictionary)}")
        
        freeze_module_params(self.teacher) # if requires_grad not turned off, optimizer states will be saved even if not being trained.

        embed_dim = encoder.embed_tokens.embedding_dim
        teacher_embed_dim = self.teacher.sentence_encoder.embed_tokens.embedding_dim
        self.teacher_proj = nn.Linear(teacher_embed_dim, embed_dim)
        
        if getattr(args, "apply_bert_init", False):
            for m in (self.teacher_proj,):
                m.apply(init_bert_params)

    def state_dict(self, *args, **kwargs):
        new_state_dict = {}
        state_dict = super().state_dict(*args, **kwargs)
        for layer_name in state_dict.keys():
            match = re.search(r"^teacher\.", layer_name)
            if not match:
                new_state_dict[layer_name] = state_dict[layer_name]
                continue
            # otherwise, layer should be pruned.
        return new_state_dict

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        **unused,
    ):
        new_state_dict = {}
        cur_state_dict = super().state_dict() # use super or current?
        for layer_name in cur_state_dict.keys():
            match = re.search(r"^teacher\.", layer_name)
            if not match:
                new_state_dict[layer_name] = state_dict[layer_name] # load other params normally
            else:
                new_state_dict[layer_name] = cur_state_dict[layer_name]
        return super().load_state_dict(new_state_dict, strict)


    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        # parser.add_argument('--freeze-teacher', action='store_true',
        #         help='whether to freeze teacher.')
        parser.add_argument("--teacher-dir", type=str, required=True, help="The pre-trained XLMR model directory.")
        parser.add_argument("--hint-from-layer", type=int, default=8, help="The layer index in teacher model from which the hint came.")

        parser.add_argument("--hint-loss-factor",
            default=1.,
            type=float,
            help="factor for hint-based loss."
        )

    def compute_hint_loss(self, src_tokens, tgt_tokens, feats, masks):
        # get hints
        with torch.no_grad():            
            self.teacher.eval()
            total_len = src_tokens.size(1) + tgt_tokens.size(1)
            if tgt_tokens.size(-1) > self.teacher.max_positions():
                raise ValueError(
                    "target tokens exceeds maximum length: {} > {}".format(
                        tgt_tokens.size(-1), self.teacher.max_positions()
                    )
                )
            elif total_len > self.teacher.max_positions():
                logger.warning("tokens exceeds maximum length: {} > {}. trimming source tokens.".format(
                        total_len, self.teacher.max_positions()
                    ))
                src_max_len = self.teacher.max_positions() - tgt_tokens.size(-1)
                src_tokens = src_tokens[:,:src_max_len]                
                teacher_tokens = torch.cat((src_tokens, tgt_tokens), dim=1)
            else:
                src_max_len = src_tokens.size(1)
                teacher_tokens = torch.cat((src_tokens, tgt_tokens), dim=1)

            hints = self.teacher(
                teacher_tokens, 
                features_only=True, 
                return_all_hiddens=True
            )[1]['inner_states'] # 13 hidden layers, (T,B,768)
            # hints = hints[-2] # get second to last layer
            hints = hints[self.hint_from_layer][src_max_len:,...] # get 8-th layer, only target reps

        # project to same dim
        hints = self.teacher_proj(hints).transpose(1,0).contiguous() # (B,T,h)
        
        # pdb.set_trace()
        feats = feats[masks]
        hints = hints[masks]
        loss = F.mse_loss(
            feats, 
            hints.detach(), reduction='mean')

        return loss

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        # word_ins_out = self.decoder(
        #     normalize=False,
        #     prev_output_tokens=prev_output_tokens,
        #     encoder_out=encoder_out,
        # )
        features, extras = self.decoder.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=self.decoder.src_embedding_copy,
        )
        word_ins_out = self.decoder.output_layer(features)

        # word_ins_mask = prev_output_tokens.type_as(tgt_tokens).eq(self.unk)
        word_ins_mask = tgt_tokens.ne(self.pad)

        # distillation
        hint_loss = self.compute_hint_loss(
            src_tokens=src_tokens, 
            tgt_tokens=tgt_tokens, 
            feats=features, 
            masks=word_ins_mask
        )
        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": word_ins_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            },
            # This will just addup your loss for you. i.e. factor is not applied. need to apply ourselves.
            "hint":{
                "loss": hint_loss*self.hint_loss_factor, 
                "factor": self.hint_loss_factor
            }
        }

@register_model_architecture("bert2nat", "bert2nat")
def bert2nat(args):
    # make compatible with roberta/xlmr
    args.max_target_positions = 510
    nonautoregressive_transformer_base_architecture(args)


@register_model_architecture(
    "bert2nat", "bert2nat_iwslt16"
)
def bert2nat_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    nonautoregressive_transformer_base_architecture(args)
