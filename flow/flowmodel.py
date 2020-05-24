from fairseq.models import register_model, register_model_architecture
# from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
# from torch import Tensor
# from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch.nn.functional as F

from flowseq.flownmt.flows import NMTFlow

@register_model("flow")
class FlowModel(FairseqNATModel):
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )
        parser.add_argument("--flow-scales", type=int, help="")
        parser.add_argument("--flow-num-steps", type=str, help="steps for each scale, separated by ','.")
        parser.add_argument("--decoder-positional-attention", action="store_true", help="steps for each scale, separated by ','.")


    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        args.flow_scales = getattr(args, "flow_scales", 1)
        args.flow_num_steps = getattr(args, "flow_num_steps", 2)
        args.decoder_positional_attention = getattr(args, "decoder_positional_attention", False)
        decoder = MyDecoder(args, tgt_dict, embed_tokens)
        return decoder

    def forward_decoder(self, *args, **kwargs):
        return NotImplementedError

    def initialize_output_tokens(self, *args, **kwargs):
        return NotImplementedError

    def forward(self, *args, **kwargs):
        return NotImplementedError

class MyDecoder(nn.Module):
    def __init__(self, args, tgt_dict, embed_tokens):
        # TODO: inverse?

        super().__init__()
        self.dictionary = tgt_dict
        self.bos = tgt_dict.bos()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()

        self.dropout = args.dropout

        input_embed_dim = embed_tokens.embedding_dim
        self.embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.flow = NMTFlow(
            levels=args.flow_scales, # scales = 3
            num_steps=args.flow_num_steps, # [48, 48, 16] bottm to top should be parsed in build decoder.
            features=self.embed_dim,
            src_features=input_embed_dim, # encoder dim
            factors=2,
            hidden_features=None, # hidden dim default 2xfeatures
            inverse=False, # inverse=False
            transform='affine', 
            coupling_type='self_attn', 
            heads=args.decoder_attention_heads, # transformer decoder layer (coupling) heads 
            pos_enc='attn' if args.decoder_positional_attention else 'add', 
            max_length=self.max_target_positions,  # ?
            dropout=self.dropout)
        