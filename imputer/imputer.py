# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn


from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding
)
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder
)
from fairseq.models.nat import (
    NATransformerModel,
    NATransformerDecoder,
    FairseqNATModel
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules.transformer_sentence_encoder import init_bert_params

@register_model("imputer")
class ImputerModel(NATransformerModel):
    """
    Implementing the Imputer model from 
    Imputer: Sequence Modelling via Imputation and Dynamic Programming
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""        
        FairseqNATModel.add_args(parser)
        # parser.add_argument('--encoder-learned-pos', action='store_true',
        #                     help='use learned positional embeddings in the encoder')
        parser.add_argument("--upsample-scale", type=int,
                            help="upsampling scale s to use for encoder features")
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = ImputerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = ImputerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    # def forward(
    #     self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    # ):
    #     raise NotImplementedError("TODO")
    #     # encoding
    #     encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
    #     # decoding
    #     word_ins_out = self.decoder(
    #         normalize=False,
    #         prev_output_tokens=prev_output_tokens,
    #         encoder_out=encoder_out)
    #     word_ins_mask = prev_output_tokens.eq(self.unk)

    #     return {
    #         "word_ins": {
    #             "out": word_ins_out, "tgt": tgt_tokens,
    #             "mask": word_ins_mask, "ls": self.args.label_smoothing,
    #             "nll_loss": True
    #         },
    #         "length": {
    #             "out": length_out, "tgt": length_tgt,
    #             "factor": self.decoder.length_loss_factor
    #         }
    #     }

    # def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
    #     raise NotImplementedError("TODO")
    #     step = decoder_out.step
    #     max_step = decoder_out.max_step

    #     output_tokens = decoder_out.output_tokens
    #     output_scores = decoder_out.output_scores
    #     history = decoder_out.history

    #     # execute the decoder
    #     output_masks = output_tokens.eq(self.unk)
    #     _scores, _tokens = self.decoder(
    #         normalize=True,
    #         prev_output_tokens=output_tokens,
    #         encoder_out=encoder_out,
    #     ).max(-1)
    #     output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
    #     output_scores.masked_scatter_(output_masks, _scores[output_masks])

    #     if history is not None:
    #         history.append(output_tokens.clone())

    #     # skeptical decoding (depend on the maximum decoding steps.)
    #     if (step + 1) < max_step:
    #         skeptical_mask = _skeptical_unmasking(
    #             output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
    #         )

    #         output_tokens.masked_fill_(skeptical_mask, self.unk)
    #         output_scores.masked_fill_(skeptical_mask, 0.0)

    #         if history is not None:
    #             history.append(output_tokens.clone())

    #     return decoder_out._replace(
    #         output_tokens=output_tokens,
    #         output_scores=output_scores,
    #         attn=None,
    #         history=history
    #     )

    

class ImputerEncoder(FairseqEncoder):
    """
    Imputer encoder is simply the encoder embeddings (optionally includes positional embeddings), 
    and a linear transform to upsample the output.
    
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.upsample_scale = args.upsample_scale
        self.upsampler = nn.Linear(embed_dim, self.upsample_scale*embed_dim)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(self, src_tokens, src_lengths, **unused):
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # upsampling B x T x C -> B x T x sC -> B x Ts x C
        # 
        B, T, C = x.size()
        x = self.upsampler(x).view(B, T*self.upsample_scale, C)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return EncoderOut(
            encoder_out=x,  # B x T x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


class ImputerDecoder(NATransformerDecoder):
    """
    ImputerDecoder is just a NAT Decoder, but slight adjustments
        1. No length prediction
        2. Adds encoder out to decoder input

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=True):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out,
        early_exit=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # encoder source embeddings
        x = encoder_out.encoder_out
        # for ctc, no prior alignment is given.
        if prev_output_tokens is None:
            decoder_padding_mask = None
        else:
            prior, decoder_padding_mask = self.forward_embedding(prev_output_tokens)
            # adds encoder features
            x = x + prior
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}


@register_model_architecture("imputer", "imputer")
def imputer_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.upsample_scale = getattr(args, "upsample_scale", 2)

@register_model_architecture(
    "imputer", "imputer_iwslt16"
)
def imputer_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 10)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    imputer_base_architecture(args)