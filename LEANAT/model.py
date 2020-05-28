from fairseq.models.nat import (
    NATransformerModel,
    NATransformerDecoder,
    ensemble_decoder, 
    nonautoregressive_transformer_wmt_en_de
)
from fairseq.models import register_model, register_model_architecture

import torch
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits
from torch import Tensor
from typing import Dict, List, NamedTuple, Optional

from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_sentence_encoder import init_bert_params

import re
import pdb
import logging

logger = logging.getLogger(__name__)

PriorOut = NamedTuple(
    "PriorOut",
    [
        ("out", Tensor),  # (B, Ty, dim)
        ("attn", Tensor),  # (B, Ty, Tx)
    ],
)

@register_model("lea_nat")
class LeaNAT(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)        
        parser.add_argument("--load-encoder-only", action="store_true", #type=bool, nargs='?', const=True, default=False,
        help="whether only load encoder states from checkpoint.")        
        parser.add_argument("--lea-loss-factor", type=float,
                            help="weights on the latent embedding attention loss.")
        parser.add_argument("--lea-attention-heads", type=int,
                            help="weights on the latent embedding attention loss.")
        parser.add_argument("--sg-lea-pred", action="store_true",
                            help="stop the gradients back-propagated from the latent embedding aligner predictor")
        parser.add_argument("--lea-use-embed", action="store_true",
                            help="Use encoder embeddings instead of encoder out as input to LEA module.")
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
        # if getattr(args, "load_encoder_only", False)--gen-subset
        if getattr(args, "load_encoder_only", False):
            logger.warning("Will only load encoder weights!")
            cur = self.state_dict()
            for param in state_dict:
                if re.match(r"^encoder\.", param) is not None:
                    cur[param] = state_dict[param]
            state_dict = cur

        return super().load_state_dict(state_dict, strict=strict, args=args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LeaDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # latent embedding alignment        
        prior_out = self.decoder.forward_prior_attn(
            encoder_out=encoder_out, 
            prev_output_tokens=prev_output_tokens
        )
        if tgt_tokens is not None:
            posterior_out = self.decoder.forward_posterior_attn(
                prior_out=prior_out, 
                encoder_out=encoder_out, 
                prev_output_tokens=prev_output_tokens
            )
        else:
            posterior_out = prior_out
        # lea loss
        # lea_loss = self.decoder.compute_lea_loss(
        #     prior=prior_out.attn, 
        #     posterior=posterior_out.attn,
        #     mask=tgt_tokens.ne(self.pad)
        # )

        # when training, noise out prev_output_tokens / prev_output_embeds just like in cmlm


        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens, # prev_output_tokens, (B, T)
            prev_output_embeds=posterior_out.out, # need to be same as pos emb (B, Ty, dim)
            encoder_out=encoder_out)

        if getattr(self.args, "predict_masked_only", False):
            word_ins_mask = prev_output_tokens.eq(self.unk)
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
            },
            # this will leave nat_loss to compute kl_div for you. 
            "lea":{
                "out": probs_to_logits(prior_out.attn), "tgt": posterior_out.attn.detach(), # cannot let gradient go through your target!
                "mask": tgt_tokens.ne(self.pad),
                "factor": self.decoder.lea_loss_factor
            }
            # This will just addup your loss for you.
            # "lea":{
            #     "loss": lea_loss,
            #     "factor": self.decoder.lea_loss_factor
            # }
        }

class LeaDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.lea_loss_factor = getattr(args, "length_loss_factor", 0.1)
        lea_attention_heads = getattr(args, "lea_attention_heads", args.decoder_attention_heads)
        self.prior_attn = MultiheadAttention(
            args.decoder_embed_dim,
            lea_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=getattr(args, "quant_noise_pq", 0),
            qn_block_size=getattr(args, "quant_noise_pq_block_size", 8),
        )
        self.posterior_attn = MultiheadAttention(
            args.decoder_embed_dim,
            lea_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=getattr(args, "quant_noise_pq", 0),
            qn_block_size=getattr(args, "quant_noise_pq_block_size", 8),
        )
        self.sg_lea_prediction = getattr(args, "sg_lea_prediction", False)
        self.lea_use_embed = getattr(args, "lea_use_embed", False)

    @ensemble_decoder
    def forward_prior_attn(self, encoder_out, prev_output_tokens, **unused):
        enc_feats = encoder_out.encoder_embedding if self.lea_use_embed else encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.sg_lea_prediction:
            enc_feats = enc_feats.detach()
        # (B, Ty, dim)
        positions = self.embed_positions(prev_output_tokens.transpose(1,0))
        x, attn = self.prior_attn(
            query=positions, 
            key=enc_feats, # (Tx, B, dim)
            value=enc_feats, 
            key_padding_mask=src_masks,
            incremental_state=None,
            static_kv=True,
            need_weights=True
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        return PriorOut(
            out=x.transpose(1,0), # (B, Ty, dim)
            attn=attn # (B, Ty, Tx)
        )

    @ensemble_decoder
    def forward_posterior_attn(self, encoder_out, prev_output_tokens, **unused):        
        enc_feats = encoder_out.encoder_embedding if self.lea_use_embed else encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.sg_lea_prediction:
            enc_feats = enc_feats.detach()
        # (B, Ty, dim)
        x, _ = self.forward_embedding(prev_output_tokens.transpose(1,0)) # with positional encoding
        x, posterior_attn = self.prior_attn(
            query=x, 
            key=enc_feats, # (Tx, B, dim)
            value=enc_feats,
            key_padding_mask=src_masks,
            incremental_state=None,
            static_kv=True,
            need_weights=True
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        posterior_out = PriorOut(
            out=x.transpose(1,0), # (B, Ty, dim)
            attn=posterior_attn # (B, Ty, Tx)
        )
        return posterior_out

    # @ensemble_decoder
    # def forward_posterior_attn(self, prior_out, encoder_out, tgt_tokens=None, **unused):
    #     if tgt_tokens is not None:
    #         enc_feats = encoder_out.encoder_embedding if self.lea_use_embed else encoder_out.encoder_out  # T x B x C
    #         src_masks = encoder_out.encoder_padding_mask  # B x T or None
    #         if self.sg_lea_prediction:
    #             enc_feats = enc_feats.detach()
    #         # (B, Ty, dim)
    #         x, _ = self.forward_embedding(tgt_tokens.transpose(1,0)) # with positional encoding
    #         x, posterior_attn = self.prior_attn(
    #             query=x, 
    #             key=enc_feats, # (Tx, B, dim)
    #             value=enc_feats,
    #             key_padding_mask=src_masks,
    #             incremental_state=None,
    #             static_kv=True,
    #             need_weights=True
    #         )
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #         posterior_out = PriorOut(
    #             out=x.transpose(1,0), # (B, Ty, dim)
    #             attn=posterior_attn # (B, Ty, Tx)
    #         )
    #     else:
    #         posterior_out = prior_out
    #     return posterior_out

    # def compute_lea_loss(self, prior, posterior, mask=None, factor=1.0):
    #     """ In case want to use L1 or L2 loss.
    #         outputs: batch x len x d_model
    #         targets: batch x len
    #         masks:   batch x len

    #         policy_logprob: if there is some policy
    #             depends on the likelihood score as rewards.
    #     """
    #             
    #     def mean_ds(x: Tensor, dim=None) -> Tensor:
    #         return (
    #             x.float().mean().type_as(x)
    #             if dim is None
    #             else x.float().mean(dim).type_as(x)
    #         )
    #     posterior = posterior.detach() # no loss through posterior

    #     if masks is not None:
    #         prior, posterior = prior[masks], posterior[masks]

    #     if masks is not None and not masks.any():
    #         loss = torch.tensor(0)
    #     else:
    #         logits = F.log_softmax(prior, dim=-1)
    #         losses = F.kl_div(logits, posterior.to(logits.device), reduction='none')
    #         losses = losses.sum(-1) # sum the Tx dim, which is features.
    #         loss = mean_ds(losses)
    #     return loss = loss * factor

    r"""
    below is just the same as nat but with option to pass embeddings instead of tokens.
    """
    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, prev_output_embeds=None, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens=prev_output_tokens,
            prev_output_embeds=prev_output_embeds,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    def extract_features(
        self,
        prev_output_tokens,
        prev_output_embeds=None,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
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
        # embedding
        # if embedding_copy:
        #     src_embd = encoder_out.encoder_embedding
        #     src_mask = encoder_out.encoder_padding_mask
        #     src_mask = (
        #         ~src_mask
        #         if src_mask is not None
        #         else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
        #     )

        #     x, decoder_padding_mask = self.forward_embedding(
        #         prev_output_tokens,
        #         self.forward_copying_source(
        #             src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
        #         ),
        #     )

        # else:

        #     x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # this function will prioritize embeds over tokens.
        x, decoder_padding_mask = self.forward_embedding( 
            prev_output_tokens,
            prev_output_embeds,
        )

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

@register_model_architecture(
    "lea_nat", "lea_nat"
)
def lea_nat(args):
    nonautoregressive_transformer_wmt_en_de(args)

@register_model_architecture(
    "lea_nat", "lea_iwslt_14"
)
def for_iwslt_14(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    nonautoregressive_transformer_wmt_en_de(args)

@register_model_architecture(
    "lea_nat", "lea_iwslt_16"
)
def for_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)

    nonautoregressive_transformer_wmt_en_de(args)
