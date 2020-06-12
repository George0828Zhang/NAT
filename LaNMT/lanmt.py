from fairseq.models.nat import (
    NATransformerModel,
    NATransformerDecoder,
    ensemble_decoder, 
    nonautoregressive_transformer_wmt_en_de
)
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    base_architecture,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits
from torch import Tensor
from typing import Dict, List, NamedTuple, Optional
from argparse import Namespace

from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_sentence_encoder import init_bert_params

import re
import pdb
import json
import copy
import logging

logger = logging.getLogger(__name__)

Latent = NamedTuple(
    "Latent",
    [
        ("mu", Tensor),  
        ("logvar", Tensor),  
    ],
)

ControlVAERecord = NamedTuple(
    "ControlVAERecord",
    [
        ("v_kl", float),
        ("Kp", float),
        ("Ki", float),
        ("beta_min", float),
        ("beta_max", float),
        # below are records
        ("P", float), 
        ("I", float),
        ("beta_prev", float),
    ],
)

@register_model("lanmt")
class LaNMT(NATransformerModel):
    def __init__(self, args, encoder, decoder, prior, posterior):
        super().__init__(args, encoder, decoder)
        self.prior = prior
        self.posterior = posterior
        self.latent_projection = nn.Linear(
            args.latent_dim,
            args.decoder_embed_dim
        )
        self.length_transform_logvar = nn.Parameter(torch.zeros(1)) # initialize to var=1
        self.kl_div_loss_factor = args.kl_div_loss_factor
        self.max_update = args.max_update
        self.num_updates = 0
        self.control_vae = getattr(args, "control_vae", False)

        if self.control_vae:
            control_args = json.loads(getattr(args, 'control_vae_args', '{}') or '{}')
            self.controller = ControlVAERecord(P=0, I=0, beta_prev=0, **control_args)
        else:
            self.kl_budget = 1.

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)        
        parser.add_argument("--load-weight-level", default='all',
                            choices=['all', 'encoder_decoder', 'encoder'],
                            help="which components needs to load weights from checkpoint. all: load all. encoder_decoder: load encoder and decoder only. encoder: load encoder only.")
        parser.add_argument("--latent-dim", type=int,
                            help="dimension for latent vector.")
        parser.add_argument("--posterior-layers", type=int,
                            help="num layers for posterior transformer.")
        parser.add_argument("--kl-div-loss-factor", type=float,
                            help="weights on the kl divergence term in ELBO (or initial budget). ignored if using control-VAE")
        parser.add_argument("--control-vae", action="store_true",
                            help="use the PI algorithm introduced in ControlVAE to calculate the weight on KL-divergence on latent.")
        parser.add_argument('--control-vae-args', type=str, metavar='JSON',
                            help="""args for ControlVAE, a valid setup is: '{"v_kl": 3.0, "Kp": 0.01, "Ki": 0.0001, "beta_min": 0.0, "beta_max": 1.0 }' """)

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
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        """The same as models.transformer, but adds prior and posterior"""
        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
                
        ## because posterior requires vocab and embeddings, we need to build them here.
        prior = cls.build_prior(args)
        posterior = cls.build_posterior(args, task, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        return cls(args, encoder, decoder, prior, posterior)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LaNMTDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_prior(cls, args):
        prior = LatentPredictor(
            hidden_dim=args.encoder_embed_dim,
            latent_dim=args.latent_dim
        )
        if getattr(args, "apply_bert_init", False):
            prior.apply(init_bert_params)
        return prior

    @classmethod
    def build_posterior(cls, args, task, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens):
        posterior = Posterior.build_model(args,task, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        if getattr(args, "apply_bert_init", False):
            posterior.apply(init_bert_params)
        return posterior

    def sample_from(self, z: Latent):
        eps = torch.randn(z.mu.size()).type_as(z.mu)
        std = torch.exp(z.logvar / 2.0)
        return z.mu + eps * std

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def calculate_kl_div(self, z0: Latent, z1: Latent):
        var0 = torch.exp(z0.logvar)
        var1 = torch.exp(z1.logvar)
        kl = 0.5*((z1.logvar - z0.logvar) + (var0 + (z0.mu-z1.mu)**2)/var1 - 1).sum(-1) # sum over latent dim.

        # we use mean over tokens in batch, same as in nat_loss.
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if self.control_vae:
            kl = mean_ds(kl) # tensor, has grads
            e_t = self.controller.v_kl - kl.item() # float
            P_t = self.controller.Kp * torch.tensor(-e_t).sigmoid() # cpu tensor
            if self.controller.beta_min <= self.controller.beta_prev <= self.controller.beta_max:
                I_t = self.controller.I - self.controller.Ki*e_t # float
            else:
                I_t = self.controller.I # float
            beta_t = (P_t + I_t + self.controller.beta_min).clamp(
                min=self.controller.beta_min, 
                max=self.controller.beta_max
            ).item() # float
            # update loss factor (beta)
            self.kl_div_loss_factor = beta_t
            # update controller
            self.controller._replace(P=P_t.item(), I=I_t, beta_prev=beta_t)
        else:
            self.kl_budget = max(min(2.*(1-self.num_updates/self.max_update), 1.), 0.)
            kl = torch.max(kl.new([self.kl_budget]), kl)
            kl = mean_ds(kl)
        return kl

    def length_transform(self, src_features, prev_output_tokens, src_masks=None):
        """
        src_features: (B, S, Z)
        prev_output_tokens: (B, T)
        mapping: (T, S)
        src_masks: (B, S) src_tokens != pad
        out: (B, T, Z)
        """        
        S = src_features.size(1)
        T = prev_output_tokens.size(1)
        
        js = torch.arange(T).view(T,1).type_as(src_features)
        ks = torch.arange(S).view(1,S).type_as(src_features)
        var = torch.exp(self.length_transform_logvar)
        mapping = -1/(2*var)*(ks - S*js/T)**2 # (T, S)

        if src_masks is not None:
            mapping = mapping.unsqueeze(0) # (1, T, S)
            src_masks = src_masks.unsqueeze(1) # (B, 1, S)
            mapping = mapping * src_masks.type_as(mapping) # (B, T, S)

        mapping = torch.softmax(mapping, dim=-1)
        return torch.matmul(mapping, src_features)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        
        # posterior & prior
        posterior_out = self.posterior(src_tokens, src_lengths, prev_output_tokens)
        prior_out = self.prior(encoder_out.encoder_out.transpose(1,0)) # T x B x Z -> B x T x Z
        
        # latent
        kl_div_loss = self.calculate_kl_div(
            z0=posterior_out, 
            z1=prior_out
        )
        latent = self.sample_from(posterior_out)
        latent = self.length_transform(latent, prev_output_tokens, src_masks=src_tokens.ne(self.pad))
        latent = self.latent_projection(latent)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens, # prev_output_tokens, (B, T) # only used for length.
            prev_output_embeds=latent, # need to be same as pos emb (B, Ty, dim)
            encoder_out=encoder_out)

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
            # This will just addup your loss for you.
            "kl_div":{
                "loss": kl_div_loss,
                "factor": self.kl_div_loss_factor
            }
        }

    def forward_encoder(self, encoder_inputs):
        """ 
        input [src_tokens, src_lengths] 
        originally, these 2 are None. Useful for posterior if using iterative refinment (latent search).
        """
        encoder_out = self.encoder(*encoder_inputs)
        return encoder_out._replace(
            src_tokens=encoder_inputs[0],
            src_lengths=encoder_inputs[1]
        )

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # posterior & prior
        if step == 0:
            posterior_out = self.prior(encoder_out.encoder_out.transpose(1,0)) # T x B x Z -> B x T x Z
        else:
            src_tokens = encoder_out.src_tokens
            src_lengths = encoder_out.src_lengths
            posterior_out = self.posterior(src_tokens, src_lengths, output_tokens)
        
        # latent        
        latent = posterior_out.mu # delta posterior: don't sample, just use mean.
        latent = self.length_transform(latent, output_tokens)
        latent = self.latent_projection(latent)

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            prev_output_embeds=latent, # need to be same as pos emb (B, Ty, dim)
            encoder_out=encoder_out,
            step=step,
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )

class LatentPredictor(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self._linear = nn.Linear(hidden_dim, latent_dim*2)

    def forward(self, hiddens):
        out = self._linear(hiddens)
        mu, logvar = torch.chunk(out, 2, dim=-1) # mu logvar
        return Latent(mu=mu, logvar=logvar)

class Posterior(NATransformerModel):
    @classmethod
    def build_model(cls, args, task, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens):
        posterior_args = copy.deepcopy(args)
        posterior_args.encoder_layers = args.posterior_layers
        posterior_args.decoder_layers = args.posterior_layers
        # posterior_args.share_decoder_input_output_embed = False
        # posterior_args.share_all_embeddings = False
        # posterior_args.src_embedding_copy = False

        # TODO: swap encoder & decoder values
        # assumed same args for now.
        
        transfo = cls(
            args=posterior_args, 
            # in posterior, encoder eats target
            encoder=NATransformerModel.build_encoder(
                posterior_args, 
                tgt_dict, 
                decoder_embed_tokens
            ), 
            # in posterior, decoder eats source
            decoder=NATransformerModel.build_decoder(
                posterior_args, 
                src_dict, 
                encoder_embed_tokens
            )
        )
        transfo.predict_head = LatentPredictor(
            hidden_dim=args.decoder_embed_dim,
            latent_dim=args.latent_dim
        )
        return transfo

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, **kwargs
    ):
        tgt_lengths = prev_output_tokens.ne(self.pad).sum(1)
        # encoding
        encoder_out = self.encoder(prev_output_tokens, src_lengths=tgt_lengths, **kwargs)
        # decoding
        features, _ = self.decoder.extract_features(
            prev_output_tokens=src_tokens,
            encoder_out=encoder_out,
            embedding_copy=False,
        )
        return self.predict_head(features)


class LaNMTDecoder(NATransformerDecoder):
    r"""
    below is just the same as nat but with 2 difference:
    1. option to pass embeddings instead of tokens.
    2. length prediction from latent z
    """
    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, prev_output_embeds=None, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens=prev_output_tokens,
            prev_output_embeds=prev_output_embeds, # modification here
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        """ from mean pooling to just bos """
        enc_feats = encoder_out.encoder_out  # T x B x C
        enc_feats = enc_feats[0, ...]
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

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
    "lanmt", "lanmt"
)
def lea_nat(args):
    nonautoregressive_transformer_wmt_en_de(args)
    args.latent_dim = getattr(args, "latent_dim", 8)
    args.posterior_layers = getattr(args, "posterior_layers", 3)
    args.kl_div_loss_factor = getattr(args, "kl_div_loss_factor", 1.0)