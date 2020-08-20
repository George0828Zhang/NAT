# from collections import OrderedDict

# from fairseq import utils
# from fairseq.models import (
#     register_model, register_model_architecture, ARCH_MODEL_REGISTRY,
#     FairseqDecoder, FairseqEncoder, BaseFairseqModel
# )
# from fairseq.models.transformer import (
#     base_architecture,
#     Embedding,
#     DEFAULT_MAX_SOURCE_POSITIONS,
#     DEFAULT_MAX_TARGET_POSITIONS
# )
# from fairseq.models.nat import CMLMNATransformerModel, cmlm_base_architecture

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.utils import probs_to_logits

import pdb
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import CMLMNATransformerModel, cmlm_base_architecture
from fairseq.models.transformer import TransformerModel, base_architecture

@register_model('mutual_cmlm')
class CMLMMutualLearnModel(CMLMNATransformerModel):
    def __init__(self, args, encoder, decoder, peer):
        super().__init__(args, encoder, decoder)
        self.peer = peer
        self.peer_type = args.peer_type
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        CMLMNATransformerModel.add_args(parser) # assumed to be cmlm
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--peer-type', default="ar", choices=["cmlm", "ar"],
                            help='determine the type of peer network to mutual learn from.')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        # assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_architecture(args)
        base_mutual_cmlm_architecture(args) # assumed to be cmlm

        if args.share_encoders:
            args.share_encoder_embeddings = True
                
        ### cmlm model
        # build shared embeddings (if applicable)
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

        ### peer model
        if args.peer_type == "cmlm":
            peer_cls = CMLMNATransformerModel
        else:
            peer_cls = PeerTransformerModel
        
        peer_encoder = peer_cls.build_encoder(
            args, src_dict, 
            encoder_embed_tokens if args.share_encoder_embeddings else cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
                )
            )
        peer_decoder = peer_cls.build_decoder(
            args, tgt_dict, 
            decoder_embed_tokens if args.share_decoder_embeddings else cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
                )
            )
        peer = peer_cls(args,peer_encoder,peer_decoder)

        return cls(args, encoder, decoder, peer)


class PeerTransformerModel(TransformerModel):
    """
    This makes criterion a whole lot easier.
    """

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        word_ins_out, _ = super().forward(src_tokens, src_lengths, prev_output_tokens)
        word_ins_mask = tgt_tokens.ne(self.decoder.dictionary.pad())
        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": word_ins_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
        }


@register_model_architecture('mutual_cmlm', 'mutual_cmlm')
def base_mutual_cmlm_architecture(args):
    cmlm_base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.peer_type = getattr(args, 'peer_type', "ar")
