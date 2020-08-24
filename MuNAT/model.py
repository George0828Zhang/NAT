import pdb
import torch
from fairseq.models import register_model, register_model_architecture
# from fairseq.models.nat import CMLMNATransformerModel, cmlm_base_architecture
from fairseq.models.nat import (
    NATransformerModel,
    base_architecture as nat_base_architecture,
)
from fairseq.models.transformer import TransformerModel, base_architecture
import logging

logger = logging.getLogger(__name__)


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False

@register_model('mutual_learn_nat')
class MutualLearnNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder, peer):
        super().__init__(args, encoder, decoder)
        self.peer = peer
        self.peer_type = args.peer_type
        self.register_buffer("num_updates", torch.zeros((1,), dtype=torch.int))
        if getattr(args, 'freeze_peer', False):
            self.freeze_peer = True
            freeze_module_params(self.peer)

    def set_num_updates(self, num_updates):
        self.num_updates.fill_(num_updates)

    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        NATransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--peer-type', default="ar", choices=["nat", "ar"],
                            help='determine the type of peer network to mutual learn from.')
        parser.add_argument('--load-peer-only', action='store_true',
                            help='only load peer network.')
        parser.add_argument('--freeze-peer', action='store_true',
                help='freeze peer(teacher) network. (baseline knowledge distillation)')
                            

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        # assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_architecture(args)
        base_mutual_learn_nat_architecture(args) # assumed to be nat

        if args.share_encoders:
            args.share_encoder_embeddings = True
                
        ### nat model
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
        if args.peer_type == "nat":
            peer_cls = NATransformerModel
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

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        
        """Overrides fairseq_model.py

        """
        if getattr(args, "load_peer_only", False):
            logger.warning("Will only load peer weights!")
            cur = self.state_dict()
            for k, v in state_dict.items():
                cur["peer." + k] = v
            state_dict = cur

        return super().load_state_dict(state_dict, strict=strict, args=args)


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


@register_model_architecture('mutual_learn_nat', 'mutual_learn_nat')
def base_mutual_learn_nat_architecture(args):
    nat_base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.peer_type = getattr(args, 'peer_type', "ar")


@register_model_architecture(
    "mutual_learn_nat", "mutual_learn_nat_iwslt16"
)
def mutual_learn_nat_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    
    base_mutual_learn_nat_architecture(args)

# @register_model_architecture('mutual_learn_nat', 'mutual_learn_nat_iwslt14')
# def iwslt14_mutual_learn_nat_architecture(args):
    
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
#     args.encoder_layers = getattr(args, "encoder_layers", 5)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
#     args.decoder_layers = getattr(args, "decoder_layers", 5)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

#     base_mutual_learn_nat_architecture(args)