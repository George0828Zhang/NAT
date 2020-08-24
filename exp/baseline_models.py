from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    base_architecture as transformer_base_architecture,
)
from fairseq.models.nat.nonautoregressive_transformer import (
    NATransformerModel,
    base_architecture as nonautoregressive_transformer_base_architecture,
)
import re
import pdb
import logging

logger = logging.getLogger(__name__)

@register_model("na_transformer")
class BaselineNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)        
        parser.add_argument("--load-encoder-only", action="store_true", #type=bool, nargs='?', const=True, default=False,
        help="whether only load encoder states from checkpoint.")

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


def for_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)

@register_model_architecture(
    "na_transformer", "na_transformer_iwslt16"
)
def nonautoregressive_transformer_iwslt_16(args):
    for_iwslt_16(args)
    nonautoregressive_transformer_base_architecture(args)

@register_model_architecture(
    "transformer", "transformer_iwslt16"
)
def transformer_iwslt_16(args):
    for_iwslt_16(args)
    transformer_base_architecture(args)