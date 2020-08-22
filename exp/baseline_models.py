from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture as transformer_base_architecture
from fairseq.models.nat.nonautoregressive_transformer import base_architecture as nonautoregressive_transformer_base_architecture

@register_model_architecture(
    "nonautoregressive_transformer", "na_transformer_iwslt16"
)
def nonautoregressive_transformer_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)

    nonautoregressive_transformer_base_architecture(args)

@register_model_architecture(
    "transformer", "transformer_iwslt16"
)
def transformer_iwslt_16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)

    transformer_base_architecture(args)