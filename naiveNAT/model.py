from fairseq.models.nat import NATransformerModel, NATransformerDecoder, base_architecture, FairseqNATDecoder, nonautoregressive_transformer_wmt_en_de
from fairseq.modules import TransformerDecoderLayer, PositionalEmbedding, MultiheadAttention
from fairseq.models import register_model, register_model_architecture
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch.nn.functional as F
@register_model("myNAT")
class Model(NATransformerModel):

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument("--decoder_positional_attention", action="store_true",
                            help="add postional attention when decoding")
        parser.add_argument("--decoder_positional_attention_head_num", type=int, 
            help="num of heads of positional attention in decoder layers")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = MyDecoder(args, tgt_dict, embed_tokens)
        # decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

class MyDecoder(NATransformerDecoder):

    def __init__(self, args, tgt_dict, embed_tokens):
        super().__init__(args, tgt_dict, embed_tokens)

    def build_pos_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            num_heads=args.decoder_positional_attention_head_num,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if getattr(self, "pos_attn", None) == None:
            self.pos_attn = self.build_pos_attention(
            self.embed_dim,
            args,
        )
        return MyDecoderLayer(args, no_encoder_attn=no_encoder_attn, embed_positions = self.embed_positions, pos_attn = self.pos_attn)

class MyDecoderLayer(TransformerDecoderLayer):

    def __init__(self, args, *posargs, embed_positions=None, pos_attn = None, **kwargs):

        super().__init__(args, posargs, kwargs)
        self.decoder_positional_attention = getattr(args, "decoder_positional_attention", False)
        self.embed_positions = embed_positions
        self.pos_attn = pos_attn
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ## add positional attention
        if self.decoder_positional_attention:
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
            positions = self.embed_positions(x[..., 0].squeeze(-1))
            residual = x
            x, attn = self.pos_attn(
                query=positions,
                key=positions,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = residual + x
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)



        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

@register_model_architecture(
    "myNAT", "myNAT"
)
def myNAT(args):
    nonautoregressive_transformer_wmt_en_de(args)
    args.decoder_positional_attention = getattr(args, "decoder_positional_attention", False)
    args.decoder_positional_attention_head_num = getattr(args, "decoder_positional_attention_head_num", 1)

@register_model_architecture(
    "myNAT", "gu"
)
def gu(args):
    nonautoregressive_transformer_wmt_en_de(args)
    args.decoder_positional_attention = getattr(args, "decoder_positional_attention", True)
    args.decoder_positional_attention_head_num = getattr(args, "decoder_positional_attention_head_num", 1)
