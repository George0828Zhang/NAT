import re
import pdb
import json
import torch
from argparse import Namespace
from fairseq.models import (
    register_model, 
    register_model_architecture, 
    BaseFairseqModel,
    ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY
)
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
class MutualLearnNATransformerModel(BaseFairseqModel):
    def __init__(self, args, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_is_ar = not isinstance(teacher, NATransformerModel)
        self.freeze_teacher = getattr(args, "freeze_teacher", False)
        if self.freeze_teacher:
            logger.warning("Teacher weights will be freezed!")
            freeze_module_params(self.teacher)

        # for KD loss controlling
        self.student.kd_factor = args.student_kd_factor
        self.teacher.kd_factor = args.teacher_kd_factor
        use_control_kd_factor = getattr(
            args, "control_kd_factor", False)
        self.student.use_control_kd_factor = use_control_kd_factor
        self.teacher.use_control_kd_factor = use_control_kd_factor

        if use_control_kd_factor:
            control_args = Namespace(**json.loads(
                getattr(args, 'control_kd_args', '{}') or '{}'))
            from .ctrlvae import ctrlVAE
            self.student.controller = ctrlVAE(control_args)
            self.teacher.controller = ctrlVAE(control_args)
        
        # for inference time
        if getattr(args, "reduce_to_teacher", False):
            self.reduced_model = self.teacher
        elif getattr(args, "reduce_to_student", False):
            self.reduced_model = self.student
        else:
            self.reduced_model = None

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
        parser.add_argument('--student-arch', default="nonautoregressive_transformer",
                            help='determine the type of student network to mutual learn from.')        
        parser.add_argument('--teacher-arch', default="transformer",
                            help='determine the type of teacher network to mutual learn from.')

        parser.add_argument('--load-to-teacher', action='store_true',
                            help='load checkpoint to teacher network.')
        parser.add_argument('--freeze-teacher', action='store_true',
                help='whether to freeze teacher.')

        parser.add_argument("--student-kd-factor",
                            default=.5,
                            type=float,
                            help="weights on the knowledge distillation loss for training student"
                            )
        parser.add_argument("--teacher-kd-factor",
                            default=.5,
                            type=float,
                            help="weights on the knowledge distillation loss for training teacher"
                            )
        parser.add_argument("--control-kd-factor", action="store_true",
                            help="use the PI algorithm introduced in ControlVAE to calculate the weight on KL-divergence on latent.")
        parser.add_argument('--control-kd-args', type=str, metavar='JSON',
                            help="""args for ControlVAE, a valid setup is: '{"v_kl": 3.0, "Kp": 0.01, "Ki": 0.0001, "beta_min": 0.0, "beta_max": 1.0 }' """)


        # inference flags
        parser.add_argument('--reduce-to-student', action='store_true',
                            help='when inference, only load student network.')
        parser.add_argument('--reduce-to-teacher', action='store_true',
                            help='when inference, only load teacher network.')
                            

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        # assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_architecture(args)

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
            encoder_embed_tokens = TransformerModel.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = TransformerModel.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = TransformerModel.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )


        student_cls = ARCH_MODEL_REGISTRY[args.student_arch]
        encoder = student_cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = student_cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        student = student_cls(args,encoder,decoder)

        teacher_cls = ARCH_MODEL_REGISTRY[args.teacher_arch]
        if not issubclass(teacher_cls, NATransformerModel):
            teacher_cls = PatchedTransformerModel

        teacher_encoder = teacher_cls.build_encoder(
            args, src_dict,
            encoder_embed_tokens if args.share_encoder_embeddings else TransformerModel.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
                )
            )
        teacher_decoder = teacher_cls.build_decoder(
            args, tgt_dict,
            decoder_embed_tokens if args.share_decoder_embeddings else TransformerModel.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
                )
            )
        teacher = teacher_cls(args,teacher_encoder,teacher_decoder)

        return cls(args, student, teacher)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
    ##################################
    # some functions to avoid errors #
    ##################################
    # Both
    def max_positions(self):
        """Maximum length supported by the model."""
        return self.student.max_positions() # also needed in validation runs.

    def forward_encoder(self, *args, **kwargs):
        return self.reduced_model.forward_encoder(*args, **kwargs)    
    def forward_decoder(self, *args, **kwargs):
        return self.reduced_model.forward_decoder(*args, **kwargs)
    
    # NAT
    def initialize_output_tokens(self, *args, **kwargs):
        return self.reduced_model.initialize_output_tokens(*args, **kwargs)    
    def allow_length_beam(self, *args, **kwargs):
        return self.reduced_model.allow_length_beam(*args, **kwargs)
    def regenerate_length_beam(self, *args, **kwargs):
        return self.reduced_model.regenerate_length_beam(*args, **kwargs)    
    @property
    def encoder(self):
        return self.reduced_model.encoder

    # AR
    def max_decoder_positions(self, *args, **kwargs):
        return self.reduced_model.max_decoder_positions(*args, **kwargs)
    def reorder_encoder_out(self, *args, **kwargs):
        return self.reduced_model.reorder_encoder_out(*args, **kwargs)
    def reorder_incremental_state(self, *args, **kwargs):
        return self.reduced_model.reorder_incremental_state(*args, **kwargs)
    ##################################
    #  end functions to avoid errors #
    ##################################    

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        """Overrides fairseq_model.py

        """
        if getattr(args, "load_to_teacher", False):
            logger.warning("Will load checkpoint weights to teacher!")
            cur = self.state_dict()
            for k, v in state_dict.items():
                cur["teacher." + k] = v
            state_dict = cur

        return super().load_state_dict(state_dict, strict=strict, args=args)


class PatchedTransformerModel(TransformerModel):
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
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.student_arch = getattr(args, 'student_arch', "nonautoregressive_transformer")
    args.teacher_arch = getattr(args, 'teacher_arch', "transformer")

    ARCH_CONFIG_REGISTRY[args.student_arch](args)
    # ARCH_CONFIG_REGISTRY[args.teacher_arch](args)

    


# @register_model_architecture(
#     "mutual_learn_nat", "mutual_learn_nat_iwslt16"
# )
# def mutual_learn_nat_iwslt_16(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 278)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 507)
#     args.encoder_layers = getattr(args, "encoder_layers", 5)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 278)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 507)
#     args.decoder_layers = getattr(args, "decoder_layers", 5)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)

#     base_mutual_learn_nat_architecture(args)
    
