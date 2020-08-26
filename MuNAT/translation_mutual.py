# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import copy
import torch
from argparse import Namespace
from fairseq.data import (
    encoders,
    LanguagePairDataset,
)
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset, EVAL_BLEU_ORDER
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq import utils, metrics
import logging
logger = logging.getLogger(__name__)

@register_task('translation_mutual')
class TranslationMutualLearningTask(TranslationLevenshteinTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""        
        TranslationLevenshteinTask.add_args(parser)    
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

    def __init__(self, args, src_dict, tgt_dict):        
        super().__init__(args, src_dict, tgt_dict)
        self.student_kd_factor = getattr(args, 'student_kd_factor', 0.5)
        self.teacher_kd_factor = getattr(args, 'teacher_kd_factor', 0.5)

    def build_generator(self, models, args, autoregressive=False):
        # add 'models' input to match the API for SequenceGenerator
        # add 'autoregressive' to have ar generator to evaluate ar student
        if autoregressive:
            ar_args = copy.deepcopy(args)
            ar_args.beam = 1
            ar_args.max_len_a = 1.2
            ar_args.max_len_b = 10
            return TranslationTask.build_generator(self, models, ar_args)

        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

    def build_model(self, args):
        """
        Add another generator for autoregressive teacher.
        """
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model.student], Namespace(**gen_args))
            ### Added ar seq gen for bleu2 evaluation
            self.ar_sequence_generator = self.build_generator(
                [model.teacher], Namespace(**gen_args), autoregressive=True
                ) if model.teacher_is_ar else self.sequence_generator
        return model

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        sample['prev_target'] = self.inject_noise(sample['target'])

        student, teacher = model.student, model.teacher

        """ forward model """
        student.train()
        teacher.eval()
        loss, sample_size, logging_output = criterion(student, teacher, sample, kd_factor=self.student_kd_factor)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        
        """ forward teacher """
        if getattr(model, "teacher_objective", "kd"):
            teacher.train()
            student.eval()
            teacher_loss, _, teacher_logging_output = criterion(teacher, student, sample, kd_factor=self.teacher_kd_factor)
            if ignore_grad:
                teacher_loss *= 0
            optimizer.backward(teacher_loss)

            for k, v in teacher_logging_output.items():
                if k == "loss":
                    logging_output["loss"] += v
                elif "-loss" in k:
                    logging_output["teacher-"+k] = v

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(model.student, model.teacher, sample, kd_factor=self.student_kd_factor)
            _, _, teacher_logging_output = criterion(model.teacher, model.student, sample, kd_factor=self.teacher_kd_factor)
            for k, v in teacher_logging_output.items():
                if k == "loss":
                    logging_output["loss"] += v
                elif "-loss" in k:
                    logging_output["teacher-"+k] = v

            if self.args.eval_bleu:
                """ original bleu for nat model. """

                bleu = self._inference_with_bleu(self.sequence_generator, sample, model.student, "nat example")
                logging_output['_bleu_sys_len'] = bleu.sys_len
                logging_output['_bleu_ref_len'] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                    logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]

                """ added bleu2 for teacher model. """

                bleu2 = self._inference_with_bleu(self.ar_sequence_generator, sample, model.teacher, "ar example")
                logging_output['_bleu2_sys_len'] = bleu2.sys_len
                logging_output['_bleu2_ref_len'] = bleu2.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu2.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu2_counts_' + str(i)] = bleu2.counts[i]
                    logging_output['_bleu2_totals_' + str(i)] = bleu2.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model, print_header=None):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
                # extra_symbols_to_ignore=[
                #     self.tgt_dict.index('[{}]'.format(self.args.source_lang)),
                #     self.tgt_dict.index('[{}]'.format(self.args.target_lang))
                # ],
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s if s else "UNKNOWNTOKENINREF"

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            if print_header is not None:
                logger.info(print_header)
            logger.info('hypothesis: ' + hyps[0])
            logger.info('reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            """ original bleu for nat model. """

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)


            """ added bleu2 for teacher model. """

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu2_counts_' + str(i)))
                totals.append(sum_logs('_bleu2_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu2_counts', np.array(counts))
                metrics.log_scalar('_bleu2_totals', np.array(totals))
                metrics.log_scalar('_bleu2_sys_len', sum_logs('_bleu2_sys_len'))
                metrics.log_scalar('_bleu2_ref_len', sum_logs('_bleu2_ref_len'))

                def compute_bleu2(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu2 = sacrebleu.compute_bleu(
                        correct=meters['_bleu2_counts'].sum,
                        total=meters['_bleu2_totals'].sum,
                        sys_len=meters['_bleu2_sys_len'].sum,
                        ref_len=meters['_bleu2_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu2.score, 2)

                metrics.log_derived('bleu2', compute_bleu2)