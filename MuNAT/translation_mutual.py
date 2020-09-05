# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import Namespace
import torch

from fairseq.data import (
    encoders,
    LanguagePairDataset,
)
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset, EVAL_BLEU_ORDER
from fairseq import utils, metrics

import numpy as np
logger = logging.getLogger(__name__)

@register_task('translation_mutual')
class TranslationMutualLearningTask(TranslationTask):
    """
    Translation (Sequence Generation) task for
    """    
    def __init__(self, args, src_dict, tgt_dict):        
        super().__init__(args, src_dict, tgt_dict)
        self.student_kd_factor = args.student_kd_factor
        self.teacher_kd_factor = args.teacher_kd_factor

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask']) 

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

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split. ***Updated with original translation. Removes prepend_bos.***

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
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

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        """ *Removes append_bos* """
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError("Constrained decoding with the translation_lev task is not supported")        
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary,
                                   tgt_dict=self.target_dictionary, append_bos=True)

    def build_model(self, args):
        """
        Add another generator for autoregressive teacher.
        """
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            ### Added ar seq gen for bleu2 evaluation
            if model.teacher_is_ar:
                task_cls = TranslationTask
                gen_args = Namespace(beam=1, max_len_a=1.2, max_len_b=10, **vars(args))
            else:
                task_cls = TranslationMutualLearningTask
                gen_args = Namespace(iter_decode_max_iter=0, iter_decode_with_beam=1, **vars(args))
            self.teacher_sequence_generator = task_cls.build_generator(
                self,
                [model.teacher],
                gen_args
            )
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
        if not getattr(model, "freeze_teacher", False):
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
        student, teacher = model.student, model.teacher
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(student, teacher, sample, kd_factor=self.student_kd_factor)
            _, _, teacher_logging_output = criterion(teacher, student, sample, kd_factor=self.teacher_kd_factor)
            for k, v in teacher_logging_output.items():
                if k == "loss":
                    logging_output["loss"] += v
                elif "-loss" in k:
                    logging_output["teacher-"+k] = v

            if self.args.eval_bleu:
                """ original bleu for nat model. """

                bleu = self._inference_with_bleu(self.sequence_generator, sample, student)
                logging_output['_bleu_sys_len'] = bleu.sys_len
                logging_output['_bleu_ref_len'] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                    logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]

                """ added bleu2 for teacher model. """

                bleu2 = self._inference_with_bleu(self.teacher_sequence_generator, sample, teacher, eval_teacher=True)
                logging_output['_bleu2_sys_len'] = bleu2.sys_len
                logging_output['_bleu2_ref_len'] = bleu2.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu2.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu2_counts_' + str(i)] = bleu2.counts[i]
                    logging_output['_bleu2_totals_' + str(i)] = bleu2.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model, eval_teacher=False):
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
            logger.info("teacher example" if eval_teacher else "student example")
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