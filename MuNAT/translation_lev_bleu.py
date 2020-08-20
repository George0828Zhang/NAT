# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from fairseq.data import LanguagePairDataset
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset, EVAL_BLEU_ORDER
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq import utils
import logging
logger = logging.getLogger(__name__)

@register_task('translation_lev_bleu')
class TranslationMutualLearningTask(TranslationLevenshteinTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    # @staticmethod
    # def add_args(parser):
    #     """Add task-specific arguments to the parser."""
    #     # fmt: off
    #     TranslationTask.add_args(parser)
    #     parser.add_argument(
    #         '--noise',
    #         default='random_delete',
    #         choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])        

    # def __init__(self, args, src_dict, tgt_dict):        
    #     super().__init__(args, src_dict, tgt_dict)

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        model.set_num_updates(update_num) # useful if model needs step number.
        sample['prev_target'] = self.inject_noise(sample['target'])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(model, sample)

            if self.args.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output['_bleu_sys_len'] = bleu.sys_len
                logging_output['_bleu_ref_len'] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                    logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
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
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    ###################################
    # inherited from translation task #
    ###################################

    # def valid_step(self, sample, model, criterion):
    #     loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
    #     if self.args.eval_bleu:
    #         bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
    #         logging_output['_bleu_sys_len'] = bleu.sys_len
    #         logging_output['_bleu_ref_len'] = bleu.ref_len
    #         # we split counts into separate entries so that they can be
    #         # summed efficiently across workers using fast-stat-sync
    #         assert len(bleu.counts) == EVAL_BLEU_ORDER
    #         for i in range(EVAL_BLEU_ORDER):
    #             logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
    #             logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
    #     return loss, sample_size, logging_output

    # def reduce_metrics(self, logging_outputs, criterion):
    #     super().reduce_metrics(logging_outputs, criterion)
    #     if self.args.eval_bleu:

    #         def sum_logs(key):
    #             return sum(log.get(key, 0) for log in logging_outputs)

    #         counts, totals = [], []
    #         for i in range(EVAL_BLEU_ORDER):
    #             counts.append(sum_logs('_bleu_counts_' + str(i)))
    #             totals.append(sum_logs('_bleu_totals_' + str(i)))

    #         if max(totals) > 0:
    #             # log counts as numpy arrays -- log_scalar will sum them correctly
    #             metrics.log_scalar('_bleu_counts', np.array(counts))
    #             metrics.log_scalar('_bleu_totals', np.array(totals))
    #             metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
    #             metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

    #             def compute_bleu(meters):
    #                 import inspect
    #                 import sacrebleu
    #                 fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
    #                 if 'smooth_method' in fn_sig:
    #                     smooth = {'smooth_method': 'exp'}
    #                 else:
    #                     smooth = {'smooth': 'exp'}
    #                 bleu = sacrebleu.compute_bleu(
    #                     correct=meters['_bleu_counts'].sum,
    #                     total=meters['_bleu_totals'].sum,
    #                     sys_len=meters['_bleu_sys_len'].sum,
    #                     ref_len=meters['_bleu_ref_len'].sum,
    #                     **smooth
    #                 )
    #                 return round(bleu.score, 2)

    #             metrics.log_derived('bleu', compute_bleu)

    # def inference_step(self, generator, models, sample, prefix_tokens=None):
    #     with torch.no_grad():
    #         return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    