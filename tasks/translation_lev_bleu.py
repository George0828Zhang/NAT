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
from fairseq import utils
import logging
logger = logging.getLogger(__name__)

@register_task('translation_lev_bleu')
class TranslationLevenshteinBLEUTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument('--add-mask-token', action='store_true',
                            help='add a mask token for model compatibility.')
        parser.add_argument('--use-mask-token', default=False, action='store_true')
        parser.add_argument('--add-lang-token', default=False, action='store_true')
        parser.add_argument('--use-lang-token', default=False, action='store_true')
        parser.add_argument('--langs', default=None, metavar='LANG',
                            help='comma-separated list of monolingual language, for example, "en,de,fr"'
                                 'be careful these langs are what you used for pretraining (the same order),'
                                 'not for finetuning.'
                                 'you should always add all pretraining language idx during finetuning.')

    def __init__(self, args, src_dict, tgt_dict):
        if args.add_mask_token:
            for d in [src_dict, tgt_dict]:
                d.add_symbol('<mask>')
            self.mask_id = tgt_dict.index('<mask>')
        elif args.use_mask_token:
            raise (f'please enable add-mask-token for use-mask-token to work.')
        if args.add_lang_token:
            if args.langs is None:
                raise (f'please provide langs for add-lang-token or use-lang-token.')
            else:
                languages = args.langs.split(',')
            for d in [src_dict, tgt_dict]:
                for lang in languages:
                    d.add_symbol('[{}]'.format(lang))
        elif args.use_lang_token:
            raise (f'please enable add-lang-token for use-lang-token to work.')
        super().__init__(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
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
            prepend_bos=True,
            append_source_id=self.args.use_lang_token,
        )

    def inject_noise(self, target_tokens, mask_id=None):        
        unk = mask_id if mask_id is not None else self.tgt_dict.unk()
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
        gen = IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

        if self.args.use_lang_token:
            gen.eos=self.target_dictionary.index('[{}]'.format(self.args.target_lang))
        return gen

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        if self.args.use_lang_token:
            src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.source_lang))
            source_tokens = []
            for s_t in src_tokens:
                s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
                source_tokens.append(s_t)
            dataset = LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)
            return dataset
        else:
            return super().build_dataset_for_inference(src_tokens, src_lengths)

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        mask_id = self.target_dictionary.index('<mask>') if self.args.use_mask_token else None # None uses <unk> as mask
        sample['prev_target'] = self.inject_noise(sample['target'], mask_id=mask_id)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    # def valid_step(self, sample, model, criterion):
    #     model.eval()
    #     with torch.no_grad():
    #         sample['prev_target'] = self.inject_noise(sample['target'])
    #         loss, sample_size, logging_output = criterion(model, sample)
    #     return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            mask_id = self.target_dictionary.index('<mask>') if self.args.use_mask_token else None # None uses <unk> as mask
            sample['prev_target'] = self.inject_noise(sample['target'], mask_id=mask_id)
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
                extra_symbols_to_ignore=[
                    self.tgt_dict.index('[{}]'.format(self.args.source_lang)),
                    self.tgt_dict.index('[{}]'.format(self.args.target_lang))
                ]
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s if s else '<unk>'

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

    