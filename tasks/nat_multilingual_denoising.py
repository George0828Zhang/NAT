# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import json
from argparse import Namespace

import numpy as np
import torch

from fairseq.data import (
    encoders,
    data_utils,
    Dictionary,
    AppendTokenDataset,
    ConcatDataset,
    DenoisingDataset,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
)

from fairseq.data.encoders.utils import get_whole_word_mask
# from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from .translation_lev_bleu import TranslationLevenshteinBLEUTask
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingTask
from fairseq.tasks import register_task
from fairseq import utils

logger = logging.getLogger(__name__)


@register_task('nat_multilingual_denoising')
class NATMultilingualDenoisingTask(MultilingualDenoisingTask):

    @staticmethod
    def add_args(parser):
        """
        parser.add_argument('--multilang-sampling-alpha', type=float, default=1.0,
                            help='smoothing alpha for sample rations across multiple datasets')
        parser.add_argument('--add-lang-token', default=False, action='store_true')
        parser.add_argument('--langs', type=str, help="language ids we are considering", default=None)
        parser.add_argument('--no-whole-word-mask-langs', type=str, default='', metavar='N',
                            help='languages without spacing between words dont support whole word masking')
        """
        MultilingualDenoisingTask.add_args(parser)

        parser.add_argument(
            '--decoder-input-noise', dest="noise",
            default='random_mask',
            choices=['random_mask', 'no_noise', 'full_mask'])

        # options for showing sentences during validation        
        # parser.add_argument('--eval-lm', action='store_true',
        #                     help='evaluation with BLEU scores')
        parser.add_argument('--eval-lm-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-lm-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-lm-print-samples; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-lm-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-lm-print-samples', action='store_true',
                            help='print sample generations during validation')

    ### only changed logging to make sampling probs show in info.
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = self.langs.split(',')
            for name in languages:
                assert os.path.exists(os.path.join(data_path, name)), "all the languages must exist"

        logger.info("| Training on {0} languages: {1}".format(len(languages), languages))
        logger.info("| Language to id mapping: %s", {
                lang: id for id, lang in enumerate(languages)
            }
        )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(',')
        lang_datasets = []
        for language in languages:
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            end_token = self.source_dictionary.index('[{}]'.format(language)) \
                if self.args.add_lang_token else self.source_dictionary.eos()

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 2,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=end_token,
                break_mode=self.args.sample_break_mode,
            )
            logger.info('| loaded {} blocks from: {}'.format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            dataset = AppendTokenDataset(dataset, end_token)

            lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None
            lang_dataset = DenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=None if not self.args.add_lang_token else self.source_dictionary.index('[{}]'.format(language)),
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            '| loaded total {} blocks for all languages'.format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info("| Sample probability by language: %s", {
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                }
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info("| Up/Down Sampling ratio by language: %s", {
                    lang: "{0:.2f}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                }
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + '_' + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ','.join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    

    #############################################
    # patched to support nat model and printing #
    #############################################
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


    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_lm_print_samples', False):
            assert getattr(args, 'eval_lm_detok', None) is not None, (
                '--eval-lm-detok is required if using --eval-lm-print-samples; '
                'try --eval-lm-detok=moses (or --eval-lm-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_lm_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_lm_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_lm_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    @property
    def tgt_dict(self):
        """ tanslation_lev 's inject_noise requires this. """
        return self.target_dictionary

    def nat_sample(self, sample):
        if 'prev_output_tokens' in sample['net_input']:
            del sample['net_input']['prev_output_tokens'] # reduce vram usage
        sample['prev_target'] = TranslationLevenshteinBLEUTask.inject_noise(self, sample['target'], mask_id=self.mask_idx)
        return sample

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        sample = self.nat_sample(sample)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample = self.nat_sample(sample)
            loss, sample_size, logging_output = criterion(model, sample)

            if self.args.eval_lm_print_samples:
                self._inference_print(self.sequence_generator, sample, model)                
        return loss, sample_size, logging_output

    def _inference_print(self, generator, sample, model):
        # import pdb; pdb.set_trace()
        def decode(toks, escape_unk=False):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.args.eval_lm_remove_bpe,
                escape_unk=escape_unk,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        srcs, hyps, refs = [], [], []
        for i in range(1):#len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.target_dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            srcs.append(decode(
                utils.strip_pad(sample['net_input']['src_tokens'][i], self.source_dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        logger.info('example source: ' + srcs[0])
        logger.info('example hypothesis: ' + hyps[0])
        logger.info('example reference: ' + refs[0])        