# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
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
from fairseq.tasks import register_task

### patched
import torch
import json
from argparse import Namespace
from fairseq.data import encoders
from fairseq import utils
from fairseq.utils import new_arange
from .nat_multilingual_denoising import NATMultilingualDenoisingTask
from .next_sentence_dataset import NextSentenceDataset

logger = logging.getLogger(__name__)


@register_task('nat_next_sentence_generation')
class NATNextSentenceGenerationTask(NATMultilingualDenoisingTask):
    @staticmethod
    def add_args(parser):
        NATMultilingualDenoisingTask.add_args(parser)
        parser.add_argument(
            '--randomize-mask-ratio', action="store_true",
            help='use random ratio to mask input.'
        )

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
        # Changed
        logger.info("| Language to id mapping: %s", {
                lang: id for id, lang in enumerate(languages)
            }
        )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
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

            lang_dataset = NextSentenceDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                mask_whole_words,
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
    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        sample['prev_target'] = sample['net_input'].pop('prev_output_tokens')
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = sample['net_input'].pop('prev_output_tokens')
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
        ctxs, srcs, hyps, refs = [], [], [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.target_dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            ctxs.append(decode(
                utils.strip_pad(sample['net_input']['src_tokens'][i], self.source_dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            srcs.append(decode(
                utils.strip_pad(sample['prev_target'][i], self.target_dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        logger.info('example context: ' + ctxs[0])
        logger.info('example source: ' + srcs[0])
        logger.info('example hypothesis: ' + hyps[0])
        logger.info('example reference: ' + refs[0])        