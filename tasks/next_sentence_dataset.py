# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math

from fairseq.data import data_utils, FairseqDataset, DenoisingDataset


def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    """ ntokens is by default determined by target """
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            # prev_output_tokens = merge(
            #     'target',
            #     left_pad=left_pad_target,
            #     move_eos_to_beginning=True,
            # )
            prev_output_tokens = merge('noised', left_pad=left_pad_target)
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class NextSentenceDataset(DenoisingDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        if args.mask_length != 'subword' and self.replace_length != -1:
            raise (f'if not using subwords, replace_length can only be -1 (same)')
        if self.insert_ratio != 0.0:
            raise (f'insertion not supported')

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            """
            include context for next sentence denoising
            """
            context = self.dataset[index]
            token = self.dataset[index+1]
            assert tokens[-1] == self.eos
            assert context[-1] == self.eos
            source, target = tokens, tokens.clone()

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.random_mask_ratio:
                source = self.add_whole_word_mask(source, np.random.random()) # TODO
            elif self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            ## added support for token permutation
            if self.permute > 0:
                source = self.add_permuted_noise(source, p, self.permute)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        assert len(source) == len(target)
        return {
            'id': index,
            'source': context,
            'noised': source, # noised target token
            'target': target,
        }

    def __len__(self):
        return len(self.dataset) - 1

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples, self.vocab.pad(), self.vocab.eos(), self.vocab)
    
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching.
        includes context tokens
        """
        return max(self.sizes[index], self.sizes[index+1])

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``.
        includes context tokens
        """
        return (self.sizes[index], self.sizes[index+1])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, 'supports_prefetch')
            and self.src.supports_prefetch
            and hasattr(self.tgt, 'supports_prefetch')
            and self.tgt.supports_prefetch
        )
