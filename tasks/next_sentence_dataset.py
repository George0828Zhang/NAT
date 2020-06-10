# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math
import pdb

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
        # batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['prev_target'] = prev_output_tokens

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

    # def __init__(self, *args, **kwargs):
    #     super().__init__(args, kwargs)
    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_whole_words,
        shuffle,
        seed,
        args,
        eos=None
    ):
        self.permute = args.permute
        self.randomize_mask_ratio = args.randomize_mask_ratio
        
        if args.context_type == "sentence":
            self.context_fragment = False
        elif args.context_type == "fragment":
            self.context_fragment = True
        else:
            raise NotImplementedError(f"context type {args.context_type} not implemented.")

        if args.mask_length != 'subword' and args.replace_length != -1:
            raise (f'if not using subwords, replace_length can only be -1 (same)')
        if args.insert != 0.0:
            raise (f'insertion not supported')

        # if self.context_fragment:
        #     self.context_cutoff = np.clip(
        #         (np.random.uniform(0.4,0.6,sizes.size) * sizes).astype(int),
        #         1,          # min
        #         sizes - 1   # max
        #     )
        #     self.context_coinflip = np.random.randint(0,2,sizes.size)*2-1
        # else:
        #     self.context_coinflip = np.random.randint(0,2,sizes.size)*2-1
        #     self.context_coinflip[0] = 1

        super().__init__(
            dataset,
            sizes,
            vocab,
            mask_idx,
            mask_whole_words,
            shuffle,
            seed,
            args,
            eos=eos,
        )

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            """
            include context for next sentence denoising
            """            
            if self.context_fragment:
                cutoff = np.clip(
                    (np.random.uniform(0.4,0.6) * self.sizes[index]).astype(int),
                    1,                      # min
                    self.sizes[index] - 1   # max
                )
                context = self.dataset[index][:cutoff]
                tokens = self.dataset[index][cutoff:]

                context = torch.cat([context, context.new([self.eos])])
                tokens = torch.cat([tokens.new([self.vocab.bos()]), tokens])
            else:
                tokens = self.dataset[index]
                offset = 1 if index==0 else -1
                context = self.dataset[index+offset]

            if np.random.random() < 0.5:
                tmp = tokens
                tokens = context
                context = tmp

            assert tokens[-1] == self.eos
            assert context[-1] == self.eos
            source, target = tokens, tokens.clone()

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.randomize_mask_ratio:
                p = np.random.uniform(low=0.01, high=0.99) #np.random.random()
                source = self.add_whole_word_mask(source, p) 
            elif self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            ## added support for token permutation
            if self.permute > 0:
                source = self.add_permuted_noise(source, self.permute)

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
        return len(self.dataset)

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
        return self.sizes[index]        

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``.
        includes context tokens
        """
        if self.context_fragment:            
            return self.sizes[index]
        else:
            offset = 1 if index==0 else -1
            return (self.sizes[index+offset], self.sizes[index])
