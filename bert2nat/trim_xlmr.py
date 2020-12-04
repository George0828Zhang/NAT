# Modified from ddaspit's comment in https://github.com/pytorch/fairseq/issues/2120

import argparse
import os
from typing import List
import shutil
import torch
import pdb

from fairseq.data import Dictionary

def load_dict(langs: List[str], path: str) -> Dictionary:
    d = Dictionary.load(path)
    for l in langs:
        d.add_symbol(f"[{l}]")
    d.add_symbol("<mask>")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained XLMR model for fine-tuning.")
    parser.add_argument("--xlmr-dir", type=str, required=True, help="The pre-trained XLMR model directory.")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary.")
    parser.add_argument("--ft-bpe", type=str, required=True, help="The fine-tuning bpe code/spmmodel.")
    # parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output-dir", type=str, required=True, help="The dir to save trimmed XLMR model.")
    parser.add_argument("--retain-mask-id", action="store_true", help="Whether to retain mask id or trim it.")
    args = parser.parse_args()

    langs = [] #args.langs.split(",")
    pre_dict = load_dict(langs, os.path.join(args.xlmr_dir, "dict.txt"))
    ft_dict = load_dict(langs, args.ft_dict)
    data = torch.load(os.path.join(args.xlmr_dir, "model.pt"))
    model = data["model"]

    # pdb.set_trace()

    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))

    # for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
    for name in ["decoder.sentence_encoder.embed_tokens.weight","decoder.lm_head.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), 768], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        model[name] = ft_tensor

    for name in ["decoder.lm_head.bias",]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict),], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        model[name] = ft_tensor

    torch.save(data, os.path.join(args.output_dir, "model.pt"))
    shutil.copy(args.ft_dict, os.path.join(args.output_dir, "dict.txt"))
    shutil.copy(args.ft_bpe, 
        os.path.join(args.output_dir, os.path.basename(args.ft_bpe)
    ))


if __name__ == "__main__":
    main()

    """
    python trim_xlmr.py --xlmr-dir /media/george/Data/xlmr.base --ft-dict ../DATA/data-bin/wmt14.en-de/dict.en.txt --ft-bpe /media/george/Data/wmt14.en-de/prep/spm.model --output-dir /media/george/Data/xlmr.base/trimmed_nomask
    """