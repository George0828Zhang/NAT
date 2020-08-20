# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.distributions.utils import probs_to_logits, logits_to_probs
import pdb

# def label_smoothed_target(logits, targets, smoothing):
#     labels = targets.size(-1) - 1
#     true_dist = logits.new_full(logits.size(), smoothing/labels)
#     true_dist.scatter_(1, targets.unsqueeze(1), 1. - smoothing)
#     # true_dist[:, self.padding_idx] = 0
#     # mask = torch.nonzero(target.data == self.padding_idx)
#     # if mask.dim() > 0:
#     #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
#     return true_dist

@register_criterion("memory_efficient_mutual_loss")
class MemoryEfficientMutualLearningCriterion(FairseqCriterion):

    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing',
        )

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len

            policy_logprob: if there is some policy
                depends on the likelihood score as rewards.
        """
        soft = targets.dim() == outputs.dim()            
        
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        if masks is not None:            
            outputs = outputs[masks]
            targets = targets[masks]
        else:
            outputs = outputs.view(-1, outputs.size(-1))
            if soft:
                targets = targets.view(-1, outputs.size(-1))
            else:
                targets = targets.view(-1)

        # if masks is not None and not masks.any():
        if masks is not None and (masks==0).all():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)            

            if soft:  # soft-labels
                losses = F.kl_div(logits, targets.to(
                    logits.device), reduction='none')
                losses = losses.sum(-1)
            else:
                losses = F.nll_loss(logits, targets.to(
                    logits.device), reduction='none')

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                    1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        # B x T
        src_tokens, src_lengths, prev_output_tokens = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
            sample["net_input"]["prev_output_tokens"],
        )
        tgt_tokens, nat_prev_output_tokens = sample["target"], sample["prev_target"]

        
        primary_step = model.num_updates % 2 == 0
        if primary_step:
            learner, helper = model, model.peer
            learner_name = "model"
        else:
            learner, helper = model.peer, model
            learner_name = "peer"
    
        helper.eval()
        learner.train() # because peer is a submodule of model, .train() comes later
        """ forward helper """
        with torch.no_grad():
            helper_outputs = helper(
                src_tokens, src_lengths,
                prev_output_tokens if (primary_step and model.peer_type == "ar") else nat_prev_output_tokens,
                tgt_tokens
            )
            helper_logits = helper_outputs["word_ins"]["out"]
        helper.train() # to avoid .backward() error.

        """ forward learner """
        outputs = learner(
            src_tokens, src_lengths,
            prev_output_tokens if (not primary_step and model.peer_type == "ar") else nat_prev_output_tokens,
            tgt_tokens
        )
        learner_logits, learner_masks, smoothing = (
            outputs["word_ins"]["out"],
            outputs["word_ins"].get("mask", None),
            outputs["word_ins"].get("ls", 0.0)
        )

        """ model loss
        1. label smoothed ground-truth(gt) loss
        2. kd loss
        """
        gt_losses = self._compute_loss(
            learner_logits,
            tgt_tokens,
            learner_masks,
            smoothing,
            name=learner_name+'-gt-loss',
            factor=0.5
        )
        kd_losses = self._compute_loss(
            learner_logits,
            logits_to_probs(helper_logits).detach(),
            learner_masks,
            name=learner_name+'-kd-loss',
            factor=0.5
        )

        losses = [
            gt_losses, 
            kd_losses,
        ]

        """ length prediction module
        length prediction loss
        """
        if primary_step:
            length_losses = self._compute_loss(
                outputs["length"].get("out"),
                outputs["length"].get("tgt"),
                name="length-loss",
                factor=outputs["length"].get("factor", 1.0)
            )
            losses += [length_losses]

        loss = sum(l["loss"] for l in losses)
        nll_loss = loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0)
                                     for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0)
                                  for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size /
                           math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss /
                           sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived(
            'ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size /
                    math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
