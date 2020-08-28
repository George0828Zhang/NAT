# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
from torch import Tensor
from torch.distributions.utils import logits_to_probs

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.models.nat import NATransformerModel

def top_k_kl_div(logits, targets, k, reduction='none'):
    if k != -1:
        values, indices = torch.topk(targets.to(logits.device), k, dim=-1)
        logits = torch.gather(logits, -1, indices)
        targets = F.normalize(values, p=1, dim=-1)
    return F.kl_div(logits, targets, reduction=reduction)

@register_criterion("knowledge_distillation_loss")
class KnowledgeDistillationCriterion(FairseqCriterion):

    def __init__(self, task, label_smoothing, distill_top_k):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.distill_top_k = distill_top_k
        assert self.distill_top_k==-1 or self.distill_top_k > 0, 'top K need to be -1 or >0'

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
        parser.add_argument(
            '--distill-top-k',
            default=-1,
            type=int,
            metavar='K',
            help='distill the top K log probabilities only. -1 to compute full distillation.',
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
                losses = top_k_kl_div(logits, targets, k=self.distill_top_k, reduction='none')
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

    def forward(self, model, target_model, sample, reduce=True, kd_factor=0.5):
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
            
        """ forward target_model """
        with torch.no_grad():
            target_model_outputs = target_model(
                src_tokens, src_lengths,
                nat_prev_output_tokens if isinstance(target_model, NATransformerModel) else prev_output_tokens,
                tgt_tokens
            )
            target_model_logits, target_model_masks = (
                target_model_outputs["word_ins"]["out"],
                target_model_outputs["word_ins"].get("mask", None),
            )
        
        """ forward model """
        outputs = model(
            src_tokens, src_lengths,
            nat_prev_output_tokens if isinstance(model, NATransformerModel) else prev_output_tokens,
            tgt_tokens
        )
        model_logits, model_masks, smoothing = (
            outputs["word_ins"]["out"],
            outputs["word_ins"].get("mask", None),
            outputs["word_ins"].get("ls", 0.0)
        )

        """ model loss
        1. label smoothed ground-truth loss (label loss)
        2. kd loss
        """
        lb_losses = self._compute_loss(
            model_logits,
            tgt_tokens,
            model_masks,
            smoothing,
            name='label-loss',
            factor=1. - kd_factor
        )
        kd_losses = self._compute_loss(
            model_logits,
            logits_to_probs(target_model_logits).detach(),
            torch.logical_and(model_masks, target_model_masks),
            name='kd-loss',
            factor=kd_factor,
        )

        losses = [
            lb_losses, 
            kd_losses,
        ]

        """ length prediction module
        length prediction loss
        """
        if "length" in outputs:
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
