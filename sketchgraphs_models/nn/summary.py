"""This module implements utilities to compute summary statistics.
"""

import posixpath
import torch


class ClassificationSummary:
    """ Simple class to keep track of summaries of a classification problem. """
    def __init__(self, num_outcomes=2, device=None):
        """ Initializes a new summary class with the given number of outcomes.

        Parameters
        ----------
        num_outcomes : int
            the number of possible outcomes of the classification problem.
        device : torch.device, optional
            device on which to place the recorded statistics.
        """
        self.recorded = torch.zeros(num_outcomes * num_outcomes, dtype=torch.int32, device=device)
        self.num_outcomes = num_outcomes

    @property
    def prediction_matrix(self):
        return self.recorded.view((self.num_outcomes, self.num_outcomes))

    def record_statistics(self, labels, predictions):
        """ Records statistics for a batch of predictions.

        Parameters
        ----------
        labels: an array of true labels in integer format. Each label must correspond to an
            integer in 0 to num_outcomes - 1 inclusive.
        predictions: an array of predicted labels. Must follow the same format as `labels`.
        """
        indices = torch.add(labels.int(), self.num_outcomes, predictions.int()).long().to(device=self.recorded.device)
        self.recorded = self.recorded.scatter_add_(
            0, indices, torch.ones_like(indices, dtype=torch.int32))

    def reset_statistics(self):
        """ Resets statistics recorded in this accumulator. """
        self.recorded = torch.zeros_like(self.recorded)

    def accuracy(self):
        """ Compute the accuracy of the recorded problem. """
        num_correct = self.prediction_matrix.diag().sum()
        num_total = self.recorded.sum()

        return num_correct.float() / num_total.float()

    def confusion_matrix(self):
        return self.prediction_matrix.float() / self.prediction_matrix.sum().float()

    def cohen_kappa(self):
        pm = self.prediction_matrix.float()
        N = self.recorded.sum().float()

        p_observed = pm.diag().sum() / N
        p_expected = torch.dot(pm.sum(dim=0), pm.sum(dim=1)) / (N * N)

        if p_expected == 1:
            return 1
        else:
            return 1 - (1 - p_observed) / (1 - p_expected)

    def marginal_labels(self):
        return self.prediction_matrix.sum(dim=0).float() / self.recorded.sum().float()

    def marginal_predicted(self):
        return self.prediction_matrix.sum(dim=1).float() / self.recorded.sum().float()

    def write_tensorboard(self, writer, prefix="", global_step=None, **kwargs):
        writer.add_scalar(posixpath.join(prefix, "kappa"), self.cohen_kappa(), global_step, **kwargs)
        writer.add_scalar(posixpath.join(prefix, "accuracy"), self.accuracy(), global_step, **kwargs)
