"""This module implements utilities to compute summary statistics.
"""

import posixpath
import torch
import torchmetrics


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
        """Returns a `torch.Tensor` representing the prediction matrix."""
        return self.recorded.view((self.num_outcomes, self.num_outcomes))

    def record_statistics(self, labels, predictions):
        """ Records statistics for a batch of predictions.

        Parameters
        ----------
        labels : torch.Tensor
            an array of true labels in integer format. Each label must correspond to an
            integer in 0 to num_outcomes - 1 inclusive.
        predictions : torch.Tensor
            an array of predicted labels. Must follow the same format as `labels`.
        """
        indices = torch.add(labels.int(), predictions.int(), alpha=self.num_outcomes).long().to(device=self.recorded.device)
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
        """Returns a `torch.Tensor` representing the confusion matrix."""
        return self.prediction_matrix.float() / self.prediction_matrix.sum().float()

    def cohen_kappa(self):
        """Computes the Cohen kappa measure of agreement.
        """
        pm = self.prediction_matrix.float()
        N = self.recorded.sum().float()

        p_observed = pm.diag().sum() / N
        p_expected = torch.dot(pm.sum(dim=0), pm.sum(dim=1)) / (N * N)

        if p_expected == 1:
            return 1
        else:
            return 1 - (1 - p_observed) / (1 - p_expected)

    def marginal_labels(self):
        """Computes the empirical marginal distribution of the true labels."""
        return self.prediction_matrix.sum(dim=0).float() / self.recorded.sum().float()

    def marginal_predicted(self):
        """Computes the empirical marginal distribution of the predicted labels."""
        return self.prediction_matrix.sum(dim=1).float() / self.recorded.sum().float()

    def write_tensorboard(self, writer, prefix="", global_step=None, **kwargs):
        """Write the accuracy and kappa metrics to a tensorboard writer.

        Parameters
        ----------
        writer: torch.utils.tensorboard.SummaryWriter
            The writer to which the metrics will be written
        prefix: str, optional
            Optional prefix for the name under which the metrics will be written
        global_step: int, optional
            Global step at which the metric is recorded
        **kwargs
            Further arguments to `torch.utils.tensorboard.SummaryWriter.add_scalar`.
        """
        writer.add_scalar(posixpath.join(prefix, "kappa"), self.cohen_kappa(), global_step, **kwargs)
        writer.add_scalar(posixpath.join(prefix, "accuracy"), self.accuracy(), global_step, **kwargs)


class CohenKappa(torchmetrics.Metric):
    """A pytorch-lightning compatible metric which computes Cohen's kappa score of agreement.
    """
    recorded: torch.Tensor

    def __init__(self, num_outcomes=2, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.num_outcomes = num_outcomes
        self.add_state("recorded", torch.zeros(num_outcomes * num_outcomes, dtype=torch.int32), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        indices = torch.add(targets.int(), preds.int(), alpha=self.num_outcomes).long().to(device=self.recorded.device)
        self.recorded.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int32))

    def compute(self):
        pm = self.recorded.view(self.num_outcomes, self.num_outcomes).float()
        N = self.recorded.sum().float()

        if N == 0:
            return pm.new_tensor(0.0)

        p_observed = pm.diag().sum() / N
        p_expected = torch.dot(pm.sum(dim=0), pm.sum(dim=1)) / (N * N)

        if p_expected == 1:
            return pm.new_tensor(1.0)
        else:
            return 1 - (1 - p_observed) / (1 - p_expected)
