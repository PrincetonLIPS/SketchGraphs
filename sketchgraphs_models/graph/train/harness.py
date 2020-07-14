"""This module contains the main training harness for training graph models. """

import collections
import itertools
import os
import pickle

import numpy as np
import torch

from sketchgraphs_models import training
from sketchgraphs_models.graph import model as graph_model
from sketchgraphs_models.nn import summary


def _detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    else:
        return x


def _mean_value(v):
    values = [a.mean().cpu().numpy() for a in v]
    return np.mean(values) if values else np.nan


def _total_loss(losses):
    result = 0

    for v in losses.values():
        if v is None:
            continue

        if isinstance(v, dict):
            result += _total_loss(v)
        else:
            result += v.sum()

    return result


class GraphModelHarness(training.TrainingHarness):
    """This class is the main harness for training graph models.

    The harness is responsible for coordinating all the procedures that surround training,
    such as learning rate scheduling, data loading, and logging.
    """
    def __init__(self, model, opt, node_feature_dimension, edge_feature_dimension,
                 config_train, config_eval=None, scheduler=None, output_dir=None, dist_config=None,
                 profile_enabled=False, additional_model_information=None):
        super(GraphModelHarness, self).__init__(model, opt, config_train, config_eval, dist_config)
        self.scheduler = scheduler
        self.output_dir = output_dir

        self.node_feature_dimension = node_feature_dimension
        self.edge_feature_dimension = edge_feature_dimension
        self.feature_dimension = {**node_feature_dimension, **edge_feature_dimension}
        self.profile_enabled = profile_enabled
        self._last_profile_step = 0
        self.additional_model_information = additional_model_information or {}

        def _make_feature_summary(fd):
            return collections.OrderedDict(
                (t, collections.OrderedDict(
                    (feature_name, summary.ClassificationSummary(dim))
                    for feature_name, dim in feature_description.items()
                )) for t, feature_description in fd.items())

        self.edge_feature_summaries = _make_feature_summary(edge_feature_dimension)
        self.node_feature_summaries = _make_feature_summary(node_feature_dimension)


    def _get_profile_path(self, global_step):
        if not self.profile_enabled:
            return None

        if self._last_profile_step is None or global_step - self._last_profile_step > 100000:
            self._last_profile_step = global_step
            return 'profile_step_{0}.pkl'.format(global_step)

        return None

    def single_step(self, batch, global_step):
        self.opt.zero_grad()

        profile_path = self._get_profile_path(global_step)
        with torch.autograd.profiler.profile(enabled=profile_path is not None, use_cuda=True) as trace:
            with torch.autograd.profiler.record_function("forward"):
                readout = self.model(batch)
                losses, accuracy, edge_metrics, node_metrics = graph_model.compute_losses(
                    readout, batch, self.feature_dimension)
                total_loss = _total_loss(losses)
            if self.model.training:
                with torch.autograd.profiler.record_function("backward"):
                    total_loss.backward()

                with torch.autograd.profiler.record_function("opt_update"):
                    self.opt.step()

        if profile_path is not None:
            with open(profile_path, 'wb') as f:
                pickle.dump(trace, f, pickle.HIGHEST_PROTOCOL)

        losses = training.map_structure_flat(losses, _detach)
        losses = graph_model.compute_average_losses(losses, batch['graph_counts'])
        avg_loss = total_loss.detach() / float(sum(batch['graph_counts']))
        losses['average'] = avg_loss

        if self.is_leader():
            def _record_classification_summaries(metrics, summaries):
                for t, (labels, preds) in metrics.items():
                    for i, cs in enumerate(summaries[t].values()):
                        cs.record_statistics(labels[:, i], preds[:, i])

            _record_classification_summaries(edge_metrics, self.edge_feature_summaries)
            _record_classification_summaries(node_metrics, self.node_feature_summaries)

        return losses, accuracy

    def on_epoch_end(self, epoch, global_step):
        if self.scheduler is not None:
            self.scheduler.step()

            if self.config_train.tb_writer is not None and self.is_leader():
                lr = self.scheduler.get_last_lr()[0]
                self.config_train.tb_writer.add_scalar('learning_rate', lr, global_step)

        if self.is_leader() and self.output_dir is not None and (epoch + 1) % 10 == 0:
            self.log('Saving checkpoint for epoch {}'.format(epoch + 1))
            torch.save(
                {
                    'opt': self.opt.state_dict(),
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    **self.additional_model_information,
                },
                os.path.join(self.output_dir, 'model_state_{0}.pt'.format(epoch + 1)))

    def write_summaries(self, global_step, losses, accuracies, tb_writer):
        if tb_writer is None:
            return

        for k, v in losses.items():
            if k == 'edge_features' or k == 'node_features':
                v = _mean_value(v.values())
            tb_writer.add_scalar('loss/' + k, v, global_step)

        for k, v in accuracies.items():
            if k == 'edge_features' or k == 'node_features':
                v = _mean_value(v.values())
            tb_writer.add_scalar('accuracy/' + k, v, global_step)

        for t, feature_schema in self.edge_feature_summaries.items():
            for n, cs in feature_schema.items():
                cs.write_tensorboard(tb_writer, 'kappa' + '/' + t.name + '/' + n, global_step)

        for t, feature_schema in self.node_feature_summaries.items():
            for n, cs in feature_schema.items():
                cs.write_tensorboard(tb_writer, 'kappa' + '/' + t.name + '/' + n, global_step)

    def print_statistics(self, loss_acc, accuracy_acc):
        self.log(f'\tLoss ({loss_acc["average"]:.3f}): Node({loss_acc["node_label"]:.3f}, {loss_acc["node_stop"]:.3f}) '
                 f'Edge({loss_acc["edge_label"]:.3f}, {loss_acc["edge_partner"]:.3f}, '
                 f'{_mean_value(loss_acc["edge_features"].values()):.3f}) '
                 f'Subnode({loss_acc["subnode_stop"]:.3f}).')
        self.log(f'\tAccuracy: Node({accuracy_acc["node_label"]:4.1%}, {accuracy_acc["node_stop"]:4.1%}) '
                 f'Edge({accuracy_acc["edge_label"]:4.1%}, {accuracy_acc["edge_partner"]:4.1%}, '
                 f'{_mean_value(accuracy_acc["edge_features"].values()):4.1%}) '
                 f'Subnode({accuracy_acc["subnode_stop"]:4.1%})')
        self.log()

        def _summary_text(target, features):
            return f'{target.name}: ' + '; '.join(f'{n} ({cs.cohen_kappa():.3f})' for n, cs in features.items())

        if self.node_feature_summaries:
            self.log('Kappa Entity')
            for t, features in self.node_feature_summaries.items():
                self.log(_summary_text(t, features))

            self.log()

        if self.edge_feature_summaries:
            self.log('Kappa Edges')
            for t, features in self.edge_feature_summaries.items():
                self.log(_summary_text(t, features))
            self.log()

    def reset_statistics(self):
        for features in itertools.chain(self.node_feature_summaries.values(), self.edge_feature_summaries.values()):
            for classification_summary in features.values():
                classification_summary.reset_statistics()

        super(GraphModelHarness, self).reset_statistics()
