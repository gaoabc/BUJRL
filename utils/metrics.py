import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.metrics import concordance_index_censored
from scipy.interpolate import interp1d


def compute_c_index(time, event, risk):
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk)
    result = concordance_index_censored(event, time, risk)
    return result[0]


def compute_auc(time, event, risk):
    try:
        return roc_auc_score(event, risk)
    except:
        return 0.5


def compute_brier_score(survival, time, event):
    n_samples = survival.shape[0]
    n_times = survival.shape[1]
    time_points = np.linspace(0, n_times - 1, n_times)
    brier_scores = []

    for t in range(1, n_times):
        idx = time >= t
        if np.sum(idx) == 0:
            continue
        true = (time[idx] <= t) & (event[idx] == 1)
        pred = survival[idx, t]
        brier = brier_score_loss(true, 1 - pred)
        brier_scores.append(brier)

    return np.mean(brier_scores) if len(brier_scores) > 0 else 1.0


def compute_discrimination_score(time, event, risk, n_bins=5):
    bins = np.percentile(risk, np.linspace(0, 100, n_bins + 1))
    bin_labels = np.digitize(risk, bins) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)
    event_rates = []

    for i in range(n_bins):
        mask = bin_labels == i
        if np.sum(mask) == 0:
            continue
        rate = np.mean(event[mask])
        event_rates.append(rate)

    return np.max(event_rates) - np.min(event_rates) if len(event_rates) > 1 else 0.0


def compute_integrated_brier_score(survival, time, event, n_times=10):
    max_t = int(np.max(time))
    eval_times = np.linspace(0, max_t, n_times)
    ibs = 0.0
    count = 0

    for t in eval_times:
        t_idx = np.clip(int(t), 0, survival.shape[1] - 1)
        idx = time >= t
        if np.sum(idx) == 0:
            continue
        true = (time[idx] <= t) & (event[idx] == 1)
        pred = survival[idx, t_idx]
        ibs += brier_score_loss(true, 1 - pred)
        count += 1

    return ibs / count if count > 0 else 1.0


def compute_time_dependent_auc(survival, time, event, n_times=5):
    max_t = int(np.percentile(time[event == 1], 75))
    eval_times = np.linspace(0, max_t, n_times)
    aucs = []

    for t in eval_times:
        t_idx = np.clip(int(t), 0, survival.shape[1] - 1)
        idx = time >= t
        if np.sum(idx) < 5:
            continue
        true = (time[idx] <= t) & (event[idx] == 1)
        pred = 1 - survival[idx, t_idx]
        if len(np.unique(true)) < 2:
            continue
        auc = roc_auc_score(true, pred)
        aucs.append(auc)

    return np.mean(aucs) if len(aucs) > 0 else 0.5


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'c_index': [],
            'auc': [],
            'brier': [],
            'ibs': [],
            'td_auc': []
        }

    def update(self, outputs, batch):
        survival = outputs['survival'].detach().cpu().numpy()
        hazard = outputs['hazard'].detach().cpu().numpy()
        time = batch['time'].detach().cpu().numpy()
        event = batch['event'].detach().cpu().numpy()
        risk = 1 - np.mean(survival, axis=-1)

        self.metrics['c_index'].append(compute_c_index(time, event, risk))
        self.metrics['auc'].append(compute_auc(time, event, risk))
        self.metrics['brier'].append(compute_brier_score(survival, time, event))
        self.metrics['ibs'].append(compute_integrated_brier_score(survival, time, event))
        self.metrics['td_auc'].append(compute_time_dependent_auc(survival, time, event))

    def avg(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}