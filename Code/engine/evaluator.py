import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import compute_c_index, compute_auc, compute_brier_score

class Evaluator:
    def __init__(self, model, loss_fn, test_loader, cfg):
        self.model = model
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = cfg.train.device

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_survival = []
        all_hazard = []
        all_risk = []
        all_time = []
        all_event = []
        all_sample_ids = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            pathology = [p.to(self.device) for p in batch["pathology"]]
            genomic = batch["genomic"].to(self.device)
            time = batch["time"].to(self.device)
            event = batch["event"].to(self.device)
            sample_ids = batch["sample_ids"]

            outputs = self.model(pathology, genomic)
            losses = self.loss_fn(outputs, {
                "time": time,
                "event": event
            })
            total_loss += losses["total"].item()

            survival = outputs["survival"]
            hazard = outputs["hazard"]
            risk = 1 - torch.mean(survival, dim=-1)

            all_survival.append(survival.cpu().numpy())
            all_hazard.append(hazard.cpu().numpy())
            all_risk.append(risk.cpu().numpy())
            all_time.append(time.cpu().numpy())
            all_event.append(event.cpu().numpy())
            all_sample_ids.extend(sample_ids)

        avg_loss = total_loss / len(self.test_loader)
        all_risk = np.concatenate(all_risk)
        all_time = np.concatenate(all_time)
        all_event = np.concatenate(all_event)
        all_survival = np.concatenate(all_survival)
        all_hazard = np.concatenate(all_hazard)

        c_index = compute_c_index(all_time, all_event, all_risk)
        auc = compute_auc(all_time, all_event, all_risk)
        brier_score = compute_brier_score(all_survival, all_time, all_event)

        results = {
            "loss": avg_loss,
            "c_index": c_index,
            "auc": auc,
            "brier_score": brier_score,
            "risk_scores": all_risk,
            "survival_curves": all_survival,
            "hazard_values": all_hazard,
            "event_time": all_time,
            "event_status": all_event,
            "sample_ids": all_sample_ids
        }
        return results

    @torch.no_grad()
    def evaluate_missing_modality(self, missing_type):
        self.model.eval()
        total_loss = 0.0
        all_risk = []
        all_time = []
        all_event = []

        for batch in tqdm(self.test_loader, desc=f"Evaluating {missing_type}"):
            pathology = [p.to(self.device) for p in batch["pathology"]]
            genomic = batch["genomic"].to(self.device)
            time = batch["time"].to(self.device)
            event = batch["event"].to(self.device)

            if missing_type == "pathology":
                pathology = [torch.zeros_like(p) for p in pathology]
            elif missing_type == "genomic":
                genomic = torch.zeros_like(genomic)

            outputs = self.model(pathology, genomic)
            losses = self.loss_fn(outputs, {"time": time, "event": event})
            total_loss += losses["total"].item()
            risk = 1 - torch.mean(outputs["survival"], dim=-1)

            all_risk.append(risk.cpu().numpy())
            all_time.append(time.cpu().numpy())
            all_event.append(event.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        all_risk = np.concatenate(all_risk)
        all_time = np.concatenate(all_time)
        all_event = np.concatenate(all_event)
        c_index = compute_c_index(all_time, all_event, all_risk)

        return {
            "loss": avg_loss,
            "c_index": c_index
        }