import torch
import os
from tqdm import tqdm
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from utils.metrics import compute_c_index

class Trainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, optimizer, cfg):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.cfg = cfg.train
        self.logger = Logger(cfg.log_dir)
        self.best_cindex = 0.0
        self.device = cfg.train.device

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for step, batch in enumerate(pbar):
            pathology = [p.to(self.device) for p in batch["pathology"]]
            genomic = batch["genomic"].to(self.device)
            time = batch["time"].to(self.device)
            event = batch["event"].to(self.device)
            outputs = self.model(pathology, genomic)
            losses = self.loss_fn(outputs, {"time": time, "event": event})
            loss = losses["total"] / self.cfg.gradient_accum_steps
            loss.backward()
            if (step + 1) % self.cfg.gradient_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item() * self.cfg.gradient_accum_steps
            pbar.set_postfix({"loss": total_loss / (step + 1)})
        avg_loss = total_loss / len(self.train_loader)
        self.logger.log_scalar("train/loss", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_times = []
        all_events = []
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        for batch in pbar:
            pathology = [p.to(self.device) for p in batch["pathology"]]
            genomic = batch["genomic"].to(self.device)
            time = batch["time"].to(self.device)
            event = batch["event"].to(self.device)
            outputs = self.model(pathology, genomic)
            losses = self.loss_fn(outputs, {"time": time, "event": event})
            total_loss += losses["total"].item()
            risk = 1 - outputs["survival"].mean(dim=-1)
            all_preds.append(risk.cpu())
            all_times.append(time.cpu())
            all_events.append(event.cpu())
        avg_loss = total_loss / len(self.val_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_times = torch.cat(all_times).numpy()
        all_events = torch.cat(all_events).numpy()
        cindex = compute_c_index(all_times, all_events, all_preds)
        self.logger.log_scalar("val/loss", avg_loss, epoch)
        self.logger.log_scalar("val/cindex", cindex, epoch)
        if cindex > self.best_cindex:
            self.best_cindex = cindex
            save_checkpoint(self.model, self.optimizer, epoch, self.best_cindex, self.cfg.ckpt_dir)
        return avg_loss, cindex

    def run(self):
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, cindex = self.val_epoch(epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | C-Index: {cindex:.4f}")