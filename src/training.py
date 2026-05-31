# Обучение и инкрементальное обучение
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from copy import deepcopy
from datetime import datetime
import gc
from model import WeatherTransformer, count_parameters, estimate_flops, print_model_architecture
from generator import optimized_batch_generator


def train_model(loader, pipeline, start_date, end_date, global_df=None,
                window=32, epochs=20, batch_size=16, accumulation_steps=4, fixed_threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("\n[TRAINING] Step 1: Fitting scaler on training data...")

    gen_for_scaler = optimized_batch_generator(loader, pipeline, start_date, end_date, global_df,
                                               window, batch_size, fit_scaler_first=True,
                                               is_train=True, fixed_threshold=fixed_threshold)

    first_batch = None
    try:
        first_batch = next(gen_for_scaler)
    except StopIteration:
        raise RuntimeError("!!! Генератор не выдал ни одного батча для scaler")

    pos = 0
    neg = 0
    for x_tmp, y_cls_tmp, y_reg_tmp, _ in gen_for_scaler:
        y_np = y_cls_tmp.cpu().numpy()
        pos += (y_np == 1).sum()
        neg += (y_np == 0).sum()
        if pos + neg > 10000:
            break

    pos_weight_value = max(1.0, neg / (pos + 1e-6))
    print(f"\n[CLASS IMBALANCE] pos={pos}, neg={neg}, pos_weight={pos_weight_value:.3f}")

    if first_batch is None:
        raise RuntimeError("Нет данных после всех фильтров")

    input_size = first_batch[0].shape[2]

    model = WeatherTransformer(input_size, d_model=192, num_layers=4, max_seq_len=window).to(device)
    count_parameters(model)
    estimate_flops(model, input_size, seq_len=window, batch_size=batch_size)
    print_model_architecture(model, input_size, seq_len=window)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_reg = nn.HuberLoss()

    best_loss = float('inf')
    patience = 4
    patience_counter = 0

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    train_losses = []

    print(f"\n[START TRAINING] batch_size={batch_size}, epochs={epochs}, accumulation_steps={accumulation_steps}")

    for epoch in range(epochs):
        train_gen = optimized_batch_generator(loader, pipeline, start_date, end_date, global_df,
                                              window, batch_size, fit_scaler_first=False,
                                              is_train=True, fixed_threshold=fixed_threshold)

        epoch_loss = 0
        count = 0
        optimizer.zero_grad()
        model.train()

        for batch_idx, (x_batch, y_cls, y_reg, region_ids) in enumerate(train_gen):
            loss = None

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits_last, reg_last = model(x_batch, region_ids)

                    loss_cls = criterion_cls(logits_last, y_cls)
                    if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                        continue
                    loss_reg = criterion_reg(reg_last, y_reg)

                    loss = loss_cls + 1.5 * loss_reg
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                del loss_cls, loss_reg, logits_last, reg_last

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                logits_last, reg_last = model(x_batch, region_ids)

                loss_cls = criterion_cls(logits_last, y_cls)
                if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                    continue
                loss_reg = criterion_reg(reg_last, y_reg)

                loss = loss_cls + 1.5 * loss_reg
                loss = loss / accumulation_steps

                loss.backward()

                del loss_cls, loss_reg, logits_last, reg_last

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if loss is not None:
                epoch_loss += loss.item() * accumulation_steps
                count += 1

            if batch_idx % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gc.collect()

        if count > 0:
            avg_loss = epoch_loss / count
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  -> Best model saved (loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(torch.load("best_model.pt"))
                    break
        else:
            print(f"Epoch {epoch+1} - No valid batches processed")
            break

    return model, train_losses


class IncrementalLearner:
    def __init__(self, model, pipeline, loader, global_df=None,
                 buffer_size=2000,
                 ewc_lambda=1000,
                 replay_ratio=0.3):
        self.model = model
        self.pipeline = pipeline
        self.loader = loader
        self.global_df = global_df
        self.buffer_size = buffer_size
        self.ewc_lambda = ewc_lambda
        self.replay_ratio = replay_ratio
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y_cls = deque(maxlen=buffer_size)
        self.buffer_y_reg = deque(maxlen=buffer_size)
        self.fisher = {}
        self.old_params = {}
        self.update_history = []

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fisher': self.fisher,
            'old_params': self.old_params,
            'buffer_x': list(self.buffer_x),
            'buffer_y_cls': list(self.buffer_y_cls),
            'buffer_y_reg': list(self.buffer_y_reg),
            'update_history': self.update_history,
            'pipeline_columns': self.pipeline.fixed_columns,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fisher = checkpoint['fisher']
        self.old_params = checkpoint['old_params']
        self.buffer_x = deque(checkpoint['buffer_x'], maxlen=self.buffer_size)
        self.buffer_y_cls = deque(checkpoint['buffer_y_cls'], maxlen=self.buffer_size)
        self.buffer_y_reg = deque(checkpoint['buffer_y_reg'], maxlen=self.buffer_size)
        self.update_history = checkpoint['update_history']
        print(f"[LOAD] Restored from {path}, buffer size={len(self.buffer_x)}")

    def compute_fisher(self, dataloader, sample_size=500):
        print("[EWC] Computing Fisher Information Matrix...")
        self.model.eval()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
                self.old_params[name] = param.clone().detach()

        n_samples = 0
        for x_batch, y_cls, y_reg, _ in dataloader:
            if n_samples >= sample_size:
                break

            self.model.zero_grad()

            logits, reg = self.model(x_batch)
            loss_cls = nn.BCEWithLogitsLoss()(logits, y_cls)
            loss_reg = nn.HuberLoss()(reg, y_reg)
            loss = loss_cls + 1.5 * loss_reg
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.clone().detach() ** 2

            n_samples += len(x_batch)

        for name in self.fisher:
            self.fisher[name] /= n_samples

        print(f"[EWC] Fisher computed on {n_samples} samples")
        return self.fisher

    def ewc_loss(self, model):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher and self.fisher[name].sum() > 0:
                penalty = self.fisher[name] * (param - self.old_params[name]) ** 2
                loss += penalty.sum()
        return self.ewc_lambda * loss

    def fill_buffer_from_dates(self, start_date, end_date,
                                max_samples_per_date=100):
        print(f"[BUFFER] Filling from {start_date} to {end_date}")

        gen = optimized_batch_generator(
            self.loader, self.pipeline, start_date, end_date, self.global_df,
            window=48, batch_size=32, fit_scaler_first=False, is_train=True
        )

        samples_added = 0
        for x_batch, y_cls, y_reg, _ in gen:
            step = max(1, len(x_batch) // max_samples_per_date)
            for i in range(0, len(x_batch), step):
                if len(self.buffer_x) >= self.buffer_size:
                    break
                self.buffer_x.append(x_batch[i:i+1].cpu())
                self.buffer_y_cls.append(y_cls[i:i+1].cpu())
                self.buffer_y_reg.append(y_reg[i:i+1].cpu())
                samples_added += 1

            if len(self.buffer_x) >= self.buffer_size:
                break

        print(f"[BUFFER] Added {samples_added} samples, total={len(self.buffer_x)}")
        return samples_added

    def get_combined_batch(self, new_x, new_y_cls, new_y_reg, batch_size):
        n_new = len(new_x)
        n_replay = int(batch_size * self.replay_ratio)
        n_new_actual = batch_size - n_replay

        if n_replay > 0 and len(self.buffer_x) > 0:
            replay_indices = np.random.choice(
                len(self.buffer_x),
                min(n_replay, len(self.buffer_x)),
                replace=False
            )

            replay_x = torch.cat([self.buffer_x[i] for i in replay_indices], dim=0)
            replay_y_cls = torch.cat([self.buffer_y_cls[i] for i in replay_indices], dim=0)
            replay_y_reg = torch.cat([self.buffer_y_reg[i] for i in replay_indices], dim=0)

            new_x_actual = new_x[:n_new_actual]
            new_y_cls_actual = new_y_cls[:n_new_actual]
            new_y_reg_actual = new_y_reg[:n_new_actual]

            x_combined = torch.cat([new_x_actual, replay_x], dim=0)
            y_cls_combined = torch.cat([new_y_cls_actual, replay_y_cls], dim=0)
            y_reg_combined = torch.cat([new_y_reg_actual, replay_y_reg], dim=0)

            return x_combined, y_cls_combined, y_reg_combined
        else:
            return new_x[:batch_size], new_y_cls[:batch_size], new_y_reg[:batch_size]

    def incremental_update(self, new_start_date, new_end_date,
                           epochs=3,
                           batch_size=32,
                           lr=1e-5,
                           validation_dates=None,
                           patience=2):
        print(f"\n{'='*60}")
        print(f"[INCREMENTAL UPDATE] New data: {new_start_date} to {new_end_date}")
        print(f"  Buffer size: {len(self.buffer_x)}")
        print(f"  EWC lambda: {self.ewc_lambda}")
        print(f"  Replay ratio: {self.replay_ratio}")
        print(f"{'='*60}")

        if len(self.buffer_x) > 0 and not self.fisher:
            print("[STEP 1] Computing EWC Fisher on old data...")
            old_dataloader = self._create_dataloader_from_buffer(batch_size)
            self.compute_fisher(old_dataloader)

        new_gen = optimized_batch_generator(
            self.loader, self.pipeline, new_start_date, new_end_date, self.global_df,
            window=48, batch_size=batch_size, fit_scaler_first=False, is_train=True
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=0.5
        )

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            for x_batch, y_cls, y_reg, _ in new_gen:
                x_combined, y_cls_combined, y_reg_combined = self.get_combined_batch(
                    x_batch, y_cls, y_reg, batch_size
                )

                if len(x_combined) < 8:
                    continue

                x_combined = x_combined.to(self.model.classifier.weight.device)
                y_cls_combined = y_cls_combined.to(self.model.classifier.weight.device)
                y_reg_combined = y_reg_combined.to(self.model.classifier.weight.device)
                logits, reg = self.model(x_combined)

                loss_cls = nn.BCEWithLogitsLoss()(logits, y_cls_combined)
                loss_reg = nn.HuberLoss()(reg, y_reg_combined)

                loss = loss_cls + 1.5 * loss_reg

                if self.fisher:
                    loss += self.ewc_loss(self.model)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print("[STEP 5] Updating buffer with new samples...")
        self._add_new_samples_to_buffer(new_start_date, new_end_date)

        if validation_dates:
            val_metrics = self._validate(validation_dates)
            print(f"[STEP 6] Validation AUC: {val_metrics.get('auc', 0):.4f}")
        else:
            val_metrics = None

        self.update_history.append({
            'date': datetime.now().isoformat(),
            'new_data_period': f"{new_start_date} to {new_end_date}",
            'epochs': epochs,
            'final_loss': best_loss,
            'validation_metrics': val_metrics,
            'buffer_size': len(self.buffer_x)
        })

        print(f"[INCREMENTAL UPDATE] Completed. Final loss: {best_loss:.4f}")
        return val_metrics

    def _create_dataloader_from_buffer(self, batch_size):
        if len(self.buffer_x) == 0:
            return None

        class BufferDataset(torch.utils.data.Dataset):
            def __init__(self, buffer_x, buffer_y_cls, buffer_y_reg):
                self.x = torch.cat(list(buffer_x), dim=0)
                self.y_cls = torch.cat(list(buffer_y_cls), dim=0)
                self.y_reg = torch.cat(list(buffer_y_reg), dim=0)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y_cls[idx], self.y_reg[idx]

        dataset = BufferDataset(self.buffer_x, self.buffer_y_cls, self.buffer_y_reg)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _add_new_samples_to_buffer(self, start_date, end_date, max_samples=500):
        gen = optimized_batch_generator(
            self.loader, self.pipeline, start_date, end_date, self.global_df,
            window=48, batch_size=32, fit_scaler_first=False, is_train=True
        )

        samples_added = 0
        for x_batch, y_cls, y_reg, _ in gen:
            step = max(1, len(x_batch) // 10)
            for i in range(0, len(x_batch), step):
                if len(self.buffer_x) >= self.buffer_size:
                    self.buffer_x.popleft()
                    self.buffer_y_cls.popleft()
                    self.buffer_y_reg.popleft()

                self.buffer_x.append(x_batch[i:i+1].cpu())
                self.buffer_y_cls.append(y_cls[i:i+1].cpu())
                self.buffer_y_reg.append(y_reg[i:i+1].cpu())
                samples_added += 1

                if samples_added >= max_samples:
                    break

            if samples_added >= max_samples:
                break

        print(f"[BUFFER] Added {samples_added} new samples, total={len(self.buffer_x)}")
        return samples_added

    def _validate(self, validation_dates):
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        all_true = []
        all_pred = []

        for start_date, end_date in validation_dates:
            gen = optimized_batch_generator(
                self.loader, self.pipeline, start_date, end_date, self.global_df,
                window=48, batch_size=32, fit_scaler_first=False, is_train=False
            )

            with torch.no_grad():
                for x_batch, y_cls, _, _ in gen:
                    logits, _ = self.model(x_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_true.extend(y_cls.cpu().numpy().flatten())
                    all_pred.extend(probs.flatten())

        if len(all_true) > 0 and len(np.unique(all_true)) > 1:
            auc = roc_auc_score(all_true, all_pred)
            return {'auc': auc, 'n_samples': len(all_true)}
        return None


class UpdateScheduler:
    def __init__(self, incremental_learner, model_path='incremental_model.pt'):
        self.learner = incremental_learner
        self.model_path = model_path
        self.last_update = None

    def needs_update(self, current_date, update_interval_days=7):
        if self.last_update is None:
            return True

        days_since_update = (current_date - self.last_update).days
        return days_since_update >= update_interval_days

    def run_update(self, new_start_date, new_end_date,
                   validation_dates=None,
                   save_if_improved=True):
        new_data_exists = self._check_new_data_exists(new_start_date, new_end_date)
        if not new_data_exists:
            print(f"[SCHEDULER] No new data in {new_start_date} to {new_end_date}")
            return False

        old_state = deepcopy(self.learner.model.state_dict())

        val_metrics = self.learner.incremental_update(
            new_start_date, new_end_date, validation_dates=validation_dates
        )

        if save_if_improved and val_metrics:
            old_auc = self._get_last_auc()
            if val_metrics['auc'] > old_auc:
                self.learner.save_checkpoint(self.model_path)
                print(f"[SCHEDULER] Model saved (AUC improved: {old_auc:.4f} -> {val_metrics['auc']:.4f})")
            else:
                self.learner.model.load_state_dict(old_state)
                print(f"[SCHEDULER] Rolled back (AUC decreased: {old_auc:.4f} -> {val_metrics['auc']:.4f})")
        else:
            self.learner.save_checkpoint(self.model_path)

        self.last_update = datetime.now()
        return True

    def _check_new_data_exists(self, start_date, end_date):
        all_files = self.learner.loader.get_all_files()
        date_str = start_date.replace('-', '/')
        for f in all_files:
            if date_str in f and f.endswith('.h5'):
                return True
        return False

    def _get_last_auc(self):
        if self.learner.update_history:
            last = self.learner.update_history[-1]
            if last.get('validation_metrics'):
                return last['validation_metrics'].get('auc', 0.5)
        return 0.5