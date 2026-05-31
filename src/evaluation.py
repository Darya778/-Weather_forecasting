# Оценка модели
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    mean_squared_error, mean_absolute_error, precision_score, recall_score
)
import json
from generator import optimized_batch_generator


def evaluate_model(model, loader, pipeline, start_date, end_date, global_df=None,
                   window=32, batch_size=16, fixed_threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_true, all_pred = [], []
    all_reg_true, all_reg_pred = [], []

    gen = optimized_batch_generator(loader, pipeline, start_date, end_date, global_df,
                                    window, batch_size, fit_scaler_first=False,
                                    is_train=False, fixed_threshold=fixed_threshold)

    with torch.no_grad():
        for x_batch, y_cls, y_reg, region_ids in gen:
            logits_last, reg_last = model(x_batch, region_ids)

            logits_last = torch.clamp(logits_last, -10, 10)
            probs = torch.sigmoid(logits_last)

            all_true.extend(y_cls.cpu().numpy().flatten())
            all_pred.extend(probs.cpu().numpy().flatten())
            all_reg_true.extend(y_reg.cpu().numpy().squeeze(1))
            all_reg_pred.extend(reg_last.cpu().numpy().squeeze(1))

    print(f"Total samples in evaluation: {len(all_true)}")

    if len(all_true) == 0:
        print("! WARNING: No samples found in evaluation!")
        return {"auc": 0.0, "f1": 0.0, "accuracy": 0.0}, pd.DataFrame()

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_reg_true = np.array(all_reg_true)
    all_reg_pred = np.array(all_reg_pred)

    print("\n[SCALER CHECK] Before inverse_transform:")
    for i, name in enumerate(["temp", "wind", "humidity", "rain"]):
        if len(all_reg_pred) > 0:
            print(f"  {name} (scaled): min={all_reg_pred[:, i].min():.3f}, "
                  f"max={all_reg_pred[:, i].max():.3f}, std={all_reg_pred[:, i].std():.6f}")
            if np.std(all_reg_pred[:, i]) < 1e-6:
                print(f"[WARNING] {name} predictions collapsed to constant!")

    temp_pred = pipeline.scaler_temp.inverse_transform(all_reg_pred[:, [0]])
    wind_pred = pipeline.scaler_wind.inverse_transform(all_reg_pred[:, [1]])
    humidity_pred = pipeline.scaler_humidity.inverse_transform(all_reg_pred[:, [2]])
    rain_pred = pipeline.scaler_rain.inverse_transform(all_reg_pred[:, [3]])
    pm10_pred = pipeline.scaler_pm10.inverse_transform(all_reg_pred[:, [4]])
    pm25_pred = pipeline.scaler_pm25.inverse_transform(all_reg_pred[:, [5]])

    temp_true = pipeline.scaler_temp.inverse_transform(all_reg_true[:, [0]])
    wind_true = pipeline.scaler_wind.inverse_transform(all_reg_true[:, [1]])
    humidity_true = pipeline.scaler_humidity.inverse_transform(all_reg_true[:, [2]])
    rain_true = pipeline.scaler_rain.inverse_transform(all_reg_true[:, [3]])
    pm10_true = pipeline.scaler_pm10.inverse_transform(all_reg_true[:, [4]])
    pm25_true = pipeline.scaler_pm25.inverse_transform(all_reg_true[:, [5]])

    temp_pred = np.clip(temp_pred, -50, 50)
    wind_pred = np.clip(wind_pred, 0, 60)
    humidity_pred = np.clip(humidity_pred, 0, 100)
    rain_pred = np.clip(rain_pred, 0, 100)
    pm10_pred = np.clip(pm10_pred, 0, 500)
    pm25_pred = np.clip(pm25_pred, 0, 300)

    all_reg_pred_original = np.concatenate([temp_pred, wind_pred, humidity_pred, rain_pred, pm10_pred, pm25_pred], axis=1)
    all_reg_true_original = np.concatenate([temp_true, wind_true, humidity_true, rain_true, pm10_true, pm25_true], axis=1)

    df_preds = pd.DataFrame({
        "true": all_true,
        "pred": all_pred,
        "true_temp": all_reg_true_original[:, 0],
        "true_wind": all_reg_true_original[:, 1],
        "true_humidity": all_reg_true_original[:, 2],
        "true_rain": all_reg_true_original[:, 3],
        "true_pm10": all_reg_true_original[:, 4],
        "true_pm25": all_reg_true_original[:, 5],
        "pred_temp": all_reg_pred_original[:, 0],
        "pred_wind": all_reg_pred_original[:, 1],
        "pred_humidity": all_reg_pred_original[:, 2],
        "pred_rain": all_reg_pred_original[:, 3],
        "pred_pm10": all_reg_pred_original[:, 4],
        "pred_pm25": all_reg_pred_original[:, 5],
    })

    best_threshold = 0.5
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    for thresh in np.arange(0.3, 0.8, 0.01):
        pred_temp = (all_pred > thresh).astype(int)
        if len(np.unique(pred_temp)) < 2:
            continue
        f1_temp = f1_score(all_true, pred_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = thresh
            best_precision = precision_score(all_true, pred_temp)
            best_recall = recall_score(all_true, pred_temp)

    print(f"\n[OPTIMAL THRESHOLD FOUND]")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  F1-Score: {best_f1:.4f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall: {best_recall:.4f}")

    threshold = best_threshold
    binary_pred = (all_pred > threshold).astype(int)
    binary_true = all_true.astype(int)

    print(f"True labels distribution: 0={(binary_true==0).sum()}, 1={(binary_true==1).sum()}")
    print(f"Pred labels distribution: 0={(binary_pred==0).sum()}, 1={(binary_pred==1).sum()}")

    if len(np.unique(binary_true)) < 2:
        auc_score = 0.5
        f1 = best_f1
        accuracy = accuracy_score(binary_true, binary_pred)
    else:
        auc_score = roc_auc_score(binary_true, all_pred)
        f1 = f1_score(binary_true, binary_pred)
        accuracy = accuracy_score(binary_true, binary_pred)

    rmse = np.sqrt(mean_squared_error(
        all_reg_true.reshape(-1),
        all_reg_pred.reshape(-1)
    ))
    mae = mean_absolute_error(
        all_reg_true.reshape(-1),
        all_reg_pred.reshape(-1)
    )

    metrics = {
        "auc": float(auc_score),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "best_threshold": float(best_threshold),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "rmse": float(rmse),
        "mae": float(mae)
    }

    print(f"\n[FINAL METRICS]")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Best Threshold: {best_threshold:.3f}")
    print(f"\n[REGRESSION METRICS]")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    df_preds.to_csv("predictions.csv", index=False)

    return metrics, df_preds