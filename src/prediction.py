# Предсказания и визуализация
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score
from config import RAIN_WEIGHT, WIND_WEIGHT, TEMP_WEIGHT, PRESSURE_WEIGHT, HUMIDITY_WEIGHT, PM10_WEIGHT, PM25_WEIGHT
from generator import optimized_batch_generator


class WeatherPredictor:
    def __init__(self, model, pipeline, loader, global_df=None, threshold=0.45, fixed_threshold=None):
        self.model = model
        self.pipeline = pipeline
        self.loader = loader
        self.global_df = global_df
        self.threshold = threshold
        self.fixed_threshold = fixed_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.region_id = None
        if pipeline.lat is not None and pipeline.lon is not None:
            self.region_id = torch.tensor([hash((pipeline.lat, pipeline.lon)) % 100], dtype=torch.long).to(self.device)

    def predict_from_h5(self, h5_key, step=1):
        raw = self.loader._read_h5(h5_key)
        time_str = self.loader._extract_time(h5_key)

        date_match = re.search(r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})', h5_key)
        if date_match:
            date_part = date_match.group(1).replace('/', '-')
        else:
            date_part = datetime.now().strftime("%Y-%m-%d")

        start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")
        df = self.loader._to_dataframe(raw, start_dt)
        if df.empty:
            return None
        df = self.pipeline.preprocess(df)

        if self.global_df is not None and not self.global_df.empty:
            df = df.merge(self.global_df, left_index=True, right_index=True, how='left')
            df = df.ffill().fillna(0)

        df = self.pipeline.create_features(df)
        df = self.pipeline.remove_leakage(df)
        X, _, _ = self.pipeline.transform(df, require_target=False)

        if X is None or len(X) == 0:
            return None

        predictions = []
        timestamps = df.index
        window_size = 32

        for i in range(window_size - 1, len(X), step):
            x_window = X[i - window_size + 1:i + 1]
            x_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_last, reg_last = self.model(x_tensor, self.region_id)
                prob = torch.sigmoid(logits_last).cpu().numpy()[0, 0]
            predictions.append({
                'timestamp': timestamps[i],
                'probability': prob,
                'prediction': 1 if prob > self.threshold else 0
            })

        return pd.DataFrame(predictions)

    def predict_sequence(self, df, horizon_minutes=10):
        df = self.pipeline.preprocess(df)

        if self.global_df is not None and not self.global_df.empty:
            df = df.merge(self.global_df, left_index=True, right_index=True, how='left')
            df = df.ffill().fillna(0)
        df = self.pipeline.create_features(df)
        df = self.pipeline.remove_leakage(df)
        X, _, _ = self.pipeline.transform(df, require_target=False)

        if X is None or len(X) < 32:
            print(f"Предупреждение: недостаточно данных - нужно минимум 32 записи, есть {len(X) if X is not None else 0}")
            return None

        window = X[-32:]
        current_timestamp = df.index[-1]
        predictions = []
        current_window = window.copy()

        for step in range(horizon_minutes):
            x_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits_last, reg_last = self.model(x_tensor, self.region_id)
                prob = torch.sigmoid(logits_last).cpu().numpy()[0, 0]
                reg_pred = reg_last.cpu().numpy().reshape(-1, 6)[0]

            pred_time = current_timestamp + timedelta(minutes=step + 1)
            temp_raw = self.pipeline.scaler_temp.inverse_transform([[reg_pred[0]]])[0, 0]
            wind_raw = self.pipeline.scaler_wind.inverse_transform([[reg_pred[1]]])[0, 0]
            humidity_raw = self.pipeline.scaler_humidity.inverse_transform([[reg_pred[2]]])[0, 0]
            rain_raw = self.pipeline.scaler_rain.inverse_transform([[reg_pred[3]]])[0, 0]
            pm10_raw = self.pipeline.scaler_pm10.inverse_transform([[reg_pred[4]]])[0, 0]
            pm25_raw = self.pipeline.scaler_pm25.inverse_transform([[reg_pred[5]]])[0, 0]

            new_row = current_window[-1].copy()
            if new_row.shape[-1] >= 6:
                feature_names = self.pipeline.fixed_columns
                for i, name in enumerate(feature_names):
                    name_lower = name.lower()
                    if "temperature" in name_lower and "humidity" not in name_lower and "air" not in name_lower:
                        new_row[i] = temp_raw
                    elif "speed" in name_lower and "sound" not in name_lower:
                        new_row[i] = wind_raw
                    elif "humidity" in name_lower and "temp" not in name_lower:
                        new_row[i] = humidity_raw
                    elif "rain" in name_lower or "precip" in name_lower:
                        new_row[i] = rain_raw
                    elif "pm10" in name_lower or "pm_10" in name_lower:
                        new_row[i] = pm10_raw
                    elif "pm25" in name_lower or "pm_25" in name_lower or "pm2.5" in name_lower:
                        new_row[i] = pm25_raw
            else:
                new_row = current_window[-1].copy()
            current_window = np.vstack([current_window[1:], new_row])

            predictions.append({
                'minutes_ahead': step + 1,
                'timestamp': pred_time,
                'probability': prob,
                'prediction': 1 if prob > self.threshold else 0,
                'risk_level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW',
                'temp': float(temp_raw),
                'wind': float(wind_raw),
                'humidity': float(humidity_raw),
                'rain': float(rain_raw),
                'pm10': float(pm10_raw),
                'pm25': float(pm25_raw)
            })

        return pd.DataFrame(predictions)


def visualize_predictions(df_predictions, save_path='predictions_plot.png', max_points=500):
    if df_predictions is None or df_predictions.empty:
        print("Нет данных для визуализации")
        return

    n = min(max_points, len(df_predictions))
    df_plot = df_predictions.iloc[:n]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    ax1 = axes[0]
    ax1.plot(df_plot['timestamp'], df_plot['probability'], 'b-', linewidth=2, label='Probability')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='High risk (0.7)')
    ax1.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='Low risk (0.3)')
    ax1.fill_between(df_plot['timestamp'], 0, df_plot['probability'], alpha=0.3, color='blue')
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_title(f'Weather Risk Predictions (first {n} points)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2 = axes[1]
    colors = ['red' if p > 0.7 else 'orange' if p > 0.3 else 'green' for p in df_plot['probability']]
    ax2.bar(df_plot['timestamp'], df_plot['prediction'], color=colors, alpha=0.7, width=0.02)
    ax2.set_ylabel('Prediction (0=Safe, 1=Risk)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_title('Binary Predictions with Risk Levels', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Визуализация сохранена в {save_path}")


def predict_and_visualize_example(model, loader, pipeline, global_df,
                                  threshold=0.45, fixed_threshold=None, step=10):
    print("ПРИМЕР ПРЕДСКАЗАНИЯ ПОГОДНЫХ УСЛОВИЙ (Transformer)")

    all_files = loader._list_files("")
    h5_files = [f for f in all_files if f.endswith('.h5')]

    if not h5_files:
        print("Нет HDF5 файлов для предсказания")
        return None

    latest_file = sorted(h5_files)[-1]
    print(f"\nИспользуем файл: {latest_file}")

    predictor = WeatherPredictor(model, pipeline, loader, global_df, threshold=threshold, fixed_threshold=fixed_threshold)

    print("\nПредсказание на основе исторических данных...")
    df_pred = predictor.predict_from_h5(latest_file, step=step)

    if df_pred is not None and not df_pred.empty:
        print(f"\nСделано {len(df_pred)} предсказаний")
        print("\nПоследние 5 предсказаний:")
        print(df_pred.tail(5).to_string())

        visualize_predictions(df_pred, 'latest_predictions.png')

        risk_dist = df_pred['prediction'].value_counts()
        print(f"\nСтатистика предсказаний:")
        print(f"  - Без риска: {risk_dist.get(0, 0)} ({risk_dist.get(0, 0)/len(df_pred)*100:.1f}%)")
        print(f"  - С риском: {risk_dist.get(1, 0)} ({risk_dist.get(1, 0)/len(df_pred)*100:.1f}%)")

        high_risk = (df_pred['probability'] > 0.7).sum()
        print(f"  - Высокий риск (>0.7): {high_risk} предсказаний")
        print(f"  - Используемый порог: {threshold:.3f}")

        return df_pred
    else:
        print("Не удалось сделать предсказания")
        return None


def predict_future_minutes(model, loader, pipeline, global_df,
                          minutes_ahead=10, threshold=0.45,
                          fixed_threshold=None):
    print(f"\nПРОГНОЗ НА {minutes_ahead} МИНУТ ВПЕРЕД (Transformer, вероятностная модель)")

    all_files = loader._list_files("")
    h5_files = [f for f in all_files if f.endswith('.h5')]

    if not h5_files:
        print("Нет данных для предсказания")
        return None

    latest_file = sorted(h5_files)[-1]
    print(f"Источник данных: {latest_file}")

    raw = loader._read_h5(latest_file)
    time_str = loader._extract_time(latest_file)

    date_match = re.search(r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})', latest_file)
    if date_match:
        date_part = date_match.group(1).replace('/', '-')
    else:
        date_part = datetime.now().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")
    df = loader._to_dataframe(raw, start_dt)

    if df.empty:
        print("Не удалось загрузить данные")
        return None
    print(f"Данные: {len(df)} записей ({df['timestamp'].min()} — {df['timestamp'].max()})")
    df = pipeline.preprocess(df)

    if global_df is not None and not global_df.empty:
        df = df.merge(global_df, left_index=True, right_index=True, how='left')
        df = df.ffill().fillna(0)

    df = pipeline.create_features(df)
    df = pipeline.remove_leakage(df)
    X, _, _ = pipeline.transform(df, require_target=False)
    if X is None or len(X) < 32:
        print("Недостаточно данных")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region_id = None
    if pipeline.lat is not None and pipeline.lon is not None:
        region_id = torch.tensor([hash((pipeline.lat, pipeline.lon)) % 100],
                                 dtype=torch.long).to(device)

    window = X[-32:]
    current_timestamp = df.index[-1]
    predictions = []
    current_window = window.copy()

    for step in range(minutes_ahead):
        x_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_last, reg_last = model(x_tensor, region_id)

            prob = torch.sigmoid(logits_last).cpu().numpy()[0, 0]
            reg_pred = reg_last.detach().cpu().numpy().squeeze()

        temp = pipeline.scaler_temp.inverse_transform([[reg_pred[0]]])[0, 0]
        wind = pipeline.scaler_wind.inverse_transform([[reg_pred[1]]])[0, 0]
        humidity = pipeline.scaler_humidity.inverse_transform([[reg_pred[2]]])[0, 0]
        rain = pipeline.scaler_rain.inverse_transform([[reg_pred[3]]])[0, 0]
        pm10 = pipeline.scaler_pm10.inverse_transform([[reg_pred[4]]])[0, 0]
        pm25 = pipeline.scaler_pm25.inverse_transform([[reg_pred[5]]])[0, 0]

        temp_pred = np.clip(temp, -50, 50)
        wind_pred = np.clip(wind, 0, 60)
        humidity_pred = np.clip(humidity, 0, 100)
        rain_pred = np.clip(rain, 0, 100)
        pm10_pred = np.clip(pm10, 0, 500)
        pm25_pred = np.clip(pm25, 0, 300)

        if prob > 0.7:
            risk_level = "ВЫСОКИЙ"
        elif prob > 0.3:
            risk_level = "СРЕДНИЙ"
        else:
            risk_level = "НИЗКИЙ"

        contributions = {
            "осадки": rain_pred * RAIN_WEIGHT,
            "ветер": wind_pred * WIND_WEIGHT,
            "температура": abs(temp_pred) * TEMP_WEIGHT,
            "влажность": abs(humidity_pred - 50) * HUMIDITY_WEIGHT,
            "PM10": pm10_pred * PM10_WEIGHT,
            "PM2.5": pm25_pred * PM25_WEIGHT
        }

        main_factor = max(contributions, key=contributions.get)
        explanation = f"Основной фактор риска: {main_factor}"

        if prob > 0.7:
            recommendation = "Рекомендуется принять меры (возможны неблагоприятные условия)"
        elif prob > 0.3:
            recommendation = "Условия нестабильны, требуется наблюдение"
        else:
            recommendation = "Погодные условия стабильны"

        pred_time = current_timestamp + timedelta(minutes=step + 1)

        predictions.append({
            'minutes_ahead': step + 1,
            'timestamp': pred_time,
            'probability': prob,
            'risk_level': risk_level,
            'temp': temp_pred,
            'wind': wind_pred,
            'humidity': humidity_pred,
            'rain': rain_pred,
            'pm10': pm10_pred,
            'pm25': pm25_pred,
            'main_factor': main_factor
        })

        print(f"\nЧерез {step+1:2d} мин ({pred_time.strftime('%H:%M:%S')}):")
        print(f"  Риск: {prob*100:.1f}% ({risk_level})")
        print(f"  Прогноз:")
        print(f"    Температура: {temp_pred:.2f} °C")
        print(f"    Ветер: {wind_pred:.2f} м/с")
        print(f"    Влажность: {humidity_pred:.1f}%")
        print(f"    Осадки: {rain_pred:.2f} мм")
        print(f"    PM10: {pm10_pred:.1f} мкг/м³")
        print(f"    PM2.5: {pm25_pred:.1f} мкг/м³")
        print(f"  {explanation}")
        print(f"  Рекомендация: {recommendation}")

        new_row = current_window[-1].copy()
        if new_row.shape[-1] >= 4:
            feature_names = pipeline.fixed_columns
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if "temperature" in name_lower and "humidity" not in name_lower and "air" not in name_lower:
                    new_row[i] = temp_pred
                elif "speed" in name_lower and "sound" not in name_lower:
                    new_row[i] = wind_pred
                elif "humidity" in name_lower and "temp" not in name_lower:
                    new_row[i] = humidity_pred
                elif "rain" in name_lower or "precip" in name_lower:
                    new_row[i] = rain_pred
        else:
            new_row = current_window[-1].copy()
        current_window = np.vstack([current_window[1:], new_row])

    return pd.DataFrame(predictions)


def plot_regression_vs_real(y_true, y_pred, title="Regression: Predicted vs Real", max_points=500):
    n = min(max_points, len(y_true))
    y_true_plot = y_true[:n]
    y_pred_plot = y_pred[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(y_true_plot, label="Real", alpha=0.7, linewidth=1.5)
    axes[0].plot(y_pred_plot, label="Predicted", alpha=0.7, linewidth=1.5)
    axes[0].legend(fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].set_xlabel("Sample index", fontsize=12)
    axes[0].set_ylabel("Temperature (°C)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    n_scatter = min(1000, len(y_true))
    axes[1].scatter(y_true[:n_scatter], y_pred[:n_scatter], alpha=0.5, s=10)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    axes[1].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='Ideal fit')
    axes[1].set_xlabel("Real temperature (°C)", fontsize=12)
    axes[1].set_ylabel("Predicted temperature (°C)", fontsize=12)
    axes[1].set_title("Scatter Plot: Real vs Predicted", fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_weather_params(y_true, y_pred, max_points=2000, x_label="Sample index", save_path=None):
    feature_names = ["Temperature", "Wind Speed", "Humidity", "Precipitation", "PM10", "PM2.5"]
    units = ["°C", "m/s", "%", "mm", "µg/m³", "µg/m³"]
    n = min(max_points, len(y_true))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_features = min(6, y_true.shape[1])

    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4 * n_features))

    if n_features == 1:
        axes = [axes]

    for i in range(n_features):
        axes[i].plot(y_true[:n, i], label="Real", alpha=0.7, linewidth=1.5, color='blue')
        axes[i].plot(y_pred[:n, i], label="Predicted", alpha=0.7, linewidth=1.5, color='red')
        axes[i].set_title(f"{feature_names[i]} (first {n} points)", fontsize=13)
        axes[i].set_xlabel(x_label, fontsize=11)
        axes[i].set_ylabel(f"{feature_names[i]} ({units[i]})", fontsize=11)
        axes[i].legend(fontsize=11)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён в {save_path}")

    plt.show()


def plot_classification_vs_real(df_preds, threshold=0.5, start_idx=0, max_points=500,
                                x_label="Sample index", time_per_point=1):
    end_idx = min(start_idx + max_points, len(df_preds))
    df_plot = df_preds.iloc[start_idx:end_idx]
    n = len(df_plot)

    time_axis = np.arange(start_idx, start_idx + n) * time_per_point

    plt.figure(figsize=(14, 6))
    plt.plot(time_axis, df_plot['pred'].values,
             label='Predicted probability', alpha=0.7, linewidth=1.5)
    plt.plot(time_axis, df_plot['true'].values,
             label='True label', alpha=0.7, linewidth=1.5, drawstyle='steps-post')
    plt.axhline(threshold, linestyle='--', color='red', label=f'Threshold ({threshold:.2f})')
    plt.legend(fontsize=12)
    plt.title(f"Classification: Prediction vs Real (points {start_idx}–{end_idx})", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Probability / Label", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_dashboard(df_predictions, save_path="weather_dashboard.png"):
    if df_predictions is None or df_predictions.empty:
        print("Нет данных для визуализации")
        return

    plt.figure(figsize=(16, 12))
    timestamps = df_predictions['timestamp']

    ax1 = plt.subplot(3, 1, 1)
    probs = df_predictions['probability']
    ax1.plot(timestamps, probs, 'b-', linewidth=2, label="Probability")
    ax1.fill_between(timestamps, 0, probs, where=(probs <= 0.3), alpha=0.3, color='green', label="Low risk")
    ax1.fill_between(timestamps, 0, probs, where=((probs > 0.3) & (probs <= 0.7)), alpha=0.3, color='orange', label="Medium risk")
    ax1.fill_between(timestamps, 0, probs, where=(probs > 0.7), alpha=0.3, color='red', label="High risk")
    ax1.axhline(0.3, linestyle="--", color='green', label="Low threshold (0.3)")
    ax1.axhline(0.7, linestyle="--", color='red', label="High threshold (0.7)")
    ax1.set_title("Weather Risk Probability", fontsize=14)
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2 = plt.subplot(3, 1, 2)
    colors = ['green' if p <= 0.3 else 'orange' if p <= 0.7 else 'red' for p in probs]
    ax2.bar(timestamps, probs, color=colors, alpha=0.7, width=0.02)
    ax2.set_title("Risk Levels (color-coded)", fontsize=14)
    ax2.set_ylabel("Risk probability", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    ax3 = plt.subplot(3, 2, 5)
    ax3.plot(timestamps, df_predictions['temp'], 'r-', linewidth=1.5, label="Temperature")
    ax3.set_title("Temperature", fontsize=13)
    ax3.set_ylabel("Temperature (°C)", fontsize=11)
    ax3.set_xlabel("Time", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 2, 6)
    ax4.plot(timestamps, df_predictions['wind'], 'b-', linewidth=1.5, label="Wind speed")
    ax4.set_title("Wind speed", fontsize=13)
    ax4.set_ylabel("Wind speed (m/s)", fontsize=11)
    ax4.set_xlabel("Time", fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    trend = "↑ растет" if probs.iloc[-1] > probs.iloc[0] else "↓ падает"
    print(f"Тренд риска: {trend}")
    print(f"Дашборд сохранен: {save_path}")

    plt.figure(figsize=(16, 4))

    ax5 = plt.subplot(1, 2, 1)
    ax5.plot(timestamps, df_predictions['humidity'], 'g-', linewidth=1.5, label="Humidity")
    ax5.set_title("Humidity", fontsize=13)
    ax5.set_ylabel("Humidity (%)", fontsize=11)
    ax5.set_xlabel("Time", fontsize=11)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(1, 2, 2)
    ax6.plot(timestamps, df_predictions['rain'], 'c-', linewidth=1.5, label="Precipitation")
    ax6.set_title("Precipitation", fontsize=13)
    ax6.set_ylabel("Precipitation (mm)", fontsize=11)
    ax6.set_xlabel("Time", fontsize=11)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, val_losses=None, save_path="training_history.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8, label='Training Loss')

    if val_losses is not None and len(val_losses) > 0:
        plt.plot(epochs, val_losses, 'r-o', linewidth=2, markersize=8, label='Validation Loss')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    for i, loss in enumerate(train_losses):
        plt.annotate(f'{loss:.4f}', (epochs[i], loss), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"График обучения сохранен в {save_path}")


def plot_auc_history(auc_scores, save_path="auc_history.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(auc_scores) + 1)
    plt.plot(epochs, auc_scores, 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('AUC по эпохам', fontsize=14)
    plt.grid(True, alpha=0.3)
    for i, auc in enumerate(auc_scores):
        plt.annotate(f'{auc:.4f}', (epochs[i], auc), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_feature_importance(model, loader, pipeline, start_date, end_date, global_df,
                               window=32, batch_size=16, n_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    gen = optimized_batch_generator(loader, pipeline, start_date, end_date, global_df,
                                    window, batch_size, fit_scaler_first=False, is_train=False)

    X_all, y_all = [], []
    with torch.no_grad():
        for x_batch, y_cls, _, _ in gen:
            X_all.append(x_batch.cpu().numpy())
            y_all.append(y_cls.cpu().numpy())
            if sum(len(x) for x in X_all) >= n_samples:
                break

    X_base = np.concatenate(X_all, axis=0)[:n_samples]
    y_base = np.concatenate(y_all, axis=0)[:n_samples].flatten()

    if pipeline.lat is not None and pipeline.lon is not None:
        region_id_value = hash((pipeline.lat, pipeline.lon)) % 100
    else:
        region_id_value = 0

    region_ids = torch.tensor([region_id_value] * len(X_base), dtype=torch.long).to(device)

    with torch.no_grad():
        x_tensor = torch.tensor(X_base).to(device)
        logits, _ = model(x_tensor, region_ids)
        probs_base = torch.sigmoid(logits).cpu().numpy().flatten()

    if len(np.unique(y_base)) < 2:
        print("[WARNING] Less than 2 classes in sample, skipping feature importance")
        return pd.DataFrame()

    auc_base = roc_auc_score(y_base, probs_base)
    print(f"Baseline AUC: {auc_base:.4f}")

    feature_names = pipeline.fixed_columns
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_base.shape[2])]

    importance = []

    for i in range(X_base.shape[2]):
        X_perm = X_base.copy()
        np.random.shuffle(X_perm[:, :, i])

        with torch.no_grad():
            x_tensor = torch.tensor(X_perm).to(device)
            logits, _ = model(x_tensor, region_ids)
            probs_perm = torch.sigmoid(logits).cpu().numpy().flatten()

        auc_perm = roc_auc_score(y_base, probs_perm)
        imp = auc_base - auc_perm
        importance.append({"feature": feature_names[i], "importance": imp})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{X_base.shape[2]} features...")

    df_imp = pd.DataFrame(importance).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 8))
    top_n = min(30, len(df_imp))
    top_features = df_imp.head(top_n)
    plt.barh(range(len(top_features)), top_features["importance"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("AUC drop (importance)", fontsize=12)
    plt.title(f"Permutation Feature Importance (top-{top_n})", fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()

    not_important = df_imp[df_imp["importance"] <= 0]
    print(f"\n[FEATURE IMPORTANCE]")
    print(f"  Baseline AUC: {auc_base:.4f}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Important (>0): {len(df_imp[df_imp['importance'] > 0])}")
    print(f"  Not important: {len(not_important)}")
    if len(not_important) > 0:
        print(f"  Not important features (first 10): {list(not_important['feature'].values[:10])}")

    return df_imp


def print_data_summary(loader, pipeline, global_df, start_date, end_date, train_end, test_start, test_end,
                       horizon, window, epochs, batch_size):
    print(f"\n{'='*60}")
    print(f" СВОДКА ЗАГРУЖЕННЫХ ДАННЫХ")

    if global_df is not None and not global_df.empty:
        print(f"\n Глобальные данные (Open-Meteo):")
        print(f"   Строк: {len(global_df)}")
        print(f"   Период: {global_df.index.min()} — {global_df.index.max()}")
        print(f"   Колонки: {list(global_df.columns)}")
        for col in global_df.columns:
            vals = global_df[col].dropna()
            if len(vals) > 0:
                print(f"   • {col}: mean={vals.mean():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")
    else:
        print(f"\n Глобальные данные: НЕ ЗАГРУЖЕНЫ")

    all_files = loader.get_all_files()
    h5_files = [f for f in all_files if f.endswith('.h5') and 'bad_region' not in f]

    all_date_counts = {}
    for f in h5_files:
        date_match = re.search(r'(\d{4}[/\-]\d{2}[/\-]\d{2})', f)
        if date_match:
            date_str = date_match.group(1)
            all_date_counts[date_str] = all_date_counts.get(date_str, 0) + 1

    dates = pd.date_range(start=start_date, end=end_date)
    date_strs = set()
    for d in dates:
        date_strs.add(d.strftime("%Y/%m/%d"))
        date_strs.add(d.strftime("%Y-%m-%d"))
        date_strs.add(d.strftime("%Y%m%d"))

    matching_files = []
    for f in h5_files:
        for ds in date_strs:
            if ds in f:
                matching_files.append(f)
                break

    matching_date_counts = {}
    for f in matching_files:
        date_match = re.search(r'(\d{4}[/\-]\d{2}[/\-]\d{2})', f)
        if date_match:
            date_str = date_match.group(1)
            matching_date_counts[date_str] = matching_date_counts.get(date_str, 0) + 1

    print(f"\n Локальные H5 файлы:")
    print(f"   Всего в бакете: {len(all_files)}")
    print(f"   Валидных H5: {len(h5_files)}")
    print(f"   Диапазон дат в бакете: {min(all_date_counts.keys()) if all_date_counts else 'N/A'} — {max(all_date_counts.keys()) if all_date_counts else 'N/A'}")
    print(f"    Подходит под запрос ({start_date}—{end_date}): {len(matching_files)} файлов")

    if len(matching_files) == 0:
        print(f"   ! ВНИМАНИЕ: Нет файлов за запрашиваемый период!")
    elif matching_date_counts:
        print(f"   Реальные даты подходящих файлов:")
        for date_str in sorted(matching_date_counts.keys())[:10]:
            print(f"   • {date_str}: {matching_date_counts[date_str]} файлов")
        if len(matching_date_counts) > 10:
            print(f"   ... и ещё {len(matching_date_counts)-10} дат")

        real_dates = set(matching_date_counts.keys())
        requested_dates_set = set()
        for d in dates:
            requested_dates_set.add(d.strftime("%Y/%m/%d"))
        overlap = real_dates & requested_dates_set
        if not overlap:
            print(f"   ! Реальные даты файлов НЕ совпадают с запрошенным периодом!")
            print(f"   ! Файлам присваивается фейковая дата start_date={start_date}")

    print(f"\n Периоды:")
    print(f"   Общий: {start_date} — {end_date}")
    print(f"   Train: {start_date} — {train_end}")
    print(f"   Test:  {test_start} — {test_end}")

    print(f"\n СВОДКА ПО ПРИЗНАКАМ (анализ {min(20, len(matching_files)) if matching_files else min(20, len(h5_files))} файлов):")

    sample_files = (matching_files or h5_files)[:20]

    stats = {
        "temperature": {"found": 0, "files": 0, "values": [], "name": None},
        "wind_speed": {"found": 0, "files": 0, "values": [], "name": None},
        "humidity": {"found": 0, "files": 0, "values": [], "name": None},
        "precipitation": {"found": 0, "files": 0, "values": [], "name": None},
        "pm10": {"found": 0, "files": 0, "values": [], "name": None},
        "pm25": {"found": 0, "files": 0, "values": [], "name": None},
    }

    total_files_processed = 0

    for f in sample_files:
        try:
            raw = loader._read_h5(f)
            time_str = loader._extract_time(f)
            date_match = re.search(r'(\d{4}[/\-]\d{2}[/\-]\d{2})', f)
            date_part = date_match.group(1).replace('/', '-') if date_match else start_date
            start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")
            df = loader._to_dataframe(raw, start_dt)

            if df.empty:
                continue

            total_files_processed += 1

            for key, keyword in [("temperature", "temp"), ("wind_speed", "speed"),
                                  ("humidity", "humidity"), ("precipitation", "precip"),
                                  ("pm10", "pm10"), ("pm25", "pm25")]:
                col = pipeline.find_column(df, keyword)
                if col:
                    stats[key]["found"] += 1
                    stats[key]["name"] = col
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        stats[key]["values"].extend(vals.values[:100])
                else:
                    if key == "precipitation":
                        col = pipeline.find_column(df, "rain")
                        if col:
                            stats[key]["found"] += 1
                            stats[key]["name"] = col
                            vals = df[col].dropna()
                            if len(vals) > 0:
                                stats[key]["values"].extend(vals.values[:100])
        except:
            continue

    for key, label in [("temperature", "Температура"), ("wind_speed", "Ветер"),
                        ("humidity", "Влажность"), ("precipitation", "Осадки"),
                        ("pm10", "PM10"), ("pm25", "PM2.5")]:
        s = stats[key]
        pct = (s["found"] / total_files_processed * 100) if total_files_processed > 0 else 0
        status = " OK " if pct > 50 else " !WARNING! " if pct > 0 else " !!!ERROR!!! "

        print(f"   {status} {label}: найден в {s['found']}/{total_files_processed} файлов ({pct:.0f}%)", end="")

        if s["name"]:
            print(f" — {s['name']}", end="")

        if s["values"]:
            vals = np.array(s["values"])
            print(f" | range: [{vals.min():.2f}, {vals.max():.2f}]", end="")
            if pct > 0:
                print(f" | mean={vals.mean():.2f}", end="")

        print()

    sample_file = matching_files[0] if matching_files else (sorted(h5_files)[0] if h5_files else None)
    if sample_file:
        print(f"\n Пример файла: {sample_file}")
        try:
            raw = loader._read_h5(sample_file)
            time_str = loader._extract_time(sample_file)
            date_match = re.search(r'(\d{4}[/\-]\d{2}[/\-]\d{2})', sample_file)
            date_part = date_match.group(1).replace('/', '-') if date_match else start_date
            start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")

            sample_df = loader._to_dataframe(raw, start_dt)
            if not sample_df.empty:
                print(f"   Записей: {len(sample_df)}")
                print(f"   Период: {sample_df['timestamp'].min()} — {sample_df['timestamp'].max()}")
                print(f"   Колонок: {len(sample_df.columns)-1}")

                for keyword, label in [('temp', 'Температура'), ('speed', 'Ветер'),
                       ('humidity', 'Влажность'), ('precip', 'Осадки'), ('rain', 'Осадки')]:
                    col = pipeline.find_column(sample_df, keyword)
                    if col:
                        vals = sample_df[col].dropna()
                        if len(vals) > 0:
                            print(f"   • {label} ({col}): mean={vals.mean():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")
        except Exception as e:
            print(f"   ! Не удалось прочитать: {e}")

    print(f"\n Конфигурация модели:")
    print(f"   Горизонт: {horizon} мин")
    print(f"   Окно: {window} точек")
    print(f"   Эпох: {epochs}")
    print(f"   Батч: {batch_size}")
    print(f"   Веса: rain={RAIN_WEIGHT}, wind={WIND_WEIGHT}, temp={TEMP_WEIGHT}, pressure={PRESSURE_WEIGHT}")
    print(f"{'='*60}\n")