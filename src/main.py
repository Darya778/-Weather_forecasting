# Основной скрипт запуска
import torch
from config import *
from data_loader import MamkaLoader, GlobalWeatherLoader
from pipeline import WeatherPipeline
from training import train_model, IncrementalLearner
from evaluation import evaluate_model
from prediction import (
    plot_training_history, compute_feature_importance, plot_regression_vs_real,
    plot_classification_vs_real, predict_and_visualize_example, predict_future_minutes,
    visualize_dashboard, plot_all_weather_params, print_data_summary
)


def main():
    # Конфигурация
    START_DATE = "2026-04-24"
    END_DATE = "2026-05-08"
    TRAIN_END = "2026-05-03"
    TEST_START = "2026-05-04"
    TEST_END = "2026-05-08"

    HORIZON = DEFAULT_HORIZON
    WINDOW = DEFAULT_WINDOW
    EPOCHS = DEFAULT_EPOCHS
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    LAT = DEFAULT_LAT
    LON = DEFAULT_LON

    # Инициализация загрузчиков
    loader = MamkaLoader(
        endpoint_url="http://weather.okpars.com:9000",
        access_key="RO_User",
        secret_key="vy6ogWbguest"
    )

    pipeline = WeatherPipeline(lat=LAT, lon=LON, horizon=HORIZON)
    global_loader = GlobalWeatherLoader(LAT, LON)
    global_df = global_loader.load(START_DATE, END_DATE)

    # Вывод сводки данных
    print_data_summary(loader, pipeline, global_df,
                       START_DATE, END_DATE, TRAIN_END, TEST_START, TEST_END,
                       HORIZON, WINDOW, EPOCHS, BATCH_SIZE)

    # Первичное обучение
    print("ПЕРВИЧНОЕ ОБУЧЕНИЕ")
    model, train_losses = train_model(
        loader, pipeline, START_DATE, TRAIN_END, global_df,
        window=WINDOW, epochs=EPOCHS, batch_size=BATCH_SIZE,
        accumulation_steps=4, fixed_threshold=None
    )

    # Инициализация инкрементального обучения
    print("ИНИЦИАЛИЗАЦИЯ INCREMENTAL LEARNER")
    learner = IncrementalLearner(
        model=model,
        pipeline=pipeline,
        loader=loader,
        global_df=global_df,
        buffer_size=3000,
        ewc_lambda=1000,
        replay_ratio=0.25
    )

    learner.fill_buffer_from_dates(
        start_date=START_DATE,
        end_date=TRAIN_END,
        max_samples_per_date=100
    )

    learner.save_checkpoint("base_model.pt")
    print("Базовая модель сохранена в base_model.pt")

    # Оценка модели
    print("BASELINE ОЦЕНКА")
    metrics_base, _ = evaluate_model(
        model, loader, pipeline, TEST_START, TEST_END, global_df,
        window=WINDOW, batch_size=BATCH_SIZE
    )
    print(f"Baseline AUC: {metrics_base['auc']:.4f}")

    # Визуализация результатов
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

    if train_losses:
        print("\n[ГРАФИК 1] Training Loss History")
        plot_training_history(train_losses, save_path="training_history.png")

    print("\n[ГРАФИК 2] Running Evaluation to get predictions...")
    metrics, df_preds = evaluate_model(
        model, loader, pipeline, TEST_START, TEST_END, global_df,
        window=WINDOW, batch_size=BATCH_SIZE
    )

    if df_preds is not None and not df_preds.empty:
        print(f"Получено {len(df_preds)} предсказаний для визуализации")

        print("\n[DIAGNOSTIC] Rain data analysis:")
        print(f"  true_rain non-zero: {(df_preds['true_rain'] > 0).sum()} / {len(df_preds)}")
        print(f"  true_rain mean: {df_preds['true_rain'].mean():.4f}")
        print(f"  pred_rain mean: {df_preds['pred_rain'].mean():.4f}")
        print(f"  true_rain max: {df_preds['true_rain'].max():.4f}")

    if model is not None:
        print("\n[ГРАФИК 3] Feature Importance Analysis")
        df_importance = compute_feature_importance(
            model, loader, pipeline, TEST_START, TEST_END, global_df,
            window=WINDOW, batch_size=BATCH_SIZE
        )

    if df_preds is not None and not df_preds.empty:
        print("\n[ГРАФИК 4] Regression: Real vs Predicted Temperature")
        if 'true_reg' in df_preds.columns and 'pred_reg' in df_preds.columns:
            plot_regression_vs_real(
                df_preds['true_reg'].values,
                df_preds['pred_reg'].values,
                title=f"Weather Temperature: Real vs Predicted (H={HORIZON})"
            )
        else:
            print("  Пропущено: нет колонок true_reg/pred_reg")

    if df_preds is not None and not df_preds.empty:
        print("\n[ГРАФИК 5] Classification: Predicted Probability vs True Label")
        if 'pred' in df_preds.columns and 'true' in df_preds.columns:
            optimal_threshold = metrics.get('best_threshold', 0.45)
            plot_classification_vs_real(
                df_preds,
                threshold=optimal_threshold,
                start_idx=0,
                max_points=500
            )
        else:
            print("  Пропущено: нет колонок pred/true")

    if model is not None:
        print("\n[ГРАФИК 6] Example Prediction on Latest File")
        optimal_threshold = metrics.get('best_threshold', 0.45)
        predict_and_visualize_example(
            model, loader, pipeline, global_df,
            threshold=optimal_threshold,
            step=10
        )

    if model is not None:
        print("\n[ГРАФИК 7] Future Forecast (30 minutes ahead)")
        optimal_threshold = metrics.get('best_threshold', 0.45)
        future_predictions = predict_future_minutes(
            model, loader, pipeline, global_df,
            minutes_ahead=30,
            threshold=optimal_threshold
        )

        if future_predictions is not None and not future_predictions.empty:
            visualize_dashboard(future_predictions, save_path="weather_dashboard.png")
        else:
            print("  Не удалось получить прогноз")

    # Визуализация всех погодных параметров
    if df_preds is not None and not df_preds.empty:
        required_all = ["true_temp", "true_wind", "true_humidity", "true_rain", "true_pm10", "true_pm25"]
        predicted_all = ["pred_temp", "pred_wind", "pred_humidity", "pred_rain", "pred_pm10", "pred_pm25"]

        if all(c in df_preds.columns for c in required_all + predicted_all):
            print("\n[ГРАФИК 8] Все погодные параметры")
            n = min(2000, len(df_preds))
            plot_all_weather_params(
                y_true=df_preds[required_all].values[:n],
                y_pred=df_preds[predicted_all].values[:n],
                max_points=n,
                x_label="Time step (minutes)",
                save_path="weather_params_comparison.png"
            )
        else:
            print(f"Доступные колонки: {list(df_preds.columns)}")
            print("Для графика всех параметров нужны колонки с PM")

    # Финальный график классификации
    if df_preds is not None and not df_preds.empty and 'pred' in df_preds.columns and 'true' in df_preds.columns:
        print("\n[ГРАФИК 9] Финальный график классификации")
        optimal_threshold = metrics.get('best_threshold', 0.45)
        plot_classification_vs_real(
            df_preds,
            threshold=optimal_threshold,
            start_idx=0,
            max_points=500,
            x_label="Time (minutes)",
            time_per_point=1
        )

    print("ВСЕ ГРАФИКИ СОХРАНЕНЫ:")
    print("  - training_history.png")
    print("  - feature_importance.png")
    print("  - predictions_plot.png")
    print("  - weather_dashboard.png")
    print("  - latest_predictions.png")
    print("  - weather_params_comparison.png")


def run_incremental_update():
    """Функция для инкрементального дообучения"""
    print("ИНКРЕМЕНТАЛЬНОЕ ДООБУЧЕНИЕ")

    loader = MamkaLoader(
        endpoint_url="http://weather.okpars.com:9000",
        access_key="RO_User",
        secret_key="vy6ogWbguest"
    )

    pipeline = WeatherPipeline(lat=DEFAULT_LAT, lon=DEFAULT_LON, horizon=DEFAULT_HORIZON)
    global_loader = GlobalWeatherLoader(DEFAULT_LAT, DEFAULT_LON)
    global_df = global_loader.load("2026-04-24", "2026-05-08")

    learner = IncrementalLearner(
        model=None,
        pipeline=pipeline,
        loader=loader,
        global_df=global_df
    )
    learner.load_checkpoint("base_model.pt")

    NEW_START_DATE = "2026-05-09"
    NEW_END_DATE = "2026-05-15"

    validation_dates = [
        ("2026-05-01", "2026-05-03"),
        ("2026-05-12", "2026-05-16")
    ]

    val_metrics = learner.incremental_update(
        new_start_date=NEW_START_DATE,
        new_end_date=NEW_END_DATE,
        epochs=3,
        batch_size=32,
        lr=1e-5,
        validation_dates=validation_dates,
        patience=2
    )

    learner.save_checkpoint("updated_model.pt")
    print(f"Обновлённая модель сохранена в updated_model.pt")
    print(f"Validation AUC: {val_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
    # Для инкрементального обучения раскомментируйте:
    # run_incremental_update()