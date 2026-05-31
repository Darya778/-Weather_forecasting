# Генератор батчей
import numpy as np
import pandas as pd
import torch
import re
from datetime import datetime
import gc


def optimized_batch_generator(loader, pipeline, start_date, end_date, global_df=None,
                              window=32, batch_size=16, fit_scaler_first=False,
                              is_train=True, fixed_threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = loader.get_all_files()

    valid_files = []
    dates = pd.date_range(start=start_date, end=end_date)
    date_strs = set()
    for d in dates:
        date_strs.add(d.strftime("%Y/%m/%d"))
        date_strs.add(d.strftime("%Y-%m-%d"))
        date_strs.add(d.strftime("%Y%m%d"))

    # print(f"[DIAGNOSTIC] Searching dates: {sorted(list(date_strs))[:5]}...")
    # print(f"[DIAGNOSTIC] Total files in bucket: {len(all_files)}")
    
    # Покажем примеры имен файлов
    h5_files_example = [f for f in all_files if f.endswith('.h5')][:5]
    # print(f"[DIAGNOSTIC] Example H5 files: {h5_files_example}")
    
    # Покажем примеры дат в именах файлов
    date_pattern = re.compile(r'(\d{4}[/\-]\d{2}[/\-]\d{2})')
    dates_in_files = set()
    for f in all_files:
        match = date_pattern.search(f)
        if match:
            dates_in_files.add(match.group(1))
    # print(f"[DIAGNOSTIC] Dates found in file names: {sorted(list(dates_in_files))[:10]}")

    for key in all_files:
        if not key.endswith(".h5"):
            continue
        if "bad_region" in key:
            continue

        for ds in date_strs:
            if ds in key:
                valid_files.append(key)
                break

    # print(f"[GENERATOR] Total files: {len(all_files)}, valid: {len(valid_files)}")
    
    if len(valid_files) == 0:
        print("[ERROR] No valid files found!")
        print("[ERROR] Check:")
        print(f"  - Date range: {start_date} to {end_date}")
        print(f"  - Date strings searched: {sorted(list(date_strs))[:10]}...")
        print(f"  - Dates in files: {sorted(list(dates_in_files))[:10]}...")
        # Покажем все файлы, чтобы понять формат дат
        all_h5 = [f for f in all_files if f.endswith('.h5')]
        print(f"  - All H5 files ({len(all_h5)}):")
        for f in all_h5[:10]:
            print(f"    {f}")

    region_id = None
    if pipeline.lat is not None and pipeline.lon is not None:
        region_id = hash((pipeline.lat, pipeline.lon)) % 100

    batch_size_files = 64

    if fit_scaler_first and not pipeline.scaler_fitted:
        # print("[GENERATOR] Fitting scaler using optimized file processing...")
        dfs_for_scaler = []

        max_files_for_scaler = min(len(valid_files), batch_size_files * 12)
        files_processed = 0
        files_skipped = 0
        
        print(f"[SCALER FITTING] Will process up to {max_files_for_scaler} files")

        for i in range(0, max_files_for_scaler, batch_size_files):
            file_batch = valid_files[i:i + batch_size_files]

            if not file_batch:
                print(f"[SCALER FITTING] Empty file batch at index {i}, breaking")
                break

            print(f"[SCALER FITTING] Processing batch {i//batch_size_files + 1}, files: {len(file_batch)}")
            batch_data = loader._read_h5_batch(file_batch)
            print(f"[SCALER FITTING] Successfully read {len(batch_data)} files from batch")

            for key, raw in batch_data.items():
                try:
                    time_str = loader._extract_time(key)

                    date_match = re.search(r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})', key)
                    if date_match:
                        date_part = date_match.group(1).replace('/', '-')
                    else:
                        date_part = start_date

                    start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")

                    df = loader._to_dataframe(raw, start_dt)
                    if df.empty:
                        files_skipped += 1
                        if files_skipped <= 3:
                            print(f"[SCALER FITTING] Empty dataframe for {key}")
                        continue

                    df = pipeline.preprocess(df)

                    if global_df is not None and not global_df.empty:
                        try:
                            df = df.merge(global_df, left_index=True, right_index=True, how='left')
                            df = df.ffill().bfill().fillna(0)
                        except Exception as e:
                            print(f"[WARNING] Global data merge failed: {e}")
                    
                    df = pipeline.create_features(df)
                    df = pipeline.create_target(df, horizon=pipeline.horizon)

                    if df.empty:
                        files_skipped += 1
                        if files_skipped <= 3:
                            print(f"[SCALER FITTING] Empty dataframe after create_target for {key}")
                            # Проверим, какие колонки есть
                            temp_col = pipeline.find_column(df, "temp")
                            humidity_col = pipeline.find_column(df, "humidity")
                            wind_col = pipeline.find_column(df, "speed")
                            print(f"  Found columns - temp: {temp_col}, humidity: {humidity_col}, wind: {wind_col}")
                        continue

                    df = pipeline.remove_leakage(df)

                    if len(df) > 1000:
                        df = df.iloc[:1000]

                    if len(df) > 10:
                        dfs_for_scaler.append(df)
                        files_processed += 1
                        
                        if files_processed == 1:
                            print(f"[SCALER FITTING] First valid dataframe: {len(df)} rows, columns: {list(df.columns)[:10]}...")

                        if len(dfs_for_scaler) >= 10:
                            print(f"[SCALER FITTING] Reached target of 10 dataframes")
                            break

                except Exception as e:
                    files_skipped += 1
                    if files_skipped <= 3:
                        print(f"[SCALER FITTING] Error processing {key}: {str(e)[:200]}")
                    continue

            if len(dfs_for_scaler) >= 10:
                break

        # print(f"[GENERATOR] Scaler fitting stats: processed={files_processed}, skipped={files_skipped}")

        if len(dfs_for_scaler) >= 2:
            pipeline.fit_scaler(dfs_for_scaler)
            # print(f"[GENERATOR] Scaler fitted on {len(dfs_for_scaler)} dataframes")
        elif len(dfs_for_scaler) == 1:
            print(f"[WARNING] Only 1 dataframe found for scaler. Attempting to fit...")
            pipeline.fit_scaler(dfs_for_scaler)
        else:
            print("\n" + "="*60)
            print("[CRITICAL ERROR] No dataframes for scaler!")
            print(f"Files processed: {files_processed}")
            print(f"Files skipped: {files_skipped}")
            print("\nPossible issues:")
            print("1. Check if files exist for the date range")
            print("2. Check if files contain required sensors (temperature, humidity, wind)")
            print("3. Check file format and data extraction")
            print("\nLet's examine a sample file...")
            
            # Попробуем прочитать один файл для диагностики
            if valid_files:
                sample_file = valid_files[0]
                print(f"\nExamining sample file: {sample_file}")
                try:
                    raw = loader._read_h5(sample_file)
                    print(f"Keys in H5 file: {list(raw.keys())[:20]}")
                    
                    time_str = loader._extract_time(sample_file)
                    date_match = re.search(r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})', sample_file)
                    date_part = date_match.group(1).replace('/', '-') if date_match else start_date
                    start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")
                    
                    df = loader._to_dataframe(raw, start_dt)
                    print(f"DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {list(df.columns)[:20]}")
                    
                    if not df.empty:
                        df = pipeline.preprocess(df)
                        print(f"After preprocess: {df.shape}")
                        print(f"Columns after preprocess: {list(df.columns)[:20]}")
                        
                        # Проверим наличие нужных колонок
                        for keyword, label in [('temp', 'Temperature'), 
                                               ('speed', 'Wind Speed'),
                                               ('humidity', 'Humidity'),
                                               ('precip', 'Precipitation'),
                                               ('rain', 'Rain')]:
                            col = pipeline.find_column(df, keyword)
                            print(f"  {label}: {'FOUND' if col else 'NOT FOUND'} ({col if col else 'N/A'})")
                except Exception as e:
                    print(f"Error examining sample file: {e}")
            
            print("="*60 + "\n")
            
            raise RuntimeError(
                f"Нет данных для scaler. Найдено: {len(dfs_for_scaler)}. "
                f"Проверьте:\n"
                f"1. Наличие файлов за период {start_date} - {end_date}\n"
                f"2. Наличие колонок temperature, humidity, wind в данных\n"
                f"3. Корректность извлечения дат из имен файлов\n"
                f"Подробности в логах выше"
            )

        del dfs_for_scaler
        gc.collect()

    # ... остальная часть функции без изменений ...
    n_features = None
    batch_x = None
    batch_y_cls = None
    batch_y_reg = None
    batch_region = None
    idx = 0
    total_batches = 0
    errors_count = 0

    if is_train:
        np.random.shuffle(valid_files)

    for i in range(0, len(valid_files), batch_size_files):
        file_batch = valid_files[i:i + batch_size_files]

        batch_data = loader._read_h5_batch(file_batch)

        for key, raw in batch_data.items():
            if raw is None:
                continue

            try:
                time_str = loader._extract_time(key)

                date_match = re.search(r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})', key)
                if date_match:
                    date_part = date_match.group(1).replace('/', '-')
                else:
                    date_part = start_date

                start_dt = datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")

                df = loader._to_dataframe(raw, start_dt)
                if df.empty:
                    continue

                df = pipeline.preprocess(df)

                if global_df is not None and not global_df.empty:
                    df = df.merge(global_df, left_index=True, right_index=True, how='left')
                    missing_ratio = df.isna().mean().mean()
                    if missing_ratio > 0.5:
                        continue
                    df = df.ffill().bfill().fillna(0)

                df = pipeline.create_features(df)
                df = pipeline.create_target(df, horizon=pipeline.horizon)

                if df.empty:
                    continue

                df = pipeline.remove_leakage(df)

                X, y_cls, y_reg = pipeline.transform(df, require_target=True)

                if n_features is None:
                    n_features = X.shape[1]
                    batch_x = np.zeros((batch_size, window, n_features), dtype=np.float32)
                    batch_y_cls = np.zeros((batch_size, 1), dtype=np.float32)
                    batch_y_reg = np.zeros((batch_size, 1, 6), dtype=np.float32)
                    batch_region = [] if region_id is not None else None
                    idx = 0
                    # print(f"[GENERATOR] Initialized with {n_features} features")

                if len(X) < window + 1:
                    pad_size = window + 1 - len(X)
                    X = np.pad(X, ((pad_size, 0), (0, 0)), mode='edge')
                    y_cls = np.pad(y_cls, (pad_size, 0), mode='edge')
                    y_reg = np.pad(y_reg, ((pad_size, 0), (0, 0)), mode='edge')

                step = 1 if is_train else max(1, pipeline.horizon // 5)
                indices = np.arange(0, len(X) - window, step)

                if len(indices) > 50:
                    indices = indices[:50]

                if is_train:
                    pos_indices = [i for i in indices if y_cls[i+window] == 1]
                    neg_indices = [i for i in indices if y_cls[i+window] == 0]

                    if len(pos_indices) == 0 or len(neg_indices) == 0:
                        selected = indices.tolist()
                    else:
                        n_pos = min(len(pos_indices), batch_size // 2)
                        n_neg = min(len(neg_indices), batch_size - n_pos)

                        selected = []
                        selected += np.random.choice(pos_indices, n_pos,
                                                    replace=len(pos_indices) < n_pos).tolist()
                        selected += np.random.choice(neg_indices, n_neg,
                                                    replace=len(neg_indices) < n_neg).tolist()
                        np.random.shuffle(selected)
                else:
                    selected = indices.tolist()

                for start_idx in selected:
                    x_win = X[start_idx:start_idx+window]
                    y_cls_win = y_cls[start_idx+window]
                    y_reg_win = y_reg[start_idx+window]

                    x_win = np.nan_to_num(x_win, nan=0.0, posinf=0.0, neginf=0.0)

                    batch_x[idx] = x_win
                    batch_y_cls[idx, 0] = y_cls_win
                    batch_y_reg[idx, 0] = y_reg_win

                    if region_id is not None:
                        batch_region.append(region_id)

                    idx += 1

                    if idx >= batch_size:
                        region_tensor = torch.tensor(batch_region, dtype=torch.long).to(device) \
                            if region_id is not None else None

                        x_tensor = torch.from_numpy(batch_x[:idx]).to(device, non_blocking=True)
                        y_cls_tensor = torch.from_numpy(batch_y_cls[:idx]).to(device, non_blocking=True)
                        y_reg_tensor = torch.from_numpy(batch_y_reg[:idx]).to(device, non_blocking=True)

                        yield (x_tensor, y_cls_tensor, y_reg_tensor, region_tensor)

                        del x_tensor, y_cls_tensor, y_reg_tensor, region_tensor

                        idx = 0
                        if region_id is not None:
                            batch_region.clear()

                        total_batches += 1

            except Exception as e:
                errors_count += 1
                if errors_count % 50 == 0:
                    print(f"[WARNING] Total errors so far: {errors_count}")
                continue

        del batch_data
        gc.collect()

    if idx > 0:
        region_tensor = torch.tensor(batch_region, dtype=torch.long).to(device) \
            if region_id is not None else None

        x_tensor = torch.from_numpy(batch_x[:idx]).to(device, non_blocking=True)
        y_cls_tensor = torch.from_numpy(batch_y_cls[:idx]).to(device, non_blocking=True)
        y_reg_tensor = torch.from_numpy(batch_y_reg[:idx]).to(device, non_blocking=True)

        yield (x_tensor, y_cls_tensor, y_reg_tensor, region_tensor)

        del x_tensor, y_cls_tensor, y_reg_tensor, region_tensor

        total_batches += 1

    # print(f"[GENERATOR] Total batches produced: {total_batches}")

    if total_batches == 0:
        print("! WARNING: generator produced 0 batches")