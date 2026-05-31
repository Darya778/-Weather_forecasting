# Загрузка данных (MamkaLoader, GlobalWeatherLoader)
import numpy as np
import pandas as pd
import h5py
import boto3
import re
import requests
from io import BytesIO
from datetime import datetime
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class MamkaLoader:
    def __init__(self, endpoint_url, access_key, secret_key, bucket="mamka"):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.cache = OrderedDict()
        self.cache_limit = 3
        self.processed_cache = None
        self.files_cache = {}
        self.files_cache_limit = 50
        self.all_files_cache = None
        self._lock = threading.Lock()

    def _list_files(self, prefix):
        if len(self.files_cache) > self.files_cache_limit:
            self.files_cache.clear()

        if prefix in self.files_cache:
            return self.files_cache[prefix]

        files, token = [], None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token

            res = self.s3.list_objects_v2(**kwargs)
            files += [obj["Key"] for obj in res.get("Contents", [])]

            if res.get("IsTruncated"):
                token = res.get("NextContinuationToken")
            else:
                break

        self.files_cache[prefix] = files
        return files

    def get_all_files(self):
        if self.all_files_cache is not None:
            return self.all_files_cache

        with self._lock:
            if self.all_files_cache is not None:
                return self.all_files_cache
            print("[LOADER] Fetching all files list (one-time)...")
            self.all_files_cache = self._list_files("")
            return self.all_files_cache

    def _extract_time(self, key):
        match = re.search(r'(\d{2}:\d{2}:\d{2})', key)
        return match.group(1) if match else "00:00:00"

    def _read_h5(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        file_bytes = obj["Body"].read()
        data = {}
        with h5py.File(BytesIO(file_bytes), "r") as f:
            def extract(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        val = obj[()]
                        data[name] = val
                    except:
                        pass
            f.visititems(extract)

        self.cache[key] = data
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return data

    def _read_h5_batch(self, keys):
        results = {}

        def read_single(key):
            if not key.endswith(".h5"):
                return key, None
            if "bad_region" in key:
                return key, None

            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                file_bytes = obj["Body"].read()
                data = {}
                with h5py.File(BytesIO(file_bytes), "r") as f:
                    def extract(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            try:
                                val = obj[()]
                                data[name] = val
                            except:
                                pass
                    f.visititems(extract)
                return key, data
            except Exception as e:
                return key, None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(read_single, key): key for key in keys}
            for future in as_completed(futures):
                key, data = future.result()
                if data is not None:
                    results[key] = data

        return results

    MAX_COLUMNS = 30

    def _to_dataframe(self, data, start_dt, use_cache=False):
        dfs = []
        col_count = 0
        
        # Приоритетные ключи - обрабатываем их первыми
        priority_patterns = ['precipitation', 'temperature', 'humidity', 'speed', 'wind', 'pm10', 'pm25', 'pressure']
        
        # Сначала обрабатываем приоритетные ключи
        priority_items = []
        other_items = []
        
        for key, val in data.items():
            key_lower = key.lower().strip()
            if key_lower == "timestamp" or key_lower.endswith("/timestamp") or key_lower.endswith("\\timestamp"):
                continue
                
            is_priority = any(pattern in key_lower for pattern in priority_patterns)
            if is_priority:
                priority_items.append((key, val))
            else:
                other_items.append((key, val))
        
        # Обрабатываем все элементы: сначала приоритетные, потом остальные
        all_items = priority_items + other_items
        
        for key, val in all_items:
            if col_count >= self.MAX_COLUMNS:
                break

            try:
                arr = np.array(val)
                arr = np.where(np.isfinite(arr), arr, np.nan)
                arr = arr.astype(np.float32)
                arr[(arr < -5000) | (arr > 5000)] = np.nan

                if np.isnan(arr).mean() > 0.5:
                    continue

                if arr.ndim > 1:
                    arr = np.nanmean(arr, axis=-1)

                arr = arr.astype(np.float32)

                # Для precipitation не применяем фильтр по std
                key_lower = key.lower()
                if 'precipitation' not in key_lower and 'rain' not in key_lower:
                    if np.nanstd(arr) < 1e-6:
                        continue

                timestamps = pd.date_range(start=start_dt, periods=len(arr), freq="min")

                dfs.append(pd.DataFrame({
                    "timestamp": timestamps,
                    key.replace("/", "_"): arr
                }).set_index("timestamp"))

                col_count += 1

            except:
                continue

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, axis=1)
        df = df.sort_index().ffill().bfill().reset_index()

        return df


class GlobalWeatherLoader:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.cache = {}

    def load(self, start_date, end_date):
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={self.lat}&longitude={self.lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,wind_speed_10m,relative_humidity_2m,precipitation,pm10,pm2_5"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "hourly" not in data:
                return pd.DataFrame()

            df = pd.DataFrame({
                "timestamp": data["hourly"]["time"],
                "g_temp": data["hourly"]["temperature_2m"],
                "g_wind": data["hourly"]["wind_speed_10m"],
                "g_humidity": data["hourly"]["relative_humidity_2m"],
                "g_precip": data["hourly"]["precipitation"],
                "g_pm10": data["hourly"]["pm10"],
                "g_pm2_5": data["hourly"]["pm2_5"]
            })

            for col in ["g_temp", "g_wind", "g_humidity", "g_precip", "g_pm10", "g_pm2_5"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df = df.ffill().fillna(0)

            self.cache[cache_key] = df
            return df

        except Exception as e:
            print(f"[ERROR] GlobalWeatherLoader: {e}")
            return pd.DataFrame()
