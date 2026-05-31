# WeatherPipeline (предобработка, фичи, таргеты)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class WeatherPipeline:
    def __init__(self, lat=None, lon=None, horizon=10):
        self.scaler = StandardScaler()
        self.scaler_temp = StandardScaler()
        self.scaler_wind = StandardScaler()
        self.scaler_rain = StandardScaler()
        self.scaler_humidity = StandardScaler()
        self.scaler_pm10 = StandardScaler()
        self.scaler_pm25 = StandardScaler()
        self.fixed_columns = None
        self.lat = lat
        self.lon = lon
        self.horizon = horizon
        self.scaler_fitted = False

    def find_column(self, df, keyword):
        col_lower_map = {col.lower(): col for col in df.columns}

        # ТЕМПЕРАТУРА
        if keyword == "temp":
            for col in df.columns:
                if "temphumidity" in col.lower() and "temperature" in col.lower():
                    return col

            for col in df.columns:
                col_l = col.lower()
                if "temperature" in col_l and "humidity" not in col_l and "air" not in col_l:
                    return col

        # ВЕТЕР
        elif keyword == "speed":
            for col in df.columns:
                if "winddetector" in col.lower() and col.lower().endswith("_speed"):
                    return col
            for col in df.columns:
                col_l = col.lower()
                if col_l.endswith("_speed") and "sound" not in col_l:
                    return col

        # ВЛАЖНОСТЬ
        elif keyword == "humidity":
            for col in df.columns:
                if "temphumidity" in col.lower() and "humidity" in col.lower():
                    return col
            for col in df.columns:
                if "humidity" in col.lower() and "temp" not in col.lower():
                    return col

        # ДАВЛЕНИЕ
        elif keyword == "press":
            for col in df.columns:
                if "temphumidity" in col.lower() and "pressure" in col.lower():
                    return col
            for col in df.columns:
                if "pressure" in col.lower():
                    return col

        # ОСАДКИ
        elif keyword in ("precip", "rain"):
            # Добавим отладку для первого вызова
            if not hasattr(self, '_precip_debug'):
                self._precip_debug = True
                precip_cols = [c for c in df.columns if 'precipitation' in c.lower() or 'rain' in c.lower()]
                print(f"[DEBUG PRECIP] Available precipitation columns: {precip_cols}")
            
            for col in df.columns:
                if "precipitation" in col.lower() and "accumulated" in col.lower():
                    return col
            for col in df.columns:
                if "precipitation" in col.lower() and "intensity" in col.lower():
                    return col
            for col in df.columns:
                if "precipitation" in col.lower() or "rain" in col.lower():
                    return col

        # PM10
        elif keyword == "pm10":
            for col in df.columns:
                col_l = col.lower()
                if "pm10" in col_l or "pm_10" in col_l or "particulate_10" in col_l:
                    return col
            for col in df.columns:
                col_l = col.lower()
                if ("particulate" in col_l and "10" in col_l) or "dust" in col_l:
                    return col

        # PM2.5
        elif keyword == "pm25":
            for col in df.columns:
                col_l = col.lower()
                if "pm25" in col_l or "pm_25" in col_l or "pm2.5" in col_l or "particulate_25" in col_l:
                    return col
            for col in df.columns:
                col_l = col.lower()
                if ("particulate" in col_l and "25" in col_l) or "fine_dust" in col_l:
                    return col

        return None

    def preprocess(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df.select_dtypes(include=np.number)
        df = df.interpolate(limit=5)
        df = df.ffill().bfill()
        return df

    def create_features(self, df):
        df = df.copy()
        # Исключаем целевые колонки
        exclude_cols = ['target', 'target_temp', 'target_wind', 'target_humidity',
                        'target_rain', 'target_pm10', 'target_pm25', 'risk_score']
        for col in exclude_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        rain_col = self.find_column(df, "precip") or self.find_column(df, "rain")
        wind_col = self.find_column(df, "speed")
        temp_col = self.find_column(df, "temp")
        humidity_col = self.find_column(df, "humidity")

        target_cols = [rain_col, wind_col, temp_col, humidity_col]
        target_cols = [c for c in target_cols if c is not None]

        direction_cols = [col for col in df.columns if 'direction' in col.lower()]

        for col in direction_cols:
            if col in df.columns:
                angle_rad = np.radians(df[col] % 360)
                df[f"{col}_sin"] = np.sin(angle_rad)
                df[f"{col}_cos"] = np.cos(angle_rad)
                df = df.drop(columns=[col])

        for col in target_cols:
            if col in df.columns and len(df) > 5:
                df[f"{col}_ma5"] = df[col].rolling(5, min_periods=1).mean()
                df[f"{col}_lag1"] = df[col].shift(1).fillna(0)
                df[f"{col}_diff"] = df[col].diff().fillna(0)

        df = df.ffill(limit=10).bfill(limit=10)
        df = df.dropna(axis=1, thresh=int(len(df) * 0.7))
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["sin_time"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_time"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["horizon"] = self.horizon

        if self.lat is not None and self.lon is not None:
            df["lat"] = self.lat
            df["lon"] = self.lon
            df["lat_scaled"] = self.lat / 90.0
            df["lon_scaled"] = self.lon / 180.0
            df["region_id"] = hash((self.lat, self.lon)) % 100

        return df

    def create_target(self, df, horizon=10):
        df = df.copy()

        rain_col = self.find_column(df, "precip") or self.find_column(df, "rain")
        wind_col = self.find_column(df, "speed")
        temp_col = self.find_column(df, "temp")
        humidity_col = self.find_column(df, "humidity")
        pm10_col = self.find_column(df, "pm10")
        pm25_col = self.find_column(df, "pm25")

        found_critical = 0
        if temp_col is not None: found_critical += 1
        if humidity_col is not None: found_critical += 1
        if wind_col is not None: found_critical += 1

        if found_critical < 2:
            missing = []
            if temp_col is None: missing.append("temp")
            if humidity_col is None: missing.append("humidity")
            if wind_col is None: missing.append("wind")
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            self._skip_count += 1
            if self._skip_count % 50 == 0:
                print(f"[SKIP] Too few critical columns (total skips: {self._skip_count})")
            return pd.DataFrame()

        if rain_col is None:
            df["__rain_dummy__"] = 0.0
            rain_col = "__rain_dummy__"

        if temp_col is None:
            df["__temp_dummy__"] = 0.0
            temp_col = "__temp_dummy__"

        if wind_col is None:
            df["__wind_dummy__"] = 0.0
            wind_col = "__wind_dummy__"

        if humidity_col is None:
            df["__humidity_dummy__"] = 50.0
            humidity_col = "__humidity_dummy__"

        if pm10_col is None:
            df["__pm10_dummy__"] = 0.0
            pm10_col = "__pm10_dummy__"

        if pm25_col is None:
            df["__pm25_dummy__"] = 0.0
            pm25_col = "__pm25_dummy__"

        # Очистка
        df[temp_col] = df[temp_col].replace(-10000, np.nan).ffill().fillna(0)
        df[humidity_col] = df[humidity_col].clip(0, 100).replace(-10000, np.nan).ffill().fillna(50)
        df[wind_col] = df[wind_col].clip(lower=0).replace(-10000, np.nan).fillna(0)
        df[rain_col] = df[rain_col].clip(lower=0).replace(-10000, np.nan).fillna(0)
        df[pm10_col] = df[pm10_col].clip(lower=0).replace(-10000, np.nan).ffill().fillna(0)
        df[pm25_col] = df[pm25_col].clip(lower=0).replace(-10000, np.nan).ffill().fillna(0)

        rain = df[rain_col]
        wind = df[wind_col]
        temp = df[temp_col]
        humidity = df[humidity_col]
        pm10 = df[pm10_col]
        pm25 = df[pm25_col]

        rain_f = rain.shift(-horizon)
        wind_f = wind.shift(-horizon)
        temp_f = temp.shift(-horizon)
        humidity_f = humidity.shift(-horizon)
        pm10_f = pm10.shift(-horizon)
        pm25_f = pm25.shift(-horizon)

        rain_score = rain_f / 20.0
        wind_score = wind_f / 25.0
        temp_score = np.abs(temp_f - 20) / 40.0
        humidity_score = np.abs(humidity_f - 50) / 50.0
        pm10_score = pm10_f / 100.0
        pm25_score = pm25_f / 50.0

        rain_score = np.clip(rain_score, 0, 1)
        wind_score = np.clip(wind_score, 0, 1)
        temp_score = np.clip(temp_score, 0, 1)
        humidity_score = np.clip(humidity_score, 0, 1)
        pm10_score = np.clip(pm10_score, 0, 1)
        pm25_score = np.clip(pm25_score, 0, 1)

        w_rain, w_wind, w_temp, w_humidity, w_pm10, w_pm25 = 0.35, 0.2, 0.1, 0.05, 0.15, 0.15

        if rain_col == "__rain_dummy__":
            w_rain = 0.0
        if wind_col == "__wind_dummy__":
            w_wind = 0.0
        if temp_col == "__temp_dummy__":
            w_temp = 0.0
        if humidity_col == "__humidity_dummy__":
            w_humidity = 0.0
        if pm10_col == "__pm10_dummy__":
            w_pm10 = 0.0
        if pm25_col == "__pm25_dummy__":
            w_pm25 = 0.0

        total_w = w_rain + w_wind + w_temp + w_humidity + w_pm10 + w_pm25
        if total_w > 0:
            w_rain /= total_w
            w_wind /= total_w
            w_temp /= total_w
            w_humidity /= total_w
            w_pm10 /= total_w
            w_pm25 /= total_w

        risk_score = (
            w_rain * rain_score +
            w_wind * wind_score +
            w_temp * temp_score +
            w_humidity * humidity_score +
            w_pm10 * pm10_score +
            w_pm25 * pm25_score
        )

        threshold = risk_score.quantile(0.8)
        df["target"] = (risk_score > threshold).astype(int)
        df["risk_score"] = risk_score

        df["target_temp"] = temp_f
        df["target_wind"] = wind_f
        df["target_humidity"] = humidity_f
        df["target_rain"] = rain_f
        df["target_pm10"] = pm10_f
        df["target_pm25"] = pm25_f

        df = df.dropna(subset=["target"])
        df["target_temp"] = df["target_temp"].fillna(0)
        df["target_wind"] = df["target_wind"].fillna(0)
        df["target_humidity"] = df["target_humidity"].fillna(50)
        df["target_rain"] = df["target_rain"].fillna(0)
        df["target_pm10"] = df["target_pm10"].fillna(0)
        df["target_pm25"] = df["target_pm25"].fillna(0)

        return df

    def remove_leakage(self, df):
        df = df.copy()
        rain_col = self.find_column(df, "precip") or self.find_column(df, "rain")
        wind_col = self.find_column(df, "speed")
        temp_col = self.find_column(df, "temp")
        humidity_col = self.find_column(df, "humidity")

        target_cols = [c for c in [rain_col, wind_col, temp_col, humidity_col] if c is not None]

        for col in target_cols:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1).fillna(0)
        df = df.ffill().fillna(0)
        return df

    def fit_scaler(self, df_list, fixed_columns=None):
        if self.scaler_fitted:
            print("[SCALER] Already fitted, skipping...")
            return

        if isinstance(df_list, pd.DataFrame):
            df_list = [df_list]

        X_all = []
        temp_all, wind_all, humidity_all, rain_all, pm10_all, pm25_all = [], [], [], [], [], []

        for df in df_list:
            X = df.drop(columns=['target', 'target_reg'], errors='ignore')
            X = X.replace([np.inf, -np.inf], np.nan).dropna()

            if len(X) > 500:
                X = X.iloc[:500]

            X_all.append(X)

            if all(col in df.columns for col in ["target_temp", "target_wind", "target_humidity", "target_rain", "target_pm10", "target_pm25"]):
                temp_all.append(df[["target_temp"]])
                wind_all.append(df[["target_wind"]])
                humidity_all.append(df[["target_humidity"]])
                rain_all.append(df[["target_rain"]])
                pm10_all.append(df[["target_pm10"]])
                pm25_all.append(df[["target_pm25"]])

        if not X_all:
            raise RuntimeError("Нет данных для обучения scaler'ов")

        X_all = pd.concat(X_all, ignore_index=True)

        if len(X_all) > 2000:
            X_all = X_all.sample(2000, random_state=42)

        self.fixed_columns = list(X_all.columns)

        self.scaler.fit(X_all)
        print(f"[SCALER] Fitted on {len(X_all)} samples, {len(self.fixed_columns)} features")

        if temp_all:
            self.scaler_temp.fit(pd.concat(temp_all))
            self.scaler_wind.fit(pd.concat(wind_all))
            self.scaler_humidity.fit(pd.concat(humidity_all))
            self.scaler_rain.fit(pd.concat(rain_all))
            self.scaler_pm10.fit(pd.concat(pm10_all))
            self.scaler_pm25.fit(pd.concat(pm25_all))
            print(f"[SCALER] Target scalers fitted: temp, wind, humidity, rain, pm10, pm25")
        else:
            raise RuntimeError("Нет данных для обучения регрессионных scaler'ов")

        self.scaler_fitted = True

    def transform(self, df, require_target=True):
        X = df.drop(columns=['target', 'target_reg'], errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in self.fixed_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.fixed_columns]

        X_scaled = self.scaler.transform(X).astype(np.float32, copy=False)

        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
        X_scaled = np.clip(X_scaled, -5, 5)

        if require_target:
            y_cls = df['target'].values.astype(np.float32)
            temp = self.scaler_temp.transform(df[["target_temp"]]).astype(np.float32)
            wind = self.scaler_wind.transform(df[["target_wind"]]).astype(np.float32)
            humidity = self.scaler_humidity.transform(df[["target_humidity"]]).astype(np.float32)
            rain = self.scaler_rain.transform(df[["target_rain"]]).astype(np.float32)
            pm10 = self.scaler_pm10.transform(df[["target_pm10"]]).astype(np.float32)
            pm25 = self.scaler_pm25.transform(df[["target_pm25"]]).astype(np.float32)

            y_reg = np.concatenate([temp, wind, humidity, rain, pm10, pm25], axis=1)
            y_reg = np.nan_to_num(y_reg, nan=0.0, posinf=5.0, neginf=-5.0)
            y_reg = np.clip(y_reg, -5, 5)

            return X_scaled, y_cls, y_reg

        return X_scaled, None, None
