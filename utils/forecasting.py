import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==========================================================
# Feature Engineering
# ==========================================================

def create_time_features(df):
    """
    Creates time-based features for ML forecasting.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter

    data["Lag1"] = data["Sales"].shift(1)
    data["Lag2"] = data["Sales"].shift(2)
    data["Lag3"] = data["Sales"].shift(3)

    data["Rolling3"] = (
        data["Sales"]
        .rolling(3)
        .mean()
    )

    data["Rolling6"] = (
        data["Sales"]
        .rolling(6)
        .mean()
    )

    data = data.dropna().reset_index(drop=True)

    return data


# ==========================================================
# Metrics
# ==========================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        )
        * 100
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }


# ==========================================================
# Gradient Boosting Forecast
# ==========================================================

def gradient_boosting_forecast(monthly_sales, forecast_periods=3):

    df = monthly_sales.copy()

    df = create_time_features(df)

    feature_cols = [
        "Year",
        "Month",
        "Quarter",
        "Lag1",
        "Lag2",
        "Lag3",
        "Rolling3",
        "Rolling6",
    ]

    X = df[feature_cols]
    y = df["Sales"]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    history = monthly_sales.copy()

    history["Date"] = pd.to_datetime(history["Date"])

    history = history.sort_values("Date")

    future_predictions = []

    temp = history.copy()


    # ==========================================
    # Recursive Future Forecast
    # ==========================================

    for _ in range(forecast_periods):

        next_date = temp["Date"].max() + pd.DateOffset(months=1)

        sales_series = temp["Sales"]

        lag1 = sales_series.iloc[-1]
        lag2 = sales_series.iloc[-2]
        lag3 = sales_series.iloc[-3]

        rolling3 = sales_series.tail(3).mean()
        rolling6 = sales_series.tail(6).mean()

        future_row = pd.DataFrame({
            "Year": [next_date.year],
            "Month": [next_date.month],
            "Quarter": [next_date.quarter],
            "Lag1": [lag1],
            "Lag2": [lag2],
            "Lag3": [lag3],
            "Rolling3": [rolling3],
            "Rolling6": [rolling6],
        })

        prediction = model.predict(future_row)[0]

        future_predictions.append({
            "Date": next_date,
            "Forecast": prediction
        })

        temp = pd.concat([
            temp,
            pd.DataFrame({
                "Date": [next_date],
                "Sales": [prediction]
            })
        ], ignore_index=True)

    future_df = pd.DataFrame(future_predictions)

    return {
        "model": model,
        "metrics": metrics,
        "test_actual": y_test.values,
        "test_prediction": test_predictions,
        "forecast": future_df
    }


# ==========================================================
# SARIMA Forecast (Secondary Model)
# ==========================================================

def sarima_forecast(monthly_sales, forecast_periods=3):

    df = monthly_sales.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    series = df.set_index("Date")["Sales"]

    split = int(len(series) * 0.8)

    train = series.iloc[:split]
    test = series.iloc[split:]

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)

    predictions = fitted_model.forecast(len(test))

    metrics = calculate_metrics(
        test,
        predictions
    )

    final_model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    future = final_model.forecast(forecast_periods)

    future_dates = pd.date_range(
        start=series.index.max() + pd.DateOffset(months=1),
        periods=forecast_periods,
        freq="MS",
    )

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future.values
    })

    return {
        "model": final_model,
        "metrics": metrics,
        "test_actual": test.values,
        "test_prediction": predictions.values,
        "forecast": future_df
    }


# ==========================================================
# Default Forecast Function
# ==========================================================

def generate_forecast(monthly_sales, forecast_periods=3):
    """
    Default forecasting method used by Streamlit.

    Uses Gradient Boosting because it achieved
    the best MAE/RMSE during evaluation.
    """

    return gradient_boosting_forecast(
        monthly_sales,
        forecast_periods=forecast_periods,
    )


# ==========================================================
# Compare Both Models
# ==========================================================

def compare_models(monthly_sales):

    gb = gradient_boosting_forecast(monthly_sales)

    sarima = sarima_forecast(monthly_sales)

    comparison = pd.DataFrame({

        "Model": [
            "Gradient Boosting",
            "SARIMA"
        ],

        "MAE": [
            gb["metrics"]["MAE"],
            sarima["metrics"]["MAE"]
        ],

        "RMSE": [
            gb["metrics"]["RMSE"],
            sarima["metrics"]["RMSE"]
        ],

        "MAPE": [
            gb["metrics"]["MAPE"],
            sarima["metrics"]["MAPE"]
        ]

    })

    comparison = comparison.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    return comparison