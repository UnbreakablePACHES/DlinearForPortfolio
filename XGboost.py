import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

RDSEED = 123
TICKERS = ["EEM", "EFA", "JPXN", "SPY", "VTI", "XLK", "AGG", "DBC"]
TRAIN_START = '2019-10-01'
TRAIN_END = '2021-01-01'
EVAL_START = '2021-01-04'
EVAL_END = '2022-01-19'
ROLLING_STEPS = 11
ROLLING_SHIFT_DAYS = 24
NUM_EPOCHS = 200
CONFIG = {
    'batch_size': 64,
    'kernel_size': 21,
    'learning_rate': 0.0001,
    'seq_len': 48,
    'pred_len': 24,
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def shift_date(start_date: str, days_to_shift: int) -> str:
    date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    shifted_date = date_obj + timedelta(days=days_to_shift)
    return shifted_date.strftime('%Y-%m-%d')


def build_xgb_dataset(df_norm: np.ndarray, seq_len: int, pred_len: int):
    x_samples, y_samples = [], []
    for i in range(len(df_norm) - seq_len - pred_len + 1):
        x_samples.append(df_norm[i:i + seq_len].flatten())
        y_samples.append(df_norm[i + seq_len:i + seq_len + pred_len].flatten())
    return np.array(x_samples), np.array(y_samples)


def train_model_xgb(df_norm: np.ndarray, config: dict, num_epoch: int = 200):
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    x_data, y_data = build_xgb_dataset(df_norm, seq_len, pred_len)

    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=RDSEED,
    )

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=num_epoch,
        learning_rate=config['learning_rate'],
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True, early_stopping_rounds=10)
    return model


def plot_and_save(real: pd.DataFrame, ticker: str, rdseed: int) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(real['pred'], label='Predicted', color='blue')
    plt.plot(real['Return'], label='Real', color='red')
    plt.title(f'{ticker} Predicted vs Real')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'image/XGboost/{rdseed}_{ticker}.png', dpi=300)


def predict_and_save_xgb(model, df_norm: np.ndarray, config: dict, history: pd.DataFrame, start_date: str, end_date: str, ticker: str, rdseed: int) -> pd.DataFrame:
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    res_total = pd.DataFrame()

    for _ in range(ROLLING_STEPS):
        r = df_norm[-seq_len:, :]
        x_test = r.flatten().reshape(1, -1)
        y_pred = model.predict(x_test)
        r = np.append(r, y_pred.reshape(pred_len, -1), axis=0)
        res_df = pd.DataFrame(r[-pred_len:])
        res_df.columns = ['ret_pred']
        res_total = pd.concat([res_total, res_df], axis=0, ignore_index=True) if not res_total.empty else res_df
        start_date = shift_date(start_date, ROLLING_SHIFT_DAYS)
        end_date = shift_date(end_date, ROLLING_SHIFT_DAYS)

    real = history[(history['date'] >= EVAL_START) & (history['date'] <= EVAL_END)].copy()
    real['pred'] = res_total.values
    plot_and_save(real, ticker, rdseed)
    real.to_csv(f'res/XGboost/{rdseed}_{ticker}_pred.csv', index=False)
    return real


def evaluate_predictions(real: pd.Series, pred: pd.Series):
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((real - pred) / real)) * 100
    return mae, mse, rmse, mape


def run_single_ticker(ticker: str, all_metrics: pd.DataFrame) -> pd.DataFrame:
    print(f'Processing {ticker}')
    history = pd.read_csv(f'data/{ticker}.csv')
    history['date'] = pd.to_datetime(history['date'])

    data = history[(history['date'] >= TRAIN_START) & (history['date'] <= TRAIN_END)]
    df = data[['Return']].reset_index(drop=True)
    df_norm = df.to_numpy()

    model = train_model_xgb(df_norm, CONFIG, NUM_EPOCHS)
    real = predict_and_save_xgb(model, df_norm, CONFIG, history, TRAIN_START, TRAIN_END, ticker, RDSEED)

    mae, mse, rmse, mape = evaluate_predictions(real['Return'], real['pred'])
    metric_row = pd.DataFrame({'Ticker': [ticker], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})
    return pd.concat([all_metrics, metric_row], ignore_index=True)


def main() -> None:
    set_all_seeds(RDSEED)
    all_metrics = pd.DataFrame(columns=['Ticker', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for ticker in TICKERS:
        all_metrics = run_single_ticker(ticker, all_metrics)

    all_metrics.to_csv(f'res/XGboost/{len(TICKERS)}_{RDSEED}_metrics.csv', index=False)
    print(all_metrics)


if __name__ == '__main__':
    main()
