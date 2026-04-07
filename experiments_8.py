import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RDSEED = 123
TICKERS = ["EEM", "EFA", "JPXN", "SPY", "VTI", "XLK", "AGG", "DBC"]
TRAIN_START = '2019-10-01'
TRAIN_END = '2021-01-01'
EVAL_START = '2021-01-04'
EVAL_END = '2022-01-19'
ROLLING_STEPS = 11
ROLLING_SHIFT_DAYS = 24
NUM_EPOCHS = 500
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class CustomDataset(Dataset):
    def __init__(self, df: np.ndarray, seq_len: int, pred_len: int):
        self.df = df
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __getitem__(self, index: int):
        input_begin = index
        input_end = input_begin + self.seq_len
        label_begin = input_end
        label_end = label_begin + self.pred_len
        seq_x = self.df[input_begin:input_end]
        seq_y = self.df[label_begin:label_end]
        return seq_x, seq_y

    def __len__(self) -> int:
        return len(self.df) - self.seq_len - self.pred_len + 1


def shift_date(start_date: str, days_to_shift: int) -> str:
    date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    shifted_date = date_obj + timedelta(days=days_to_shift)
    return shifted_date.strftime('%Y-%m-%d')


def reset_model_weights(model: nn.Module) -> None:
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_model(df_norm: np.ndarray, config: dict, num_epoch: int = 200) -> nn.Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_set = CustomDataset(df=df_norm, seq_len=config['seq_len'], pred_len=config['pred_len'])
    data_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, drop_last=False)

    model = DLinear(seq_len=config['seq_len'], pred_len=config['pred_len'], kernel_size=config['kernel_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.L1Loss().to(device)

    train_loss_ep = []
    for _ in tqdm(range(num_epoch)):
        train_loss = []
        model.train()
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
        train_loss_ep.append(np.average(train_loss))
    return model


def plot_and_save(real: pd.DataFrame, ticker: str, rdseed: int) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(real['pred'], label='Predicted', color='blue')
    plt.plot(real['Return'], label='Real', color='red')
    plt.title(f'{ticker} Predicted vs Real')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'image/{rdseed}_{ticker}.png', dpi=300)


def predict_and_save(model: nn.Module, df_norm: np.ndarray, config: dict, history: pd.DataFrame, start_date: str, end_date: str, ticker: str, rdseed: int) -> pd.DataFrame:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res_total = pd.DataFrame()

    for _ in range(ROLLING_STEPS):
        r = df_norm[-config['seq_len']:, :]
        x = torch.tensor(r[-config['seq_len']:, :]).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(x.float().to(device))
        r = np.append(r, output.squeeze(0).to('cpu').detach().numpy(), axis=0)
        res_df = pd.DataFrame(r[-config['pred_len']:])
        res_df.columns = ['ret_pred']
        res_total = pd.concat([res_total, res_df], axis=0, ignore_index=True) if not res_total.empty else res_df
        start_date = shift_date(start_date, ROLLING_SHIFT_DAYS)
        end_date = shift_date(end_date, ROLLING_SHIFT_DAYS)
        reset_model_weights(model)

    real = history[(history['date'] >= EVAL_START) & (history['date'] <= EVAL_END)].copy()
    real['pred'] = res_total.values
    plot_and_save(real, ticker, rdseed)
    real.to_csv(f'res/{rdseed}_{ticker}_pred.csv', index=False)
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

    model = train_model(df_norm, CONFIG, NUM_EPOCHS)
    real = predict_and_save(model, df_norm, CONFIG, history, TRAIN_START, TRAIN_END, ticker, RDSEED)

    mae, mse, rmse, mape = evaluate_predictions(real['Return'], real['pred'])
    metric_row = pd.DataFrame({'Ticker': [ticker], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})
    return pd.concat([all_metrics, metric_row], ignore_index=True)


def main() -> None:
    set_all_seeds(RDSEED)
    all_metrics = pd.DataFrame(columns=['Ticker', 'MAE', 'MSE', 'RMSE', 'MAPE'])
    for ticker in TICKERS:
        all_metrics = run_single_ticker(ticker, all_metrics)

    all_metrics.to_csv(f'res/{len(TICKERS)}_{RDSEED}_metrics.csv', index=False)
    print(all_metrics)


if __name__ == '__main__':
    main()
