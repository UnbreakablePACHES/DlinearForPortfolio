import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


RDSEED = 123
TICKERS = [
    'IAI', 'IVV', 'ESGU', 'PICK', 'QUAL', 'SLV', 'IWB', 'HEWJ', 'RING', 'IAU',
    'IYY', 'EWT', 'ITOT', 'IWV', 'IAK', 'ILCB', 'DIVB', 'ICVT', 'DGRO', 'IFRA'
]

TRAIN_WINDOW_START = '2019-10-01'
TRAIN_WINDOW_END = '2021-01-01'
REAL_WINDOW_START = '2021-01-04'
REAL_WINDOW_END = '2022-01-19'
ROLLING_STEPS = 11
ROLLING_SHIFT_DAYS = 24
NUM_EPOCHS = 200

CONFIG = {
    'batch_size': 64,
    'kernel_size': 21,
    'learning_rate': 0.0001,
    'seq_len': 96,
    'pred_len': 24,
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size: int, stride: int):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size: int):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """Decomposition-Linear model."""

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class Dataset_Custom(Dataset):
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


def shift_date_24(start_date: str, days_to_shift: int) -> str:
    date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    shifted_date = date_obj + timedelta(days=days_to_shift)
    shifted_date_str = shifted_date.strftime('%Y-%m-%d')
    return shifted_date_str


def reset_weights(model: nn.Module) -> None:
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def load_history(ticker: str) -> pd.DataFrame:
    history = pd.read_csv(f'data\\{ticker}.csv')
    history['date'] = pd.to_datetime(history['date'])
    return history


def build_train_array(history: pd.DataFrame, start_date: str, end_date: str) -> np.ndarray:
    data = history[(history['date'] >= start_date) & (history['date'] <= end_date)]
    df = data[['Return']].reset_index(drop=True)
    return df.to_numpy()


def train_once(df_norm: np.ndarray, device: str):
    data_set = Dataset_Custom(
        df=df_norm,
        seq_len=CONFIG['seq_len'],
        pred_len=CONFIG['pred_len'],
    )

    data_loader = DataLoader(
        data_set,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=False,
    )

    model = DLinear(
        seq_len=CONFIG['seq_len'],
        pred_len=CONFIG['pred_len'],
        kernel_size=CONFIG['kernel_size'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.L1Loss().to(device)

    for _ in tqdm(range(NUM_EPOCHS)):
        model.train()
        for _, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


def forecast_next_window(model: nn.Module, df_norm: np.ndarray, device: str) -> pd.DataFrame:
    recent = df_norm[-CONFIG['seq_len']:, :]
    x = torch.tensor(recent[-CONFIG['seq_len']:, :]).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(x.float().to(device))
    recent = np.append(recent, output.squeeze(0).to('cpu').detach().numpy(), axis=0)
    res_df = pd.DataFrame(recent[-24:])
    res_df.columns = ['ret_pred']
    return res_df


def run_for_ticker(ticker: str, device: str) -> None:
    print(ticker)
    start_date = TRAIN_WINDOW_START
    end_date = TRAIN_WINDOW_END
    res_all = pd.DataFrame()

    history = load_history(ticker)

    for _ in range(ROLLING_STEPS):
        df_norm = build_train_array(history, start_date, end_date)
        model = train_once(df_norm, device)
        res_df = forecast_next_window(model, df_norm, device)

        if res_all.empty:
            res_all = res_df
        else:
            res_all = pd.concat([res_all, res_df], axis=0, ignore_index=True)

        start_date = shift_date_24(start_date, ROLLING_SHIFT_DAYS)
        end_date = shift_date_24(end_date, ROLLING_SHIFT_DAYS)
        reset_weights(model)

    real = history[(history['date'] >= REAL_WINDOW_START) & (history['date'] <= REAL_WINDOW_END)].copy()
    print(f'{ticker}:{real.head()}')
    real['pred'] = res_all.values

    plt.figure(figsize=(14, 4))
    plt.plot(real['pred'], label='Predicted', color='blue')
    plt.plot(real['Return'], label='Real', color='red')
    plt.title(f'{ticker} Predicted vs Real')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(0.8, 2.0)
    plt.legend()
    plt.savefig(f'image\\DLinear\\{RDSEED}_{ticker}png', dpi=300)

    real.to_csv(f'res\\DLinear\\{RDSEED}_{ticker}pred.csv', index=False)


def main() -> None:
    set_all_seeds(RDSEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for ticker in TICKERS:
        run_for_ticker(ticker, device)


if __name__ == '__main__':
    main()
