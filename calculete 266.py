import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

TICKERS = [
    "IVV", "IEFA", "AGG", "IWF", "IJH", "IEMG", "IJR", "IWM", "IWD", "ITOT",
    "EFA", "TLT", "IVW", "QUAL", "IXUS", "MUB", "IWB", "IWR", "IVE", "MBB",
    "IEF", "IAU", "LQD", "IUSB", "DGRO", "GOVT", "SHY", "USMV", "IGSB", "EEM",
    "SHV", "ACWI", "DVY", "TIP", "IUSG", "IYW", "EFV", "IUSV", "HYG", "EWJ",
    "EMB", "SOXX", "PFF", "IWP", "IWV", "IEI", "IGIB", "SLV", "IWS", "EFG",
    "OEF", "IWN", "IWO", "USIG", "MTUM", "INDA", "HDV", "IWY", "IJK", "SCZ",
    "SUB", "IQLT", "EZU", "STIP", "TLH", "IJJ", "FLOT", "IBB", "VLUE", "TFLO",
    "HEFA", "IJS", "EFAV", "ITA", "IGV", "SHYG", "IAGG", "IJT", "MCHI", "ICSH",
    "IOO", "EWY", "IHI", "IEUR", "EWT", "FXI", "EWZ", "ACWX", "IXN", "IGM",
    "EEMV", "DSI", "IDV", "ACWV", "ISTB", "IXJ", "IGF", "IXC", "REET", "SUSA",
    "URTH", "XT", "IYH", "GVI", "NEAR", "IYR", "EWU", "CMF", "EWC", "IYF",
    "IBDQ", "ITB", "IBDP", "AAXJ", "ICLN", "USRT", "IGLB", "ILCG", "IWX",
    "SLQD", "IMTM", "AOR", "EWW", "IMCG", "LRGF", "EPP", "ICVT", "ICF", "IYY",
    "IPAC", "AOA", "IVLU", "IEV", "EWA", "EUFN", "IYJ", "AIA", "ILF", "IYE",
    "IWL", "AOM", "IYG", "PICK", "IYK", "IDU", "IAI", "EWL", "INTF", "GSG",
    "SMLF", "QLTA", "EWP", "COMT", "EWG", "CRBN", "IYC", "ILCV", "SMIN", "ILCB",
    "IYT", "INDY", "KXI", "IWC", "IMCB", "IEO", "KSA", "IHF", "EUSA", "DVYE",
    "NYF", "IHE", "IYM", "EMGF", "EWQ", "REZ", "ECH", "IAK", "AGZ", "MEAR",
    "IMCV", "REM", "IAT", "AOK", "EXI", "IGE", "ILTB", "ISCG", "ISCF", "EWH",
    "IGOV", "RING", "EWI", "HEZU", "EWS", "EEMA", "LEMB", "CEMB", "FM", "ISCV",
    "EIDO", "CMBS", "EEMS", "IXG", "EMHY", "HEWJ", "EWD", "IXP", "LQDH", "EPOL",
    "GNMA", "SIZE", "EWM", "EDEN", "EWN", "EZA", "IEZ", "MXI", "HYGH", "RXI",
    "TUR", "THD", "SLVP", "HAWX", "GBF", "ISCB", "IYZ", "TOK", "WOOD", "BYLD",
    "EWZS", "HEEM", "IGBH", "EIS", "HSCZ", "JXI", "GHYG", "VEGI", "GLOF", "EPU",
    "SCJ", "EIRL", "IYLD", "FILL", "EPHE", "IFGL", "IEUS", "JPXN", "ENZL", "ISHG",
    "QAT", "BKF", "EWO", "FIBR", "ECNS", "HYXU", "DVYA", "EWUS", "IDGT", "UAE",
    "WPS", "HEWG", "ENOR", "EMIF", "EFNL", "EWK", "ISZE"
]


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def calculate_monthly_avg(pred_data: pd.DataFrame) -> pd.DataFrame:
    pred_data = pred_data.copy()
    pred_data['date'] = pd.to_datetime(pred_data['date'])
    pred_data['year_month'] = pred_data['date'].dt.to_period('M')
    monthly_avg = pred_data.groupby('year_month').agg({'pred': 'mean', 'Return': 'mean'}).reset_index()
    return monthly_avg


def calculate_mean_return(real_data: pd.DataFrame) -> pd.DataFrame:
    real_data = real_data.copy()
    real_data['date'] = pd.to_datetime(real_data['date'])
    real_data.set_index('date', inplace=True)
    monthly_starts = real_data.resample('MS').first().index

    results = []
    for start_date in monthly_starts:
        date_range_start = start_date - pd.Timedelta(days=50)
        date_range_end = start_date - pd.Timedelta(days=1)
        filtered_data = real_data.loc[date_range_start:date_range_end]
        mean_return = filtered_data['Return'].mean()
        results.append({'month': start_date, 'mean_return': mean_return})

    results_df = pd.DataFrame(results)
    results_df['month'] = pd.to_datetime(results_df['month'])
    results_df['year_month'] = results_df['month'].dt.strftime('%Y-%m')
    return results_df


def merge_data(monthly_avg: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    monthly_avg = monthly_avg.copy()
    results_df = results_df.copy()
    monthly_avg['year_month'] = monthly_avg['year_month'].astype(str)
    results_df['year_month'] = results_df['year_month'].astype(str)
    merged_df = pd.merge(monthly_avg, results_df[['year_month', 'mean_return']], how='left', on='year_month')
    return merged_df


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }
    return metrics


def collect_ticker_metrics(ticker: str) -> list:
    pred_data = load_data(f'res/total/123_{ticker}pred.csv')
    real_data = load_data(f'data/total/{ticker}.csv')

    monthly_avg = calculate_monthly_avg(pred_data)
    results_df_ticker = calculate_mean_return(real_data)
    merged_df = merge_data(monthly_avg, results_df_ticker)

    pred_metrics = calculate_metrics(merged_df['Return'], merged_df['pred'])
    mean_return_metrics = calculate_metrics(merged_df['Return'], merged_df['mean_return'])

    return [
        {'Ticker': ticker, 'Metric': 'pred_mae', 'Value': pred_metrics['MAE']},
        {'Ticker': ticker, 'Metric': 'pred_mse', 'Value': pred_metrics['MSE']},
        {'Ticker': ticker, 'Metric': 'pred_rmse', 'Value': pred_metrics['RMSE']},
        {'Ticker': ticker, 'Metric': 'pred_mape', 'Value': pred_metrics['MAPE']},
        {'Ticker': ticker, 'Metric': 'mean_mae', 'Value': mean_return_metrics['MAE']},
        {'Ticker': ticker, 'Metric': 'mean_mse', 'Value': mean_return_metrics['MSE']},
        {'Ticker': ticker, 'Metric': 'mean_rmse', 'Value': mean_return_metrics['RMSE']},
        {'Ticker': ticker, 'Metric': 'mean_mape', 'Value': mean_return_metrics['MAPE']},
    ]


def main() -> pd.DataFrame:
    results_list = []
    for ticker in TICKERS:
        results_list.extend(collect_ticker_metrics(ticker))
    return pd.DataFrame(results_list)


if __name__ == '__main__':
    results_df = main()
    transposed_df = results_df.pivot(index='Ticker', columns='Metric', values='Value')
    transposed_df.to_csv(f'{len(TICKERS)}_Mean vs Pred.csv')
