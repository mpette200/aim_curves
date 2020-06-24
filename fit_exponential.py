import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from scipy.optimize import curve_fit

df_prices = pd.read_csv('data/daily_prices.csv', index_col=0)
print(df_prices.info())
df = df_prices[['symbol', 'Date', 'Close']]

def exp_func(t, a, b, c, d):
    '''exponential curve function'''
    return a*t + b*np.exp(c*t) + d

def date_stamps(series):
    tstamps = pd.to_datetime(series).astype('int')
    scaled_tstamps = (tstamps - tstamps.min()) / (tstamps.max() - tstamps.min())
    return tstamps, scaled_tstamps

def fit_curve(df):
    tstamps = pd.to_datetime(df['Date']).astype('int')
    scaled_tstamps = (tstamps - tstamps.min()) / (tstamps.max() - tstamps.min())
    x = scaled_tstamps
    y = df['Close']
    params, covar = curve_fit(
        exp_func, x, y,
        bounds=([0, 1, 1, 0],
                [1e4, 1e4, 10, 1e4]),
        verbose=1)
    y_fitted = exp_func(x, *params)
    # mean absolute percentage error
    mape = (np.abs((y_fitted - y) / y)).mean()*100
    return pd.Series([mape, *params],
        index=['mean_abs_percent_error',
        'params.a', 'params.b', 'params.c', 'params.d'])

df_curves = df.groupby('symbol').apply(fit_curve)
best = df_curves.nsmallest(21, columns='mean_abs_percent_error', keep='all')

plt.figure(figsize=(15, 35))
for n, symbol in enumerate(best.index, 1):
    plt.subplot(7, 3, n)
    prices = df[df.symbol==symbol]
    df_params = df_curves.loc[symbol, 'params.a':]
    _, x = date_stamps(prices['Date'])
    y = prices['Close']
    y_pred = exp_func(x, *df_params)
    plt.plot(x, y, x, y_pred)
    plt.title(symbol)
plt.show()
