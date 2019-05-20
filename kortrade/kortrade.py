# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import keras
import pandas as pd
from pandas_datareader import data as pdr
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import fix_yahoo_finance


def grupredict(share, model1, model2, model3):
    dataset = pdr.get_data_yahoo(share, "2018-01-01")
    dataset = dataset.dropna()

    data = pd.DataFrame(dataset["Open"])
    data = data.join(dataset["High"])
    data = data.join(dataset["Low"])
    data = data.join(dataset["Adj Close"])
    data = data.join(dataset["Volume"])
    data = data.loc[~data.index.duplicated(keep="first")]

    data["change1"] = data["Adj Close"].pct_change()
    data["change5"] = data["Adj Close"].pct_change(periods=5)
    data["ema200"] = ema(data["Adj Close"], 200)
    data["mfi"] = money_flow_index(
        data["High"], data["Low"], data["Adj Close"], data["Volume"], fillna=True
    )
    data["mfi3"] = data["mfi"].pct_change(periods=3)

    data["conv"], data["base"], data["spana"], data["spanb"], data["chikou"] = ichimoku(
        data["High"], data["Low"], data["Adj Close"]
    )

    sc = MinMaxScaler(feature_range=(0, 1))
    x = sc.fit_transform(data[-30:])
    x = np.array([x])

    y1 = model1.predict(x)
    y5 = model2.predict(x)
    mfi3 = model3.predict(x)

    #   df = pd.DataFrame(columns=['stock','y1', 'y5', 'mfi', 'score','prob1','prob5'])
    #   if math.isnan(y1) or math.isnan(y5) or math.isnan(mfi3):
    #       return df

    i1 = np.argmax(y1[0])
    c1 = f"{int(y1[0][i1]*100)}%"
    i5 = np.argmax(y5[0])
    c5 = f"{int(y5[0][i5]*100)}%"

    # y1 = model1.predict(x)
    # y5 = model2.predict(x)

    result = interpret(i1, i5, np.argmax(mfi3[0]))
    # result = interpret(np.argmax(y1[0]), np.argmax(y5[0]), np.argmax(mfi3[0]))
    result1 = list(result)
    result1.insert(0, share)
    result1.append(c1)
    result1.append(c5)
    df = pd.DataFrame(
        [result1], columns=["stock", "y1", "y5", "mfi", "score", "prob1", "prob5"]
    )
    return df


def interpret(y1, y5, mfi):
    y1_class = ["+2", "+1", "0", "-1", "-2"]
    y5_class = ["+5", "+3", "0", "-3", "-5"]
    mfi_class = ["inc", "unchanged", "dec"]
    score = int(y1_class[y1]) + int(y5_class[y5])
    return y1_class[y1], y5_class[y5], mfi_class[mfi], score


def korpredict(share, model1, model2, model3):
    dataset = pdr.get_data_yahoo(share, "2018-01-01")
    dataset = dataset.dropna()

    data = pd.DataFrame(dataset["Open"])
    data = data.join(dataset["High"])
    data = data.join(dataset["Low"])
    data = data.join(dataset["Adj Close"])
    data = data.join(dataset["Volume"])
    data = data.loc[~data.index.duplicated(keep="first")]
    data["change1"] = data["Adj Close"].pct_change()
    data["change5"] = data["Adj Close"].pct_change(periods=5)
    data["ema200"] = ema(data["Adj Close"], 200)
    data["mfi"] = money_flow_index(
        data["High"], data["Low"], data["Adj Close"], data["Volume"], fillna=True
    )
    data["mfi3"] = data["mfi"].pct_change(periods=3)

    data["conv"], data["base"], data["spana"], data["spanb"], data["chikou"] = ichimoku(
        data["High"], data["Low"], data["Adj Close"]
    )

    sc = MinMaxScaler(feature_range=(0, 1))
    x = sc.fit_transform(data)
    X = np.expand_dims(x[-1], 0)
    y1 = model1.predict(X)
    y5 = model2.predict(X)
    mfi3 = model3.predict(X)
    i1 = np.argmax(y1[0])
    c1 = f"{int(y1[0][i1]*100)}%"
    i5 = np.argmax(y5[0])
    c5 = f"{int(y5[0][i5]*100)}%"

    result = interpret(np.argmax(y1[0]), np.argmax(y5[0]), np.argmax(mfi3[0]))
    result1 = list(result)
    result1.insert(0, share)
    result1.append(c1)
    result1.append(c5)

    df = pd.DataFrame(
        [result1], columns=["stock", "y1", "y5", "mfi", "score", "prob1", "prob5"]
    )
    return df


def label_change1(x):
    if x > 0.02:
        return 0  # '+2'
    if x > 0.01:
        return 1  # '+1'
    if x > -0.01:
        return 2  # '0'
    if x > -0.02:
        return 3  # '-1'
    if x <= -0.02:
        return 4  # '-2'
    return math.nan


def label_change5(x):
    if x > 0.05:
        return 0  # '+5'
    if x > 0.03:
        return 1  # '+3'
    if x > -0.03:
        return 2  # '0'
    if x > -0.05:
        return 3  # '-3'
    if x <= -0.05:
        return 4  # '-5'
    return math.nan


def label_mfi3(x):
    threshold = 0.20
    if x >= threshold:
        return 0  # 'increase'
    if x < threshold:
        return 1  # 'unchanged'
    if x < -threshold:
        return 2  # 'decrease'
    return math.nan


def money_flow_index(high, low, close, volume, n=14, fillna=False):
    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ["High", "Low", "Close", "Volume"]
    df["Up_or_Down"] = 0
    df.loc[
        (df["Close"] > df["Close"].shift(1, fill_value=df["Close"].mean())),
        "Up_or_Down",
    ] = 1
    df.loc[
        (df["Close"] < df["Close"].shift(1, fill_value=df["Close"].mean())),
        "Up_or_Down",
    ] = 2

    # 1 typical price
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0

    # 2 money flow
    mf = tp * df["Volume"]

    # 3 positive and negative money flow with n periods
    df["1p_Positive_Money_Flow"] = 0.0
    df.loc[df["Up_or_Down"] == 1, "1p_Positive_Money_Flow"] = mf
    n_positive_mf = df["1p_Positive_Money_Flow"].rolling(n, min_periods=0).sum()

    df["1p_Negative_Money_Flow"] = 0.0
    df.loc[df["Up_or_Down"] == 2, "1p_Negative_Money_Flow"] = mf
    n_negative_mf = df["1p_Negative_Money_Flow"].rolling(n, min_periods=0).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = 100 - (100 / (1 + mr))
    if fillna:
        mr = mr.replace([np.inf, -np.inf], np.nan).fillna(50)
    mfi = pd.Series(mr, name="mfi_" + str(n))
    mfi.replace(0, math.nan, inplace=True)
    return mfi


def adx(high, low, close, n=14):
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: get_min_max(x1, x2, "max"))
    pdn = low.combine(cs, lambda x1, x2: get_min_max(x1, x2, "min"))
    tr = pdm - pdn

    trs_initial = np.zeros(n - 1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs) - 1):
        trs[i] = trs[i - 1] - (trs[i - 1] / float(n)) + tr[n + i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio) - 1):
        dip_mio[i] = dip_mio[i - 1] - (dip_mio[i - 1] / float(n)) + pos[n + i]

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio) - 1):
        din_mio[i] = din_mio[i - 1] - (din_mio[i - 1] / float(n)) + neg[n + i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100 * (dip_mio[i] / trs[i])

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100 * (din_mio[i] / trs[i])

    dx = 100 * np.abs((dip - din) / (dip + din))

    adx = np.zeros(len(trs))
    adx[n] = dx[0:n].mean()

    for i in range(n + 1, len(adx)):
        adx[i] = ((adx[i - 1] * (n - 1)) + dx[i - 1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)
    adx = pd.Series(data=adx, index=close.index)
    return pd.Series(adx, name="adx")


def ichimoku(high, low, close, n1=9, n2=26, n3=52):
    # displacement = n2
    conv = 0.5 * (
        high.rolling(n1, min_periods=0).max() + low.rolling(n1, min_periods=0).min()
    )
    base = 0.5 * (
        high.rolling(n2, min_periods=0).max() + low.rolling(n2, min_periods=0).min()
    )
    spana = 0.5 * (conv + base)
    spanb = 0.5 * (
        high.rolling(n3, min_periods=0).max() + low.rolling(n3, min_periods=0).min()
    )
    spana = spana.shift(n2)
    spanb = spanb.shift(n2)
    #     chikou = close.shift(-n2)
    chikou = close - close[-n2]
    return conv, base, spana, spanb, chikou


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)]  # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()


def get_min_max(x1, x2, f="min"):
    if not np.isnan(x1) and not np.isnan(x2):
        if f == "max":
            max(x1, x2)
        elif f == "min":
            min(x1, x2)
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    else:
        return np.nan
