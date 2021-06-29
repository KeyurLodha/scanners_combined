from pydantic import BaseModel
from pytz import timezone
import json
import copy
import numpy as np
from datetime import timedelta
import datetime as dt
import mysql.connector as cnx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pandas as pd
pd.set_option('chained_assignment', None)


def round_minutes(dt, direction, resolution):
    new_minute = (dt.minute // resolution +
                  (1 if direction == 'up' else 0)) * resolution
    return dt + timedelta(minutes=new_minute - dt.minute)


class Input(BaseModel):
    input_date: dt.date
    cutoff_time: dt.time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get('/')
def show_form(request: Request):
    return templates.TemplateResponse('relstrength.html', context={'request': request})


@app.get('/btst')
def show_form(request: Request):
    return templates.TemplateResponse('btst.html', context={'request': request})


@app.get("/api/relstrength")
def callRelStrengthAPI(request: Request):
    print('RELSTRENGTH POSTED')
    format = "%Y-%m-%d %H:%M"
    now_utc = dt.datetime.strptime(dt.datetime.now().strftime(format), format)
    now_ist = now_utc.astimezone(timezone('Asia/Kolkata'))
    curr_datetime = dt.datetime.strptime(now_ist.strftime(format), format)
    curr_date = curr_datetime.date()
    curr_time = curr_datetime.time()

    if curr_date.weekday() == 5:
        curr_date = curr_date - timedelta(days=1)
    elif curr_date.weekday() == 6:
        curr_date = curr_date - timedelta(days=2)

    if curr_time < dt.time(9, 30):
        rtn_data = {'scrips': [], 'shortlist_buy': [],
                    'shortlist_sell': [], 'time': ''}
        payload = json.dumps(rtn_data)
    elif curr_time > dt.time(15, 29):
        start_time = dt.time(15, 15)
        end_time = dt.time(15, 29)
        scrips, shortlist_buy, shortlist_sell = relstrength_func(
            curr_date, start_time, end_time)
        s1 = scrips.to_dict(orient="records")
        s2 = shortlist_buy.to_dict(orient="records")
        s3 = shortlist_sell.to_dict(orient="records")
        time_str = f'{start_time} - {end_time}'
        rtn_data = {'scrips': s1, 'shortlist_buy': s2,
                    'shortlist_sell': s3, 'time': time_str}
        payload = json.dumps(rtn_data)
    else:
        if curr_time.minute % 15 == 0:
            start_time = (curr_datetime - timedelta(minutes=15)).time()
            end_time = curr_time
        else:
            end_time_dt = round_minutes(curr_datetime, 'down', 15)
            end_time = end_time_dt.time()
            start_time = (end_time_dt - timedelta(minutes=15)).time()

        # curr_date = dt.date(2021, 6, 16)
        # start_time = dt.time(9, 30)
        # end_time = dt.time(15, 15)

        scrips, shortlist_buy, shortlist_sell = relstrength_func(
            curr_date, start_time, end_time)
        s1 = scrips.to_dict(orient="records")
        s2 = shortlist_buy.to_dict(orient="records")
        s3 = shortlist_sell.to_dict(orient="records")
        time_str = f'{start_time} - {end_time}'
        rtn_data = {'scrips': s1, 'shortlist_buy': s2,
                    'shortlist_sell': s3, 'time': time_str}
        payload = json.dumps(rtn_data)

    return payload

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

##### FUNC DEFNS #######


def get_time(time_list, time_var, input_date):
    time_range = [((dt.datetime.combine(input_date, time_var)) -
                   timedelta(minutes=i)).time() for i in range(0, 3)]
    intersect = list(set(time_list).intersection(time_range))

    if len(intersect) == 1:
        rt_time = intersect[0]
        time_flag = True
    else:
        intersect.sort()
        idx_list = [abs(ele.minute-time_var.minute) for ele in intersect]
        try:
            idx = idx_list.index(min(idx_list))
            rt_time = intersect[idx]
            time_flag = True
        except ValueError as e:
            rt_time = dt.time(0, 0)
            time_flag = False

    return rt_time, time_flag


def resampler(data, sample_interval, date_col, agg_dict, na_subset):

    sampled_df = data.resample(
        sample_interval, on=date_col).aggregate(agg_dict)
    sampled_df.dropna(subset=[na_subset], inplace=True)
    sampled_df.reset_index(inplace=True)

    return sampled_df


def get_vol(input_date, start_time, end_time):

    day_20 = input_date - dt.timedelta(days=20)
    yest = input_date - dt.timedelta(days=1)

    try:
        stocks_db = cnx.connect(host="164.52.207.158", user="stock",
                                password="stockdata@data", database='stock_production')

        eq_query = f'select instrument_id, ins_date, volume from instrument_scan where date(ins_date) between "{day_20}" and "{yest}";'
        eq_df = pd.read_sql(eq_query, stocks_db, parse_dates=['ins_date'])

        sl_query = 'select id, tradingsymbol from instruments where f_n_o=1 and tradingsymbol not like "%NIFTY%";'
        sl_df = pd.read_sql(sl_query, stocks_db)

        stocks_db.close()
    except Exception as e:
        stocks_db.close()
        print(str(e))

    eq_df.drop_duplicates(subset=['instrument_id', 'ins_date'], inplace=True)
    eq_df['date'] = eq_df['ins_date'].dt.date
    eq_df['time'] = eq_df['ins_date'].dt.time

    eq_df.drop(eq_df[(eq_df['time'] < start_time)].index, inplace=True)
    eq_df.drop(eq_df[(eq_df['time'] > end_time)].index, inplace=True)
    eq_df.reset_index(inplace=True, drop=True)

    id_dict = dict(sl_df.values)
    params_list = []

    for id, name in id_dict.items():

        scrip_df = eq_df[eq_df['instrument_id'] == id]
        agg_dict = {'instrument_id': 'first', 'volume': 'sum'}
        scrip_resample = resampler(
            scrip_df, '1D', 'ins_date', agg_dict, 'instrument_id')
        scrip_resample['10Davg'] = scrip_resample['volume'].rolling(
            window=10).mean()
        last_10d_vol = scrip_resample[-1:]['10Davg'].to_list()[0]

        params_list.append([id, name, last_10d_vol])

    vol_df = pd.DataFrame(params_list, columns=['id', 'name', '10Dvol'])

    return vol_df


def relstrength_func(input_date, start_time, end_time):

    try:
        stocks_db = cnx.connect(host="164.52.207.158", user="stock",
                                password="stockdata@data", database='stock_production')

        eq_query = f'select instrument_id, ins_date, open, high, low, close, volume from instrument_scan where date(ins_date)="{input_date}" ;'
        eq_df = pd.read_sql(eq_query, stocks_db, parse_dates=['ins_date'])

        high_low_query = 'select * from instrument_high;'
        high_low_df = pd.read_sql(high_low_query, stocks_db)

        sl_query = 'select id, tradingsymbol from instruments where f_n_o=1 and tradingsymbol not like "%NIFTY%";'
        sl_df = pd.read_sql(sl_query, stocks_db)

        stocks_db.close()
    except Exception as e:
        stocks_db.close()
        print(str(e))

    eq_df.drop_duplicates(subset=['instrument_id', 'ins_date'], inplace=True)
    eq_df['date'] = eq_df['ins_date'].dt.date
    eq_df['time'] = eq_df['ins_date'].dt.time

    eq_df.drop(eq_df[(eq_df['time'] < dt.time(9, 15))].index, inplace=True)
    eq_df.drop(eq_df[(eq_df['time'] > dt.time(15, 29))].index, inplace=True)
    eq_df.reset_index(inplace=True, drop=True)
    eq_df.sort_values(by=['instrument_id', 'ins_date'], inplace=True)

    vol_df = get_vol(input_date, start_time, end_time)

    temp_st = copy.deepcopy(start_time)
    temp_et = copy.deepcopy(end_time)

    bchmrk_df = eq_df[eq_df['instrument_id'] == 417]
    bchmrk_time_list = set(list(bchmrk_df['time']))

    if (start_time in bchmrk_time_list) and (end_time in bchmrk_time_list):
        bchmrk_open = bchmrk_df[bchmrk_df['time'] == start_time]['open'].to_list()[
            0]
        bchmrk_close = bchmrk_df[bchmrk_df['time'] == end_time]['close'].to_list()[
            0]
        bchmrk_pc = round(((bchmrk_close - bchmrk_open)/bchmrk_open)*100, 4)
    elif (start_time not in bchmrk_time_list) and (end_time in bchmrk_time_list):
        start_time, time_flag = get_time(
            bchmrk_time_list, start_time, input_date)
        if time_flag == True:
            bchmrk_open = bchmrk_df[bchmrk_df['time'] == start_time]['open'].to_list()[
                0]
            bchmrk_close = bchmrk_df[bchmrk_df['time'] == end_time]['close'].to_list()[
                0]
            bchmrk_pc = round(
                ((bchmrk_close - bchmrk_open)/bchmrk_open)*100, 4)
        else:
            bchmrk_pc = np.nan
    elif (start_time in bchmrk_time_list) and (end_time not in bchmrk_time_list):
        end_time, time_flag = get_time(bchmrk_time_list, end_time, input_date)
        if time_flag == True:
            bchmrk_open = bchmrk_df[bchmrk_df['time'] == start_time]['open'].to_list()[
                0]
            bchmrk_close = bchmrk_df[bchmrk_df['time'] == end_time]['close'].to_list()[
                0]
            bchmrk_pc = round(
                ((bchmrk_close - bchmrk_open)/bchmrk_open)*100, 4)
        else:
            bchmrk_pc = np.nan
    else:
        start_time, time_flag1 = get_time(
            bchmrk_time_list, start_time, input_date)
        end_time, time_flag2 = get_time(bchmrk_time_list, end_time, input_date)
        if (time_flag1 == True) and (time_flag2 == True):
            bchmrk_open = bchmrk_df[bchmrk_df['time'] == start_time]['open'].to_list()[
                0]
            bchmrk_close = bchmrk_df[bchmrk_df['time'] == end_time]['close'].to_list()[
                0]
            bchmrk_pc = round(
                ((bchmrk_close - bchmrk_open)/bchmrk_open)*100, 4)
        else:
            bchmrk_pc = np.nan

    id_dict = dict(sl_df.values)
    params_list = []

    for id, name in id_dict.items():

        start_time = copy.deepcopy(temp_st)
        end_time = copy.deepcopy(temp_et)
        stock_df = eq_df[eq_df['instrument_id'] == id]
        stock_time_list = set(list(stock_df['time']))

        if (start_time in stock_time_list) and (end_time in stock_time_list):
            stock_open = stock_df[stock_df['time'] == start_time]['open'].to_list()[
                0]
            stock_close = stock_df[stock_df['time'] == end_time]['close'].to_list()[
                0]
            stock_pc = round(((stock_close - stock_open)/stock_open)*100, 4)
        elif (start_time not in stock_time_list) and (end_time in stock_time_list):
            start_time, time_flag = get_time(
                stock_time_list, start_time, input_date)
            if time_flag == True:
                stock_open = stock_df[stock_df['time'] == start_time]['open'].to_list()[
                    0]
                stock_close = stock_df[stock_df['time'] == end_time]['close'].to_list()[
                    0]
                stock_pc = round(
                    ((stock_close - stock_open)/stock_open)*100, 4)
            else:
                stock_open = stock_close = stock_pc = np.nan
        elif (start_time in stock_time_list) and (end_time not in stock_time_list):
            end_time, time_flag = get_time(
                stock_time_list, end_time, input_date)
            if time_flag == True:
                stock_open = stock_df[stock_df['time'] == start_time]['open'].to_list()[
                    0]
                stock_close = stock_df[stock_df['time'] == end_time]['close'].to_list()[
                    0]
                stock_pc = round(
                    ((stock_close - stock_open)/stock_open)*100, 4)
            else:
                stock_open = stock_close = stock_pc = np.nan
        else:
            start_time, time_flag1 = get_time(
                stock_time_list, start_time, input_date)
            end_time, time_flag2 = get_time(
                stock_time_list, end_time, input_date)
            if (time_flag1 == True) and (time_flag2 == True):
                stock_open = stock_df[stock_df['time'] == start_time]['open'].to_list()[
                    0]
                stock_close = stock_df[stock_df['time'] == end_time]['close'].to_list()[
                    0]
                stock_pc = round(
                    ((stock_close - stock_open)/stock_open)*100, 4)
            else:
                stock_open = stock_close = stock_pc = np.nan

        rs_wo_beta = round(stock_pc - bchmrk_pc, 4)

        stock_range = stock_df[(stock_df['time'] >= start_time) & (
            stock_df['time'] <= end_time)]
        today_high = stock_range['high'].max()
        today_low = stock_range['low'].min()

        stock_hl_df = high_low_df[high_low_df['instrument_id'] == id]
        try:
            high_20d = stock_hl_df['twentyH'].to_list()[0]
        except IndexError as e:
            high_20d = np.nan

        try:
            high_50d = stock_hl_df['fiftyH'].to_list()[0]
        except IndexError as e:
            high_50d = np.nan

        try:
            high_250d = stock_hl_df['twofiftyH'].to_list()[0]
        except IndexError as e:
            high_250d = np.nan

        try:
            low_20d = stock_hl_df['twentyL'].to_list()[0]
        except IndexError as e:
            low_20d = np.nan

        try:
            low_50d = stock_hl_df['fiftyL'].to_list()[0]
        except IndexError as e:
            low_50d = np.nan

        try:
            low_250d = stock_hl_df['twofiftyL'].to_list()[0]
        except IndexError as e:
            low_250d = np.nan

        high_vs_20d = 'True' if today_high > high_20d else 'False'
        high_vs_50d = 'True' if today_high > high_50d else 'False'
        high_vs_250d = 'True' if today_high > high_250d else 'False'

        low_vs_20d = 'True' if today_low < low_20d else 'False'
        low_vs_50d = 'True' if today_low < low_50d else 'False'
        low_vs_250d = 'True' if today_low < low_250d else 'False'

        last_10davg = vol_df[vol_df['id'] == id]['10Dvol'].to_list()[0]
        today_vol = stock_range['volume'].sum()
        vol_ratio = round(today_vol/last_10davg, 3)

        params_list.append([id, name, bchmrk_pc, stock_pc, rs_wo_beta, high_vs_20d, high_vs_50d,
                            high_vs_250d, low_vs_20d, low_vs_50d, low_vs_250d, stock_close, vol_ratio])

    scrips = pd.DataFrame(params_list, columns=['id', 'name', 'bchmrk_pc', 'stock_pc', 'rs_wo_beta', 'high_vs_20d',
                                                'high_vs_50d', 'high_vs_250d', 'low_vs_20d', 'low_vs_50d', 'low_vs_250d', 'LTP', 'vol_ratio'])
    scrips.sort_values(by='rs_wo_beta', ascending=False, inplace=True)

    shortlist_buy = scrips.head(10)
    shortlist_buy['priority'] = np.nan
    shortlist_buy.loc[shortlist_buy['high_vs_250d']
                      == 'True', ['priority']] = 1
    shortlist_buy.loc[(shortlist_buy['high_vs_50d'] == 'True') & (
        shortlist_buy['high_vs_250d'] == 'False'), ['priority']] = 2
    shortlist_buy.loc[(shortlist_buy['high_vs_20d'] == 'True') & (
        shortlist_buy['high_vs_50d'] == 'False') & (shortlist_buy['high_vs_250d'] == 'False'), ['priority']] = 3
    shortlist_buy.drop(
        shortlist_buy[shortlist_buy['priority'].isna()].index, inplace=True)
    shortlist_buy.sort_values(by='priority', inplace=True)

    shortlist_sell = scrips.head(10)
    shortlist_sell['priority'] = np.nan
    shortlist_sell.loc[shortlist_sell['low_vs_250d']
                       == 'True', ['priority']] = 1
    shortlist_sell.loc[(shortlist_sell['low_vs_50d'] == 'True') & (
        shortlist_sell['low_vs_250d'] == 'False'), ['priority']] = 2
    shortlist_sell.loc[(shortlist_sell['low_vs_20d'] == 'True') & (
        shortlist_sell['low_vs_50d'] == 'False') & (shortlist_sell['low_vs_250d'] == 'False'), ['priority']] = 3
    shortlist_sell.drop(
        shortlist_sell[shortlist_sell['priority'].isna()].index, inplace=True)
    shortlist_sell.sort_values(by='priority', inplace=True)

    scrips.fillna('', inplace=True)
    shortlist_buy.fillna('', inplace=True)
    shortlist_sell.fillna('', inplace=True)

    return scrips, shortlist_buy, shortlist_sell


# BTST

@app.post("/api/btst")
def callBTSTAPI(request: Request, args: Input):
    print('BTST POSTED')
    input_date = args.input_date
    cutoff_time = args.cutoff_time

    try:
        full_list, btst, stbt, manual_btst, manual_stbt = btst_func(
            input_date, cutoff_time)
        s1 = full_list.to_dict(orient="records")
        s2 = btst.to_dict(orient="records")
        s3 = stbt.to_dict(orient="records")
        s4 = manual_btst.to_dict(orient="records")
        s5 = manual_stbt.to_dict(orient="records")
        # time_str = f'{start_time} - {end_time}'
        rtn_data = {'full_list': s1, 'btst': s2, 'stbt': s3,
                    'manual_btst': s4, 'manual_stbt': s5}
        payload = json.dumps(rtn_data)
    except KeyError as e:
        rtn_data = {'full_list': [], 'btst': [], 'stbt': [],
                    'manual_btst': [], 'manual_stbt': []}
        payload = json.dumps(rtn_data)

    return payload


def resampler(data, sample_interval, date_col, agg_dict, na_subset):
    sampled_df = data.resample(
        sample_interval, on=date_col).aggregate(agg_dict)
    sampled_df.dropna(subset=[na_subset], inplace=True)
    sampled_df.reset_index(inplace=True)
    return sampled_df


def TrueRange(data):
    data = data.copy()
    data["TR"] = np.nan
    for i in range(1, len(data)):
        h = data.loc[i, "high"]
        l = data.loc[i, "low"]
        pc = data.loc[i-1, "close"]
        x = h-l
        y = abs(h-pc)
        z = abs(l-pc)
        TR = max(x, y, z)
        data.loc[i, "TR"] = TR
    return data


def average_true_range(data, period, drop_tr=True, smoothing="RMA"):
    data = data.copy()
    if smoothing == "RMA":
        data['atr_' + str(period) + '_' + str(smoothing)
             ] = data['TR'].ewm(com=period - 1, min_periods=period).mean()
    elif smoothing == "SMA":
        data['atr_' + str(period) + '_' + str(smoothing)
             ] = data['TR'].rolling(window=period).mean()
    elif smoothing == "EMA":
        data['atr_' + str(period) + '_' + str(smoothing)
             ] = data['TR'].ewm(span=period, adjust=False).mean()
    if drop_tr:
        data.drop(['TR'], inplace=True, axis=1)
    data = data.round(decimals=2)
    return data


def get_supertrend(high, low, close, lookback, multiplier):

    # ATR
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns=['upper', 'lower'])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1, 0]) | (close[i-1] > final_bands.iloc[i-1, 0]):
                final_bands.iloc[i, 0] = upper_band[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i-1, 0]

    # FINAL LOWER BAND
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1, 1]) | (close[i-1] < final_bands.iloc[i-1, 1]):
                final_bands.iloc[i, 1] = lower_band[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i-1, 1]

    # SUPERTREND
    supertrend = pd.DataFrame(columns=[f'supertrend_{lookback}'])
    supertrend.iloc[:, 0] = [
        x for x in final_bands['upper'] - final_bands['upper']]

    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]
    st = pd.Series(supertrend.iloc[:, 0])

    return st


def btst_func(trading_date, cutoff_time):

    try:
        stocks_db = cnx.connect(host="164.52.207.158", user="stock",
                                password="stockdata@data", database='stock_production')
        stock_query = f'select instrument_id, ins_date, open, high, low, close, volume from instrument_scan where date(ins_date) between "{trading_date - dt.timedelta(days=50)}" and "{trading_date}";'
        stock_df = pd.read_sql(stock_query, stocks_db,
                               parse_dates=['ins_date'])

        oi_query = f'select id, ins_date, oi from future_all where date(ins_date) between "{trading_date - dt.timedelta(days=5)}" and "{trading_date}";'
        oi_df = pd.read_sql(oi_query, stocks_db, parse_dates=['ins_date'])

        sl_query = 'select id, tradingsymbol from instruments where f_n_o=1 and tradingsymbol not like "%NIFTY%";'
        sl_df = pd.read_sql(sl_query, stocks_db)

        stocks_db.close()
    except Exception as e:
        stocks_db.close()
        print(str(e))

    mkt_start = dt.time(9, 15)
    mkt_end = dt.time(15, 29)

    stock_df.drop_duplicates(
        subset=['instrument_id', 'ins_date'], inplace=True)
    stock_df['date'] = stock_df['ins_date'].dt.date
    stock_df['time'] = stock_df['ins_date'].dt.time

    stock_df.drop(stock_df[(stock_df['time'] < mkt_start)].index, inplace=True)
    stock_df.drop(stock_df[(stock_df['time'] > mkt_end)].index, inplace=True)
    stock_df.reset_index(inplace=True, drop=True)

    oi_df['date'] = oi_df['ins_date'].dt.date
    oi_df['time'] = oi_df['ins_date'].dt.time

    oi_df.drop(oi_df[(oi_df['time'] < mkt_start)].index, inplace=True)
    oi_df.drop(oi_df[(oi_df['time'] > mkt_end)].index, inplace=True)
    oi_df.reset_index(inplace=True, drop=True)

    # will prevent against empty df for that day
    instru_ids = stock_df[stock_df['date'] ==
                          trading_date]['instrument_id'].unique()
    sl_df = sl_df[sl_df['id'].isin(instru_ids)]
    stock_dict = dict(sl_df.values)

    # stock_dict = {4: 'WIPRO'}
    # stock_dict = dict(sl_df.values)

    scanner_list = []

    for id, name in stock_dict.items():

        scrip_df = stock_df[stock_df['instrument_id']
                            == id].reset_index(drop=True)
        scrip_oi = oi_df[oi_df['id'] == id].reset_index(drop=True)

        # OI
        try:
            yest_close_oi = scrip_oi.iloc[scrip_oi[scrip_oi['date'] == trading_date].head(
                1).index-1]['oi'].to_list()[0]
            tdy_coff_oi = scrip_oi[(scrip_oi['date'] == trading_date) & (
                scrip_oi['time'] >= cutoff_time)]['oi'].to_list()[0]
            oic_yest_coff = round(
                ((tdy_coff_oi - yest_close_oi)/yest_close_oi)*100, 3)
        except IndexError as e:
            oic_yest_coff = np.nan

        # VOL
        vol_df = scrip_df.copy()
        vol_df.drop(vol_df[(vol_df['time'] > cutoff_time)].index, inplace=True)

        agg_dict = {'instrument_id': 'first', 'open': 'first',
                    'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        vol_resample = resampler(vol_df, '1D', 'ins_date', agg_dict, 'close')
        vol_resample.drop(
            vol_resample[vol_resample['volume'] == 0].index, inplace=True)
        vol_resample.reset_index(inplace=True, drop=True)
        vol_resample['10Davg'] = vol_resample['volume'].rolling(
            window=10).mean()
        vol_resample['date'] = vol_resample['ins_date'].dt.date
        vol_resample['time'] = vol_resample['ins_date'].dt.time

        tdy_vol = vol_resample[vol_resample['date'] == trading_date]['volume'].to_list()[
            0]
        last_10d_vol = vol_resample.iloc[vol_resample[vol_resample['date']
                                                      == trading_date].index-1]['10Davg'].to_list()[0]
        vc_tdy_10day = round(tdy_vol/last_10d_vol, 3)

        # CANDLE 80
        cutoff_df = vol_resample[(vol_resample['date'] == trading_date)]
        cutoff_high = cutoff_df['high'].to_list()[0]
        cutoff_low = cutoff_df['low'].to_list()[0]
        cutoff_close = cutoff_df['close'].to_list()[0]

        candle_80_type = 'Green' if cutoff_close >= cutoff_low+(0.8*(cutoff_high-cutoff_low)) else (
            'Red' if cutoff_close <= cutoff_high-(0.8*(cutoff_high-cutoff_low)) else 'NA')

        if candle_80_type == 'Green':
            candle_80_val = (((cutoff_close - (cutoff_low+(0.8*(cutoff_high-cutoff_low)))) + (
                0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            candle_80_val = round(candle_80_val, 2)
        elif candle_80_type == 'Red':
            candle_80_val = ((((cutoff_high-(0.8*(cutoff_high-cutoff_low))) - cutoff_close) + (
                0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            candle_80_val = round(candle_80_val, 2)
        else:
            v1 = (((cutoff_close - (cutoff_low+(0.8*(cutoff_high-cutoff_low)))
                    ) + (0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            v2 = ((((cutoff_high-(0.8*(cutoff_high-cutoff_low))) - cutoff_close) +
                   (0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            candle_80_val = (round(v1, 2), round(v2, 2))

        # PC 1PM-COFF
        try:
            tdy_100_open = scrip_df[(scrip_df['date'] == trading_date) & (
                scrip_df['time'] >= dt.time(13, 0))]['open'].to_list()[0]
            pc_100_coff = round(
                ((cutoff_close - tdy_100_open)/tdy_100_open)*100, 3)
        except IndexError as e:
            pc_100_coff = np.nan

        # ATR
        agg_dict = {'instrument_id': 'first', 'open': 'first',
                    'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        scrip_resample = resampler(
            scrip_df, '1D', 'ins_date', agg_dict, 'close')

        scrip_tr = TrueRange(scrip_resample)
        scrip_atr = average_true_range(
            data=scrip_tr, period=14, drop_tr=True, smoothing="RMA")
        yest = scrip_atr.iloc[scrip_atr[scrip_atr['ins_date'].dt.date ==
                                        trading_date].index-1]
        atr_val = yest['atr_14_RMA'].to_list()[0]
        yest_close = yest['close'].to_list()[0]
        atr_flag_btst = 'True' if cutoff_close > (
            yest_close+(0.5*atr_val)) else 'False'
        atr_flag_stbt = 'True' if cutoff_close < (
            yest_close-(0.5*atr_val)) else 'False'

        # ST
        st = get_supertrend(
            scrip_resample['high'], scrip_resample['low'], scrip_resample['close'], 10, 2)
        stock_st = round(st.iloc[-1], 2)
        abv_st = 'Yes' if cutoff_close > stock_st else 'No'

        params = [id, name, cutoff_close, atr_val, atr_flag_btst, atr_flag_stbt, pc_100_coff,
                  oic_yest_coff, vc_tdy_10day, candle_80_type, candle_80_val, stock_st, abv_st]
        scanner_list.append(params)
        # print(id, name)

    full_list = pd.DataFrame(scanner_list, columns=['id', 'name', 'cutoff_close', 'atr_val', 'atr_flag_btst', 'atr_flag_stbt',
                                                    'pc_100_coff', 'oic_yest_coff', 'vc_tdy_10day', 'candle_80_type', 'candle_80_val', 'stock_st', 'abv_st'])
    btst = full_list[(full_list['atr_flag_btst'] == 'True') & (abs(full_list['oic_yest_coff']) > 4) & (
        full_list['vc_tdy_10day'] > 1) & (full_list['candle_80_type'] == 'Green')]
    stbt = full_list[(full_list['atr_flag_stbt'] == 'True') & (abs(full_list['oic_yest_coff']) > 4) & (
        full_list['vc_tdy_10day'] > 1) & (full_list['candle_80_type'] == 'Red')]
    manual_btst = full_list[(full_list['atr_flag_btst'] == 'True') & (
        full_list['vc_tdy_10day'] > 1) & (full_list['candle_80_type'] == 'Green')]
    manual_stbt = full_list[(full_list['atr_flag_stbt'] == 'True') & (
        full_list['vc_tdy_10day'] > 1) & (full_list['candle_80_type'] == 'Red')]

    btst_ids = btst['id'].unique()
    manual_btst = manual_btst[~manual_btst['id'].isin(btst_ids)]
    manual_btst.reset_index(inplace=True, drop=True)

    stbt_ids = stbt['id'].unique()
    manual_stbt = manual_stbt[~manual_stbt['id'].isin(stbt_ids)]
    manual_stbt.reset_index(inplace=True, drop=True)

    full_list.fillna('', inplace=True)
    btst.fillna('', inplace=True)
    stbt.fillna('', inplace=True)
    manual_btst.fillna('', inplace=True)
    manual_stbt.fillna('', inplace=True)

    return full_list, btst, stbt, manual_btst, manual_stbt
