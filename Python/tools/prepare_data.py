# -*- coding: utf-8 -*-
"""
Preparation of the TU-E data files

step 1: data = process_csv(filename)    # load data and do basic preprocessing
step 2: data = make_equidist(data)      # sample-n-hold interpolation to equidistant time
step 3: data = reduce_Ts(data)          # decimation with low-pass filtering to reduce amount of data

.. moduleauthor:: Sascha Eichstaedt (sascha.eichstaedt@ptb.de)
"""

import pandas as pd
from numpy import NaN, nonzero, isnan, searchsorted
from datetime import datetime as dt
import datetime as dtt

def str2dt(text,full_second=True,**kwargs):
    # 
    day,time = text[:-1].split("T")
    date = dt.strptime(day+" "+time,"%Y-%m-%d %H:%M:%S.%f")
    if full_second:
        rnd = dt(date.year,date.month,date.day,date.hour,date.minute,date.second)
        if date.microsecond > 500000:
            date = rnd + dtt.timedelta(seconds=1)
        else:
            date = rnd
    return date
    
def str2float(text):
    if text=="Bad":
        return NaN
    val_str = text.replace(",",".")
    try:
        val = float(val_str)
        return val
    except ValueError:
        return NaN

def deal_NaNs(values):
    inds = nonzero(isnan(values))[0]
    for ind in inds:
        values[ind] = values[ind-1]
    return values

def make_equidist(data,Ts=4):
    # Ts in seconds
    time = pd.date_range(data.index[1],data.index[-1],freq="%ds"%(Ts//2))
    inds = searchsorted(data.index.values,time.values,side="right")-1
    vals = data.ix[inds].values[::2].flatten()
    new_data = pd.Series(vals)
    new_data.index = pd.Index(time[::2])
    return new_data

def reduce_Ts(data,decim=50,use_lowpass=True):
    if use_lowpass:
        from scipy.signal import decimate	
        vals = decimate(data.values,decim)
    else:
	  vals = data.values[::decim]
    time = data.index.values[::decim]
    new_data = pd.DataFrame(vals)
    new_data.index=pd.Index(time)
    return new_data


def process_csv(fname,treat_nans=True,**kwargs):
    data = pd.read_csv(fname,sep=";",header=-1)
    data.columns = ["time","value"]
    data["time"] = data["time"].apply(str2dt,**kwargs)
    data["value"]= data["value"].apply(str2float)
    data = data.set_index("time")
    if treat_nans:    
    	data["value"] = deal_NaNs(data["value"].values)
    return data
