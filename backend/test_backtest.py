# -*-coding:utf-8-*-
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import Option
from dateutil.relativedelta import relativedelta
import datetime
import matplotlib.pyplot as plt
import Backtest

def test(start_date, period, otm_pct, maturity, flag, delta):
    result = Backtest.Backtest(start_date, period, otm_pct, maturity, flag, delta)
    result.run()
    print(result.daily_returns[['daily PnL', 'daily SPY PnL']])
    print(result.monthly_returns)

start_date = datetime.datetime.strptime('01/01/2015','%m/%d/%Y')
test(start_date, 78, 20, 3, 1, 10)