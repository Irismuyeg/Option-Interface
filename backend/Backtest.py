# -*-coding:utf-8-*-
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
from backend import Option
from dateutil.relativedelta import relativedelta
import datetime
import matplotlib.pyplot as plt


# tmp_Spy_PnL = 0
# tmp_PnL = 0
class Backtest:
    def __init__(self, start_date, period, otm_pct, maturity, flag, delta):
        self.start_date = start_date
        self.period = period
        self.end_date = self.start_date + relativedelta(months = period)
        self.SPY = pd.read_excel('data.xlsx',sheet_name='SPY', engine='openpyxl')
        self.SPY.set_index('Date',inplace = True)
        self.SPY = self.SPY.loc[self.start_date:self.end_date]
        self.SPY = self.SPY['Last Price'].to_frame()
        self.SPY_pos = 10**6/self.SPY.iloc[0].values[0]

        self.maturity = maturity

        if self.maturity == 1:
            self.sigma = pd.read_excel('data.xlsx',sheet_name='VIX',engine='openpyxl')
        else:
            self.sigma = pd.read_excel('data.xlsx',sheet_name='VIX3',engine='openpyxl')

        self.sigma.set_index('Date',inplace = True)
        self.sigma = self.sigma.loc[self.start_date:self.end_date] * 0.01

        self.r = pd.read_excel('data.xlsx', sheet_name='USSOC BGN Curncy(3M)',engine='openpyxl')
        self.r.set_index('Date',inplace = True)
        self.r = self.r*12/100
        self.r = np.log(1+self.r)
        self.r = self.r.loc[self.start_date:self.end_date]

        self.otm_pct = otm_pct
        self.flag = flag
        self.daily_returns = None
        self.op_pos = None
        # self.tmp_PnL = 0
        # self.tmp_Spy_PnL = 0
        self.delta = delta
        self.monthly_returns = None
        self.max_drawdown = None

        self.plt_mr = None
        self.plt_dsr = None
        self.plt_dpnl = None

    def plot_returns(self, df, figure, df_SPY=None):
        df.index = df.index.strftime('%y-%m-%d')
        time = df.index.values
        t = range(0, len(time))
        plt.plot(t, df)
        if df_SPY is not None:
            plt.plot(t, df_SPY)
        timetic = list(range(0, len(time), len(time)//20))
        plt.xticks(timetic, labels=time[timetic], rotation=90, fontsize=7)
        plt.xlim(0, len(time))
        plt.grid(ls='-.')
        if df_SPY is not None:
            plt.legend(['Strategy PnL', 'SPY PnL'],loc='upper left',fontsize =10)
        else:
            plt.legend(df.columns,loc='upper left', fontsize =10)
        plt.xlabel('Date')
        if self.flag == 0:
            plt.title('Strategy: buy %d month put contract, using %d percentage OTM' % (self.maturity, self.otm_pct))
        if self.flag == 1:
            plt.title('Strategy: buy %d month put contract, using %d delta' % (self.maturity, self.delta))
        plt.savefig(figure+'.png')
        plt.show()

    def strategy(self, start, end, tmp_PnL = 0, tmp_Spy_PnL = 0):
        # global tmp_Spy_PnL, tmp_PnL
        SPY = self.SPY.loc[start:end] # SPY price
        # print(SPY)
        option_price = []
        portfolio_value = np.multiply(SPY,self.SPY_pos)

        # Calculate option price
        # K = SPY.iloc[0].values[0]*((100-self.otm_pct)/100)
        count = 0
        for date in SPY.index:
            S = SPY.loc[date].values[0]
            v = self.sigma.loc[date].values[0]
            r = self.r.loc[date].values[0]
            T = (end-date)/np.timedelta64(1, 'D')
            # print(T)
            if count == 0:
                S_first = S
                v_first = v
                r_first = r
                T_first = (end-date)/np.timedelta64(365, 'D')
            count += 1
            if self.flag == 0: # Using OTM strategy
                K = SPY.iloc[0].values[0]*((100-self.otm_pct)/100)
                put_price = Option.Option(True, -1, S, K, T, r, v, 0)
                put_price.bs_model()
                put_price = put_price.bsprice
            elif self.flag == 1: # Using delta based strategy
                d1= -norm.ppf(self.delta/100)
                K = S_first / (np.exp(d1 * v_first * np.sqrt(T_first) - T_first * (r_first + (v_first**2) / 2)))
                # K = option_Pricer.d1_to_k(d1, S_first, v_first, r_first, T_first) # K is constant
                put_price = Option.Option(True, -1, S, K, T, r, v, 0)
                put_price.bs_model()
                put_price = put_price.bsprice

            option_price.append(put_price)

        # print(option_price)
        Option_position = self.SPY_pos/100
        daily_returns = pd.DataFrame(index=SPY.index)
        option_value_df = pd.DataFrame(100*np.multiply(option_price,Option_position),index= SPY.index,columns = ['Option Price'])
        Spy_PnL = tmp_Spy_PnL+portfolio_value - self.SPY_pos*SPY.iloc[0].values[0]
        # print(Spy_PnL)
        # print(portfolio_value)
        daily_returns['daily SPY returns'] = portfolio_value['Last Price'].pct_change().values

        portfolio_value = np.add(option_value_df, portfolio_value)
        PnL = tmp_PnL+portfolio_value-self.SPY_pos*SPY.iloc[0].values[0]-100*option_price[0]*Option_position
        # print(PnL)

        # print(portfolio_value)
        daily_returns['daily returns'] = portfolio_value['Option Price'].pct_change().values
        daily_returns['daily PnL'] = PnL['Option Price'].values
        daily_returns['daily SPY PnL'] = Spy_PnL['Last Price'].values
        # print(daily_returns)
        tmp1 = daily_returns['daily SPY PnL'][-1]
        tmp2 = daily_returns['daily PnL'][-1]
        return tmp1, tmp2, daily_returns, Option_position

    def run(self):
        daily_returns = pd.DataFrame()
        pos_list = []
        # SPY_position = 10**6/self.SPY.iloc[0].values[0]
        multiple = math.ceil(self.period/self.maturity)
        start_time = self.start_date
        tmp_PnL = 0
        tmp_Spy_PnL = 0
        for i in range(multiple):
            tmp_time = start_time+relativedelta(months = self.maturity)
            # print(start_time, tmp_time)
            if tmp_time > self.end_date:
                tmp_time = self.end_date
            tmp_Spy_PnL, tmp_PnL, tmp_daily_returns,op_pos = self.strategy(start_time, tmp_time, tmp_PnL, tmp_Spy_PnL)
            daily_returns = pd.concat([daily_returns, tmp_daily_returns])
            start_time = tmp_time
            pos_list.append(op_pos)

        daily_returns['mean'] = daily_returns['daily returns'].expanding().mean() *252
        daily_returns['volatility'] = daily_returns['daily returns'].expanding().std() * np.sqrt(252)
        daily_returns['sharpe ratio'] = daily_returns['mean'] / daily_returns['volatility']
        monthly_returns = pd.DataFrame()
        monthly_returns['monthly returns'] = (daily_returns['daily returns'] +1).resample('M').prod()
        monthly_returns['monthly returns'] = monthly_returns['monthly returns'] -1
        monthly_returns['monthly SPY returns'] = (daily_returns['daily SPY returns'] +1).resample('M').prod()
        monthly_returns['monthly SPY returns'] = monthly_returns['monthly SPY returns'] -1
        monthly_returns['mean'] = monthly_returns['monthly returns'].expanding().mean() *12
        monthly_returns['volatility'] = monthly_returns['monthly returns'].expanding().std() * np.sqrt(12)
        monthly_returns['sharpe ratio'] = monthly_returns['mean'] / monthly_returns['volatility']
        self.daily_returns = daily_returns
        self.monthly_returns = monthly_returns
        # print(pos_list)
        self.op_pos = pos_list
        cum_ret = (1 + monthly_returns[['monthly returns','monthly SPY returns']]).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        self.max_drawdown = max_drawdown

        self.plot_returns(pd.DataFrame(monthly_returns['monthly returns']),'monthly_returns')
        self.plot_returns(pd.DataFrame(daily_returns['daily PnL']),'daily_PnL',
                          df_SPY=pd.DataFrame(daily_returns['daily SPY PnL']))
        self.plot_returns(pd.DataFrame(daily_returns['sharpe ratio']),'daily_SR')
        # QApplication.processEvents()

