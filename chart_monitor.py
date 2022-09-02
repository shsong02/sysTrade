import os
import pprint

import numpy as np
import yaml
import pandas as pd
from datetime import datetime, timedelta

##core
from lib.marcap import marcap_data
import FinanceDataReader as fdr
import mplfinance as mpf
from mplfinance import _styles

## local file
import st_utils as stu



## log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e :
    print(e)

####    로그 생성    #######
logger = stu.create_logger()
class chartMonitor:
    def __init__(self, config_file):
        ## 설정 파일을 필수적으로 한다.
        try:
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e :
            print (e)

        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")
        pprint.pprint(config)

        ## global 변수 선언
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.score_rule = config["scoreRule"]
        self.keys = config["keyList"]
        pass

    def run(self):
        '''
            제무재표 좋은 종목중에 Threshold socre 보다 높은 종목을 선발한다.
            그중에서 진입 시점을 임박한 종목을 선발한다.
                - cond1: RSI
                - cond2: 볼린저
                - cond3: 거래량
                - cond4: 공매도
        '''

        ## 1) 데이터 준비
        file_path = self.file_manager["selected_items"]["path"]
        ## 파일 하나임을 가정 함 (todo: 멀티 파일 지원은 추후 고려)
        file_name = os.listdir('./data/selected_items/')[0]
        df_stocks = pd.read_csv(file_path + file_name, index_col=0)
        df_stocks = df_stocks.sort_values(by='total_score', ascending=False)

        ##data 가져오기
        code = str(df_stocks.iloc[1]['Symbol'])
        now = datetime.today()
        now_str = now.strftime(self.param_init["time_format"])
        pre_date = now - timedelta(days=365*2)
        pre_date_str = pre_date.strftime(self.param_init["time_format"])
        df = fdr.DataReader(code, pre_date_str, now_str)

        ## make sub-plot
        macd = self._macd(df, 12, 26, 9)
        stochastic = self._stochastic(df, 14, 3)
        rsi = self._rsi(df, 14)
        bol = self._bollinger(df, 20)
        obv = self._obv(df, mav=20)


        rsi_pannel = 2
        obv_pannel = 3
        macd_pannel= 4
        stoch_pannel= 5

        ## add sub-plot
        add_plots = [
            mpf.make_addplot(bol['bol_upper'], color='#606060'),
            mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

            ## rsi
            mpf.make_addplot(rsi['rsi'], ylim=[0, 100], ylabel='RSI',  panel=rsi_pannel),
            mpf.make_addplot(rsi['rsi_high'], color='r', panel=rsi_pannel),
            mpf.make_addplot(rsi['rsi_low'], color='b', panel=rsi_pannel),

            ## obv
            mpf.make_addplot(obv['obv'], ylabel='OBV', color='#8c564b', panel=obv_pannel),
            # mpf.make_addplot(obv['obv_ema'], color='#e377c2', panel=obv_pannel),

            # macd
            # mpf.make_addplot((macd['macd']), color='#606060',ylabel='MACD', secondary_y=False,  panel=macd_pannel ),
            # mpf.make_addplot((macd['signal']), color='#1f77b4',  secondary_y=False, panel=macd_pannel),
            # mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=macd_pannel),
            # mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=macd_pannel),

            #stochastic
            # mpf.make_addplot((stochastic[['%K', '%D', '%SD', 'UL', 'DL']]), ylim=[0, 100], ylabel='Stoch', panel=stoch_pannel)
        ]

        mpf.plot(df, type='candle', mav=(5,20), volume=True, addplot=add_plots,
                 figsize=(20,12),
                 figscale=0.8, style='yahoo', panel_ratios=(3,1,1,1),
                 scale_padding={'right':2.0, 'left':0.5},
                 tight_layout=True)

        pass

    ###### Chart
    def _obv(self, df, mav=20):
        obv = []
        obv.append(0)
        for i in range(1, len(df.Close)):
            if df.Close[i] > df.Close[i-1]:
                obv.append(obv[-1] + df.Volume[i])
            elif df.Close[i] < df.Close[i-1]:
                obv.append(obv[-1] - df.Volume[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_ema'] = df['obv'].ewm(com=mav).mean()
        return df

    def _bollinger(self, df, window=20, sigma=3):
        df['bol_mid'] = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window).std(ddof=0)
        df['bol_upper'] = df['bol_mid'] + sigma * std
        df['bol_lower'] = df['bol_mid'] - sigma * std
        return df

    def _rsi(self, df, window=14):
        df_tmp = df.copy()
        df_tmp['변화량'] = df['Close'] - df['Close'].shift(1)
        df_tmp['상승폭'] = np.where(df_tmp['변화량'] >= 0, df_tmp['변화량'], 0)
        df_tmp['하락폭'] = np.where(df_tmp['변화량'] < 0, df_tmp['변화량'].abs(), 0)

        # welles moving average
        df_tmp['AU'] = df_tmp['상승폭'].ewm(alpha=1 / window, min_periods=window).mean()
        df_tmp['AD'] = df_tmp['하락폭'].ewm(alpha=1 / window, min_periods=window).mean()
        # df['RS'] = df['AU'] / df['AD']
        # df['RSI'] = 100 - (100 / (1 + df['RS']))
        df['rsi'] = df_tmp['AU'] / (df_tmp['AU'] + df_tmp['AD']) * 100
        df['rsi_high'] = 70
        df['rsi_low'] = 30

        return df

    def _macd(self, df, window_slow, window_fast, window_signal):
        macd = pd.DataFrame()
        macd['ema_slow'] = df['Close'].ewm(span=window_slow).mean()
        macd['ema_fast'] = df['Close'].ewm(span=window_fast).mean()
        macd['macd'] = macd['ema_slow'] - macd['ema_fast']
        macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
        macd['diff'] = macd['macd'] - macd['signal']
        macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0)
        macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0)

        df['macd'] = macd['macd']
        df['signal'] = macd['signal']
        df['bar_positive'] = macd['bar_positive']
        df['bar_negative'] = macd['bar_negative']
        return df

    # Stochastic
    def _stochastic(self, df, window, smooth_window):
        stochastic = pd.DataFrame()
        stochastic['%K'] = ((df['Close'] - df['Low'].rolling(window).min()) \
                            / (df['High'].rolling(window).max() - df['Low'].rolling(window).min())) * 100
        stochastic['%D'] = stochastic['%K'].rolling(smooth_window).mean()
        stochastic['%SD'] = stochastic['%D'].rolling(smooth_window).mean()
        stochastic['UL'] = 80
        stochastic['DL'] = 20
        return stochastic

if __name__ == '__main__':
    config_file = './config/config.yaml'
    cm = chartMonitor(config_file)
    cm.run()

