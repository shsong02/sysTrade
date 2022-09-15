import os
import pprint

import matplotlib
import numpy as np
import yaml
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import font_manager, rc
import matplotlib

##core
import FinanceDataReader as fdr
import mplfinance as mpf
from pykrx import stock
from pykrx import bond

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
        # pprint.pprint(config)

        ## global 변수 선언
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.score_rule = config["scoreRule"]
        self.keys = config["keyList"]

    def run(self, code='selected_item', date=[], data='none' ):
        '''

            data 에 통계 수치를 더하여 반환함.

            제무재표 좋은 종목중에 Threshold socre 보다 높은 종목을 선발한다.
            그중에서 진입 시점을 임박한 종목을 선발한다.
                - cond1: RSI
                - cond2: 볼린저
                - cond3: 거래량
                - cond4: 공매도

            args:
                - data (dataframe) : OHLCV 컬럼명을 유지해야 함
        '''

        if code == 'selected_item':
            ## 1) 데이터 준비
            file_path = self.file_manager["selected_items"]["path"]
            ## 파일 하나임을 가정 함 (todo: 멀티 파일 지원은 추후 고려)
            file_name = os.listdir('./data/selected_items/')[0]
            df_stocks = pd.read_csv(file_path + file_name, index_col=0)
            df_stocks = df_stocks.sort_values(by='total_score', ascending=False)
            code = str(df_stocks.iloc[1]['Symbol'])
        else:
            code = str(code).zfill(6)


        ## 기간 처리
        if len(date) == 0 :
                end_dt = datetime.today()
                end = end_dt.strftime(self.param_init["time_format"])
                st_dt = end_dt - timedelta(days=30)  ## default 는 1달 ..
                st = st_dt.strftime(self.param_init["time_format"])
        else:
            st = date[0]
            end = date[1]

        if data == 'none':
            df = fdr.DataReader(code, st, end)
        else:
            df = data.copy()


        ## 추가 정보들
        df_out = pd.DataFrame()

        ## advanced 정보 (일자별 PER, ..)
        df_chart2 = stock.get_market_fundamental(date[0], date[1], code, freq='d')
        df_out = df.join(df_chart2)

        ##일자별 시가 총액 (시가총액, 거래량, 거래대금, 상장주식수)
        df_chart3 = stock.get_market_cap(date[0], date[1], code)
        df_out = df_out.join(df_chart3)

        ##일자별 외국인 보유량 및 외국인 한도소진률
        df_chart4 = stock.get_exhaustion_rates_of_foreign_investment(date[0], date[1], code)
        df_chart4.drop(['상장주식수', '한도수량', '한도소진률' ], axis=1, inplace=True)  ## 중복
        df_chart4.rename(columns={'보유수량': '외국인_보유수량',  '지분율':'외국인_지분율'}, inplace=True)
        df_out = df_out.join(df_chart4)

        ## 공매도 정보
        df_chart5 = stock.get_shorting_balance_by_date(date[0], date[1], code)
        df_chart5.drop(['상장주식수', '시가총액' ], axis=1, inplace=True)  ## 중복
        df_chart5.rename(columns={'비중': '공매도비중', }, inplace=True)
        df_out = df_out.join(df_chart5)

        ## make sub-plot
        macd = self._macd(df_out, 12, 26, 9)
        stochastic = self._stochastic(df_out, 14, 3)
        rsi = self._rsi(df_out, 14)
        bol = self._bollinger(df_out, 20)
        obv = self._obv(df_out, mav=20)

        vol_abn = self._volume_anomaly(df_out, window=14, quantile=0.90)




        ## Pannel number
        pannel_id = {
            'rsi': 2,
            'per': 3,
            'foreign': 4,
            'volume': 5,
            'short': 6,
           # 'obv' = 3
           #  'macd': 5,
           #  'stochestic': 6,
        }
        pannel_cnt = len(pannel_id)


        ## add sub-plot
        add_plots = [
            mpf.make_addplot(bol['bol_upper'], color='#606060'),
            mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

            ## rsi
            mpf.make_addplot(rsi['rsi'], ylim=[0, 100], ylabel='RSI',  panel=pannel_id['rsi']),
            mpf.make_addplot(rsi['rsi_high'], color='r', panel=pannel_id['rsi']),
            mpf.make_addplot(rsi['rsi_low'], color='b', panel=pannel_id['rsi']),

            ## obv
            # mpf.make_addplot(obv['obv'], ylabel='OBV', color='#8c564b', panel=pannel_id['obv]),
            # mpf.make_addplot(obv['obv_ema'], color='#e377c2', panel=pannel_id['obv']),

            ## per
            mpf.make_addplot(df_out['PER'], ylabel='PER', color='#8c564b', panel=pannel_id['per']),
            mpf.make_addplot(df_out['PBR'], ylabel='PBR', color='#e377c2', secondary_y=True, panel=pannel_id['per']),

            ## for
            mpf.make_addplot(df_out['외국인_지분율'], ylabel='Foreign ratio', color='black', panel=pannel_id['foreign']),

            ## volume anomaly
            mpf.make_addplot(df_out['VolumeThreshold'], ylabel='Turnover ratio', color='orange', panel=pannel_id['volume']),
            mpf.make_addplot(df_out['VolumeTurnOver'],  color='black', panel=pannel_id['volume']),
            mpf.make_addplot(df_out['VolumeAnomaly'], type='scatter', marker='v', markersize=200, ylabel='Turnover ratio', color='red',
                             panel=pannel_id['volume']),

            mpf.make_addplot(df_out['공매도잔고'], ylabel='Short Selling', color='black',
                             panel=pannel_id['short']),

            # macd
            # mpf.make_addplot((macd['macd']), color='#606060',ylabel='MACD', secondary_y=False,  panel=pannel_id['macd'] ),
            # mpf.make_addplot((macd['signal']), color='#1f77b4',  secondary_y=False, panel=pannel_id['macd']),
            # mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=pannel_id['macd']),
            # mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=pannel_id['macd']),

            #stochastic
            # mpf.make_addplot((stochastic[['%K', '%D', '%SD', 'UL', 'DL']]), ylim=[0, 100], ylabel='Stoch', panel=pannel_id['stochestic'])
        ]

        title = f"Code ({code}) 's period: {date[0]} ~ {date[1]} "
        pannel_ratio = [3,1]
        for i in range(pannel_cnt):
            pannel_ratio.append(1)
        mpf.plot(df, type='candle', mav=(5,20), volume=True, addplot=add_plots,
                 figsize=(20, 8 + 3*pannel_cnt),
                 title=title,
                 figscale=0.8, style='yahoo', panel_ratios=tuple(pannel_ratio),
                 scale_padding={'right':2.0, 'left':0.5},
                 warn_too_much_data=5000,
                 tight_layout=True)


        return df_out

    ###### Chart
    def _volume_anomaly(self, df, window=14, quantile=0.90):
        df["VolumeTurnOver"] = round(df["Volume"] / df["상장주식수"] * 100, 2)
        def make_band(data, quantile):
            upper = np.mean(data) + np.nanquantile(data, quantile)
            lower = np.mean(data) - np.nanquantile(data, quantile)
            return upper, lower

        uppers = []
        anomalies = []
        for i in range(len(df)):
            if i < window:
                data = df.VolumeTurnOver.iloc[0:i+1].to_list()
            else:
                data = df.VolumeTurnOver.iloc[i-window+1: i+1].to_list()
            upper, lower = make_band(data, quantile=quantile)
            uppers.append(round(upper,2))

            chk_data = df.VolumeTurnOver.iat[i]
            if  chk_data > upper:
                anomalies.append(chk_data)
            else:
                anomalies.append(np.nan)

        df["VolumeThreshold"] = uppers
        df["VolumeAnomaly"] = anomalies

        return df



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

