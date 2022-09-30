import os

import numpy as np
import yaml
import pandas as pd
from datetime import datetime, timedelta

##core
import FinanceDataReader as fdr
import mplfinance as mpf
from pykrx import stock

# Import the backtrader platform

## local file
from tools import st_utils as stu

## chart font 재설정
import matplotlib.pyplot as plt

# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

plt.rc('font', family='AppleGothic')
print(plt.rcParams['font.family'])


## log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e :
    print(e)
####    로그 생성    #######
logger = stu.create_logger()

class tradeStrategy:
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

        self.save_image = config["searchStock"]["market_leader"]["save_image"]

    def run(self, code='selected_item', name='', dates=[], data=pd.DataFrame(), mode='realtime'):
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
            file_path = self.file_manager["finance_score"]["path"]
            ## 파일 하나임을 가정 함 (todo: 멀티 파일 지원은 추후 고려)
            file_name = os.listdir('./data/finance_score/')[0]
            df_stocks = pd.read_csv(file_path + file_name, index_col=0)
            df_stocks = df_stocks.sort_values(by='total_score', ascending=False)
            code = str(df_stocks.iloc[1]['code'])
        code = str(code).zfill(6)


        if mode == 'daily':
            ## 기간 처리
            if len(dates) == 0 :
                    end_dt = datetime.today()
                    end = end_dt.strftime(self.param_init["time_format"])
                    st_dt = end_dt - timedelta(days=30)  ## default 는 1달 ..
                    st = st_dt.strftime(self.param_init["time_format"])
            else:
                st = dates[0]
                end = dates[1]

            if data == 'none':
                df = fdr.DataReader(code, st, end)
            else:
                df = data.copy()


            ## 추가 정보들
            df_out = pd.DataFrame()

            ## advanced 정보 (일자별 PER, ..)
            df_chart2 = stock.get_market_fundamental(dates[0], dates[1], code, freq='d')
            df_out = df.join(df_chart2)

            ##일자별 시가 총액 (시가총액, 거래량, 거래대금, 상장주식수)
            df_chart3 = stock.get_market_cap(dates[0], dates[1], code)
            df_out = df_out.join(df_chart3)

            ##일자별 외국인 보유량 및 외국인 한도소진률
            df_chart4 = stock.get_exhaustion_rates_of_foreign_investment(dates[0], dates[1], code)
            df_chart4.drop(['상장주식수', '한도수량', '한도소진률' ], axis=1, inplace=True)  ## 중복
            df_chart4.rename(columns={'보유수량': '외국인_보유수량',  '지분율':'외국인_지분율'}, inplace=True)
            df_out = df_out.join(df_chart4)

            ## 공매도 정보
            df_chart5 = stock.get_shorting_balance_by_date(dates[0], dates[1], code)
            df_chart5.drop(['상장주식수', '시가총액' ], axis=1, inplace=True)  ## 중복
            df_chart5.rename(columns={'비중': '공매도비중', }, inplace=True)
            df_out = df_out.join(df_chart5)
        else: ## realtime
            if len(data) == 0:
                raise ValueError("입력된 데이터의 크기가 0 입니다.")
            else:
                df_out = data
                st = dates[0]
                end = dates[1]

        ## make sub-plot
        macd = self._macd(df_out, 12, 26, 9)
        stochastic = self._stochastic(df_out, 14, 3)
        rsi = self._rsi(df_out, 14)
        bol = self._bollinger(df_out, window=20, sigma=2.0)
        obv = self._obv(df_out, mav=20)

        vol_abn = self._volume_anomaly(df_out, window=7, quantile=0.80, mode=mode)


        ## 이동 평균 추가
        df_out['ma5'] = round(df_out['Close'].rolling(window=5).mean())
        df_out['ma20'] = round(df_out['Close'].rolling(window=20).mean())
        df_out['ma60'] = round(df_out['Close'].rolling(window=60).mean())

        ## 이동평균 방향
        df_out['ma5Pos'] = (df_out['ma5'] - df_out['ma5'].shift(1)) >= 0
        df_out['ma20Pos'] = (df_out['ma20'] - df_out['ma20'].shift(1)) >= 0
        df_out['ma60Pos'] = (df_out['ma60'] - df_out['ma60'].shift(1)) >= 0

        ## 이동평균 간 거리차
        df_out['ma520Dist'] = df_out['ma5'] - df_out['ma20']
        df_out['ma520DistCenter'] = 0
        df_out['ma520DistPos'] = df_out['ma520Dist'] - df_out['ma520Dist'].shift(1)

        ## 코스피 지수 확인용
        df_krx = stock.get_index_ohlcv(st, end, "1028") ## kospi200
        df_krx.rename(columns={'시가':'Open',
                               '고가':'High',
                               '저가':'Low',
                               '종가':'Close',
                               '거래량':'Volume',
                               }, inplace=True)

        ## 종목이 지정한 날짜보다 늦게 생겼을 때, x 길이를 맞추기 위함
        if len(df_out) < len(df_krx):
            new_st = df_out.head(1).index.to_list()[0]
            df_krx2 = df_krx[df_krx.index >= new_st]
        else:
            df_krx2 = df_krx


        ## Pannel number
        if mode == "daily": ##
            pannel_id = {
                'krx':      2,
                'maDist':    3,
                'maDistPos': 4,
                # 'rsi':      4,
                'per':      5,
                'foreign':  6,
                'volume':   7,
                'short':    8,
               # 'obv' = 3
               #  'macd': 5,
               #  'stochestic': 6,
            }

            ## add sub-plot
            add_plots = [
                mpf.make_addplot(bol['bol_upper'], color='#606060'),
                mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

                ## krx
                mpf.make_addplot(df_krx2['Close'], ylabel='Kospi 200', color='#8c564b', panel=pannel_id['krx']),

                ## ma 거리차
                mpf.make_addplot(df_out['ma520Dist'], ylabel='Distance(ma5 - ma20)', color='#8c564b', panel=pannel_id['maDist']),
                mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDist']),

                ## ma 방향성
                mpf.make_addplot(df_out['ma520DistPos'], ylabel="Dist(ma5 20)'s direction", color='#8c564b',
                                 panel=pannel_id['maDistPos']),
                mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDistPos']),

                ## rsi
                # mpf.make_addplot(rsi['rsi'], ylim=[0, 100], ylabel='RSI',  panel=pannel_id['rsi']),
                # mpf.make_addplot(rsi['rsi_high'], color='r', panel=pannel_id['rsi']),
                # mpf.make_addplot(rsi['rsi_low'], color='b', panel=pannel_id['rsi']),

                ## obv
                # mpf.make_addplot(obv['obv'], ylabel='OBV', color='#8c564b', panel=pannel_id['obv]),
                # mpf.make_addplot(obv['obv_ema'], color='#e377c2', panel=pannel_id['obv']),

                ## per
                mpf.make_addplot(df_out['PER'], ylabel='PER (brown)', color='#8c564b', panel=pannel_id['per']),
                mpf.make_addplot(df_out['PBR'], ylabel='PBR (pink)', color='#e377c2', secondary_y=True, panel=pannel_id['per']),

                ## for
                mpf.make_addplot(df_out['외국인_지분율'], ylabel='Foreign ratio', color='black', panel=pannel_id['foreign']),

                ## volume anomaly
                mpf.make_addplot(df_out['VolumeThreshold'], ylabel='Turnover ratio', color='orange', panel=pannel_id['volume']),
                mpf.make_addplot(df_out['VolumeTurnOver'],  color='black', panel=pannel_id['volume']),
                mpf.make_addplot(df_out['VolumeAnomaly'], type='scatter', marker='v', markersize=200, color='red',
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
        else:   ## realtime 관련 차트 내용
            pannel_id = {
                'maDist':       2,
                'chegyeol':     3,
                # 'maDistPos':    3,
                'volume':       4,
                'rsi':          5,
                'obv' :         6
            }

            ## add sub-plot
            add_plots = [
                mpf.make_addplot(bol['bol_upper'], color='#606060'),
                mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

                ## ma 거리차
                mpf.make_addplot(df_out['ma520Dist'], ylabel='Distance(ma5 - ma20)', color='#8c564b', panel=pannel_id['maDist']),
                mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDist']),

                ## ChegyeolStr
                mpf.make_addplot(df_out['ChegyeolStr'], ylabel='Chegyeol Str.', color='#8c564b',panel=pannel_id['chegyeol']),

                # ## ma 방향성
                # mpf.make_addplot(df_out['ma520DistPos'], ylabel="Dist(ma5 20)'s direction", color='#8c564b',
                #                  panel=pannel_id['maDistPos']),
                # mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDistPos']),

                ## rsi
                mpf.make_addplot(rsi['rsi'], ylim=[0, 100], ylabel='RSI',  panel=pannel_id['rsi']),
                mpf.make_addplot(rsi['rsi_high'], color='r', panel=pannel_id['rsi']),
                mpf.make_addplot(rsi['rsi_low'], color='b', panel=pannel_id['rsi']),

                ## obv
                mpf.make_addplot(obv['obv'], ylabel='OBV', color='#8c564b', panel=pannel_id['obv']),
                mpf.make_addplot(obv['obv_ema'], color='#e377c2', panel=pannel_id['obv']),

                ## volume anomaly
                mpf.make_addplot(df_out['VolumeThreshold'], ylabel='Turnover ratio', color='orange', panel=pannel_id['volume']),
                mpf.make_addplot(df_out['VolumeTurnOver'],  color='black', panel=pannel_id['volume']),
                mpf.make_addplot(df_out['VolumeAnomaly'], type='scatter', marker='v', markersize=200, color='red',
                                 panel=pannel_id['volume']),

            ]


        ##################################
        ####   매수, 매도 조건 만들어 내기
        ##################################
        self._check_buy_sell(df_out)

        ##################################
        ####   Chart 에 표식 남기기
        ##################################
        pntDict = dict()

        # pntDict['rsiBuy'] = dict()
        # pntDict['rsiBuy']['data'] = df_out[df_out.rsiBuy == True].index.to_list()  ## rsi 기준으로 buy 시점
        # pntDict['rsiBuy']['color'] = 'pink'
        # pntDict['rsiSell'] = dict()
        # pntDict['rsiSell']['data'] = df_out[df_out.rsiSell == True].index.to_list() ## rsi 기준으로 sell 시점
        # pntDict['rsiBuy']['color'] = '#60b8f7'
        # pntDict['volAnomaly'] = dict()
        # pntDict['volAnomaly']['data'] = df_out[df_out.VolumeAnomaly.notnull()].index.to_list() ## 거래량 증가 시점
        # pntDict['volAnomaly']['color'] = '#f29c2c'  # 오렌지
        pntDict['bolBuy'] = dict()
        pntDict['bolBuy']['data'] = df_out[df_out.bolBuy == True].index.to_list() ## bol 기준으로 buy 시점
        pntDict['bolBuy']['color'] = '#66ff33'  ## 밝은 녹색
        # pntDict['bolSell'] = dict()
        # pntDict['bolSell']['data'] = df_out[df_out.bolSell == True].index.to_list() ## bol 기준으로 sell 시점
        # pntDict['bolSell']['color'] = '#0d3300'  ## 진한 녹색
        pntDict['maBuy'] = dict()
        pntDict['maBuy']['data'] = df_out[df_out.finalBuy == True].index.to_list() ## bol 기준으로 sell 시점
        pntDict['maBuy']['color'] = '#ff33bb'  ## 핑크 계열
        pntDict['maSell'] = dict()
        pntDict['maSell']['data'] = df_out[df_out.finalSell == True].index.to_list() ## bol 기준으로 sell 시점
        pntDict['maSell']['color'] = '#002080'  ## 진파랑 계열

        points = []
        colors = []
        for key, value in pntDict.items():
            data = value['data']
            color = value['color']

            if not len(data) == 0 :
                points = points + data
                temp = [color for i in range(len(data))]
                colors = colors + temp

        ## 패널정리
        pannel_ratio = [3,1]
        pannel_cnt = len(pannel_id)
        for i in range(pannel_cnt):
            pannel_ratio.append(1)

        # 차트 그리기
        mc = mpf.make_marketcolors(up='r', down='b',
                              edge='inherit',
                              wick={'up': 'red', 'down': 'blue'},
                              volume='in', ohlc='i')
        s = mpf.make_mpf_style(marketcolors=mc)

        title = f"Code({code}) 's period: {dates[0]} ~ {dates[1]} "
        if self.save_image == True:
            path = self.file_manager["stock_theme"]["path"] + f"market_leader/{st}_{end}/"
            name = f"{code}_{st}_{end}.png"
            try:
                if not os.path.exists(path):
                    os.makedirs(path)
            except Exception as e:
                raise e

            mpf.plot(df_out, type='candle', mav=(5,20,60), volume=True, addplot=add_plots, panel_ratios=tuple(pannel_ratio),
                      figsize=(30, 8 + 3*pannel_cnt),
                      title=title,
                      vlines=dict(vlines= points, linewidths=5, colors=colors, alpha=0.30),
                      figscale=0.8, style=s,
                      scale_padding={'right':2.0, 'left':0.5},
                      warn_too_much_data=5000,
                      tight_layout=True,
                      savefig=f'{path}{name}',
                      )
        else:
            mpf.plot(df_out, type='candle', mav=(5, 20, 60), volume=True, addplot=add_plots,
                     panel_ratios=tuple(pannel_ratio),
                     figsize=(30, 8 + 3 * pannel_cnt),
                     title=title,
                     vlines=dict(vlines=points, linewidths=5, colors=colors, alpha=0.30),
                     figscale=0.8, style=s,
                     scale_padding={'right': 2.0, 'left': 0.5},
                     warn_too_much_data=5000,
                     tight_layout=True,
                     )

        return df_out


    #### 매매 타이밍 체크
    def _check_buy_sell(self, df):
        '''

        :param df:
            df (Dataframe) : 내부 처리는 링크되어 반영됨
            매수, 매도 조건은 추구 config 로 뺄 것 (22.09.28)
        :return:
        '''

        #####################
        ###  매수 타이밍 정리
        #####################
        temp = []
        temp2 = [] ## sell 확인 용
        queue = [0,0,0,0]
        for i, idx in enumerate(df.index):
            if i  == 0 :
                d = df.ma520DistPos.iat[i]
            else:
                queue.append(d)
                queue.pop(0)
                d = df.ma520DistPos.iat[i]

            # 거래량 터진 다음날 중심축으로 회귀 완벽히 확인 후, 벌어지는 시점에 구매
            if (d >=0) and (queue[3]<=0) and (queue[2]<=0) and (queue[1]<=0):
                # print(idx)
                ma60Pos = df.ma60Pos.iat[i]
                if ma60Pos == True:  ## 추세 전환 확인하고 들어가기 (60일선)
                    temp.append(True)
                else:
                    temp.append(False)
            else:
                temp.append(False)

            # 추세 반전하고, 그마나 어느정도 복귀 하는 것 확인하고 팔기 위함
            if (d < 0) and (queue[3] >= 0) and (queue[2] >= 0) and (queue[1] >= 0):
                temp2.append(True)
            else:
                temp2.append(False)
        df['ma520DistChange'] = temp
        df['ma520DistChangeInv'] = temp2

        chk1 = False
        chk1HoldCnt = 0
        buy_times = []
        hold_period = 14 ## 거래량 이후 매수 까지 유지 기간
        for idx in df.index:
            bolBuy = df.bolBuy.at[idx]
            volAnol = df.VolumeAnomaly.at[idx]

            # 1) 커디션 체크 : bol 를 상단 터치하였는데, 거래량 터지고 60일 선이 증가 추세
            if (bolBuy == True) and ~(np.isnan(volAnol)):
                chk1 = True
                chk1HoldCnt = 0  ## 리프레쉬 되면 카운트 다시 시작
                buy_times.append(False)

            else:
                # 2) 중심(ma5) 회귀 후 벌어지는 시점
                if chk1 == True:
                    if df.ma520DistChange.at[idx] == True:
                        # print(f"매수 시점: {idx}")
                        buy_times.append(True)
                        chk1HoldCnt = 0
                        chk1 = False
                    elif(chk1HoldCnt >= hold_period):  ## 7일 이상되면 무효됨
                        buy_times.append(False)
                        chk1HoldCnt = 0
                        chk1 = False
                    else:
                        buy_times.append(False)
                        chk1HoldCnt += 1
                        chk1 = True
                else:
                    buy_times.append(False)

        df['finalBuy'] = buy_times

        #####################
        ###  매도 타이밍 정리
        #####################
        chk1 = False
        sell_times = []
        for idx in df.index:
            ma60Pos = df.ma60Pos.at[idx]
            bolSell = df.bolSell.at[idx]

            # 1) 하방 확인하기
            if (bolSell == True) and (ma60Pos == False):
                chk1 = True
                sell_times.append(False)
            else:
                # 2) 기한 제한없이 조금이라도 반등 추세 주면 바로 팔기
                invPoint = df.ma520DistChangeInv.at[idx]
                if (chk1 == True) and (invPoint == True):
                    sell_times.append(True)
                    chk1 = False  ## 다음 체크를 위한 원복
                else:
                    sell_times.append(False)
        df['finalSell'] = sell_times

    ###### Chart
    def _volume_anomaly(self, df, window=14, quantile=0.90, mode='daily'):
        if mode == 'daily':
            df["VolumeTurnOver"] = round(df["Volume"] / df["상장주식수"] * 100, 2)
        else: # realtime 은 상장주식수 를 확인할 수 없음
            totalStocks = 1000000 # 고정값 처리
            df["VolumeTurnOver"] = df["Volume"]

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

        df['bolBuy'] = False
        df['bolSell'] = False
        df_temp = pd.DataFrame()
        df_temp = df[df.bol_upper <= df.Close]
        for idx in df_temp.index.to_list():
            df.loc[idx, 'bolBuy'] = True  ## 상단 터치를 사는 시점으로 봄 (범위를 짥게 가져감)

        df_temp = df[df.bol_lower >=  df.Close]
        for idx in df_temp.index.to_list():
            df.loc[idx, 'bolSell'] = True

        return df

    def _rsi(self, df, window=14, rsi_high=70, rsi_low=30):
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
        df['rsi_high'] = rsi_high
        df['rsi_low'] = rsi_low

        ## rsi 상승 시점 체크
        df.rsi = df.rsi.fillna(method='bfill')
        df['rsiChange'] = df['rsi'] - df['rsi'].shift(1)
        df['rsiBuy'] = False
        df['rsiSell'] = False
        df_temp = pd.DataFrame()
        df_temp = df[(df.rsi >= df.rsi_low) & (df.rsi.shift(1) < df.rsi_low)] # low 넘어서는 시점
        for idx in df_temp.index.to_list():
            df.loc[idx,'rsiBuy'] = True

        df_temp = df[(df.rsi < df.rsi_high) & (df.rsi.shift(1) >= df.rsi_high)]  ## high 낮아지는 시점ㅏ
        for idx in df_temp.index.to_list():
            df.loc[idx, 'rsiSell'] = True

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
    cm = tradeStrategy(config_file)
    cm.run()

