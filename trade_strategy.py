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

        # logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")
        # pprint.pprint(config)

        ## global 변수 선언
        self.file_manager = config["data_management"]
        self.param_init = config["mainInit"]
        self.trade_config = config["tradeStock"]



        self.display = config["searchStock"]["market_leader"]["display_chart"]
        self.path = ""
        self.name = "image.png"

    def run(self, code='', name='', dates=[], data=pd.DataFrame(), mode='daily'):
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
            try:
                df_chart2 = stock.get_market_fundamental(dates[0], dates[1], code, freq='d')
                if df_chart2 is not None and not df_chart2.empty:
                    # 필요한 컬럼만 선택하고 누락된 컬럼은 기본값으로 채움
                    required_columns = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
                    for col in required_columns:
                        if col not in df_chart2.columns:
                            df_chart2[col] = 0.0
                    df_out = df.join(df_chart2)
                    df_out.fillna(0, inplace=True)
                    logger.info(f"기본 재무정보 조회 성공 (종목: {code})")
                else:
                    # 데이터가 없는 경우 기본값 설정
                    cols = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
                    df_chart2 = pd.DataFrame(0, index=df.index, columns=cols)
                    df_out = df.join(df_chart2)
                    df_out.fillna(0, inplace=True)
                    logger.info(f"기본 재무정보 없음 (종목: {code}) - 기본값 사용")
            except Exception as e:
                logger.warning(f"기본 재무정보 조회 실패 (종목: {code}): {str(e)}")
                # 오류 발생 시 기본값 설정
                cols = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
                df_chart2 = pd.DataFrame(0, index=df.index, columns=cols)
                df_out = df.join(df_chart2)
                df_out.fillna(0, inplace=True)

            ##일자별 시가 총액 (시가총액, 거래량, 거래대금, 상장주식수)
            try:
                df_chart3 = stock.get_market_cap(dates[0], dates[1], code)
                if df_chart3 is not None and not df_chart3.empty:
                    df_out = df_out.join(df_chart3)
                    logger.info(f"시가총액 정보 조회 성공 (종목: {code})")
                else:
                    # 데이터가 없는 경우 기본값 설정
                    cols = ['시가총액', '거래량', '거래대금', '상장주식수']
                    df_chart3 = pd.DataFrame(0, index=df_out.index, columns=cols)
                    df_out = df_out.join(df_chart3)
                    logger.info(f"시가총액 정보 없음 (종목: {code}) - 기본값 사용")
            except Exception as e:
                logger.warning(f"시가총액 정보 조회 실패 (종목: {code}): {str(e)}")
                # 오류 발생 시 기본값 설정
                cols = ['시가총액', '거래량', '거래대금', '상장주식수']
                df_chart3 = pd.DataFrame(0, index=df_out.index, columns=cols)
                df_out = df_out.join(df_chart3)

            ## 투자자별 거래량 (누적 순매수를 계산)
            try:
                df_chart4 = stock.get_market_trading_volume_by_date(dates[0], dates[1], code)
                if df_chart4 is not None and not df_chart4.empty:
                    # 누적 순매수 계산을 위한 새로운 컬럼 초기화
                    volume_columns = ['VolumeOrgan', 'VolumeForeign', 'VolumeEtc', 'VolumePersonal']
                    for col in volume_columns:
                        df_chart4[col] = 0
                    
                    # 누적 순매수 계산
                    for cnt, idx in enumerate(df_chart4.index):
                        if cnt == 0:
                            if '기관합계' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumeOrgan'] = df_chart4.at[idx, '기관합계']
                            if '기타법인' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumeEtc'] = df_chart4.at[idx, '기타법인']
                            if '개인' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumePersonal'] = df_chart4.at[idx, '개인']
                            if '외국인합계' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumeForeign'] = df_chart4.at[idx, '외국인합계']
                        else:
                            if '기관합계' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumeOrgan'] = df_chart4.at[idxp, 'VolumeOrgan'] + df_chart4.at[idx, '기관합계']
                            if '기타법인' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumeEtc'] = df_chart4.at[idxp, 'VolumeEtc'] + df_chart4.at[idx, '기타법인']
                            if '개인' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumePersonal'] = df_chart4.at[idxp, 'VolumePersonal'] + df_chart4.at[idx, '개인']
                            if '외국인합계' in df_chart4.columns:
                                df_chart4.at[idx, 'VolumeForeign'] = df_chart4.at[idxp, 'VolumeForeign'] + df_chart4.at[idx, '외국인합계']
                        idxp = idx
                    
                    df_out = df_out.join(df_chart4)
                    logger.info(f"투자자별 거래량 정보 조회 성공 (종목: {code})")
                else:
                    # 데이터가 없는 경우 기본값 설정
                    volume_columns = ['VolumeOrgan', 'VolumeForeign', 'VolumeEtc', 'VolumePersonal']
                    df_chart4 = pd.DataFrame(0, index=df_out.index, columns=volume_columns)
                    df_out = df_out.join(df_chart4)
                    logger.info(f"투자자별 거래량 정보 없음 (종목: {code}) - 기본값 사용")
            except Exception as e:
                logger.warning(f"투자자별 거래량 정보 조회 실패 (종목: {code}): {str(e)}")
                # 오류 발생 시 기본값 설정
                volume_columns = ['VolumeOrgan', 'VolumeForeign', 'VolumeEtc', 'VolumePersonal']
                df_chart4 = pd.DataFrame(0, index=df_out.index, columns=volume_columns)
                df_out = df_out.join(df_chart4)

            ##일자별 외국인 보유량 및 외국인 한도소진률 - 안전한 처리
            # 외국인 보유량 정보가 없을 때 기본값 먼저 추가
            df_out['외국인_보유수량'] = 0
            df_out['외국인_지분율'] = 0.0
            
            try:
                df_chart4_2 = stock.get_exhaustion_rates_of_foreign_investment(dates[0], dates[1], code)
                
                if df_chart4_2 is not None and not df_chart4_2.empty:
                    # 컬럼이 존재하는지 확인 후 삭제
                    columns_to_drop = ['상장주식수', '한도수량', '한도소진률']
                    existing_columns = [col for col in columns_to_drop if col in df_chart4_2.columns]
                    if existing_columns:
                        df_chart4_2.drop(existing_columns, axis=1, inplace=True)
                    
                    rename_dict = {}
                    if '보유수량' in df_chart4_2.columns:
                        rename_dict['보유수량'] = '외국인_보유수량'
                    if '지분율' in df_chart4_2.columns:
                        rename_dict['지분율'] = '외국인_지분율'
                    
                    if rename_dict:
                        df_chart4_2.rename(columns=rename_dict, inplace=True)
                    
                    # 기존 기본값 컬럼 제거 후 조인
                    for col in ['외국인_보유수량', '외국인_지분율']:
                        if col in df_chart4_2.columns and col in df_out.columns:
                            df_out.drop(columns=[col], inplace=True)
                    
                    df_out = df_out.join(df_chart4_2)
                    logger.info(f"외국인 보유량 정보 조회 성공 (종목: {code})")
                else:
                    logger.info(f"외국인 보유량 정보 없음 (종목: {code}) - 기본값 사용")
                    
            except Exception as e:
                logger.info(f"외국인 보유량 정보 조회 불가 (종목: {code}) - 기본값 사용: {str(e)}")
                # 이미 기본값이 설정되어 있으므로 추가 작업 불필요

            ## 공매도 정보 - 안전한 처리
            # 공매도 정보가 없을 때 기본값 먼저 추가
            df_out['공매도비중'] = 0.0
            df_out['공매도잔고'] = 0
            
            try:
                # pykrx API 호출 시도 - 더 안전한 방식으로 처리
                try:
                    df_chart5 = stock.get_shorting_balance_by_date(dates[0], dates[1], code)
                except Exception as api_error:
                    logger.warning(f"공매도 API 호출 실패 (종목: {code}): {str(api_error)}")
                    df_chart5 = None
                
                if df_chart5 is not None and not df_chart5.empty and len(df_chart5.columns) > 0:
                    try:
                        # 컬럼이 존재하는지 확인 후 삭제
                        columns_to_drop = ['상장주식수', '시가총액']
                        existing_columns = [col for col in columns_to_drop if col in df_chart5.columns]
                        if existing_columns:
                            df_chart5.drop(existing_columns, axis=1, inplace=True)
                        
                        # 컬럼명 정리
                        rename_dict = {}
                        if '비중' in df_chart5.columns:
                            rename_dict['비중'] = '공매도비중'
                        if '공매도잔고' in df_chart5.columns:
                            rename_dict['공매도잔고'] = '공매도잔고'
                        elif '잔고' in df_chart5.columns:
                            rename_dict['잔고'] = '공매도잔고'
                        
                        if rename_dict:
                            df_chart5.rename(columns=rename_dict, inplace=True)
                            
                            # 기존 기본값 컬럼 제거 후 조인
                            for col in ['공매도비중', '공매도잔고']:
                                if col in df_chart5.columns and col in df_out.columns:
                                    df_out.drop(columns=[col], inplace=True)
                            
                            df_out = df_out.join(df_chart5)
                            logger.info(f"공매도 정보 조회 성공 (종목: {code})")
                        else:
                            logger.info(f"공매도 정보 컬럼 매핑 실패 (종목: {code}) - 기본값 사용")
                    except Exception as process_error:
                        logger.warning(f"공매도 정보 처리 실패 (종목: {code}): {str(process_error)}")
                else:
                    logger.info(f"공매도 정보 없음 (종목: {code}) - 기본값 사용")
                    
            except Exception as e:
                logger.warning(f"공매도 정보 조회 실패 (종목: {code}): {str(e)}")
                # 이미 기본값이 설정되어 있으므로 추가 작업 불필요

            ## 코스피 지수 확인용
            df_krx = stock.get_index_ohlcv(st, end, "1028")  ## kospi200
            df_krx.rename(columns={'시가': 'Open',
                                   '고가': 'High',
                                   '저가': 'Low',
                                   '종가': 'Close',
                                   '거래량': 'Volume',
                                   }, inplace=True)

            ## 종목이 지정한 날짜보다 늦게 생겼을 때, x 길이를 맞추기 위함
            if len(df_out) < len(df_krx):
                new_st = df_out.head(1).index.to_list()[0]
                df_krx2 = df_krx[df_krx.index >= new_st]
            else:
                df_krx2 = df_krx

            ## 장중에 진행 시, krx 는 어제 까지 ... 시세는 현재까지라
            if len(df_out) > len(df_krx2):
                df_out = df_out.head(len(df_krx2))

            ## kospi 와 종목간의 일치성 확인
            df_out['Change'] = round((df_out['Close'] - df_out['Open']) / df_out['Close'] * 100,2)
            df_out['krxChange'] = round((df_krx2['Close'] - df_krx2['Open']) / df_krx2['Close'] * 100,2)
            def comp_change(df):
                if (df['Change'] >= 0) and (df['krxChange'] >= 0):
                    x = 1
                elif (df['Change'] >= 0) and (df['krxChange'] >= 0):
                    x = 1
                else:
                    x = -1
                return x
            df_out['compChange'] = df_out.apply(comp_change, axis=1)
            acc = []
            for cnt, idx in enumerate(df_out.index):
                if cnt == 0 :
                    acc_cur = 0
                else:
                    c = df_out.at[idx,'compChange']
                    acc_cur = acc_prev + c

                    if acc_cur < 0 : acc_cur = 0
                acc.append(acc_cur)
                acc_prev = acc_cur

            df_out['compChangeAcc'] = acc



        elif mode == "realtime":
            if len(data) == 0:
                raise ValueError("입력된 데이터의 크기가 0 입니다.")
            else:
                df_out = data.copy()
                st = dates[0]
                end = dates[1]
        elif mode == "etf":
            if len(data) == 0:
                raise ValueError("입력된 데이터의 크기가 0 입니다.")
            else:
                df_out = data.copy()
                st = dates[0]
                end = dates[1]
        elif mode == "investor":
            df_out = data.copy()
            st = dates[0]
            end = dates[1]



        else:
            raise ValueError(f"지원하지 않는 모드 ({mode}) 입니다. ")


        ## make sub-plot
        macd = self._macd(df_out, 12, 26, 9)
        # stochastic = self._stochastic(df_out, 14, 3)
        rsi = self._rsi(df_out, 14)
        bol = self._bollinger(df_out, window=20, sigma=2.0)
        obv = self._obv(df_out, mav=20)
        vol_abn = self._volume_anomaly(df_out, window=7, quantile=0.80, mode=mode)


        ## 이동 평균 추가
        if mode != 'investor':  ## 당일 kospi 는 open, close 가 없음.
            df_out['CloseRatio'] = round((df_out['Close'] - df_out['Open']) / df_out['Close'] * 100, 2)
        df_out['ma5'] = round(df_out['Close'].rolling(window=5).mean())
        df_out['ma20'] = round(df_out['Close'].rolling(window=20).mean())
        df_out['ma40'] = round(df_out['Close'].rolling(window=40).mean())
        df_out['ma60'] = round(df_out['Close'].rolling(window=60).mean())

        ## 이동평균 방향
        df_out['ma5Pos'] = (df_out['ma5'] - df_out['ma5'].shift(1)) >= 0
        df_out['ma20Pos'] = (df_out['ma20'] - df_out['ma20'].shift(1)) >= 0
        df_out['ma40Pos'] = (df_out['ma40'] - df_out['ma40'].shift(1)) >= 0
        df_out['ma60Pos'] = (df_out['ma60'] - df_out['ma60'].shift(1)) >= 0

        ## 이동평균 간 거리차
        df_out['ma520Dist'] = df_out['ma5'] - df_out['ma20']
        df_out['ma560Dist'] = df_out['ma5'] - df_out['ma60']
        df_out['ma520DistCenter'] = 0
        df_out['ma520DistPos'] = df_out['ma520Dist'] - df_out['ma520Dist'].shift(1)


        ##################################
        ####   매수, 매도 조건 만들어 내기
        ##################################
        if mode != 'investor':
            pntDict = self._check_buy_sell(df_out)
        else:
            pntDict = self._check_buy_sell_investor(df_out)

        points = []
        colors = []
        for key, value in pntDict.items():
            data = value['data']
            color = value['color']

            if not len(data) == 0 :
                points = points + data
                temp = [color for i in range(len(data))]
                colors = colors + temp

        ## Pannel number
        if mode == "daily": ##
            pannel_id = {
                'volume':   2,
                'krx':      3,
                'investor':  4,
                # 'krxComp':  4,
                'maDist':    5,
                # 'maDistPos': 5,
                # 'rsi':      4,
                'per':      6,
                'foreign':  7,
                'short':    8,
               # 'obv' = 3
               #  'macd': 5,
               #  'stochestic': 6,
            }

            ## add sub-plot
            if self.display != 'off':
                add_plots = [
                    mpf.make_addplot(bol['bol_upper'], color='#606060'),
                    mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

                    ## volume anomaly
                    mpf.make_addplot(df_out['VolumeThreshold'], ylabel='Turnover ratio', color='orange',
                                     panel=pannel_id['volume']),
                    mpf.make_addplot(df_out['VolumeTurnOver'], color='black', panel=pannel_id['volume']),
                    mpf.make_addplot(df_out['VolumeAnomaly'], type='scatter', marker='v', markersize=200, color='red',
                                     panel=pannel_id['volume']),

                    ## krx
                    mpf.make_addplot(df_krx2['Close'], ylabel='Kospi 200', color='#8c564b', panel=pannel_id['krx']),

                    # ## krx
                    # mpf.make_addplot(df_out['compChangeAcc'], ylabel='compare with krx', color='#8c564b', panel=pannel_id['krxComp']),

                    # investor
                    mpf.make_addplot(df_out['VolumeForeign'], ylabel='Investor (frgn:bl, org:b, prsl:y, etc:sb', color='black', panel=pannel_id['investor']),
                    mpf.make_addplot(df_out['VolumePersonal'], secondary_y=False, color='yellow', panel=pannel_id['investor']),
                    mpf.make_addplot(df_out['VolumeOrgan'], secondary_y=False, color='blue', panel=pannel_id['investor']),
                    mpf.make_addplot(df_out['VolumeEtc'], secondary_y=False, color='#2cb7f2', panel=pannel_id['investor']),

                    ## ma 거리차
                    mpf.make_addplot(df_out['ma520Dist'], ylabel='Distance(ma5 - ma20)', color='#8c564b', panel=pannel_id['maDist']),
                    mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDist']),

                    ## ma 방향성
                    # mpf.make_addplot(df_out['ma520DistPos'], ylabel="Dist(ma5 20)'s direction", color='#8c564b',
                    #                  panel=pannel_id['maDistPos']),
                    # mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDistPos']),

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

                    ## short
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
        elif mode == "realtime":
            pannel_id = {
                'volume':       2,
                'maDist':       3,
                'chegyeol':     4,
                # 'maDistPos':    3,
                # 'rsi':          5,
                # 'obv' :         6
            }

            ## add sub-plot
            if self.display != 'off':
                add_plots = [
                    mpf.make_addplot(bol['bol_upper'], color='#606060'),
                    mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

                    ## volume anomaly
                    mpf.make_addplot(df_out['VolumeThreshold'], ylabel='Turnover ratio', color='orange', panel=pannel_id['volume']),
                    mpf.make_addplot(df_out['VolumeTurnOver'],  color='black', panel=pannel_id['volume']),
                    mpf.make_addplot(df_out['VolumeAnomaly'], type='scatter', marker='v', markersize=200, color='red',
                                     panel=pannel_id['volume']),

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
                    # mpf.make_addplot(rsi['rsi'], ylim=[0, 100], ylabel='RSI',  panel=pannel_id['rsi']),
                    # mpf.make_addplot(rsi['rsi_high'], color='r', panel=pannel_id['rsi']),
                    # mpf.make_addplot(rsi['rsi_low'], color='b', panel=pannel_id['rsi']),
                    #
                    # ## obv
                    # mpf.make_addplot(obv['obv'], ylabel='OBV', color='#8c564b', panel=pannel_id['obv']),
                    # mpf.make_addplot(obv['obv_ema'], color='#e377c2', panel=pannel_id['obv']),


                ]
        elif mode == "etf":
            pannel_id = {
                'volume':       2,
                'maDist':       3,
                'maDistPos':    4,
                'rsi':          5,
                'obv' :         6
            }

            ## add sub-plot
            if self.display != 'off':
                add_plots = [
                    mpf.make_addplot(bol['bol_upper'], color='#606060'),
                    mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

                    ## volume anomaly
                    mpf.make_addplot(df_out['VolumeThreshold'], ylabel='Turnover ratio', color='orange', panel=pannel_id['volume']),
                    mpf.make_addplot(df_out['VolumeTurnOver'],  color='black', panel=pannel_id['volume']),
                    mpf.make_addplot(df_out['VolumeAnomaly'], type='scatter', marker='v', markersize=200, color='red',
                                     panel=pannel_id['volume']),

                    ## ma 거리차
                    mpf.make_addplot(df_out['ma520Dist'], ylabel='Distance(ma5 - ma20)', color='#8c564b', panel=pannel_id['maDist']),
                    mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDist']),


                    ## ma 방향성
                    mpf.make_addplot(df_out['ma520DistPos'], ylabel="Dist(ma5 20)'s direction", color='#8c564b',
                                     panel=pannel_id['maDistPos']),
                    mpf.make_addplot(df_out['ma520DistCenter'], color='black', secondary_y=False, panel=pannel_id['maDistPos']),

                    ## rsi
                    mpf.make_addplot(rsi['rsi'], ylim=[0, 100], ylabel='RSI',  panel=pannel_id['rsi']),
                    mpf.make_addplot(rsi['rsi_high'], color='r', panel=pannel_id['rsi']),
                    mpf.make_addplot(rsi['rsi_low'], color='b', panel=pannel_id['rsi']),

                    ## obv
                    mpf.make_addplot(obv['obv'], ylabel='OBV', color='#8c564b', panel=pannel_id['obv']),
                    mpf.make_addplot(obv['obv_ema'], color='#e377c2', panel=pannel_id['obv']),
                ]
        elif mode == "investor":

            pannel_id = {
                'foreign':       2,
                'fpo':       3,
                'f2p':      4,
                'future':   5,
                'ff2p':     6,
                'f2ff2p':   7,
                'program':  8,
            }

            ## add sub-plot
            if self.display != 'off':
                add_plots = [
                    mpf.make_addplot(bol['bol_upper'], color='#606060'),
                    mpf.make_addplot(bol['bol_lower'], color='#1f77b4'),

                    mpf.make_addplot(df_out['Foreigner'], ylabel='Foreigner', color='black', panel=pannel_id['foreign']),
                    mpf.make_addplot(df_out['Foreigner_ma5'],  color='blue', panel=pannel_id['foreign']),
                    mpf.make_addplot(df_out['Foreigner_ma20'],  color='orange', panel=pannel_id['foreign']),
                    mpf.make_addplot(df_out['Foreigner_ma40'],  color='green', panel=pannel_id['foreign']),

                    mpf.make_addplot(df_out['Foreigner'], ylabel='Investor: F(bk),P(y),O(g)', color='black', panel=pannel_id['fpo']),
                    mpf.make_addplot(df_out['Personal'], secondary_y=False,  color='yellow',  panel=pannel_id['fpo']),
                    mpf.make_addplot(df_out['Organ'], secondary_y=False, color='green', panel=pannel_id['fpo']),

                    mpf.make_addplot(df_out['F2P'], ylabel='Foreign-Personal', color='pink', panel=pannel_id['f2p']),
                    # mpf.make_addplot(df_out['F2P_ma5'], color='blue', panel=pannel_id['f2p']),
                    # mpf.make_addplot(df_out['F2P_ma20'], color='orange', panel=pannel_id['f2p']),
                    # mpf.make_addplot(df_out['F2P_ma40'], color='green', panel=pannel_id['f2p']),

                    mpf.make_addplot(df_out['FutureForeigner'], ylabel='Future: F(bk),P(y),O(g)', color='black', panel=pannel_id['future']),
                    mpf.make_addplot(df_out['FuturePersonal'], secondary_y=False, color='yellow', panel=pannel_id['future']),
                    mpf.make_addplot(df_out['FutureOrgan'], secondary_y=False, color='green', panel=pannel_id['future']),

                    mpf.make_addplot(df_out['FF2P'], ylabel='Future(Foreign-Personal)', color='pink', panel=pannel_id['ff2p']),

                    mpf.make_addplot(df_out['F2P_FF2P'], ylabel='Sum(F2P, FF2P)', color='red', panel=pannel_id['f2ff2p']),

                    mpf.make_addplot(df_out['Arbitrage'], ylabel='arb(b), nonarb(r)', color='blue', panel=pannel_id['program']),
                    mpf.make_addplot(df_out['NonArbitrage'], secondary_y=False, color='red', panel=pannel_id['program']),

                ]
            pass
        else:
            raise ValueError(f"sub_plot 생성 중 에러 발생입니다. 지원하지 않는 모드 ({mode}) 입니다. ")



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
        if self.display == 'save':
            if self.path == "":  # 외부에서 입력 할 수 있음.
                self.path = self.file_manager["search_stocks"]["path"] + f"market_leader/{st}_{end}/"
                self.name = f"{code}_{st}_{end}.png"

            try:
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
            except Exception as e:
                raise e

            try:
                mpf.plot(df_out, type='candle', mav=(5,20,60), volume=True, addplot=add_plots, panel_ratios=tuple(pannel_ratio),
                          figsize=(30, 8 + 3*pannel_cnt),
                          title=title,
                          vlines=dict(vlines= points, linewidths=5, colors=colors, alpha=0.30),
                          figscale=0.8, style=s,
                          scale_padding={'right':2.0, 'left':0.5},
                          warn_too_much_data=5000,
                          tight_layout=True,
                          savefig=f'{self.path}{self.name}',
                          )
            except Exception as e:
                logger.error(e)
        elif self.display == 'on':
            try:
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
            except Exception as e:
                logger.error(e)
        else:
            ## chart 관련 코드는 위에서 생성되지 않도록 처리 함. (인스턴스 살아 있는 상태에서 on 으로 바뀌는 것을 대비)
            pass

        return df_out


    def _check_buy_sell_investor(self, df):
        df['Foreigner_ma5'] = round(df['Foreigner'].rolling(window=5).mean())
        df['Foreigner_ma20'] = round(df['Foreigner'].rolling(window=20).mean())
        df['Foreigner_ma40'] = round(df['Foreigner'].rolling(window=40).mean())

        df['Personal_ma5'] = round(df['Personal'].rolling(window=5).mean())
        df['Personal_ma20'] = round(df['Personal'].rolling(window=20).mean())
        df['Personal_ma40'] = round(df['Personal'].rolling(window=40).mean())

        df['Organ_ma5'] = round(df['Organ'].rolling(window=5).mean())
        df['Organ_ma20'] = round(df['Organ'].rolling(window=20).mean())
        df['Organ_ma40'] = round(df['Organ'].rolling(window=40).mean())

        df['F2P'] = df['Foreigner'] - df['Personal']
        df['F2P_ma5'] = round(df['F2P'].rolling(window=5).mean())
        df['F2P_ma20'] = round(df['F2P'].rolling(window=20).mean())
        df['F2P_ma40'] = round(df['F2P'].rolling(window=40).mean())

        df['FF2P'] = df['FutureForeigner'] - df['FuturePersonal']
        df['FF2P_ma5']  = round(df['FF2P'].rolling(window=5).mean())
        df['FF2P_ma20'] = round(df['FF2P'].rolling(window=20).mean())
        df['FF2P_ma40'] = round(df['FF2P'].rolling(window=40).mean())

        df['F2P_FF2P'] = df['F2P'] + df['FF2P']

        ####################################
        ####     매수 타이밍 정리
        ####################################
        queue = [0,0,0,0]
        buy_times = []  ## 매수 시점 기록
        buy_times2 = []  ## 매수 시점 기록
        for i, idx in enumerate(df.index):
            if i == 0:
                ma5_prev = 2
                ma20_prev = 1
                ma40_prev = 0

                f2p_ma5_prev = 2
                f2p_ma20_prev = 1
                f2p_ma40_prev = 0

            ma5 = df.Foreigner_ma5.iat[i]
            ma20 = df.Foreigner_ma20.iat[i]
            ma40 = df.Foreigner_ma40.iat[i]

            f2p_ma5 = df.F2P_ma5.iat[i]
            f2p_ma20 = df.F2P_ma20.iat[i]
            f2p_ma40 = df.F2P_ma40.iat[i]

            if ma5 > ma20 and ma5_prev <= ma20_prev:
                buy_times.append(True)
            else:
                buy_times.append(False)

            ## 개인, 외국인 포지션이 변경되는 시점
            if f2p_ma20 > f2p_ma40 and f2p_ma20_prev <= f2p_ma40_prev:
                buy_times2.append(True)
            elif f2p_ma20 <= f2p_ma40 and f2p_ma20_prev > f2p_ma40_prev:
                buy_times2.append(True)
            else:
                buy_times2.append(False)

            ma5_prev = ma5
            ma20_prev = ma20
            ma40_prev = ma40
            f2p_ma5_prev  = f2p_ma5
            f2p_ma20_prev = f2p_ma20
            f2p_ma40_prev = f2p_ma40

        df['finalBuy'] = buy_times ## 매도에서 사용
        df['F2PBuy'] = buy_times2 ## 매도에서 사용
        ##################################
        ####   Chart 에 표식 남기기
        ##################################
        pntDict = dict()
        pntDict['ForeignBuy'] = dict()
        pntDict['ForeignBuy']['data'] = df[df.finalBuy == True].index.to_list() ## bol 기준으로 sell 시점
        pntDict['ForeignBuy']['color'] = '#ff33bb'  ## 핑크 계열
        pntDict['F2PBuy'] = dict()
        pntDict['F2PBuy']['data'] = df[df.F2PBuy == True].index.to_list() ## bol 기준으로 sell 시점
        pntDict['F2PBuy']['color'] = '#17992d'  ## 그린 계열

        return pntDict


    #### 매매 타이밍 체크
    def _check_buy_sell(self, df):
        '''

        :param df:
            df (Dataframe) : 내부 처리는 링크되어 반영됨
            매수, 매도 조건은 추구 config 로 뺄 것 (22.09.28)
        :return:
        '''

        ####################################
        ####     매수 타이밍 정리
        ####################################
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

        pos_cond1 = False
        chk1HoldCnt = 0
        buy_times = []  ## 매수 시점 기록
        buy_costs = []  ## 매수 시점 가격 (수익률 계산용)
        buy_test0 = []  ## 매수 전략 테스트용
        buy_test1 = []  ## 매수 전략 테스트 용2
        ma20_que = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 2주치
        ma40_que = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ma60_que = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        hold_period = 14 ## 거래량 이후 매수 까지 유지 기간

        ## 매수 조건 확인
        buyCond3 = self.trade_config["buy_condition"]["timepick_trend_period"]
        buyCond3_2 = self.trade_config["buy_condition"]["timepick_trend_change"]

        for idx in df.index:
            bolBuy = df.bolBuy.at[idx]
            volAnol = df.VolumeAnomaly.at[idx]

            #0) 준비작업: ma40, ma60 의 기울기 구하기
            ma20crr = df.ma20.at[idx]
            ma40crr = df.ma40.at[idx]
            ma60crr = df.ma60.at[idx]
            if (ma20crr == 0) or np.isnan(ma20crr):  ma20chg = 0
            else:
                if np.isnan(ma20_que[0]): ma20chg = 0
                else: ma20chg = self._change_ratio(ma20crr, ma20_que[0])
            if (ma40crr == 0) or np.isnan(ma40crr): ma40chg = 0
            else:
                if np.isnan(ma40_que[0]): ma40chg = 0
                else: ma40chg = self._change_ratio(ma40crr, ma40_que[0])

            if (ma60crr == 0) or np.isnan(ma60crr): ma60chg = 0
            else:
                if np.isnan(ma60_que[0]): ma60chg = 0
                else: ma60chg = self._change_ratio(ma60crr, ma60_que[0])

            ma20_que.append(ma20crr)
            ma20_que.pop(0)
            ma40_que.append(ma40crr)
            ma40_que.pop(0)
            ma60_que.append(ma60crr)
            ma60_que.pop(0)

            if buyCond3 == 'ma20':
                ma00chg = ma20chg
            elif buyCond3 == 'ma40':
                ma00chg = ma40chg
            elif buyCond3 == 'ma60':
                ma00chg = ma60chg
            else:
                raise ValueError(f"'timepick_trend_period' 는 ma20, ma40, ma60 중에 하나만 지원합니다.")

            ## test0: bol 상단터치 했는데, 거리량 터치 시점 확인
            ## test1: 실제 매수 중에 ma40 이 증가 추세를 통해 강도 확인

            buy_test1.append(False) ## 나중 사용

            # 조건 1) 커디션 체크 : bol 를 상단 터치하였는데, 거래량 터지고 60일 선이 증가 추세
            if (bolBuy == True) and ~(np.isnan(volAnol)):
                currChange = df.CloseRatio.at[idx]
                buy_test0.append(True)  ##
                # print(idx, currChange, ma00chg)

                # 조건2: 40, 60 일선의 증가율이 특정값 이상일 경우 (상승 추세 강도)
                if ma00chg >= buyCond3_2:

                    # 조건4: 하루 상승률이 10%로 이하인 경우
                    if currChange < 15 :  ## bol 상단 찍을려면 무조건 강한 상승 필요. 즉 exclusive 함
                        # 전략1:  조건1 과 조건 2, 를 만족하는 경우
                        buy_times.append(True)
                        buy_costs.append(df.Close.at[idx])
                        pos_cond1 = False
                        chk1HoldCnt = 0  ## 리프레쉬 되면 카운트 다시 시작
                        # print(f"매수 신호(date:{idx}): 당일 등락율 ({currChange}),  장기 증가 추세율 ({ma00chg}) ")
                    else:
                        buy_times.append(False)
                        buy_costs.append(0)
                        pos_cond1 = False
                        chk1HoldCnt = 0  ## 리프레쉬 되면 카운트 다시 시작
                else:
                    # 전략2:  조건2를 만족하지 못했지만 상승 가능성이 있기 때문에 다음 상승 추세로 유보한다.
                    buy_times.append(False)
                    buy_costs.append(0)
                    pos_cond1 = True
                    chk1HoldCnt = 0  ## 리프레쉬 되면 카운트 다시 시작

            else:
                buy_test0.append(False)  ##

                buy_times.append(False)
                buy_costs.append(0)

        df['finalBuy'] = buy_times ## 매도에서 사용

        df['finalBuyTest0'] = buy_test0
        df['finalBuyTest1'] = buy_test1

        #######################################
        ###       매도 타이밍 정리
        #######################################
        '''
            긍정 전략1: 20% 수익률이면 팔기 
            긍정 전략2: 수익률이 절반이 되는 시점 팔기 
            긍정 전략3: 전고점이면 팔기 
            
            부정 전략1: 손실 시, 저항선이면 팔기 
            부정 전략2: 손실 시, 10% 면 팔기 
            부정 전략3:  
        '''
        # cofig 값
        config_profit_change = self.trade_config["sell_condition"]["default_profit_change"]
        config_holding_days = self.trade_config["sell_condition"]["default_holding_days"]
        target_ratio = 20

        ## 매물대 확인
        volCnt = 20
        df_volProf = pd.cut(df.Close, volCnt)
        df_volProf2 = df.groupby(df_volProf)
        close_dict = dict()
        acc = 0
        for grp in df_volProf2:
            key = list(grp)[0]
            df_part = list(grp)[1]
            closeVol = df_part.Close * df_part.Volume
            close_dict[key] = closeVol.sum()
            acc += closeVol.sum()
        ## 매물대 별 비율 구하기
        for key, val in close_dict.items():
            a = round(val / acc * 100, 2)
            close_dict[key] = a

            _msg = f"[매물대] {key}: {a} %"
            print(_msg)
        ## 매물대 출력

        sell_times = []
        sell_qnty = []
        buy_qnty = 0
        ma5Pos_que = [0,0,0,0,0,0,0]
        acc_porf = 0
        profit_ratio_max = 0 ## 매수 이후 수익률 최대값
        for cnt, idx in enumerate(df.index):
            close_cur = df.Close.at[idx]
            ma5PosCrr = df.ma5Pos.at[idx]

            ## 초기 10일 스킵
            if cnt < 10 :
                sell_times.append(False)
                sell_qnty.append(0)
                continue

            ## 매수 시점
            if buy_times[cnt]:
                if buy_qnty == 0:
                    buy_cost = buy_costs[cnt]  ## 첫 구매 시점의 가격을 기록. 수익률 계산용
                    hold_time = 0

                buy_qnty += 1

            ## 매수조건이 남아 있을 때만 매도 가능
            if buy_qnty > 0 :
                # 상승률 계산
                profit_ratio = self._change_ratio(close_cur, buy_cost)
                if profit_ratio_max < profit_ratio:
                    profit_ratio_max = profit_ratio
                ## 단기간 데이터 준비
                # if cnt < 20:  # 20일 안에서 최대값 계산
                #     df_part = df.iloc[:cnt, :]
                # else:
                #     df_part = df.iloc[cnt - 20:cnt, :]

                ## SSH(22.12) 매도 시점 디버깅 용도
                # print(f"SSH 상승률 확인({idx}): {profit_ratio}, 최대 수익률: {profit_ratio_max}")
                if (close_cur >= buy_cost):  ## 종가 기준 수익 구간
                    # 조건1: 수익 20프로 이상이면 매도
                    if profit_ratio >= target_ratio :
                        sell_times.append(True)
                        sell_qnty.append(buy_qnty)
                        acc_porf += profit_ratio * buy_qnty
                        _msg = f"[BUY_POS1] 수익률이 ({target_ratio})% 이상일 경우 매도함.(수익률(매수 {buy_qnty}회): {profit_ratio}%)(매도일: {idx})"
                        buy_qnty = 0 # 전량 매도
                        print(_msg)

                    # 조건2: 매수 이후 최대 수이률에서 절반이 되는 시점
                    elif (profit_ratio_max > 10) and (profit_ratio < profit_ratio_max / 2):
                        sell_times.append(True)
                        sell_qnty.append(buy_qnty)
                        acc_porf += profit_ratio * buy_qnty
                        _msg = f"[BUY_POS2] 당일 종가가 20일 내 최대 값 대비 절반 이하일 경우 매도함. (수익률(매수 {buy_qnty}회): {profit_ratio}%)(매도일: {idx})"
                        buy_qnty = 0  # 전량 매도
                        print(_msg)

                    # 조건3: 매물대에 도달할 경우 판매 (최대 수익률이 5% 이후부터 동작)
                    else:
                        chk=False
                        for key, value in close_dict.items():
                            if close_cur in key:  # 내가 속한 값 범위 확인
                                if value > 10 : # 이때, 매물대가 10% 이상일 경우
                                    chk=True
                        if (chk==True) and (profit_ratio_max>5):
                            sell_times.append(True)
                            sell_qnty.append(buy_qnty)
                            acc_porf += profit_ratio * buy_qnty
                            _msg = f"[BUY_POS3] 손실 시, 저지대를 돌파한 경우 (수익률(매수 {buy_qnty}회): {profit_ratio}%)(매도일: {idx})"
                            buy_qnty = 0  # 전량 매도
                            print(_msg)
                        else:
                            sell_times.append(False)
                            sell_qnty.append(0)


                else: # 종가 기준 손실 구간
                    # 조건4: 손실 시, 지지선을 돌파하면 매도
                    if (close_cur < buy_cost) and (close_cur < close_prev):  ## 손실 발생, 2일차부터 가능
                        chk = False
                        for key, value in close_dict.items():
                            if (close_prev in key) and (close_cur not in key): # 매물대 하향 돌파
                                if value > 10:  # 어제 종가가 매물대에 머물러 있었던 상태
                                    chk = True
                        if chk:
                            sell_times.append(True)
                            sell_qnty.append(buy_qnty)
                            acc_porf += profit_ratio * buy_qnty
                            _msg = f"[BUY_NEG1] 손실 시, 저지대를 돌파한 경우 (수익률(매수 {buy_qnty}회): {profit_ratio}%)(매도일: {idx})"
                            buy_qnty = 0  # 전량 매도
                            print(_msg)
                        else:
                            sell_times.append(False)
                            sell_qnty.append(0)

                    # 조건5:  손실 시, 손실액이 매수가에 10% 면 팔기
                    else:
                        if (close_cur < buy_cost):  ## 손실 발생,
                            profit_ratio = self._change_ratio(close_cur, buy_cost)
                            if profit_ratio < -10:
                                sell_times.append(True)
                                sell_qnty.append(buy_qnty)
                                acc_porf += profit_ratio * buy_qnty
                                _msg = f"[BUY_NEG2] 손실률이 (10% 이상)일 경우 손절 (수익률(매수 {buy_qnty}회): {profit_ratio}%)(매도일: {idx})"
                                buy_qnty = 0  # 전량 매도
                                print(_msg)
                            else:
                                sell_times.append(False)
                                sell_qnty.append(0)
                        else: # 아무일도 일어나지 않음
                            sell_times.append(False)
                            sell_qnty.append(0)
            else: # buy_qnty < 0
                sell_times.append(False)
                sell_qnty.append(0)

                # 매수한 적이 없으면 수익률을 0 으로 초기화 함
                profit_ratio_max = 0

            ma5Pos_que.append(ma5PosCrr)
            ma5Pos_que.pop(0)

            close_prev = close_cur

        _msg = f"총 수익률은 {acc_porf}% 입니다. 동일 금액으로 매수하고, 매수 누적시  처음 매수시점 기준으로 계산됨. "
        print(_msg)
        print("\n")
        df['finalSell'] = sell_times
        df['finalSellQnty'] = sell_qnty

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
        # pntDict['bolBuy'] = dict()
        # pntDict['bolBuy']['data'] = df[df.finalBuyTest0 == True].index.to_list() ## bol 기준으로 buy 시점
        # pntDict['bolBuy']['color'] = '#66ff33'  ## 밝은 녹색
        # pntDict['bolSell'] = dict()
        # pntDict['bolSell']['data'] = df[df.bolSell == True].index.to_list() ## bol 기준으로 sell 시점
        # pntDict['bolSell']['color'] = '#0d3300'  ## 진한 녹색
        pntDict['maBuy'] = dict()
        pntDict['maBuy']['data'] = df[df.finalBuy == True].index.to_list() ## bol 기준으로 sell 시점
        pntDict['maBuy']['color'] = '#ff33bb'  ## 핑크 계열
        # pntDict['maBuyTest1'] = dict()
        # pntDict['maBuyTest1']['data'] = df[df.finalBuyTest1 == True].index.to_list() ## bol 기준으로 sell 시점
        # pntDict['maBuyTest1']['color'] = '#e60000'  ## 보라색 계열
        pntDict['maSell'] = dict()
        pntDict['maSell']['data'] = df[df.finalSell == True].index.to_list() ## bol 기준으로 sell 시점
        pntDict['maSell']['color'] = '#002080'  ## 진파랑 계열

        return pntDict

    ##########################################################################


    def _change_ratio(self, curr, prev):
        return round(float((int(curr) - int(prev)) / int(curr)) * 100, 2)

    ##########################################################################
    ######     Sub-Chart
    ##########################################################################

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

        if np.isnan(anomalies).sum() == len(anomalies): ## 값이 하나도 없을 경우 chart 에러 발생함.
            dmmy = df.VolumeTurnOver.iat[0]
            anomalies.pop(0)  ## inplace
            anomalies.insert(0, dmmy)
        else:
            pass
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
    cm = tradeStrategy('./config/config.yaml')
    cm.run()

