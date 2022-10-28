import yaml
from pykrx import stock
from pykrx import bond
from datetime import datetime, timedelta
import pandas as pd

## html
import requests
from bs4 import BeautifulSoup as Soup
from selenium import webdriver


## local file
from tools import st_utils as stu
from trade_strategy import tradeStrategy


####    로그 생성    #######
logger = stu.create_logger()




class searchMacro:
    def __init__(self):
        ###########################
        ####     Init Config.
        ###########################
        try:
            config_file = './config/config.yaml'  ## 고정값
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e :
            print (e)
        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

        ## global 변수 선언
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.macro_config = config["searchMacro"]

    def run(self):
        ## ETF 찾기
        end_dt = datetime.now()
        end = end_dt.date().strftime("%Y%m%d")

        st_dt = end_dt - timedelta(days=self.macro_config["config"]["change_period"])
        ch_st = st_dt.date().strftime("%Y%m%d")

        tickers = stock.get_etf_ticker_list(end)
        names = []
        for ticker in tickers:
            name = stock.get_etf_ticker_name(ticker)
            names.append(name)

        df = pd.DataFrame(columns=['Code', 'Name'])
        df['Code'] = tickers
        df['Name'] = names
        df.set_index(keys=['Code'], inplace=True, drop=True)

        ## 종목명 추가하기
        df_tot = stock.get_etf_price_change_by_ticker(ch_st, end)
        df_tot = df_tot.join(df)
        df_tot.rename(columns={'시가': 'Open',
                               '종가': 'Close',
                               '거래량': 'Volume',
                               '등락률': 'Change',
                               '변동폭': 'ChangeRatio',
                               '거래대금': 'VolumeCost',
                               }, inplace=True)

        # 수익률 순으로 정렬하기
        df_tot.sort_values(by="Change", ascending=False, inplace=True)

        ## 중복성 ETF 제거
        # makers = []
        # contents = []
        # for name in names:
        #     if "채권" in name:
        #         print(name)
        #     nsplit = name.split(' ')
        #     makers.append(nsplit[0])
        #     contents.append(nsplit[1])
        # mk = list(set(makers))
        # ct = list(set(contents))

        ## 승률이 0 보다 큰 ETF 만 남기기 (너무 많아서 이런식으로 솎아 내야 할듯)
        df_tot = df_tot[df_tot.Change >= 0]

        cm = tradeStrategy('./config/config.yaml')
        cm.display = 'on'

        st_dt = end_dt - timedelta(days=self.macro_config["config"]["chart_period"])
        st = st_dt.date().strftime("%Y%m%d")

        ## 국내 시장 걸러보기
        ex_names = ["200선물", "코스닥"]

        for code in df_tot.index.to_list():
            code = str(code).zfill(6)
            name = stock.get_etf_ticker_name(code)

            change = round(df_tot.at[code, "Change"], 2)

            skip = False
            for exword in ex_names:
                if exword in name:
                    skip = True
            if skip:
                # print(f"국내주식 제외 --- [수익률: {change}%] ETF ({name}, {code})의 Chart 를 생성합니다.  (기간: {st} ~ {end})")
                continue

            ## chart 준비
            df_ohlcv = stock.get_etf_ohlcv_by_date(ch_st, end, code)
            df_ohlcv.rename(columns={'시가': 'Open',
                                     '고가': 'High',
                                     '저가': 'Low',
                                     '종가': 'Close',
                                     '거래량': 'Volume',
                                     '등락률': 'Change',
                                     '변동폭': 'ChangeRatio',
                                     '거래대금': 'VolumeCost',
                                     '기초지수': 'BaseCost',
                                     }, inplace=True)

            logger.info(f"[수익률: {round(change)} %] ETF ({name}, {code})의 Chart 를 생성합니다.  (수익률 기간: {ch_st} ~ {end})")
            cm.run(code, name, data=df_ohlcv, dates=[st, end], mode='etf')


if __name__ == "__main__":
    sm = searchMacro()
    sm.run()


    print("SSH!!!")