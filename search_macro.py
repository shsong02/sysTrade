# from pykrx import bond
from datetime import datetime, timedelta
import pandas as pd
import yaml

from pykrx import stock

## html
# import requests
# from bs4 import BeautifulSoup as Soup
# from selenium import webdriver


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
            with open(config_file, encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as _e :
            print (_e)
        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

        ## global 변수 선언
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.macro_config = config["searchMacro"]

    def run(self):
        ## ETF 찾기
        endtime_dt = datetime.now()
        endtime_str = endtime_dt.date().strftime("%Y%m%d")

        sttime_dt = endtime_dt - timedelta(days=self.macro_config["config"]["change_period"])
        sttime_str = sttime_dt.date().strftime("%Y%m%d")

        tickers = stock.get_etf_ticker_list(endtime_str)
        names = []
        for ticker in tickers:
            name = stock.get_etf_ticker_name(ticker)
            names.append(name)

        df_symbol = pd.DataFrame(columns=['Symbol', 'Name'])
        df_symbol['Symbol'] = tickers
        df_symbol['Name'] = names
        df_symbol.set_index(keys=['Symbol'], inplace=True, drop=True)

        ## etf 데이터를 가져올 때까지 이전날짜 로 불러오기 
        tries = 0 
        max_tries = 10
        while tries < max_tries:
            try:
                df_tot = stock.get_etf_price_change_by_ticker(sttime_str, endtime_str)
                break
            except Exception:
                print("TEST")
                endtime_dt -= timedelta(days=1)
                endtime_str = endtime_dt.date().strftime("%Y%m%d")

                sttime_dt -= timedelta(days=1)
                sttime_str = sttime_dt.date().strftime("%Y%m%d")
                tries += 1
                print(sttime_str, endtime_str)
        df_tot = df_tot.join(df_symbol)
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

        sttime_dt = endtime_dt - timedelta(days=self.macro_config["config"]["chart_period"])
        sttime_chart_str = sttime_dt.date().strftime("%Y%m%d")

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
            df_ohlcv = stock.get_etf_ohlcv_by_date(sttime_chart_str, endtime_str, code)
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

            logger.info(f"[수익률: {round(change)} %] ETF ({name}, {code})의 Chart 를 생성합니다.  (수익률 기간: {sttime_str} ~ {endtime_str})")
            df_chart = cm.run(code, name, data=df_ohlcv, dates=[sttime_chart_str, endtime_str], mode='etf')

            df_chart.to_csv(f"{self.file_manager['chart']}/{code}.csv", index=False)


if __name__ == "__main__":
    sm = searchMacro()
    sm.run()


    print("SSH!!!")