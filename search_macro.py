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
        """
        거시경제 분석 실행
        
        Returns:
            dict: 분석 결과
        """
        logger.info("거시경제 분석 시작")
        
        try:
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
                    logger.warning(f"ETF 데이터 로딩 실패, 이전 날짜로 재시도: {sttime_str} -> {endtime_str}")
                    endtime_dt -= timedelta(days=1)
                    endtime_str = endtime_dt.date().strftime("%Y%m%d")

                    sttime_dt -= timedelta(days=1)
                    sttime_str = sttime_dt.date().strftime("%Y%m%d")
                    tries += 1
                    
            if tries >= max_tries:
                logger.error("ETF 데이터 로딩 실패")
                return {"error": "ETF 데이터 로딩 실패"}
                
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

            ## 승률이 0 보다 큰 ETF 만 남기기 (너무 많아서 이런식으로 솎아 내야 할듯)
            df_positive = df_tot[df_tot.Change >= 0]

            # 거시경제 분석 결과 요약
            total_etfs = len(df_tot)
            positive_etfs = len(df_positive)
            positive_ratio = (positive_etfs / total_etfs * 100) if total_etfs > 0 else 0
            
            # 상위 성과 ETF 분석
            top_performers = df_positive.head(10) if len(df_positive) >= 10 else df_positive
            
            # 섹터별 분석 (ETF 이름에서 섹터 추출)
            sector_analysis = {}
            for idx, row in top_performers.iterrows():
                name = row['Name']
                if '에너지' in name or '원유' in name:
                    sector = 'Energy'
                elif '금' in name or '은' in name or '원자재' in name:
                    sector = 'Commodity'
                elif '부동산' in name or 'REIT' in name:
                    sector = 'RealEstate'
                elif '채권' in name or '국채' in name:
                    sector = 'Bond'
                elif '중국' in name or '베트남' in name or '인도' in name:
                    sector = 'EmergingMarket'
                elif '미국' in name or 'S&P' in name or '나스닥' in name:
                    sector = 'USMarket'
                else:
                    sector = 'Others'
                    
                if sector not in sector_analysis:
                    sector_analysis[sector] = []
                sector_analysis[sector].append({
                    'name': name,
                    'change': row['Change'],
                    'volume_cost': row['VolumeCost']
                })

            # 차트 생성 (기존 로직 유지하되 최적화)
            cm = tradeStrategy('./config/config.yaml')
            cm.display = 'save'  # 파일로 저장

            sttime_dt = endtime_dt - timedelta(days=self.macro_config["config"]["chart_period"])
            sttime_chart_str = sttime_dt.date().strftime("%Y%m%d")

            ## 국내 시장 걸러보기
            ex_names = ["200선물", "코스닥"]
            
            chart_count = 0
            max_charts = 20  # 차트 생성 제한

            for code in df_positive.index.to_list():
                if chart_count >= max_charts:
                    break
                    
                code = str(code).zfill(6)
                name = stock.get_etf_ticker_name(code)
                change = round(df_positive.at[code, "Change"], 2)

                skip = False
                for exword in ex_names:
                    if exword in name:
                        skip = True
                if skip:
                    continue

                try:
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

                    logger.info(f"[수익률: {round(change)} %] ETF ({name}, {code})의 Chart 를 생성합니다.")
                    df_chart = cm.run(code, name, data=df_ohlcv, dates=[sttime_chart_str, endtime_str], mode='etf')
                    
                    # 차트 데이터 저장 (파일 경로 수정)
                    chart_path = "./data/macro_analysis/"
                    import os
                    os.makedirs(chart_path, exist_ok=True)
                    df_chart.to_csv(f"{chart_path}/{code}.csv", index=False)
                    
                    chart_count += 1
                    
                except Exception as e:
                    logger.warning(f"ETF {name}({code}) 차트 생성 실패: {e}")
                    continue

            # 분석 결과 반환
            result = {
                'analysis_date': endtime_str,
                'period': f"{sttime_str} ~ {endtime_str}",
                'total_etfs': total_etfs,
                'positive_etfs': positive_etfs,
                'positive_ratio': round(positive_ratio, 2),
                'top_performers': [
                    {
                        'name': row['Name'],
                        'code': idx,
                        'change': round(row['Change'], 2),
                        'volume_cost': row['VolumeCost']
                    }
                    for idx, row in top_performers.iterrows()
                ],
                'sector_analysis': sector_analysis,
                'market_sentiment': 'Positive' if positive_ratio > 60 else 'Neutral' if positive_ratio > 40 else 'Negative',
                'charts_generated': chart_count
            }
            
            logger.info(f"거시경제 분석 완료 - 긍정적 ETF 비율: {positive_ratio:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"거시경제 분석 중 오류: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    sm = searchMacro()
    sm.run()


    print("SSH!!!")