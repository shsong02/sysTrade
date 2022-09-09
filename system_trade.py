import os

import FinanceDataReader as fdr
import pandas as pd
import os
import shutil
import backtrader as bt
import yaml
import pprint
import math
import time
from datetime import datetime, timedelta

## html
import requests
from bs4 import BeautifulSoup as Soup
from selenium import webdriver

##multi-processing
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

#chart
from pykrx import stock
from pykrx import bond

#Dart
import OpenDartReader


## local file
from forcast_model import forcastModel
from custom_logger import CustomFormatter
from news_crawler import newsCrawler
import st_utils as stu

######### Global ############
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

## log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e :
    print(e)

####    로그 생성    #######
logger = stu.create_logger()

class systemTrade:
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

    def stock_list(self, save=True):
        """종목 코드를 받아 옵니다. (Sector 존재하는 코드만 남겨 둡니다.)

        ex:
        Symbol	Market	Name	Sector	Industry	ListingDate	SettleMonth	Representative	HomePage	Region
        60310	KOSDAQ	3S	전자부품 제조업	반도체 웨이퍼 캐리어	2002-04-23	03월	김세완	http://www.3sref.com	서울특별시
        95570	KOSPI	AJ네트웍스	산업용 기계 및 장비 임대업	렌탈(파렛트, OA장비, 건설장비)	2015-08-21	12월	박대현, 손삼달	http://www.ajnet.co.kr	서울특별시

        Args:
            save (bool): 파일 저장 여부를 결정
            file_name (str): 저장할 파일 이름 (경로는 fixed 됨)

        Returns:
            df_krx (Dataframe): code 번호가 담겨 있는 df (column='Symbol' 로 접근)
        """

        ## init var
        file_path = self.file_manager['stock_info']['path']
        file_name = self.file_manager['stock_info']['name']

        krx = fdr.StockListing('KRX')

        df_krx = krx.dropna(axis=0, subset=['Sector'])  ## 섹터값없는 코드 삭제 (ETF...)
        df_krx.reset_index(drop=True, inplace=True)

        if save == True:
            ## 파일로 저장 합니다.
            stu.file_save(df_krx, file_path, file_name, replace=True)

        return df_krx

    def access_dart(self, code):
        key = self.keys["dart"]["key"]

        dart = OpenDartReader(key)



    def finance_state(self, code_name, mode='quarter',
                      select=[['매출액증가율','영업이익증가율','영업이익률','ROE','부채비율'],
                              ['당기순이익', 'PER(배)', '부채비율', '당좌비율', '유보율', 'PBR(배)']]):
        """종목 코드에 대한 재무제표를 가져옵니다.

        ex:
                             2019.12  2020.12   2021.12 2022.12(E)
            주요재무정보
            당기순이익        724      871      1408       2119
            PER(배)       NaN    56.35     96.79      36.18
            부채비율      109.19    60.51     63.82        NaN
            당좌비율      161.37   424.84    216.01        NaN
            유보율     20569.19  6604.12  13180.02        NaN

        Args:
            code (str): 종목 코드
            mode (str): annual, quarter 중에 선택. 연단위, 분기단위 조회
            select (list): 반환할 재무제표 항목

        Returns:
            df_finance_state (Dataframe): 재무제표 정보
            total_price (int): 시가총액 (현재 분기 결과)
        """
        if not mode in ['annual','quarter']:
            raise ValueError('annual, quater 중에 하나를 선택해 주세요.')

        ##init
        code = code_name[0]
        name = code_name[1]

        opt = webdriver.ChromeOptions()
        opt.add_argument('headless')
        driver = webdriver.Chrome(options=opt)


        code = str(code).zfill(6)
        ## 네이버 재무재표 주소
        try:
            URL = f"https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={code}"

            driver.get(URL)

            ## 분기 버튼 클릭
            if mode == 'annual':
                radio = driver.find_element('id', 'frqTyp0')
                radio.click()
            else:
                radio = driver.find_element('id', 'frqTyp1')
                radio.click()

            ## 분기/연간 선택 후, 조회 클릭
            butt = driver.find_element('id', 'hfinGubun')
            butt.click()

            profit      = driver.find_element('id', 'val_tab1')  # 수익성
            growth      = driver.find_element('id', 'val_tab2')  # 성장성
            stability   = driver.find_element('id', 'val_tab3')  # 안정성
            activity    = driver.find_element('id', 'val_tab4')  # 활동성

            ids = [profit, growth, stability, activity]
            sectors = ['수익성', '성장성', '안정성', '활동성']
            df_all = []

            for id, sector in zip(ids, sectors):
                id.click()
                time.sleep(0.3)
                html = driver.page_source
                soup = Soup(html, 'html.parser')
                table = soup.select('table')
                table_html = str(table)  ## 테이블 html 정보를 문자열로 변경하기
                table_df_list = pd.read_html(table_html) ## 테이블 정보 읽어 오기
                df = table_df_list[6] # 투자정보 테이블 번호
                df2 = df.iloc[:,:6]
                df2.columns = ['항목', 'month-12', 'month-9', 'month-6', 'month-3', 'month-0']
                # df2['투자지표'] = sector
                df2=df2.replace({'항목': '펼치기  '}, {'항목':''}, regex=True)
                df_all.append(df2)

            df_tot = pd.concat(df_all)
            df_tot = df_tot.set_index(['항목'])
            df_tot = df_tot.loc[select[0]]
            # driver.clos()
        except Exception as e:
            print(f"문제되는 URL: {URL}")
            logger.error(e)

        ## 새로운 df 생성 - score 저장
        new_cols = []
        new_cols.append('code')
        new_cols.append('url')
        for i in select[0]:
            new_cols.append(f"{i}_list")
            new_cols.append(f"{i}_score")
        new_cols.append('total_score')


        ## score 계산
        score_drop_values = []
        cols = df_tot.columns.to_list()
        df_out = pd.DataFrame(columns=new_cols)

        # 기본적으로 삽입되어야 할 내용
        df_out.loc[name, 'code'] = code
        df_out.loc[name, 'url'] = URL
        sc_sum = []

        try:
            for idx in df_tot.index:
                sc = 0
                val_list = []
                if idx in select[0]:  ## df_tot 이 제대로 만들어지지 못하는 것에 대한 대비
                    for cnt, mon in enumerate(cols):
                        val = df_tot.loc[idx, mon]
                        val_list.append(val)
                        if idx == '매출액증가율' or idx == '영업이익증가율':
                            if val >  20 :
                                sc += 3*(cnt+1)
                            elif val > 10 :
                                sc += 2*(cnt+1)
                            elif val  > 0 :
                                sc += 1*(cnt+1)
                            else:  ## minus
                                sc -= 2*(cnt+1)  ## 마이너스 일 경우, 다음 분기에 높은 스코어가 나오기 때문에, 강한 마이너스로 보정해야함
                                pass
                        elif idx == '영업이익률':  ## steady 한 것이 제일 좋음
                            if val > 20 :  ## 너무 높으면 신규 사업자 진입함
                                sc += 3
                            elif val > 10 :
                                sc += 2
                            elif val > 0 :
                                sc += 1
                            else:
                                sc -= 2
                        elif idx == 'ROE':
                            if val > 20 :  ## 너무 높으면 이미 성숙한 사업이라 낮아질 일만 있음
                                sc += 3
                            elif val > 10 :
                                sc += 2
                            elif val > 0 :
                                sc += 1
                            elif val > -10:
                                sc -= 3
                            else:
                                # score_drop_values.append(True)
                                sc -= 3
                            pass
                        elif idx == '부채비율':
                            if val > 250:  ## 부채 비율이 높으면 그냥 제외 시킴
                                # score_drop_values.append(True)
                                sc += 0
                            elif val > 150:
                                sc += 1
                            elif val > 100:
                                sc += 2
                            elif val > 0:
                                sc += 3
                            else:
                                sc -= 3
                            pass
                    sc_sum.append(sc)
                    val_str = ','.join(str(e) for e in val_list)
                    df_out.loc[name, f"{idx}_list"] = val_str
                    df_out.loc[name, f"{idx}_score"] = sc
            total_score = sum(sc_sum)
            df_out.loc[name,'total_score'] = total_score
        except Exception as e:
            print(f"문제되는 URL: {URL}")
            logger.error(e)  ##
            # df_out.loc[name, f"{idx}_list"] = 'nan'
            # df_out.loc[name, f"{idx}_score"] = 0
            total_score = 0
            df_out.loc[name,'total_score'] = total_score

        try:
            URL = f"https://finance.naver.com/item/main.nhn?code={code}"
            r = requests.get(URL)
            df = pd.read_html(r.text)[3]
            df.set_index(df.columns[0], inplace=True)
            df.index.rename('주요재무정보', inplace=True)
            df.columns = df.columns.droplevel(2)
            annual_date = pd.DataFrame(df).xs('최근 연간 실적', axis=1)
            quater_date = pd.DataFrame(df).xs('최근 분기 실적', axis=1)

            ## 시가총액
            temp = pd.read_html(r.text)[4]
            total_price = temp.set_index(['종목명']).loc['시가총액(억)'][0]
            total_price = int(total_price) * 100000000

            ## total_price 별 그룹핑 (그불마다 전략방법이 다를 수 있음)
            score_list = self.score_rule['score_market_value']
            total_price_group, empty_flag = self._make_score([total_price], score_list, mode='last')
            total_price_group = f'GR{total_price_group}'

            if mode == 'annual':
                df2 = annual_date.loc[select[1], :]
            else:
                df2 = quater_date.loc[select[1], :]

            ## 일부내요 수집하기
            df_out['total_price'] = total_price
            df_out['total_price_group'] = total_price_group
            df3 = df2.iloc[:, :5]
            val_list = []
            for cnt, i in enumerate(df3.loc['PER(배)']):
                val_list.append(i)
                if cnt == 4:
                    df_out['PER'] = i

            val_str = ','.join(str(e) for e in val_list)
            df_out['PER_list'] = val_str
            for cnt, i in enumerate(df3.loc['PBR(배)']):
                val_list.append(i)
                if cnt == 4:
                    df_out['PBR'] = i
            val_str = ','.join(str(e) for e in val_list)
            df_out['PBR_list'] = val_str
        except Exception as e :
            print(f"문제되는 URL: {URL}... {e}")
            score_values = [0, 0, 0, 0]
            score_drop = True
            df_out['total_price'] = 0
            df_out['total_price_group'] = 'GR0'
            df_out['PER_list'] = 'nan'
            df_out['PBR_list'] = 'nan'



        '''
        ## score 계산하기
        score_values = []
        # 1) 당기순이익: 3년 흑자 (3점) + 3년 상승 추세 (3점)
        score_profit_add = 0
        score_profit = 0

        i2_prev = 0
        try:
            for i in df2.loc['당기순이익']:
                i = float(i)
                if np.isnan(i):
                    continue
                score_profit += 1 if i > 0 else 0

                if i >= i2_prev:
                    score_profit_add += 1
                else:
                    score_profit_add = 0
                i2_prev = i
        except Exception as e :
            print(f"문제되는 URL: {URL}")
            print(e)

        score_values.append(score_profit+score_profit_add)
        if  score_profit+score_profit_add > 0:
            score_drop_values.append(False)
        else:
            score_drop_values.append(True)

        # 2) 부채비율: 100% 이하 (3), 150% 이하 (2), 200% 이하 (1), 200% 이상 (종목 제거)
        # 3) 당좌비율: 100% 이상 (3), 75% 이상 (2), 50% 이상 (1), 50% 미만 (종목 제거)
        # 4) 사내유보율: 100% 이상이면 (3)
        for i in range(3):
            if i == 0 : # 부채비율
                score_list = self.score_rule["score_debt_ratio"]
                score_data = df2.loc['부채비율'].to_list()
                name = '부채비율'
            elif i == 1: # 당좌 비율
                score_list = self.score_rule["score_quick_ratio"]
                score_data = df2.loc['당좌비율'].to_list()
                name = '당좌비율'
            elif i == 2: ##
                score_list = self.score_rule["score_reserve_ratio"]
                score_data = df2.loc['유보율'].to_list()
                name = '유보율'

            ## 전부 nan 인지 확인
            acc = 0
            for i in score_data:
                i = float(i)
                if np.isnan(i):
                    acc += 1

            try:
                if len(score_data) == acc :
                    raise
                ## 스코어 계산
                score, score_drop = self._make_score(score_data, score_list)
            except:
                _msg = f"종목코드({code}) 의 ({name}) 이 전부 nan 입니다. (URL:{URL})"
                logger.info(_msg)
                score = 0
                score_drop = True # 정보가 부족하면 없애기

            score_values.append(score)
            score_drop_values.append(score_drop)

        # print (score_values, score_drop_values)
        # 하나의 조건도 만족하지 못하면 해당 종목을 후보에서 제외하기 위해 사용
        '''

        score_drop = any(score_drop_values)
        df_out['score_drop'] = score_drop

        c_proc = mp.current_process()
        _msg = f"스코어= {total_score:<6} : 종목이름={name:<20}, 종목코드={code:<10} -- PID: {c_proc.pid}, PROC_NAME: {c_proc.name}"
        logger.info(_msg)

        ### SSHTEST
        if '매출총이익률_list' in df_out.columns.to_list():
            print(df_out)


        ## close
        driver.close()
        driver.quit()

        return df_out


    def load_stock_data(self, code, date, display=False):
        df = fdr.DataReader(code, date[0], date[1])

        file_name = f"stock_data_{code}_{date[0]}_{date[1]}.csv"
        ## 파일로 저장 합니다.
        self._file_save(df, self.file_manager["stock_data"]["path"], file_name)

        if display == True:
            fdr.chart.plot(df)
        pass

        return df.reset_index()


    def back_test(self, code):
        cerebro = bt.Cerebro()

        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        cerebro.run()

        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        pass

    def run(self):
        ## 거래소 모든 코드 가져오기
        '''
            1) 거래소 코드 가져오기
            1-2) Sector 가 없는  코드 날리기 (Sector missing 처리)
            1-3) 파일로 저장하기

            2) 종목별로 당기순이익, 시가총액, 미래 PER, 재무비율 가져오기
            2-2) 스코어 내기
            2-3) 스코어에 따라 순위 매기기
        '''

        ## instance 생성

        params = self.param_init
        files = self.file_manager

        ######################
        ####    STEP1     ####
        ######################
        ## 1) 테마 리스트를 작성하고 테마별 종목코드를 확인합니다. 결과는 파일로 저장합니다.
        if params["ena_step1"] == True:
            df_stocks = self.stock_list()
        else:
            df_stocks = pd.read_csv(files["stock_info"]["path"] + files["stock_info"]["name"], index_col=0)

        def code_zfill(x):
            x_out = str(x).zfill(6)
            return x_out

        df_stocks['Symbol'] = df_stocks['Symbol'].apply(code_zfill)
        ######################
        ####    STEP2     ####
        ######################

        if params["ena_step2"] == True:
            df_stocks_state = []

            cpu_cnt = 8
            pool = Pool(processes=cpu_cnt)

            opt = webdriver.ChromeOptions()
            opt.add_argument('headless')
            drivers = []
            for i in range(cpu_cnt):
                drivers.append(webdriver.Chrome(options=opt))

            codes = df_stocks['Symbol'].to_list()
            names = df_stocks['Name'].to_list()
            codes_names = list(zip(codes, names))

            df_list = pool.map(self.finance_state, codes_names)

            df_stocks_state = pd.concat(df_list)

            ## join
            df_stocks.set_index(keys=['Name'], inplace=True, drop=True)

            df_fin  = df_stocks.join(df_stocks_state)

            ## 재무제표 조건에 만족하지 못하는 종목들은 제거
            df_step2 = df_fin
            # df_step2 = df_fin[df_fin.score_drop == False]

            ## 시간 총액별 그룹을 나눈다. (그룹별 대응 방법이 달라질 수 있음)



            ## 파일 저장
            now = datetime.now().strftime("%Y-%m-%d")
            file_path = self.file_manager["selected_items"]["path"]
            file_name = self.file_manager["selected_items"]["name"]
            if file_name == "":
                file_name = f"stock_items_{now}.csv"
            stu.file_save(df_step2, file_path, file_name, replace=True)
        else: ## disable step2
            file_path = self.file_manager["selected_items"]["path"]
            ## 파일 하나임을 가정 함 (todo: 멀티 파일 지원은 추후 고려)
            file_name = os.listdir('./data/selected_items/')[0]
            df_step2 = pd.read_csv(file_path + file_name, index_col=0)


        ######################
        ####    STEP3     ####
        ######################
        '''
            제무재표 좋은 종목중에 Threshold socre 보다 높은 종목을 선발한다. 
            그중에서 진입 시점을 임박한 종목을 선발한다.
                - cond1: RSI 
                - cond2: 볼린저
                - cond3: 거래량
                - cond4: 공매도 
        '''

        ## 종목별 뉴스 수집정보
        nc = newsCrawler()
        nc.config['trend_display'] = False
        cnt = 0
        duration = self.param_init['duration']
        df = df_step2.sort_values(by='total_score', ascending=False)
        for name, code in zip(df.index, df.Symbol):
            code = str(code).zfill(6)

            ## 기본 주가 정보
            end_dt = datetime.today()
            end = end_dt.strftime(self.param_init["time_format"])
            st_dt = end_dt - timedelta(days=duration)
            st = st_dt.strftime(self.param_init["time_format"])
            df_chart = fdr.DataReader(code, st, end)

            ## advanced 정보 (일자별 PER, ..)
            df_chart2 = stock.get_market_fundamental(st.replace("-",""), end.replace("-",""), code, freq='d')

            ##일자별 시가 총액 (시가총액, 거래량, 거래대금, 상장주식수)
            df_chart3 = stock.get_market_cap(st.replace("-",""), end.replace("-",""), code)

            ##일자별 외국인 보유량 및 외국인 한도소진률
            df_chart4 = stock.get_exhaustion_rates_of_foreign_investment(st.replace("-",""), end.replace("-",""), code)

            ## 공매도 정보
            df_chart5 = stock.get_shorting_balance_by_date(st.replace("-",""), end.replace("-",""), code)

            ## 뉴스 정보 수집
            logger.info(f"종목 ({name} - {code}) 의 뉴스정보 수집을 시작합니다. ")
            df_news = nc.search_keyword(name)

            ## for test
            # if cnt==10:
            #     break
            cnt +=1
        print(df_news)


    ############## internal funct ################
    def _make_score(self, score_data, score_list, mode='last'):
        ''' 종벽별 재무제표 값을 스코어 를 만듭니다.

        :param score_data (list): 재무제표 값 리스트
        :param score_list (list): 스코어 기준 테이블
        :param mode (str): avg, last, 중 하나 선택.
            - avg : 모든 데이터 스코어의 평균을 최종값으로 함
            - last : 마지막 (최근) 스코어를 최종값으로 함
        :return:
            score (float) : 최종 스코어 값
            score_drop (bool): 조건 미달로 해당 종목 제거 필요 알림
        '''
        # 초기 설정
        if not mode in ['avg', 'last']:
            _msg = f"mode={mode} 는 지워하지 않습니다. avg, last, 중에 선택해 주세요."
            logger.error(_msg)
            raise Exception(_msg)

        score_len = len(score_list)
        if score_list[0] > score_list[1]:
            score_reverse = True
        else:
            score_reverse= False
        scores = []
        scores_drop = []

        # 데이터 돌려가면서 체크
        for data in score_data:
            data = float(data)
            for i in range(score_len): #조건문 반복 횟수
                if score_reverse == True: ## 클수록 높은 점수
                    if i == (score_len -1) : # last
                        ## else 가 필요없음. if 들의 범위안에 한번은 포함됨
                        if score_list[i] >= data:
                            scores.append(0)
                            scores_drop.append(True)
                    else:
                        if score_list[i] >= data > score_list[i+1]:
                            tmp = (score_len-1) - i
                            scores.append(tmp)
                            scores_drop.append(False)
                else:  # 작을수록 높은 점수
                    if i == score_len -1 : # last
                        if score_list[i] <= data:
                            scores.append(0)
                            scores_drop.append(True)
                    else:
                        if score_list[i] <= data < score_list[i+1]:
                            tmp = (score_len-1) - i
                            scores.append(tmp)
                            scores_drop.append(False)

        ## summary
        if mode == 'avg':
            score = math.floor(sum(scores) / len(scores))
            score_drop = any(scores_drop)
        else: #last
            try:
                score = scores[-1]
                score_drop = scores_drop[-1]
            except Exception as e :

                logger.error(e)



        return score, score_drop


if __name__ == "__main__":

    ## instance 생성
    config_file = './config/config.yaml'
    st = systemTrade(config_file)
    st.run()