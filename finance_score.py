# pylint: disable=broad-except, W1203
import os
import math
import time
from datetime import datetime
import zipfile
import requests
import platform
from subprocess import check_output

# multi-processing
from multiprocessing import Pool
import multiprocessing as mp

import yaml
import pandas as pd

# html
import requests
from bs4 import BeautifulSoup as Soup
from playwright.sync_api import sync_playwright
import FinanceDataReader as fdr

# chart

# Dart
# import OpenDartReader


# local file
from tools import st_utils as stu

######### Global ############
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

# log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e:
    print(e)

####    로그 생성    #######
logger = stu.create_logger()

# ## Chrome Driver path 를 지정해줌,  TODO: 드라이버 버전 업데이트 해당 위치에 카피 하기
# driver_path = "/usr/local/bin/chromedriver"


class financeScore:
    def __init__(self, config_file):

        # 설정 파일을 필수적으로 한다.
        with open(config_file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # pylint: disable=logging-fstring-interpolation
        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

        # global 변수 선언
        self.file_manager = config["data_management"]
        self.param_init = config["mainInit"]
        self.score_rule = config["scoreRule"]
        # self.keys = config["keyList"]


    def finance_state(self, code_name, mode='quarter',):
        """종목 코드에 대한 재무제표를 가져옵니다.
        
        playwright를 사용하여 데이터를 크롤링합니다.

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
        select = [['매출액증가율', '영업이익증가율', '영업이익률', 'ROE', '부채비율'],
                  ['당기순이익', 'PER(배)', '부채비율', '당좌비율', '유보율', 'PBR(배)']]

        if mode not in ['annual', 'quarter']:
            raise ValueError('annual, quater 중에 하나를 선택해 주세요.')

        # init
        code = code_name[0]
        name = code_name[1]
        
        df_tot = pd.DataFrame()
        URL = f"https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={str(code).zfill(6)}"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(URL)

                # "올바른 종목이 아닙니다" 알림 처리
                # Playwright는 알림을 자동으로 처리하지 않지만, 페이지 로드 실패 등으로 감지 가능
                # if "c1040001" not in page.url: # 성공적인 페이지 로드의 URL 일부
                #     logger.warning(f"'{name}({code})' 페이지 로드에 실패했거나 유효하지 않은 종목일 수 있습니다. 건너뜁니다.")
                #     browser.close()
                #     return pd.DataFrame()

                # 분기/연간 버튼 클릭 및 데이터 파싱
                if mode == 'annual':
                    page.locator('#frqTyp0').click()
                else:
                    page.locator('#frqTyp1').click()

                page.locator('#hfinGubun').click()

                ids = ['#val_tab1', '#val_tab2', '#val_tab3', '#val_tab4'] # profit, growth, stability, activity
                df_all = []

                for _id in ids:
                    page.locator(_id).click()
                    time.sleep(0.3)
                    html = page.content()
                    soup = Soup(html, 'html.parser')
                    table = soup.select('table')
                    table_html = str(table)
                    table_df_list = pd.read_html(table_html)
                    df = table_df_list[6]
                    df2 = df.iloc[:, :6]
                    df2.columns = ['항목', 'month-12', 'month-9', 'month-6', 'month-3', 'month-0']
                    df2 = df2.replace({'항목': '펼치기  '}, {'항목': ''}, regex=True)
                    df_all.append(df2)
                
                browser.close()

            df_tot = pd.concat(df_all)
            df_tot = df_tot.set_index(['항목'])
            df_tot = df_tot.loc[select[0]]

        except Exception as error:
            print(f"Playwright 처리 중 문제 발생 (URL: {URL})")
            logger.error(error)
            df_tot = pd.DataFrame()


        # 새로운 df 생성 - score 저장
        new_cols = []
        new_cols.append('code')
        new_cols.append('url')
        for i in select[0]:
            new_cols.append(f"{i}")
            new_cols.append(f"{i}_list")
            new_cols.append(f"{i}_score")
        new_cols.append('total_score')

        # score 계산
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
                if idx in select[0]:  # df_tot 이 제대로 만들어지지 못하는 것에 대한 대비
                    for cnt, mon in enumerate(cols):
                        val = df_tot.loc[idx, mon]
                        val_list.append(val)
                        if idx == '매출액증가율' or idx == '영업이익증가율':
                            if val > 20:
                                sc += 3*(cnt+1)
                            elif val > 10:
                                sc += 2*(cnt+1)
                            elif val > 0:
                                sc += 1*(cnt+1)
                            else:  # minus
                                # 마이너스 일 경우, 다음 분기에 높은 스코어가 나오기 때문에, 강한 마이너스로 보정해야함
                                sc -= 2*(cnt+1)
                        elif idx == '영업이익률':  # steady 한 것이 제일 좋음
                            if val > 20:  # 너무 높으면 신규 사업자 진입함
                                sc += 3
                            elif val > 10:
                                sc += 2
                            elif val > 0:
                                sc += 1
                            else:
                                sc -= 2
                        elif idx == 'ROE':
                            if val > 20:  # 너무 높으면 이미 성숙한 사업이라 낮아질 일만 있음
                                sc += 3
                            elif val > 10:
                                sc += 2
                            elif val > 0:
                                sc += 1
                            elif val > -10:
                                sc -= 3
                            else:
                                # score_drop_values.append(True)
                                sc -= 3
                        elif idx == '부채비율':
                            if val > 250:  # 부채 비율이 높으면 그냥 제외 시킴
                                # score_drop_values.append(True)
                                sc -= 2*(cnt+1)  # 부채 가중치 높임
                            elif val > 150:
                                sc += 1
                            elif val > 100:
                                sc += 2
                            elif val > 0:
                                sc += 3
                            else:
                                sc -= 3  # 해당 경우가 존재하나??
                    sc_sum.append(sc)
                    val_str = ','.join(str(e) for e in val_list)
                    df_out.loc[name, f"{idx}"] = val_list[-1] if val_list else 'N/A'
                    df_out.loc[name, f"{idx}_list"] = val_str
                    df_out.loc[name, f"{idx}_score"] = sc
            total_score = sum(sc_sum)
            df_out.loc[name, 'total_score'] = total_score
        except Exception as error:
            print(f"문제되는 URL: {URL}")
            logger.error(error)
            # df_out.loc[name, f"{idx}_list"] = 'nan'
            # df_out.loc[name, f"{idx}_score"] = 0
            total_score = 0
            df_out.loc[name, 'total_score'] = total_score

        try:
            URL = f"https://finance.naver.com/item/main.nhn?code={code}"
            try:
                r = requests.get(URL)
            except:
                raise ValueError(f"{URL} 를 load 하는 과정에서 에러가 발생하였습니다. ")
            df = pd.read_html(r.text)[3]
            df.set_index(df.columns[0], inplace=True)
            df.index.rename('주요재무정보', inplace=True)
            df.columns = df.columns.droplevel(2)
            annual_date = pd.DataFrame(df).xs('최근 연간 실적', axis=1)
            quater_date = pd.DataFrame(df).xs('최근 분기 실적', axis=1)

            # 시가총액
            temp = pd.read_html(r.text)[4]
            total_price = temp.set_index(['종목명']).loc['시가총액(억)'][0]
            total_price = int(total_price) * 100000000

            # total_price 별 그룹핑 (그불마다 전략방법이 다를 수 있음)
            score_list = self.score_rule['score_market_value']
            # pylint: disable=W0612
            total_price_group, empty_flag = self._make_score(
                [total_price], score_list, mode='last')
            total_price_group = f'GR{total_price_group}'

            if mode == 'annual':
                df2 = annual_date.loc[select[1], :]
            else:
                df2 = quater_date.loc[select[1], :]

            # 일부내요 수집하기
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
        except Exception as error:
            print(f"문제되는 URL: {URL}... {error}")
            score_drop = True
            df_out['total_price'] = 0
            df_out['total_price_group'] = 'GR0'
            df_out['PER_list'] = 'nan'
            df_out['PBR_list'] = 'nan'

        score_drop = any(score_drop_values)
        df_out['score_drop'] = score_drop

        c_proc = mp.current_process()
        _msg = f"스코어= {total_score:<6} : 종목이름={name:<20}, 종목코드={code:<10} -- PID: {c_proc.pid}, PROC_NAME: {c_proc.name}"
        logger.info(_msg)

        # 수집 데이터 확인용 로그
        try:
            log_roe = df_out.loc[name, 'ROE']
            log_debt = df_out.loc[name, '부채비율']
            log_per = df_out.loc[name, 'PER']
            log_pbr = df_out.loc[name, 'PBR']
            logger.info(f"  >> 수집 데이터: ROE={log_roe}, 부채비율={log_debt}, PER={log_per}, PBR={log_pbr}")
        except KeyError:
            logger.info(f"  >> 종목 '{name}'의 일부 데이터는 수집되지 않았습니다.")

        # SSHTEST
        if '매출총이익률_list' in df_out.columns.to_list():
            print(df_out)

        return df_out

    def _install_playwright_browsers(self):
        """Playwright에 필요한 브라우저가 설치되어 있는지 확인하고, 없으면 설치합니다."""
        try:
            # 간단한 Playwright 작업을 시도하여 브라우저 존재 여부 확인
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            logger.info("Playwright 브라우저가 이미 설치되어 있습니다.")
        except Exception:
            logger.info("Playwright 브라우저를 찾을 수 없습니다. 설치를 시작합니다...")
            try:
                os.system("playwright install")
                logger.info("Playwright 브라우저 설치가 완료되었습니다.")
            except Exception as e:
                logger.error(f"Playwright 브라우저 설치에 실패했습니다: {e}")
                raise

    def run(self):
        # 거래소 모든 코드 가져오기
        '''
            1) 거래소 코드 가져오기
            1-2) Sector 가 없는  코드 날리기 (Sector missing 처리)
            1-3) 파일로 저장하기

            2) 종목별로 당기순이익, 시가총액, 미래 PER, 재무비율 가져오기
            2-2) 스코어 내기
            2-3) 스코어에 따라 순위 매기기
        '''

        # instance 생성

        params = self.param_init
        files = self.file_manager
        finance_scores_path = os.path.join(files["base_path"], files["discovery"]["finance_scores"]["path"])

        ######################
        ####    STEP1     ####
        ######################
        # 1) 테마 리스트를 작성하고 테마별 종목코드를 확인합니다. 결과는 파일로 저장합니다.
        # krx = fdr.StockListing('KRX-DESC')
        # 가끔 한국거래소에 서버점검 등의 이슈가 있어 fdr.StockListing 으로 상장종목을 받아오지 못할 때가 있습니다.
        # 그럴 때는 아래의 주석을 해제하고 실습해 주세요!
        krx = pd.read_csv("https://raw.githubusercontent.com/corazzon/finance-data-analysis/main/krx.csv")
        # 섹터값없는 코드 삭제 (ETF...)
        df_stocks = krx.dropna(axis=0, subset=['Sector'])
        df_stocks.drop(['Representative','Region',], axis=1, inplace=True)
        df_stocks.reset_index(drop=True, inplace=True)

        # 1-2) FinanceDataReader (외부 패키지) 가 정상적으로 동작하지 않을 경우, 이전 작업 내용을 불러오기
        if len(df_stocks) < 100:
            logger.warning(
                "FinanceDataReader 가 정상적으로 동작하지 않아 이전 작업 파일을 불러 옵니다.")
            # CSV 파일만 필터링
            csv_files = [f for f in os.listdir(
                finance_scores_path) if f.endswith('.csv')]

            # 날짜 형식에 맞게 파일 이름 파싱 및 최신 파일 찾기
            date_format = '%Y-%m-%d'
            latest_date = datetime.strptime(
                '2020-01-01', date_format)  # 초기 날짜 설정
            latest_file = ''

            for file_name in csv_files:
                # 파일명에서 확장자 제외
                file_date_str = file_name.split('.')[0].split('_')[-1]

                try:
                    file_date = datetime.strptime(file_date_str, date_format)
                except ValueError:
                    # 파일 이름이 날짜 형식이 아닌 경우 건너뜀
                    continue

                if file_date > latest_date:
                    latest_date = file_date
                    latest_file = file_name

            # 최신 파일을 DataFrame으로 읽어오기
            if latest_file:
                latest_file_path = os.path.join(
                    finance_scores_path, latest_file)
                df_stocks = pd.read_csv(latest_file_path)
                # origin column 만 남기기
                filtered_columns = [
                    col for col in df_stocks.columns if col in krx.columns]
                df_stocks = df_stocks[filtered_columns]
            else:
                print("날짜 형식의 CSV 파일이 없습니다.")

        def code_zfill(x):
            x_out = str(x).zfill(6)
            return x_out

        df_stocks['Symbol'] = df_stocks['Symbol'].apply(code_zfill)
        ######################
        ####    STEP2     ####
        ######################

        if params["ena_step2"] is True:
            # Playwright 브라우저 설치 확인 및 자동 설치
            self._install_playwright_browsers()

            cpu_cnt = 8
            pool = Pool(processes=cpu_cnt)

            codes = df_stocks['Symbol'].to_list()
            names = df_stocks['Name'].to_list()
            codes_names = list(zip(codes, names))

            df_list = pool.map(self.finance_state, codes_names)  ## 병렬 처리 하기 

            df_stocks_state = pd.concat(df_list)  ## 병렬 처리 결과를 하나로 합치기

            # join
            df_stocks.set_index(keys=['Name'], inplace=True, drop=True)

            df_fin = df_stocks.join(df_stocks_state)

            # 재무제표 조건에 만족하지 못하는 종목들은 제거
            df_step2 = df_fin.sort_values(by='total_score', ascending=False)

            ## code , Symbol 이라는 동일 column 을 혼재하여 사용 하고 있으므로, Symbol 만 남김. 23.5.4
            if 'Symbol' in df_step2.columns:
                # 'code' 열이 있는 경우 제거
                if 'code' in df_step2.columns:
                    df_step2.drop(columns=['code'], inplace=True)
            else:
                # 'Symbol' 열이 없는 경우 'code' 열의 이름을 'Symbol'로 변경
                df_step2.rename(columns={'code': 'Symbol'}, inplace=True)
            # df_step2 = df_fin[df_fin.score_drop == False]

            # 시간 총액별 그룹을 나눈다. (그룹별 대응 방법이 달라질 수 있음)

            # 파일 저장
            now = datetime.now().strftime("%Y-%m-%d")
            file_path = finance_scores_path
            file_name_format = self.file_manager["discovery"]["finance_scores"]["filename_format"]
            file_name = file_name_format.replace("{date}", now)
            stu.file_save(df_step2, file_path, file_name, replace=False)
        else:  # disable step2
            file_path = finance_scores_path
            # 파일 하나임을 가정 함 (todo: 멀티 파일 지원은 추후 고려)
            csv_files = sorted([f for f in os.listdir(file_path) if f.endswith('.csv')])
            if csv_files:
                latest_file = csv_files[-1]
                df_step2 = pd.read_csv(os.path.join(file_path, latest_file), index_col=0)
            else:
                logger.error(f"{file_path}에 finance_score 파일이 없습니다.")
                return

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
        if mode not in ['avg', 'last']:
            _msg = f"mode={mode} 는 지워하지 않습니다. avg, last, 중에 선택해 주세요."
            logger.error(_msg)
            raise ValueError(_msg)

        score_len = len(score_list)
        if score_list[0] > score_list[1]:
            score_reverse = True
        else:
            score_reverse = False
        scores = []
        scores_drop = []

        # 데이터 돌려가면서 체크
        for data in score_data:
            data = float(data)
            for i in range(score_len):  # 조건문 반복 횟수
                if score_reverse is True:  # 클수록 높은 점수
                    if i == (score_len - 1):  # last
                        # else 가 필요없음. if 들의 범위안에 한번은 포함됨
                        if score_list[i] >= data:
                            scores.append(0)
                            scores_drop.append(True)
                    else:
                        if score_list[i] >= data > score_list[i+1]:
                            tmp = (score_len-1) - i
                            scores.append(tmp)
                            scores_drop.append(False)
                else:  # 작을수록 높은 점수
                    if i == score_len - 1:  # last
                        if score_list[i] <= data:
                            scores.append(0)
                            scores_drop.append(True)
                    else:
                        if score_list[i] <= data < score_list[i+1]:
                            tmp = (score_len-1) - i
                            scores.append(tmp)
                            scores_drop.append(False)

        # summary
        if mode == 'avg':
            score = math.floor(sum(scores) / len(scores))
            score_drop = any(scores_drop)
        else:  # last
            try:
                score = scores[-1]
                score_drop = scores_drop[-1]
            except Exception as e:
                logger.error(f"score_data={score}, 를 저장하는 과정에서 에러가 발생함")

        return score, score_drop


if __name__ == "__main__":

    # instance 생성
    config_file_path = './config/config.yaml'
    fs = financeScore(config_file_path)
    fs.run()
