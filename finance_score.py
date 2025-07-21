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
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
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

        self.driver_path = None

    def finance_state(self, code_name, mode='quarter',):
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
        select = [['매출액증가율', '영업이익증가율', '영업이익률', 'ROE', '부채비율'],
                  ['당기순이익', 'PER(배)', '부채비율', '당좌비율', '유보율', 'PBR(배)']]

        if mode not in ['annual', 'quarter']:
            raise ValueError('annual, quater 중에 하나를 선택해 주세요.')

        # init
        code = code_name[0]
        name = code_name[1]

        opt = webdriver.ChromeOptions()
        opt.add_argument('headless')
        driver = webdriver.Chrome(service=Service(self.driver_path), options=opt)

        code = str(code).zfill(6)
        # 네이버 재무재표 주소
        try:
            URL = f"https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={code}"

            driver.get(URL)

            # 분기 버튼 클릭
            if mode == 'annual':
                radio = driver.find_element('id', 'frqTyp0')
                radio.click()
            else:
                radio = driver.find_element('id', 'frqTyp1')
                radio.click()

            # 분기/연간 선택 후, 조회 클릭
            butt = driver.find_element('id', 'hfinGubun')
            butt.click()

            profit = driver.find_element('id', 'val_tab1')  # 수익성
            growth = driver.find_element('id', 'val_tab2')  # 성장성
            stability = driver.find_element('id', 'val_tab3')  # 안정성
            activity = driver.find_element('id', 'val_tab4')  # 활동성

            ids = [profit, growth, stability, activity]
            sectors = ['수익성', '성장성', '안정성', '활동성']
            df_all = []
            # pyilnt: disable=W0622, W0612
            for _id, _sector in zip(ids, sectors):
                _id.click()
                time.sleep(0.3)
                html = driver.page_source
                soup = Soup(html, 'html.parser')
                table = soup.select('table')
                table_html = str(table)  # 테이블 html 정보를 문자열로 변경하기
                table_df_list = pd.read_html(table_html)  # 테이블 정보 읽어 오기
                df = table_df_list[6]  # 투자정보 테이블 번호
                df2 = df.iloc[:, :6]
                df2.columns = ['항목', 'month-12', 'month-9',
                               'month-6', 'month-3', 'month-0']
                # df2['투자지표'] = sector
                df2 = df2.replace({'항목': '펼치기  '}, {'항목': ''}, regex=True)
                df_all.append(df2)

            df_tot = pd.concat(df_all)
            df_tot = df_tot.set_index(['항목'])
            df_tot = df_tot.loc[select[0]]
            # driver.clos()
        except Exception as error:
            print(f"문제되는 URL: {URL}")
            logger.error(error)
            df_tot = pd.DataFrame()

        # 새로운 df 생성 - score 저장
        new_cols = []
        new_cols.append('code')
        new_cols.append('url')
        for i in select[0]:
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

        # SSHTEST
        if '매출총이익률_list' in df_out.columns.to_list():
            print(df_out)

        # close
        driver.close()
        driver.quit()

        return df_out

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
                files["finance_score"]["path"]) if f.endswith('.csv')]

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
                    files["finance_score"]["path"], latest_file)
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
            df_stocks_state = []

            cpu_cnt = 8
            pool = Pool(processes=cpu_cnt)

            opt = webdriver.ChromeOptions()
            opt.add_argument('headless')
            drivers = []

            ## Naver 크롤링 하기 위해 chrome 최신 드라이버 설치 
            chrome_version = self.get_chrome_version()
            self.driver_path = self.download_chromedriver(chrome_version)
            print(f"ChromeDriver for Chrome version {chrome_version} has been downloaded and installed.")

            # pylint: disable= W0612
            for i in range(cpu_cnt):
                drivers.append(webdriver.Chrome(
                    service=Service(self.driver_path), options=opt))

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
            file_path = self.file_manager["finance_score"]["path"]
            file_name = self.file_manager["finance_score"]["name"]
            if file_name == "":
                file_name = f"stock_items_{now}.csv"
            stu.file_save(df_step2, file_path, file_name, replace=False)
        else:  # disable step2
            file_path = self.file_manager["finance_score"]["path"]
            # 파일 하나임을 가정 함 (todo: 멀티 파일 지원은 추후 고려)
            file_name = os.listdir('./data/finance_score/')[0]
            df_step2 = pd.read_csv(file_path + file_name, index_col=0)

    ############## internal funct ################
    def get_chrome_version(self):
        """ chrome 버전을 확인하는 코드. Naver 크롤링 시, chrome dirver 를 사용하기 때문
        """
        # macOS의 경우
        if os.name == 'posix':
            process = os.popen('/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version')
            version = process.read().strip().replace('Google Chrome ', '')
            process.close()
        # Windows의 경우
        elif os.name == 'nt':
            program_files_path = os.getenv('PROGRAMFILES')
            if os.path.isdir(os.path.join(program_files_path, 'Google/Chrome/Application')):
                path = os.path.join(program_files_path, 'Google/Chrome/Application/chrome.exe')
            else:
                path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
            version = check_output([path, '--version']).decode().strip().replace('Google Chrome ', '')
        # Linux의 경우
        else:
            version = os.popen('google-chrome --version').read().strip().replace('Google Chrome ', '')

        print(f"chrome version (current): {version}")
        return version

    def download_chromedriver(self,  version):
        """ chrome 과 동일한 버전의 chromedriver 를 설치. Naver 크롤링에 사용됨 
        """
        
        # macOS ARM 아키텍처 확인
        if os.name == 'posix' and platform.machine() == 'arm64':
            url = f"https://storage.googleapis.com/chrome-for-testing-public/{version}/mac-arm64/chromedriver-mac-arm64.zip"
        # 기존 macOS (Intel 아키텍처)
        elif os.name == 'posix':
            url =f"https://storage.googleapis.com/chrome-for-testing-public/{version}/mac-x64/chromedriver-mac-x64.zip"
        # Windows
        elif os.name == 'nt':
            url = f"https://storage.googleapis.com/chrome-for-testing-public/{version}/linux64/chromedriver-linux64.zip"
        # Linux
        else:
            url = f"https://storage.googleapis.com/chrome-for-testing-public/{version}/win64/chromedriver-win64.zip"
        
        # ChromeDriver 다운로드 및 압축 해제
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            print(e)
        zip_file_path = "chromedriver.zip"
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall()  # 현재 디렉토리에 압축 해제
            extracted_files = zip_ref.namelist()  # 압축 해제한 파일 목록 가져오기
        
        for extracted_file in extracted_files:
            # 파일 또는 디렉토리의 전체 경로 구성
            extracted_path = os.path.join(os.getcwd(), extracted_file)
            # 권한 변경
            os.chmod(extracted_path, 0o755)

        os.remove(zip_file_path)  # 다운로드한 zip 파일 삭제

        ## driver 를 PATH 에 추가 
        ########################
        # 현재 작업 디렉토리 경로 구하기
        current_working_directory = os.getcwd()
    
        # chromedriver 경로 설정 (현재 작업 디렉토리 내의 chromedriver-mac-arm64)
        chromedriver_path = os.path.join(current_working_directory, 'chromedriver-mac-arm64/chromedriver')
    
        # PATH 환경 변수에 chromedriver 경로 추가
        if chromedriver_path not in os.environ['PATH']:
            os.environ['PATH'] += os.pathsep + chromedriver_path
            print("Chromedriver path added to PATH.")
        else:
            print("Chromedriver path is already in PATH.")
    
        # 현재 PATH 환경 변수 출력 (확인용)
        print(f"Current PATH: {os.environ['PATH']}")
        return chromedriver_path


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
