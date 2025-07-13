# pylint: disable=broad-except, W1203
import os
import math
import time
from datetime import datetime
import zipfile
import requests
import platform
from subprocess import check_output
import ssl
import urllib.request
import json

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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import FinanceDataReader as fdr

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

class financeScore:
    def __init__(self, config_file):
        # SSL 인증서 문제 해결을 위한 설정 (macOS 대응)
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            logger.info("SSL 인증서 검증 우회 설정 완료")
        except Exception as e:
            logger.warning(f"SSL 설정 실패: {e}")

        # 설정 파일을 필수적으로 한다.
        with open(config_file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # pylint: disable=logging-fstring-interpolation
        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

        # global 변수 선언
        self.file_manager = config["data_management"]
        self.param_init = config["mainInit"]
        self.score_rule = config["scoreRule"]

        # ChromeDriver 초기화
        self.setup_chromedriver()

    def setup_chromedriver(self):
        """ChromeDriver 설정 및 자동 업데이트"""
        try:
            # Chrome 버전 확인
            chrome_version = self.get_chrome_version()
            logger.info(f"현재 Chrome 버전: {chrome_version}")

            # ChromeDriver 다운로드 및 설정
            self.driver_path = self.download_chromedriver(chrome_version)
            logger.info(f"ChromeDriver 설정 완료: {self.driver_path}")
        except Exception as e:
            logger.error(f"ChromeDriver 설정 실패: {e}")
            raise

    def get_chrome_version(self):
        """현재 설치된 Chrome 브라우저의 버전을 확인"""
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                output = check_output(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"])
            elif system == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon")
                version = winreg.QueryValueEx(key, "version")[0]
                return version
            else:  # Linux
                output = check_output(["google-chrome", "--version"])
            
            version = output.decode("utf-8").strip().split()[-1]
            return version
        except Exception as e:
            logger.error(f"Chrome 버전 확인 실패: {e}")
            raise

    def download_chromedriver(self, chrome_version):
        """ChromeDriver 다운로드 및 설치"""
        try:
            # 버전에서 메이저 버전만 추출 (예: 138.0.7204.101 -> 138)
            major_version = chrome_version.split('.')[0]
            
            # ChromeDriver 다운로드 URL 설정
            system = platform.system().lower()
            arch = platform.machine().lower()
            
            if system == "darwin":  # macOS
                if "arm" in arch or "aarch64" in arch:
                    platform_name = "mac-arm64"
                else:
                    platform_name = "mac-x64"
            elif system == "windows":
                platform_name = "win32"
            else:  # Linux
                if "aarch64" in arch:
                    platform_name = "linux-arm64"
                else:
                    platform_name = "linux64"

            # ChromeDriver 저장 경로 설정
            driver_dir = os.path.join(os.getcwd(), "chromedriver")
            if not os.path.exists(driver_dir):
                os.makedirs(driver_dir)

            driver_path = os.path.join(driver_dir, f"chromedriver-{platform_name}")
            if system == "windows":
                driver_path += ".exe"

            # 최신 ChromeDriver 다운로드
            download_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{chrome_version}/{platform_name}/chromedriver-{platform_name}.zip"
            
            logger.info(f"ChromeDriver 다운로드 시작: {download_url}")
            
            response = requests.get(download_url)
            if response.status_code == 200:
                zip_path = os.path.join(driver_dir, "chromedriver.zip")
                with open(zip_path, "wb") as f:
                    f.write(response.content)

                # 압축 해제
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(driver_dir)

                # 압축 파일 삭제
                os.remove(zip_path)

                # 실행 권한 부여
                if system != "windows":
                    os.chmod(driver_path, 0o755)

                logger.info(f"ChromeDriver 다운로드 및 설치 완료: {driver_path}")
                return driver_path
            else:
                raise Exception(f"ChromeDriver 다운로드 실패: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"ChromeDriver 다운로드 실패: {e}")
            raise

    def finance_state(self, code_name, mode='quarter'):
        """종목 코드에 대한 재무제표를 가져옵니다."""
        select = [['매출액증가율', '영업이익증가율', '영업이익률', 'ROE', '부채비율'],
                  ['당기순이익', 'PER(배)', '부채비율', '당좌비율', '유보율', 'PBR(배)']]

        if mode not in ['annual', 'quarter']:
            raise ValueError('annual, quater 중에 하나를 선택해 주세요.')

        # init
        code = code_name[0]
        name = code_name[1]

        opt = Options()
        opt.add_argument('--headless')
        opt.add_argument('--no-sandbox')
        opt.add_argument('--disable-dev-shm-usage')
        opt.add_argument('--disable-gpu')
        opt.add_argument('--window-size=1920,1080')
        
        try:
            service = Service(executable_path=self.driver_path)
            driver = webdriver.Chrome(service=service, options=opt)
            wait = WebDriverWait(driver, 10)

            code = str(code).zfill(6)
            URL = f"https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={code}"
            driver.get(URL)

            # 분기/연간 선택
            radio_id = 'frqTyp0' if mode == 'annual' else 'frqTyp1'
            radio = wait.until(EC.element_to_be_clickable((By.ID, radio_id)))
            radio.click()

            # 조회 버튼 클릭
            butt = wait.until(EC.element_to_be_clickable((By.ID, 'hfinGubun')))
            butt.click()

            # 각 탭 데이터 수집
            df_all = []
            for tab_id, sector in [('val_tab1', '수익성'), ('val_tab2', '성장성'), 
                                 ('val_tab3', '안정성'), ('val_tab4', '활동성')]:
                tab = wait.until(EC.element_to_be_clickable((By.ID, tab_id)))
                tab.click()
                time.sleep(0.3)

                html = driver.page_source
                soup = Soup(html, 'html.parser')
                table = soup.select('table')
                table_df_list = pd.read_html(str(table))
                df = table_df_list[6]
                df2 = df.iloc[:, :6]
                df2.columns = ['항목', 'month-12', 'month-9', 'month-6', 'month-3', 'month-0']
                df2 = df2.replace({'항목': '펼치기  '}, {'항목': ''}, regex=True)
                df_all.append(df2)

            df_tot = pd.concat(df_all)
            df_tot = df_tot.set_index(['항목'])
            df_tot = df_tot.loc[select[0]]
            driver.quit()

        except Exception as error:
            logger.error(f"재무제표 데이터 수집 실패 - {URL}: {error}")
            df_tot = pd.DataFrame()
            if 'driver' in locals():
                driver.quit()

        # 새로운 df 생성 - score 저장
        new_cols = ['code', 'url']
        new_cols.extend([f"{i}_list" for i in select[0]])
        new_cols.extend([f"{i}_score" for i in select[0]])
        new_cols.append('total_score')

        df_out = pd.DataFrame(columns=new_cols)
        df_out.loc[name, 'code'] = code
        df_out.loc[name, 'url'] = URL

        # 점수 계산
        try:
            sc_sum = []
            cols = df_tot.columns.to_list()

            for idx in df_tot.index:
                if idx in select[0]:
                    sc = 0
                    val_list = []
                    
                    for cnt, mon in enumerate(cols):
                        val = df_tot.loc[idx, mon]
                        val_list.append(val)
                        
                        # 점수 계산 로직
                        if idx in ['매출액증가율', '영업이익증가율']:
                            sc += self._calculate_growth_score(val, cnt)
                        elif idx == '영업이익률':
                            sc += self._calculate_profit_margin_score(val)
                        elif idx == 'ROE':
                            sc += self._calculate_roe_score(val)
                        elif idx == '부채비율':
                            sc += self._calculate_debt_ratio_score(val, cnt)
                    
                    sc_sum.append(sc)
                    val_str = ','.join(str(e) for e in val_list)
                    df_out.loc[name, f"{idx}_list"] = val_str
                    df_out.loc[name, f"{idx}_score"] = sc
            
            df_out.loc[name, 'total_score'] = sum(sc_sum)

        except Exception as error:
            logger.error(f"점수 계산 실패 - {code}: {error}")
            df_out.loc[name, 'total_score'] = 0

        return df_out

    def _calculate_growth_score(self, val, cnt):
        """성장률 관련 점수 계산"""
        if val > 20:
            return 3 * (cnt + 1)
        elif val > 10:
            return 2 * (cnt + 1)
        elif val > 0:
            return 1 * (cnt + 1)
        return -2 * (cnt + 1)

    def _calculate_profit_margin_score(self, val):
        """영업이익률 점수 계산"""
        if val > 20:
            return 3
        elif val > 10:
            return 2
        elif val > 0:
            return 1
        return -2

    def _calculate_roe_score(self, val):
        """ROE 점수 계산"""
        if val > 20:
            return 3
        elif val > 10:
            return 2
        elif val > 0:
            return 1
        elif val > -10:
            return -3
        return -3

    def _calculate_debt_ratio_score(self, val, cnt):
        """부채비율 점수 계산"""
        if val > 250:
            return -2 * (cnt + 1)
        elif val > 150:
            return 1
        elif val > 100:
            return 2
        elif val > 0:
            return 3
        return -3

    def run(self):
        """재무제표 기반 종목 스크리닝 실행"""
        logger.info("재무제표 기반 종목 스크리닝 시작")
        
        try:
            # 설정 파일에서 새로운 데이터 관리 구조 로드
            with open('./config/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                data_config = config.get('data_management', {})
                
            # 저장 경로 설정
            storage_config = data_config.get('storage', {}).get('financial_statements', {})
            save_path = storage_config.get('path', './data/raw/financial_statements/')
            
            # 저장 경로 생성
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"저장 경로 확인: {save_path}")
            
            # KRX 데이터 다운로드 (캐시 파일 경로 설정)
            cache_file = os.path.join(save_path, "krx_cache.csv")
            krx = self.safe_download_csv("https://raw.githubusercontent.com/corazzon/finance-data-analysis/main/krx.csv", cache_file)
            
            # 캐시 파일 저장 (성공한 경우)
            if len(krx) > 100:
                try:
                    krx.to_csv(cache_file, index=False)
                    logger.info(f"KRX 데이터 캐시 저장: {cache_file}")
                except Exception as e:
                    logger.warning(f"캐시 저장 실패: {e}")
            
            # 섹터값없는 코드 삭제 (ETF...)
            df_stocks = krx.dropna(axis=0, subset=['Sector'])
            df_stocks.drop(['Representative','Region',], axis=1, inplace=True)
            df_stocks.reset_index(drop=True, inplace=True)
            
            # 데이터 검증
            if len(df_stocks) < 100:
                raise ValueError("유효한 종목 데이터가 너무 적습니다.")
                
            # 재무제표 분석 실행
            df_stocks_state = []
            cpu_cnt = min(8, mp.cpu_count())  # CPU 코어 수에 따라 조정
            
            with Pool(processes=cpu_cnt) as pool:
                codes = df_stocks['Symbol'].to_list()
                names = df_stocks['Name'].to_list()
                codes_names = list(zip(codes, names))
                
                # 진행 상황 로깅
                total_items = len(codes_names)
                logger.info(f"총 {total_items}개 종목 분석 시작 (CPU: {cpu_cnt}개 사용)")
                
                df_list = pool.map(self.finance_state, codes_names)
                
            # 결과 병합
            df_stocks_state = pd.concat(df_list)
            df_stocks.set_index(keys=['Name'], inplace=True, drop=True)
            df_fin = df_stocks.join(df_stocks_state)
            
            # 컬럼 정리
            if 'Symbol' in df_fin.columns:
                if 'code' in df_fin.columns:
                    df_fin.drop(columns=['code'], inplace=True)
            else:
                df_fin.rename(columns={'code': 'Symbol'}, inplace=True)
                
            # 최종 결과 정렬
            df_result = df_fin.sort_values(by='total_score', ascending=False)
            
            # 파일 저장
            try:
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"finance_score_{now}.csv"
                result_filepath = os.path.join(save_path, result_filename)
                
                df_result.to_csv(result_filepath, encoding='utf-8-sig')
                logger.info(f"재무제표 분석 결과 저장 완료: {result_filepath}")
                logger.info(f"총 {len(df_result)}개 종목 분석됨")
                
                # 분석 요약 저장
                summary = {
                    'timestamp': now,
                    'total_stocks': len(df_result),
                    'high_score_stocks': len(df_result[df_result['total_score'] >= 80]),
                    'avg_score': df_result['total_score'].mean(),
                    'top_sectors': df_result['Sector'].value_counts().head(5).to_dict()
                }
                
                summary_filename = f"finance_score_summary_{now}.json"
                summary_filepath = os.path.join(save_path, summary_filename)
                
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                
                logger.info("분석 요약 저장 완료")
                return df_result
                
            except Exception as e:
                logger.error(f"결과 저장 중 오류 발생: {e}")
                logger.exception("상세 오류:")
                raise
                
        except Exception as e:
            logger.error("재무제표 분석 중 오류 발생")
            logger.exception("상세 오류:")
            raise

    def safe_download_csv(self, url: str, cache_file: str = None) -> pd.DataFrame:
        """SSL 인증서 문제를 우회하여 안전하게 CSV 다운로드"""
        try:
            # 1차: 정상적인 방법으로 시도
            logger.info(f"CSV 다운로드 시도: {url}")
            return pd.read_csv(url)
        except Exception as e:
            logger.warning(f"정상 다운로드 실패: {e}")
            
            try:
                # 2차: SSL 검증 우회하여 시도
                logger.info("SSL 검증 우회하여 다운로드 시도...")
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                with urllib.request.urlopen(url, context=ssl_context) as response:
                    return pd.read_csv(response)
            except Exception as e2:
                logger.warning(f"SSL 우회 다운로드 실패: {e2}")
                
                # 3차: 캐시 파일 사용
                if cache_file and os.path.exists(cache_file):
                    logger.info(f"캐시 파일 사용: {cache_file}")
                    return pd.read_csv(cache_file)
                
                # 4차: fdr.StockListing 시도
                try:
                    logger.info("FinanceDataReader StockListing 시도...")
                    return fdr.StockListing('KRX-DESC')
                except Exception as e3:
                    logger.error(f"모든 다운로드 방법 실패: {e3}")
                    raise Exception(f"KRX 데이터를 가져올 수 없습니다. 원본 에러: {e}")


if __name__ == "__main__":

    # instance 생성
    config_file_path = './config/config.yaml'
    fs = financeScore(config_file_path)
    fs.run()
