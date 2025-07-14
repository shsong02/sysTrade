# pylint: disable=broad-except, W1203
import os
import math
import time
from datetime import datetime
import json
import ssl
import urllib.request

# multi-processing
from multiprocessing import Pool
import multiprocessing as mp

import yaml
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from io import StringIO

# html
import requests
from bs4 import BeautifulSoup as Soup
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
        # SSL 인증서 검증 우회 설정
        ssl._create_default_https_context = ssl._create_unverified_context
        print("SSL 인증서 검증 우회 설정 완료")

        # config 파일 로드
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

    def _create_driver(self):
        """WebDriver 인스턴스를 생성하는 메서드"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # 프로젝트 내의 크롬드라이버 경로 설정
        chrome_driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromedriver-mac-arm64', 'chromedriver')
        service = Service(executable_path=chrome_driver_path)
        
        return webdriver.Chrome(service=service, options=options)

    def finance_state(self, stock_info, mode='quarter'):
        """단일 종목의 재무 상태를 분석하는 메서드"""
        try:
            driver = self._create_driver()
            code = stock_info[0]
            name = stock_info[1]
            
            try:
                # 네이버 금융 페이지 접속
                url = f"https://finance.naver.com/item/main.naver?code={code}"
                driver.get(url)
                
                # 재무제표 데이터 가져오기
                tables = pd.read_html(StringIO(driver.page_source))
                
                # 재무비율 테이블 찾기
                df_ratio = None
                for table in tables:
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.get_level_values(-1)
                    if '부채비율' in table.iloc[:, 0].values:
                        df_ratio = table
                        break
                
                if df_ratio is None:
                    print(f"재무비율 테이블을 찾을 수 없음 - {code} ({name})")
                    return self._create_empty_result(name)
                
                # 첫 번째 열을 인덱스로 설정
                df_ratio.set_index(df_ratio.columns[0], inplace=True)
                
                # 매출액 데이터 추출 및 증가율 계산
                매출액_values = df_ratio.loc['매출액'].values.astype(float)
                매출액증가율 = ((매출액_values[-1] / 매출액_values[-2]) - 1) * 100 if 매출액_values[-2] != 0 else 0
                
                # 영업이익 데이터 추출 및 증가율 계산
                영업이익_values = df_ratio.loc['영업이익'].values.astype(float)
                영업이익증가율 = ((영업이익_values[-1] / 영업이익_values[-2]) - 1) * 100 if 영업이익_values[-2] != 0 else 0
                
                # 나머지 지표 추출 (NaN 값 처리)
                영업이익률 = self._safe_convert(df_ratio.loc['영업이익률'].iloc[-2])  # 마지막에서 두 번째 값 사용
                
                # ROE와 부채비율은 연간 데이터 사용
                ROE = self._safe_convert(df_ratio.loc['ROE(지배주주)'].iloc[2])  # 2024.12 값
                부채비율 = self._safe_convert(df_ratio.loc['부채비율'].iloc[2])  # 2024.12 값
                
                print(f"\n추출된 값 ({code}, {name}):")
                print(f"매출액증가율: {매출액증가율:.2f}%")
                print(f"영업이익증가율: {영업이익증가율:.2f}%")
                print(f"영업이익률: {영업이익률:.2f}%")
                print(f"ROE: {ROE:.2f}%")
                print(f"부채비율: {부채비율:.2f}%")
                
                # 결과 데이터프레임 생성
                result = pd.DataFrame({
                    '매출액증가율_list': [매출액증가율],
                    '영업이익증가율_list': [영업이익증가율],
                    '영업이익률_list': [영업이익률],
                    'ROE_list': [ROE],
                    '부채비율_list': [부채비율],
                    'total_score': [self._calculate_score(매출액증가율, 영업이익증가율, 영업이익률, ROE, 부채비율)]
                }, index=[name])
                
                return result
                
            finally:
                driver.quit()
                
        except Exception as e:
            print(f"Error processing {name} ({code}): {str(e)}")
            return self._create_empty_result(name)
            
    def _create_empty_result(self, name):
        """빈 결과 데이터프레임을 생성하는 메서드"""
        return pd.DataFrame({
            '매출액증가율_list': [0],
            '영업이익증가율_list': [0],
            '영업이익률_list': [0],
            'ROE_list': [0],
            '부채비율_list': [0],
            'total_score': [0]
        }, index=[name])

    def _safe_convert(self, value):
        """문자열을 숫자로 안전하게 변환하는 메서드"""
        if pd.isna(value) or value in ['-', '', 'N/A', 'nan', 'NaN']:
            return 0
        try:
            # 퍼센트 문자 제거 후 숫자로 변환
            return float(str(value).replace(',', '').replace('%', ''))
        except:
            return 0

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

    def _calculate_score(self, 매출액증가율, 영업이익증가율, 영업이익률, ROE, 부채비율):
        """새로운 점수 계산 로직 적용"""
        score = 0
        cnt = 0  # 가중치 계산용
        
        # 매출액증가율 점수
        score += self._calculate_growth_score(매출액증가율, cnt)
        
        # 영업이익증가율 점수
        score += self._calculate_growth_score(영업이익증가율, cnt)
        
        # 영업이익률 점수
        score += self._calculate_profit_margin_score(영업이익률)
        
        # ROE 점수
        score += self._calculate_roe_score(ROE)
        
        # 부채비율 점수
        score += self._calculate_debt_ratio_score(부채비율, cnt)
        
        return score

    def run(self):
        """전체 종목 분석을 실행하는 메서드"""
        try:
            # 저장 경로 확인
            save_dir = "./data/1_discovery/finance_scores/"
            os.makedirs(save_dir, exist_ok=True)
            print(f"저장 경로 확인: {save_dir}")
            
            # KRX 상장 종목 데이터 다운로드
            print("KRX 데이터 다운로드 시작...")
            krx_url = "https://raw.githubusercontent.com/corazzon/finance-data-analysis/main/krx.csv"
            df_krx = pd.read_csv(krx_url)
            print(f"KRX 데이터 다운로드 완료 (총 {len(df_krx)}개 종목)")
            
            # 분석 대상 종목 필터링
            df_krx = df_krx[df_krx['Symbol'].notna()]
            df_krx['Symbol'] = df_krx['Symbol'].astype(str).str.zfill(6)
            print(f"분석 대상 종목 수: {len(df_krx)}개")
            
            # 결과를 저장할 데이터프레임 초기화
            results = []
            
            # 진행 상황 표시
            total = len(df_krx)
            for idx, row in df_krx.iterrows():
                if idx % (total // 10) == 0:  # 10% 단위로 진행률 표시
                    print(f"진행률: {idx/total*100:.1f}% ({idx}/{total})")
                
                # 종목 정보 준비
                stock_info = [row['Symbol'], row['Name']]
                
                # 재무 상태 분석
                result = self.finance_state(stock_info)
                if not result.empty:
                    results.append(result)
            
            # 결과 합치기
            if results:
                final_df = pd.concat(results)
                
                # 결과 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"finance_scores_{timestamp}.csv")
                final_df.to_csv(save_path, encoding='utf-8-sig')
                print(f"분석 결과 저장 완료: {save_path}")
                
                return final_df
            else:
                print("분석 결과가 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"재무제표 분석 중 오류 발생\n상세 오류:\n{str(e)}")
            return pd.DataFrame()


if __name__ == "__main__":
    # instance 생성
    config_file_path = './config/config.yaml'
    fs = financeScore(config_file_path)
    fs.run()
