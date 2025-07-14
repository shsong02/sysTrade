import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
from io import StringIO
import os

def test_hugel_debt_ratio():
    # 크롬 드라이버 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 헤드리스 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 프로젝트 내의 크롬드라이버 경로 설정
    chrome_driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromedriver-mac-arm64', 'chromedriver')
    service = Service(executable_path=chrome_driver_path)
    
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        # 휴젤 재무제표 페이지 접속
        url = "https://finance.naver.com/item/main.naver?code=145020"
        driver.get(url)
        time.sleep(3)  # 초기 페이지 로딩 대기
        
        # 모든 테이블 가져오기
        tables = driver.find_elements(By.CLASS_NAME, "tb_type1")
        print(f"찾은 테이블 수: {len(tables)}")
        
        # 재무비율 테이블 찾기
        financial_ratio_table = None
        for idx, table in enumerate(tables):
            try:
                table_text = table.text
                print(f"\n테이블 {idx + 1} 내용:")
                print(table_text[:200] + "...")  # 처음 200자만 출력
                
                if "부채비율" in table_text:
                    financial_ratio_table = table
                    print(f"\n부채비율이 포함된 테이블을 찾았습니다 (테이블 {idx + 1})")
                    break
            except Exception as e:
                print(f"테이블 {idx + 1} 처리 중 오류: {str(e)}")
                continue
        
        if financial_ratio_table is None:
            raise Exception("재무비율 테이블을 찾을 수 없습니다.")
            
        # 테이블 HTML 가져오기
        table_html = financial_ratio_table.get_attribute('outerHTML')
        
        # DataFrame으로 변환
        df = pd.read_html(StringIO(table_html))[0]
        
        print("\n변환된 DataFrame 구조:")
        print(df.head())
        print("\nDataFrame 컬럼:")
        print(df.columns.tolist())
        
        # 부채비율 데이터 추출
        for idx, row in df.iterrows():
            # 첫 번째 열의 값을 문자열로 변환하여 '부채비율' 포함 여부 확인
            first_col_value = str(row.iloc[0])
            if '부채비율' in first_col_value:
                # 최근 분기 데이터 찾기 (NaN이 아닌 마지막 값)
                values = row.iloc[1:].astype(float)  # 숫자형으로 변환
                non_nan_values = values[~values.isna()]  # NaN이 아닌 값들
                if len(non_nan_values) > 0:
                    debt_ratio = non_nan_values.iloc[-1]  # 마지막 유효한 값
                    print(f"\n부채비율 행을 찾았습니다:")
                    print(row)
                    break
        else:
            raise Exception("부채비율 데이터를 찾을 수 없습니다.")
        
        print(f"\n휴젤 부채비율: {debt_ratio}%")
        
        # 전체 재무비율 데이터 출력
        print("\n전체 재무비율 데이터:")
        print(df)
        
        return debt_ratio
        
    finally:
        driver.quit()

if __name__ == "__main__":
    debt_ratio = test_hugel_debt_ratio() 