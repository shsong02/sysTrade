import yaml
import requests
import json
import pandas as pd
import os
import time
from glob import glob
from datetime import datetime, date, timedelta

## scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

## html
import requests
from bs4 import BeautifulSoup as Soup
from selenium import webdriver

## 한국투자증권 rest 사용하기
import src.kis.rest.kis_api as kis

## krx 주기
from pykrx import stock

## local file
from tools import st_utils as stu
from trade_strategy import tradeStrategy


## log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e :
    print(e)
####    로그 생성    #######
logger = stu.create_logger()

_DEBUG = False
_DEBUG_NEWDATA = False


class systemTrade:

    def __init__(self, mode="virtual"):

        ## init
        self.mode = mode  ## 'real', 'virtual'

        ## 설정 파일을 필수적으로 한다.
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

        ##### 초기 변수 설정
        ## 스케쥴링 하기 위해서 시간 간격을 생성
        self.trade_schedule = config["tradeStock"]["scheduler"]
        self.trade_config = config["tradeStock"]["config"]

        self.mon_intv           = self.trade_schedule["interval"]  # minutes  (max 30 min.)
        self.trade_target       = self.trade_schedule["target"]
        self.mode               = self.trade_schedule["mode"] ## real, backfill, kospi


        ## 거래에 필요한 kis config 파일 읽어와서 계정 관련 설정
        with open(r'config/kisdev_vi.yaml', encoding='UTF-8') as f:
            kis_config = yaml.load(f, Loader=yaml.FullLoader)
        if self.mode == 'real':
            self.url_base= kis_config["prod"]
            self.account = kis_config["my_acct_stock"]
            self.app_key = kis_config["my_app"]
            self.app_secret = kis_config["my_sec"]
        else: ## 'virtaul'
            self.url_base = kis_config["vps"]
            self.account = kis_config["my_paper_stock"]
            self.app_key = kis_config["paper_app"]
            self.app_secret = kis_config["paper_sec"]


        self._access_token()  ## self.access_token 생성

    def _access_token(self):
        ## 파일로 저장. (24시간만 유지)
        format = "%Y%m%d-%H%M%S"
        currtime = datetime.now()
        currtimestr = currtime.strftime(format=format)
        path = self.file_manager["monitor_stocks"]["path"]

        ## 없으면 폴더 생성
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            raise e

        ## 파일 존재 여부 확인
        filelist = os.listdir(path)
        chk1 = False  ## 파일 존재 여부 확인
        chk2 = False  ## 24시간 초과 확인
        for file in filelist:
            if "kis-token" in file:
                n, ext = os.path.splitext(file)
                filetime = n.split("_")[-1]
                time = datetime.strptime(filetime, format)
                time_add1day = time + timedelta(days=1)

                if time_add1day < currtime:
                    # print(time_add1day, currtime)
                    os.remove(path+file)
                    chk2 = True #
                else:
                    print(time_add1day, currtime)
                    with open(path + file, 'r') as f:
                        self.access_token = json.load(f)
                    logger.info(f"한국투자증권 API 용 토큰이 존재하므로 파일 로드 합니다. (파일명: {path + file}) ")

                chk1 = True  ## 파일이 일단 존재함
                break

        if (chk1 == False)  or (chk2==True) :
            name = f"kis-token_{currtimestr}.json"
            if chk2 == True :
                logger.info(f"한국투자증권 API 용 토큰이 날짜 초과(24시간) 하여 새로 생성 및 파일 저장합니다.. (파일명: {path + name}) ")
            else:
                logger.info(f"한국투자증권 API 용 토큰이 존재하지 않기 때문에 파일 저장합니다 . (파일명: {path+name}) ")


            # 신규 생성 : 보안인증키 받기
            url = f"{self.url_base}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            body = {"grant_type": "client_credentials",
                    "appkey": self.app_key,
                    "appsecret": self.app_secret}
            res = requests.post(url, headers=headers, data=json.dumps(body))
            self.access_token = res.json()["access_token"]

            with open(path+name, 'w') as f:
                json.dump(self.access_token, f)

    def hashkey(self, datas):
        path = "uapi/hashkey"
        url = f"{self.url_base}/{path}"
        headers = {
            'content-Type': 'application/json',
            'appKey': self.app_key,
            'appSecret': self.app_secret,
        }
        res = requests.post(url, headers=headers, data=json.dumps(datas))
        hashkey = res.json()["HASH"]

        return hashkey

    def get_curr_price(self, code):
        path = "uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = "FHKST01010100"

        url = f"{self.url_base}/{path}"


        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "tr_id": tr_id}
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code
        }
        res = requests.get(url, headers=headers, params=params)
        out = res.json()['output']['stck_prpr']
        url = f"{self.url_base}/{path}"

        return out

    def get_curr_min_price(self, code, times=[]):
        path = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        tr_id = "FHKST03010200"
        url = f"{self.url_base}/{path}"

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_ETC_CLS_CODE": "",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_HOUR_1": '',  # 아래 에서 삽입
            "FID_PW_DATA_INCU_YN": "N"
        }

        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "hashkey": self.hashkey(params),
                   "tr_id": tr_id}


        ### 분봉 30개 밖에 허용 되지 않기 때문에 이어 붙이기 필요
        time_list = self._make_time_list(times, interval=30)

        df_times = []
        for time in time_list:
            params['FID_INPUT_HOUR_1'] = time
            res = requests.get(url, headers=headers, params=params)
            data = res.json()["output2"]
            df = self._make_df(tr_id, data)
            df_times.append(df)

        df_out = pd.concat(df_times)
        df_out.sort_index(inplace=True)
        df_out.drop_duplicates(inplace=True)

        st = timedelta(hours=int(times[0][0:2]), minutes=int(times[0][2:4]))
        end = timedelta(hours=int(times[1][0:2]), minutes=int(times[1][2:4]))
        today = datetime.now()
        st_dt = datetime(today.year, today.month, today.day) + st
        end_dt = datetime(today.year, today.month, today.day) + end

        df_out2 = df_out[(st_dt <= df_out.index) & (end_dt > df_out.index) ]

        return df_out2

    def get_curr_min_chegyeol(self, code, times=[]):
        path = "/uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion"
        tr_id = "FHPST01060000"
        url = f"{self.url_base}/{path}"

        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
            "fid_input_hour_1": '',  # 아래 에서 삽입
        }

        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "hashkey": self.hashkey(params),
                   "tr_id": tr_id}

        ### 분봉 30개 밖에 허용 되지 않기 때문에 이어 붙이기 필요
        time_list = self._make_time_list(times, interval=2)

        df_times = []
        for time in time_list:
            params['fid_input_hour_1'] = time
            res = requests.get(url, headers=headers, params=params)
            data = res.json()['output2']
            df = self._make_df(tr_id, data)
            df_times.append(df)

        df_acc = pd.concat(df_times)
        df_acc.sort_index(inplace=True)
        df_acc.drop_duplicates(inplace=True)

        df_grp = df_acc.groupby(df_acc.index)

        idx = []
        str = []
        acc = []
        size = []
        for key, df_part in df_grp:
            idx.append(key)
            str_prev = float(df_part['ChegyeolStr'].iat[-1])
            if str_prev > 140 :  ## chart 상 보기 불편해서
                str_prev = 140
            str.append(str_prev)
            acc.append(df_part['VolumeAcc'].iat[-1])
            size.append(df_part['VolumeSize'].astype(int).sum())

        df_out = pd.DataFrame(columns=['Date', 'ChegyeolStr', 'VolumeAcc', 'VolumeSize'])
        df_out["Date"] = idx
        df_out["ChegyeolStr"] = str
        df_out["VolumeAcc"] = acc
        df_out["VolumeSize"] = size

        df_out.set_index('Date', inplace=True)
        df_out.sort_index(ascending=True, inplace=True)

        return df_out

    def get_curr_investor(self, code, times=[]):
        path = "/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr_id = "FHKST01010900"
        url = f"{self.url_base}/{path}"

        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }

        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "hashkey": self.hashkey(params),
                   "tr_id": tr_id}

        ## 데이터 불러오기
        res = requests.get(url, headers=headers, params=params)
        data = res.json()['output']

        ## DF 로 변경
        df = self._make_df(tr_id, data)

    def get_curr_member(self, code, times=[], df_prev=pd.DataFrame()):
        path = "/uapi/domestic-stock/v1/quotations/inquire-member"
        tr_id = "FHKST01010600"
        url = f"{self.url_base}/{path}"


        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": code,
        }

        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "hashkey": self.hashkey(params),
                   "tr_id": tr_id}

        ## 데이터 불러오기
        res = requests.get(url, headers=headers, params=params)
        data = res.json()['output']

        ## DF 로 변경
        df = self._make_df(tr_id, data)

        ## 출력 포멧으로 맞추기
        cols = df.columns.to_list()
        df_out = pd.DataFrame(columns=cols)

        # index 생성
        time_list = self._make_time_list(times, interval=1)  ## 늘릴 indexa 갯수
        today = datetime.now().date().strftime("%Y%m%d")
        time_list_new = [today+'-'+i for i in time_list]
        df_out['Date'] = time_list_new

        for col in cols:
            df_out[col] = df.iloc[0][col]

        df_out['Date'] = pd.to_datetime(df_out["Date"], format="%Y%m%d-%H%M%S")
        df_out.set_index('Date', inplace=True)
        df_out.sort_index(ascending=True, inplace=True)

        return df_out

    ########### 계좌 관리
    def get_kr_buyable_cash(self) -> int:
        """
        구매 가능 현금(원화) 조회
        return: 해당 계좌의 구매 가능한 현금(원화)
        """
        path = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        if self.mode == 'real':
            tr_id = "TTTC8908R"
        else:
            tr_id = "VTTC8001R"
        url = f"{self.url_base}/{path}"

        if self.account is None:
            msg = "계좌가 설정되지 않았습니다. set_account를 통해 계좌 정보를 설정해주세요."
            raise RuntimeError(msg)

        stock_code = ""
        qry_price = 0

        params = {
            "CANO": self.account,
            "ACNT_PRDT_CD": "01",
            "PDNO": stock_code,
            "ORD_UNPR": str(qry_price),
            "ORD_DVSN": "02",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N"
        }

        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "hashkey": self.hashkey(params),
                   "tr_id": tr_id}


        res = requests.get(url, headers=headers, params=params)
        data = res.json()["output2"]

    def get_inquire_balance(self):
        '''
            - 잔고조회
            :return:
        '''

        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{self.url_base}/{path}"

        if self.mode == 'real':
            tr_id = "TTTC8434R"
        else:
            tr_id = "VTTC8434R"
        code = ""

        params = {
            "cano": self.account,
            "acnt_prdt_cd": "01",
            "fid_input_iscd": code,
            "fid_input_hour_1": date,
            "fid_pw_data_incu_yn": "Y"
        }

        headers = {"Content-Type": "application/json",
                   "authorization": f"Bearer {self.access_token}",
                   "appKey": self.app_key,
                   "appSecret": self.app_secret,
                   "hashkey": self.hashkey(params),
                   "tr_id": tr_id}


        res = requests.get(url, headers=headers, params=params)
        data = res.json()["output2"]

    #############################
    ####   Internal Func.   ####
    #############################
    def _make_time_list(self, times, interval=30):
        # st ~ end 가 30분을 초과 할 경우 나눠서 생성
        st = timedelta(hours=int(times[0][0:2]), minutes=int(times[0][2:4]))
        end = timedelta(hours=int(times[1][0:2]), minutes=int(times[1][2:4]))

        time_list = []
        for i in range(int(60/interval)*7): ## 10분 간격   (max 7 시간)
            if i == 0 :
                t = st
            else:
                t = t + timedelta(minutes=interval)

            str_t = str(t).replace(':', '').zfill(6)
            if t < end:
                time_list.append(str_t)
            else:
                str_end = str(end).replace(':', '').zfill(6)
                time_list.append(str_end)  # 마지막 한번더 저장하기 위함
                break
        return time_list

    def _make_df(self, tr_id, data):
        if tr_id == "FHKST03010200": # 당일 분봉
            today = data[0]['stck_bsop_date']
            O = []
            L = []
            H = []
            C = []
            V = []
            T = []
            for i in data:
                O.append(i['stck_oprc'])
                L.append(i['stck_lwpr'])
                H.append(i['stck_hgpr'])
                C.append(i['stck_prpr'])
                V.append(i['cntg_vol'])
                T.append(today+'-'+i['stck_cntg_hour'])


            df = pd.DataFrame(columns=['Date','Low', 'High', 'Close', 'Volume'])
            df['Date'] = T
            df['Open'] = O
            df['Low'] = L
            df['High'] = H
            df['Close'] = C
            df['Volume'] = V
            df['Date'] = pd.to_datetime(df["Date"], format="%Y%m%d-%H%M%S")
            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
        elif tr_id == "FHPST01060000":

            today = datetime.now().date().strftime("%Y%m%d")

            T = []
            str = []
            accVol = []
            cnqn = []

            for i in data:
                T.append(today+'-'+i['stck_cntg_hour'][0:4]+'00') # 초 버리기
                str.append(i["tday_rltv"])
                accVol.append(i["acml_vol"])
                cnqn.append(i["cnqn"])

            df = pd.DataFrame(columns=['Date','ChegyeolStr', 'VolumeAcc', 'VolumeSize'])
            df["Date"] = T
            df["ChegyeolStr"] = str
            df["VolumeAcc"] = accVol
            df["VolumeSize"] = cnqn

            df['Date'] = pd.to_datetime(df["Date"], format="%Y%m%d-%H%M%S")
            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
        elif tr_id == "FHKST01010600": # 현재 회원사 정보
            # row 가 1 개밖에 존재하지 않음
            cols = []
            for id in ["buyCode", "buyName", "buyVolumeAcc", "buyRatio", "buyChange", "sellCode", "sellName", "sellVolumeAcc", "sellRatio", "sellChange" ]:
                for i in range(1,6):
                    cols.append(id+f"{i}")  ## 각 5개씩

            df = pd.DataFrame(columns=cols)
            for key, value in data.items():
                name, no = (key[:-1], key[-1])
                if name == "seln_mbcr_no": ## 매도 회원사
                    df.loc[0,f"sellCode{no}"] = data[f"{name}{no}"]
                elif name == "seln_mbcr_name": ## 매도 회원사
                    df.loc[0,f"sellName{no}"] = data[f"{name}{no}"]
                elif name == "total_seln_qty": ##
                    df.loc[0,f"sellVolumeAcc{no}"] = data[f"{name}{no}"]
                elif name == "seln_mbcr_rlim": ##
                    df.loc[0,f"sellRatio{no}"] = data[f"{name}{no}"]
                elif name == "seln_qty_icdc": ##
                    df.loc[0,f"sellChange{no}"] = data[f"{name}{no}"]
                elif name == "shnu_mbcr_no": ## 매도 회원사
                    df.loc[0,f"buyCode{no}"] = data[f"{name}{no}"]
                elif name == "shnu_mbcr_name": ## 매도 회원사
                    df.loc[0,f"buyName{no}"] = data[f"{name}{no}"]
                elif name == "total_shnu_qty": ##
                    df.loc[0,f"buyVolumeAcc{no}"] = data[f"{name}{no}"]
                elif name == "shnu_mbcr_rlim": ##
                    df.loc[0,f"buyRatio{no}"] = data[f"{name}{no}"]
                elif name == "shnu_qty_icdc": ##
                    df.loc[0,f"buyChange{no}"] = data[f"{name}{no}"]
        # elif tr_id == "FHKST01010900":
        #     for d in data:
        #         print(d)
        #     for key, value in data.items():
        #         print(key, value)
        #
        else:
            raise ValueError(f"지원하지 않는 tr_id={tr_id} 입니다")

        return df


    def run(self):

        if self.mode == 'real':
            ## 실제 시간 확인하여 웨이팅하기
            init_time = "094000"  ## ma40 때문인듯 에러가 나서... 일단 40분으로 시간 제약 해둠
            if self.trade_schedule["mode"] == "real":
                while True:
                    real_time = datetime.now().time().strftime("%H%M%S")
                    if real_time >= init_time:  ## bol window 가 20 이라, 최소 index 가 20 이상이어야 함
                        break

                    print(f"현재시간 ({real_time}) 이 목표시간 ({init_time}) 에 도달하지 못했기 때문에 기다립니다.")
                    time.sleep(60)

        ######################################
        ####     시간 정보는 여기서 한번에 처리
        ######################################
        now = datetime.now()
        now_str = now.time().strftime("%H:%M:%S")
        now_str2 = now_str.replace(":", "")
        _msg = f"{now_str} 시에 프로그램 실행합니다."
        stu.send_telegram_message(config=self.param_init, message=_msg)

        ################################
        ####     Monitoring 종목 가져오기
        ################################
        # 트레이밍 대상을 kospi 또는 개별종목 따라 진행 사항이 달라진다.
        if self.trade_target in ['all', 'stock']: # 모니터링할 개별 종목 확인
            path = self.file_manager["monitor_stocks"]["path"]
            flist = glob(path + "*/*/*/*/*.csv")  ## 년/월/일/시간/*.csv

            ## 시간 포멧이 일정하기 때문에 str 비교로도 가장 최근을 선택할 수 있음
            nstr = ''
            for f in flist:
                if f > nstr:
                    nstr = f

            ## 탐색할 종목명 확인
            df = pd.read_csv(nstr)
            codes0 = df.Code.to_list()
            selected_stocks = [str(x).zfill(6) for x in codes0]
            selected_stock_names = []
            stocks_dict = dict()
            for code in selected_stocks:
                stocks_dict[code] = dict()
                stocks_dict[code]['data'] = pd.DataFrame()
                name = stock.get_market_ticker_name(code)  ## 기본으로 입력해줘야 함
                selected_stock_names.append(name)

        ########################################
        ####     초기 데이터 준비 (kospi 는 관련 없음)
        ########################################
        '''
        실행조건: 
          - real 일 경우, now 를 확인하고 진행
            -- 0910 이후 진행 (이전에는 데이터가 없음. 에러남) 
            -- reduce_api 가 실행된 상태이면, 오늘 중 가낭 최근 데이터까지는 불러 오고 나머지는 신규로 가져온다.  (TBD)
          - backfill, test 는 지정날짜에 저장된 데이터를 가져온다. 
            -- 만약 해당일에 데이터가 하나도 없으면 에러 발생하고 종료 
            -- 선택된 종목 중 일부가 없으면 종목을 제거하고 다음 진행.
             
        '''
        ### 한번에 많이 불러와서 죽는 문제 -> 초당 15회도 처리 가능함을 확인 받음 (22.10.30). timed_out 발생 이유는 로컬 네트워크 문제라고 함 ??
        base_path = self.file_manager["system_trade"]["path"]
        reduce_api = self.trade_config["reduce_api"]

        if self.mode == "real":
            ## 현 시점이 장 시작 전이면 초기 데이터를 만들어 놓기 시작합니다.
            time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"

            if '090010' <= now_str2:
                for code in selected_stocks:
                    times = ["090000", now_str2]
                    df_fin = self._make_curr_df(code, times)
                    stocks_dict[code]['data'] = df_fin
                    logger.info(f"장 시작 9시 이후에 시작되었기 때문에 시간({times} 에 대한 종목({code}) 실시간 정보를 미리 준비합니다.)")

                    name = stock.get_market_ticker_name(code)  ## 기본으로 입력해줘야 함
                    stu.file_save(df_fin, file_path=base_path+time_path, file_name=f"{name}_{code}.csv", replace=False)

            ## Real 은 Kospi 사전 준비가 없음 (아래 단계에서 데이터 준비 및 차트가 동시 일어남)

        else:  ## backfill, test
            '''
                Backfill 은 날짜 범짜 범위 지정. Test 는 특정 날짜. 
                Backtest 제약 조건
                    - 날짜범위가 30일을 넘으면 안됨
                    - 개별 날짜 마다 존재하는 데이터를 불러옴 (날짜마다 종목 리스트가 다를 수 있음) 
                    - 모든 날짜에 데이터가 존재하지 않으면 에러 처리 
                
            '''
            ## 데이터 구조 생성 (날짜별 stocks_dict 저장)
            test_dict = dict()
            dates = self.trade_config["test_date"]
            dates = dates.split("-")
            test_dates = []
            if self.mode == 'backfill':
                for i in range(31): # max length = 30
                    if i ==0 :
                        tdate= int(dates[0])
                    elif i == 30:
                        raise ValueError (f"Backfill 의 test_date 는 30 일 이하여야 합니다. (현재: {dates})")
                    else:
                        tdate = tdate + 1
                        if tdate > int(dates[1]):
                            break

                    test_dict[str(tdate)] = dict()
                    test_dates.append(str(tdate))
            else: # test
                test_dict[dates[1]] = dict()
                test_dates.append(dates[1])

            ## 저장된 데이터 불러 오기
            if self.trade_target in ['all', 'stock']:
                empty_cnt = 0
                for tdate in test_dates:
                    ## 년/월/일/시간/*.csv
                    dlist = glob(base_path + f"year={tdate[0:4]}/month={tdate[4:6]}/day={tdate[6:8]}/*/")
                    if len(dlist) == 0 :
                        _msg = f"모드({self.mode}) 의 지정날짜({tdate}) 에 대한 저장된 데이터가 존재하지 않습니다. "
                        logger.info(_msg)
                        empty_cnt += 1
                        continue

                    ## 해당 일의 마지막 데이터 folder 찾기
                    dfin = ''
                    for d in dlist:
                        if d > dfin:
                            dfin = d

                    ## 선택된 종목이 다른지 확인. 불러온 데이터에서 선택된 종목이 없을 경우 종목을 삭제함.
                    flist = os.listdir(dfin)
                    load_stock = []
                    for f in flist:
                        load_stock.append(f.split("_")[0])

                    ## todo: selected_stocks 가 고정임. 데일리로 변경해야 하는지 추후 고민
                    stocks_dict = dict()
                    for code in selected_stocks:
                        chk1 = False
                        stocks_dict[code] = dict()
                        for f in flist:
                            if code in f:  ## 파일명이 종목 코드 이므로,
                                df =stu.file_load(file_path=dfin, file_name=f, type='csv')
                                df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
                                stocks_dict[code]["data"] = df
                                chk1 = True
                                break
                        if not chk1:
                            logger.info(f"종목코드({code}) 는 지정된 날짜({tdate})에 존재하지 않습니다. 탐색대상에서 제외합니다. (경로:{dfin})")
                            del(stocks_dict[code])

                    ## 모든 종목이 삭제된 경우를 확인
                    if len(stocks_dict) == 0 :
                        _msg = f"모드({self.mode}) 의 지정날짜({tdate}) 에  선택한 종목 중 저장된 데이터가 하나도 없습니다. "
                        logger.info(_msg)
                        empty_cnt += 1
                    else:
                        ### 데이터 쌓기
                        test_dict[tdate] = stocks_dict

                ## 범위 날짜중 데이터거 하나도 없는 경우 에러 처리
                if len(test_dates) == empty_cnt:
                    _msg = f"모드({self.mode}) 의 지정날짜({dates}) 에 대한 저장된 데이터가 하나도 존재하지 않습니다. "
                    raise ValueError(_msg)

            if self.trade_target in ['all', 'kospi']:
                empty_cnt = 0
                for tdate in test_dates:
                    ## 년/월/일/시간/*.csv
                    dlist = glob(base_path + 'kospi/csv/' + f"year={tdate[0:4]}/month={tdate[4:6]}/day={tdate[6:8]}/*/")
                    if len(dlist) == 0:
                        _msg = f"모드({self.mode}) 의 지정날짜({tdate}) 에 대한 저장된 데이터가 존재하지 않습니다. "
                        logger.info(_msg)
                        empty_cnt += 1
                    else:
                        ## 해당 일의 마지막 데이터 folder 찾기
                        dfin = ''
                        for d in dlist:
                            if d > dfin:
                                dfin = d

                        ## 선택된 종목이 다른지 확인. 불러온 데이터에서 선택된 종목이 없을 경우 종목을 삭제함.
                        # flist = os.listdir(dfin)
                        df = stu.file_load(file_path=dfin, file_name="kospi.csv", type='csv')
                        try:
                            df.index = pd.to_datetime(df.index, format="%Y:%m:%d %H:%M:%S")
                        except:
                            print(f"타임 포멧이 잘못 저장되었음. 찾아서 수정 필요 -- %Y-%m-%d %H:%M:%S. (파일위치:{dfin})")
                            df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
                        df.interpolate(inplace=True) ## 중간에 발생한 missing 처리

                        stocks_dict = dict()
                        stocks_dict["kospi"] = dict()
                        stocks_dict["kospi"]["data"] = df  ## 포멧을 맞추기 위한 용

                        test_dict[tdate] = stocks_dict

                if len(test_dates) == empty_cnt:
                    _msg = f"모드({self.mode}) 의 지정날짜({dates}) 에 대한 저장된 데이터가 하나도 존재하지 않습니다. "
                    raise ValueError(_msg)


        ###########################################
        ####     Realtime 용. (나머지는 chart 생성만)
        ###########################################
        '''
        실행조건: 
          - real 일 경우, now 를 확인하고 진행
            -- 1530  이후는 차트 표시 후, 저장. 
            -- 1530 이전은 dataframe 까지 생성 및 저장  
          - backfill, test 는 지정날짜에 저장된 데이터를 가져온다. 
            -- chart 만 보여주기 

        '''
        ## 데이터 생성은 real 에서만 (나머지는 read only)
        if self.mode == 'real':
            cm = tradeStrategy('./config/config.yaml')
            cm.display = 'save'  ## real 은 모든 파일 저장이 고정 (텔레그램 전송 때문)
            ### 얼마나 자주 반복할지 결정
            day_tlist = self._make_time_list([now_str2, "153000"], interval=self.mon_intv)
            times = [0, 0]  ## st, end
            chart_base_path = self.file_manager["system_trade"]["path"] + "chart/"
            today = now.date().strftime("%Y%m%d")

            if (now_str2 > '153000'):
                ## 저장할 경로 확인
                time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"
                image_path = chart_base_path + time_path
                cm.path = image_path

                if self.trade_target in ['all', 'stock']:
                    for code in selected_stocks:
                        df_acc = stocks_dict[code]['data']  ## 이미 앞에서 모두 준비된 상태
                        df_acc = df_acc.fillna(0)  ## missing 처리
                        df_acc = df_acc.loc[~df_acc.index.duplicated(keep='first')]  # index 중복 제거
                        name = stock.get_market_ticker_name(code)  ## 기본으로 입력해줘야 함
                        image_name = f"{name}_{code}.png"
                        cm.name = image_name
                        print(f"차트 (time: {str(now)}) 를 생성합니다. (파일명: {image_name})")
                        df_ohlcv = cm.run(code, name, data=df_acc, dates=[today, today], mode='realtime')

                        ## 파일로도 저장
                        stu.file_save(df_acc, file_path=base_path + time_path, file_name=f"{code}.csv", replace=False)

                if self.trade_target in ['all', 'kospi']:
                    msg_kospi = self._investor_position()  ## 파일 저장까지 내부 포함
                    stu.send_telegram_message(config=self.param_init, message=msg_kospi["msg"])
                    stu.send_telegram_image(config=self.param_init, image_name_path=msg_kospi["image_name_path"])

            else:  ## 장중 진행 상황 확인
                for idx, t in enumerate(day_tlist[:-1]):
                    times.append(t)
                    times.pop(0)

                    ## 저장할 경로 확인
                    now = datetime.today()
                    time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"

                    image_path = chart_base_path + time_path
                    cm.path = image_path

                    if idx == 0:
                        ## 0900 는 실행 하지 않음
                        pass
                    else:
                        if self.trade_target in ['all', 'stock']:
                            ###############################
                            #### 한국투자API 를 이용하여 DF 생성
                            ###############################
                            for code in selected_stocks:
                                df_fin = self._make_curr_df(code, times)
                                df_acc = stocks_dict[code]['data']

                                if len(df_acc) == 0:
                                    df_acc = df_fin
                                else:
                                    df_acc = pd.concat([df_acc, df_fin])

                                df_acc = df_acc.fillna(0)  ## missing 처리
                                df_acc = df_acc.loc[~df_acc.index.duplicated(keep='first')]  # index 중복 제거

                                name = stock.get_market_ticker_name(code)
                                image_name = f"{name}_{code}.png"
                                cm.name = image_name
                                if len(df_acc) != 0:
                                    print(f"차트 (time: {str(now)}) 를 생성합니다. (파일명: {image_name})")
                                    df_ohlcv = cm.run(code, name, data=df_acc, dates=[today, today], mode='realtime')

                                    ## 다음 사용을 위해 데이터 저장
                                    stocks_dict[code]['data'] = df_acc

                                    # 매수 조건이 발생하였는지 확인 (텔레그림..발생)
                                    ## 조건1: 채결 강도가 100 을 넘었을 경우, 차트로 알려주기
                                    cstrth = float(df_ohlcv.tail(1).ChegyeolStr)
                                    close0 = int(df_ohlcv.head(1).Close)
                                    close1 = int(df_ohlcv.tail(1).Close)
                                    change = self._change_ratio(close1, close0)

                                    df_chg = self._bollinger_chegyeol(df_ohlcv)  # param 은 기본으로 사용
                                    ch_sig = df_chg.copy()
                                    ch_sig_flag = ch_sig.tail(self.mon_intv).bolBuy_chegyeol.any()
                                    # ch_sig = df_chegyeol.bolBuy_chegyeol.any()
                                    # print(name, ch_sig, df_chegyeol.bol_upper_chegyeol.to_list() )
                                    if ch_sig_flag:
                                        ch_sig = df_chg.copy().tail(self.mon_intv)
                                        idx_list = ch_sig[ch_sig.bolBuy_chegyeol == True].index.to_list()
                                        del_str = datetime.now().date().strftime("%Y-%m-%d ")
                                        idx_list2 = [str(x) for x in idx_list]
                                        idx_list2 = [x.replace(del_str, '') for x in idx_list2]
                                        _msg = f"현재 ({str(now)}) 종목 ({name}) 의 채결 강도가 급하게 상승한 구간이 최근 10분 내 존재합니다. (시점: {idx_list2}) "
                                        print(_msg)
                                        stu.send_telegram_message(config=self.param_init, message=_msg)
                                        stu.send_telegram_message(config=self.param_init,
                                                                  message=f"현재 등락률은 {change}% 입니다.(장전 갭상은 반영 못함)")
                                        stu.send_telegram_image(config=self.param_init,
                                                                image_name_path=image_path + image_name)

                                    ## csv 파일로도 저장
                                    stu.file_save(df_chg, file_path=base_path + time_path,
                                                  file_name=f"{name}_{code}.csv", replace=False)

                        if self.trade_target in ['all', 'kospi']:
                            msg_kospi = self._investor_position()  ## 파일 저장까지 내부 포함
                            stu.send_telegram_message(config=self.param_init, message=msg_kospi["msg"])
                            stu.send_telegram_image(config=self.param_init,
                                                    image_name_path=msg_kospi["image_name_path"])

                        ## 실제 시간 확인하여 웨이팅하기
                        while True:
                            real_time = datetime.now().time().strftime("%H%M%S")
                            if real_time >= t:
                                break

                            print(f"현재시간 ({real_time}) 이 목표시간 ({t}) 에 도달하지 못했기 때문에 기다립니다.")
                            time.sleep(30)

        else: ## backtest, test
            ## todo: stock 용은 추후 진행
            for tdate in test_dates:
                if self.trade_target in ['all', 'kospi']:

                    # 데이터 선택하기
                    if "kospi" in test_dict[tdate]:
                        df = test_dict[tdate]["kospi"]["data"]

                        cm = tradeStrategy('./config/config.yaml')
                        cm.display = "on"  ## 파일로 차트 저장하기
                        df_ohlcv = cm.run('00000000', 'kospi', data=df, dates=[tdate, tdate], mode='investor')

                        stClose = df_ohlcv.Close.iat[0]
                        endClose = df_ohlcv.Close.iat[-1]
                        change = self._change_ratio(curr=endClose, prev=stClose)
                        _msg = f'지정날짜 {tdate}의 kospi 등락률은 {change} % 입니다.'  # for 등락률
                        print(_msg)
                    else:
                        pass

            print("finish!!!")





    #############################
    #### Internal Func
    #############################
    def _make_curr_df(self, code, times):
        df = self.get_curr_min_price(code, times)
        df2 = self.get_curr_min_chegyeol(code, times)
        df3 = self.get_curr_member(code, times)  ## 최종시점 데이터만 가져올수 있음. 실시간으로 가져오도록 해야 함
        # df4 = tr.get_curr_investor(code, times) ## 실시간 미지원으로 확인되어 사용하지 않음..(추후 까지)
        df_fin = df.join(df2)  ## time index 로 join 함
        df_fin = df_fin.join(df3)

        # 타입 변환
        cols = ["Low", "High", "Close", "Volume", "Open", "ChegyeolStr"]
        df_fin[cols] = df_fin[cols].apply(pd.to_numeric)

        return df_fin

    def _change_ratio(self, curr, prev):
        return round((curr - prev) / curr * 100, 2)


    def _bollinger_chegyeol(self, df_in, window=20, sigma=2.0):
        df = df_in.copy()
        df['bol_mid_chegyeol'] = df['ChegyeolStr'].rolling(window=window).mean()
        std = df['ChegyeolStr'].rolling(window).std(ddof=0)
        df['bol_upper_chegyeol'] = df['bol_mid_chegyeol'] + sigma * std
        df['bol_lower_chegyeol'] = df['bol_mid_chegyeol'] - sigma * std

        df['bolBuy_chegyeol'] = False
        df['bolSell_chegyeol'] = False
        df_temp = pd.DataFrame()
        df_temp = df[df.bol_upper_chegyeol <= df.ChegyeolStr]
        for idx in df_temp.index.to_list():
            df.loc[idx, 'bolBuy_chegyeol'] = True  ## 상단 터치를 사는 시점으로 봄 (범위를 짥게 가져감)

        df_temp = df[df.bol_lower_chegyeol >= df.ChegyeolStr]
        for idx in df_temp.index.to_list():
            df.loc[idx, 'bolSell_chegyeol'] = True

        return df

    def _investor_position(self, mode='real'):
        '''
          투자자별 매매 동향의 주체가 변경되는 시점을 이용하여 인버스, 레버리지 매수 매도 전략
          이유
            - 하락장에서는 장중 kospi 변동폭이 큼.
            - 방향성이 전환되면 묵직하게 쭉 유지됨. (금액이 크기 때문에 추세 변화가 많지 않음)
          평가
            - 자동 매매 에 적합함 ( 순간의 딜레이는 중요하지 않기 때문)
            - 알고리즘 명확하면 큰 금액을 태울 수 있음
        :return:
        '''



        ## 신규 생성되는 csv, image 저장을 위한 폴더 생성 용
        base_csv_path = self.file_manager["system_trade"]["path"] + 'kospi/csv/'
        base_image_path = self.file_manager["system_trade"]["path"] + 'kospi/image/'

        if mode == "real":
            ### Issue: 선물 데이터 때문에 09:20 부터 진행 해야 함. (All NaN 여서 에러남)
            now = datetime.now()
            now_str = now.strftime("%H%M%S")
            if now_str <= '092000':  ## URL 주소 용: 장 시작전이면, 어제 날짜 데이터를 불러옴
                td_dt = date.today() - timedelta(days=1)
                td = td_dt.strftime("%Y:%m:%d")
            else:
                td = datetime.today().strftime("%Y:%m:%d")
            url_date = td.replace(":", "")
            time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"

            df_tot_list = []
            for i in range(4):
                df_list = []
                if i == 0 :  ## kospi200
                    maxpage = 66
                elif i == 1: ## 투자자별
                    maxpage = 40
                elif i == 2:  ## 투자자별
                    maxpage = 43
                else:
                    maxpage = 43


                for p in range(1, maxpage):
                    if i == 0 :
                        URL = f"https://finance.naver.com/sise/sise_index_time.naver?code=KOSPI&thistime={url_date}160000&page={p}"
                        # URL = f"https://finance.naver.com/sise/sise_index_time.naver?code=KPI200&thistime={td2}160000&page={p}"
                    elif i == 1 :
                        URL = f"https://finance.naver.com/sise/investorDealTrendTime.naver?bizdate={url_date}&sosok=&page={p}"
                    elif i == 2: ## l
                        URL = f"https://finance.naver.com/sise/programDealTrendTime.naver?bizdate={url_date}&sosok=&page={p}"
                    else:  ## 선물
                        URL = f"https://finance.naver.com/sise/investorDealTrendTime.naver?bizdate={url_date}&sosok=03&page={p}"

                    html = requests.get(URL)
                    soup = Soup(html.content, "html.parser")
                    table = soup.find('table')
                    table_html = str(table)
                    df = pd.read_html(table_html)
                    df_list.append(df[0])

                df_acc = pd.concat(df_list)
                if i == 0 :
                    df = pd.DataFrame(columns=['Date', 'Close', 'Volume', 'VolumeSize'])
                    df['Date'] = td + ' ' + df_acc['체결시각'] + ':00'
                    df['Close'] = df_acc['체결가']
                    df['Volume'] = df_acc['변동량(천주)']
                    df['VolumeSize'] = df_acc['거래대금(백만)']

                    ## Open 이 없어서 강제로 생성 중
                    df.dropna(subset=['Date'], inplace=True)
                    df.sort_values(by="Date", inplace=True)
                    df['Open'] = df['Close'].shift(1)
                    # 첫번째 index nan 필요
                    df['Open'] = df['Open'].fillna(method='bfill')

                    def get_high(df):
                        if df['Close'] >= df['Open']:
                            return df['Close']
                        else:
                            return df['Open']
                    def get_low(df):
                        if df['Close'] < df['Open']:
                            return df['Close']
                        else:
                            return df['Open']
                    df['High'] = df.apply(get_high, axis=1)
                    df['Low'] = df.apply(get_low, axis=1)

                elif i == 1:
                    df = pd.DataFrame(columns=['Date', 'Personal', 'Foreigner', 'Organ'])
                    df['Date'] = td + ' ' + df_acc[('시간', '시간')] + ':00'
                    df['Personal'] = df_acc[('개인', '개인')]
                    df['Foreigner'] = df_acc[('외국인', '외국인')]
                    df['Organ'] = df_acc[('기관계', '기관계')]

                elif i == 2:
                    df = pd.DataFrame(columns=['Date', 'Arbitrage', 'NonArbitrage', 'TotalArbitrage'])
                    df['Date'] = td + ' ' + df_acc[('시간', '시간')] + ':00'
                    df['Arbitrage'] = df_acc[('차익거래', '순매수')]
                    df['NonArbitrage'] = df_acc[('비차익거래', '순매수')]
                    df['TotalArbitrage'] = df_acc[('전체', '순매수')]
                else:
                    ### 09:20 부터 진행 해야 함. (All NaN 여서 에러남)
                    df = pd.DataFrame(columns=['Date', 'FuturePersonal', 'FutureForeigner', 'FutureOrgan'])
                    df['Date'] = td + ' ' + df_acc[('시간', '시간')] + ':00'
                    df['FuturePersonal'] = df_acc[('개인', '개인')]
                    df['FutureForeigner'] = df_acc[('외국인', '외국인')]
                    df['FutureOrgan'] = df_acc[('기관계', '기관계')]

                ## common
                df.sort_values(by="Date", inplace=True)
                df.set_index(['Date'], drop=True, inplace=True)
                df.drop_duplicates(inplace=True)
                df.dropna(inplace=True)

                ## acc.
                df_tot_list.append(df)

            ## join  (시간 간격이 존재하여 missing 발생함)
            df_tot = df_tot_list[0].join(df_tot_list[1])
            df_tot = df_tot.join(df_tot_list[2])
            df_tot = df_tot.join(df_tot_list[3])
            df_tot.index = pd.to_datetime(df_tot.index, format="%Y:%m:%d %H:%M:%S")

            ## 중간에 발생한 missing 처리
            df_tot.interpolate(inplace=True)

            ## 현 시점이 장 시작 전이면 초기 데이터를 만들어 놓기 시작합니다.
            stu.file_save(df_tot, file_path=base_csv_path + time_path, file_name=f"kospi.csv", replace=False)
        else: # 이미 만들어진 데이터 활용

            ## 마지막 저장한 데이터 가져오기
            path = self.file_manager["system_trade"]["path"] + "kospi/csv/"
            tdate = self.trade_config["test_date"]

            if tdate == 'all':
                flist = glob(path + "*/*/*/*/*.csv")  ## 년/월/일/시간/*.csv
            else:
                flist = glob(path + f"year={tdate[0:4]}/month={tdate[4:6]}/day={tdate[6:8]}/*/*.csv")  ## 년/월/일/시간/*.csv
            ## 시간 포멧이 일정하기 때문에 str 비교로도 가장 최근을 선택할 수 있음
            nstr = ''
            for f in flist:
                if f > nstr:
                    nstr = f

            path, name = os.path.split(nstr)
            df_tot = stu.file_load(file_path=path+'/', file_name=name)
            try:
                df_tot.index = pd.to_datetime(df_tot.index, format="%Y:%m:%d %H:%M:%S")
            except:
                print(f"타임 포멧이 잘못 저장되었음. 찾아서 수정 필요 -- %Y-%m-%d %H:%M:%S. (파일위치:{path})")
                df_tot.index = pd.to_datetime(df_tot.index, format="%Y-%m-%d %H:%M:%S")

            ## 중간에 발생한 missing 처리
            df_tot.interpolate(inplace=True)

        cm = tradeStrategy('./config/config.yaml')
        if mode == "real":
            cm.display = "save"  ## 파일로 차트 저장하기
        else:
            cm.display = self.trade_config['display']  ## 파일로 차트 저장하기
        cm.path = base_image_path + time_path
        cm.name = 'kospi.png'
        df_ohlcv = cm.run('00000000', 'kospi', data=df_tot, dates=[td, td], mode='investor')

        stClose = df_ohlcv.Close.iat[0]
        endClose = df_ohlcv.Close.iat[-1]
        change =self._change_ratio(curr=endClose, prev=stClose)



        msg_kospi = dict()
        msg_kospi['image_name_path'] = base_image_path + time_path + "kospi.png" ## for telegram
        msg_kospi['msg'] = f'현재 {now_str} kospi 등락률은 {change} % 입니다.' # for 등락률

        # stu.send_telegram_message(config=self.param_init, message=msg_kospi["msg"])
        # stu.send_telegram_image(config=self.param_init, image_name_path=msg_kospi["image_name_path"])

        return msg_kospi

    ###############################################
    ##########       Note          ################
    ###############################################
    '''
# 계좌 잔고를 DataFrame 으로 반환
# Input: None (Option) rtCashFlag=True 면 예수금 총액을 반환하게 된다
# Output: DataFrame (Option) rtCashFlag=True 면 예수금 총액을 반환하게 된다
--> def get_acct_balance(rtCashFlag=False):
    
# 내 계좌의 일별 주문 체결 조회
# Input: 시작일, 종료일 (Option)지정하지 않으면 현재일
# output: DataFrame
--> def get_my_complete(sdt, edt=None, prd_code='01', zipFlag=True):

# 매수 가능(현금) 조회
# Input: None
# Output: 매수 가능 현금 액수
def get_buyable_cash(stock_code='', qry_price=0, prd_code='01'):
    
# 주문 base function
# Input: 종목코드, 주문수량, 주문가격, Buy Flag(If True, it's Buy order), order_type="00"(지정가)
# Output: HTTP Response
--> def do_order(stock_code, order_qty, order_price, prd_code="01", buy_flag=True, order_type="00"):
    
# 사자 주문. 내부적으로는 do_order 를 호출한다.
# Input: 종목코드, 주문수량, 주문가격
# Output: True, False
--> def do_sell(stock_code, order_qty, order_price, prd_code="01", order_type="00"):


# 팔자 주문. 내부적으로는 do_order 를 호출한다.
# Input: 종목코드, 주문수량, 주문가격
# Output: True, False
--> def do_buy(stock_code, order_qty, order_price, prd_code="01", order_type="00"):

# 특정 주문 취소(01)/정정(02)
# Input: 주문 번호(get_orders 를 호출하여 얻은 DataFrame 의 index  column 값이 취소 가능한 주문번호임)
#       주문점(통상 06010), 주문수량, 주문가격, 상품코드(01), 주문유형(00), 정정구분(취소-02, 정정-01)
# Output: APIResp object
--> def _do_cancel_revise(order_no, order_branch, order_qty, order_price, prd_code, order_dv, cncl_dv, qty_all_yn):

# 특정 주문 취소
-->--> def do_cancel(order_no, order_qty, order_price="01", order_branch='06010', prd_code='01', order_dv='00', cncl_dv='02',qty_all_yn="Y"):
# 특정 주문 정정
-->--> def do_revise(order_no, order_qty, order_price, order_branch='06010', prd_code='01', order_dv='00', cncl_dv='01', qty_all_yn="Y"):

# 모든 주문 취소
# Input: None
# Output: None
def do_cancel_all():



    '''



if __name__ == '__main__':

    ## 스케쥴러 설정
    # sched = BackgroundScheduler
    # sched.start()

    tr = systemTrade(mode='real')


    #######
    # kis.auth()
    # 계좌 정보
    # df = kis.get_acct_balance()

    # tr._investor_position()
    tr.run()





    ## 일별 주문 체결 조회
    # kis.get_my_complete(sdt='20220901')

    ## 잔액 조회
    # kis.get_buyable_cash()
