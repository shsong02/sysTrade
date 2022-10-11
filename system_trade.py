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
        self.trade_config = config["tradeStock"]

        ##### 초기 변수 설정
        ## 스케쥴링 하기 위해서 시간 간격을 생성
        self.mon_intv = self.trade_config["scheduler"]["interval"]  # minutes  (max 30 min.)


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
            "fid_cond_mrkt_div_code": "J",
            "fid_etc_cls_code": "",
            "fid_input_iscd": code,
            "fid_input_hour_1": '',  # 아래 에서 삽입
            "fid_pw_data_incu_yn": "N"
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
            params['fid_input_hour_1'] = time
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
            str.append(df_part['ChegyeolStr'].iat[-1])
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

        ## 실제 시간 확인하여 웨이팅하기
        if self.trade_config["scheduler"]["mode"] == "real":
            while True:
                real_time = datetime.now().time().strftime("%H%M%S")
                if real_time >= "091000":
                    break

                print(f"현재시간 ({real_time}) 이 목표시간 (091000) 에 도달하지 못했기 때문에 기다립니다.")
                time.sleep(60)

        ## Monitoring 종목 가져오기
        path = self.file_manager["monitor_stocks"]["path"]
        flist = glob(path + "*/*/*/*/*.csv")  ## 년/월/일/시간/*.csv

        ## 시간 포멧이 일정하기 때문에 str 비교로도 가장 최근을 선택할 수 있음
        nstr = ''
        for f in flist:
            if f > nstr:
                # print(f)
                nstr = f

        ## 탐색할 종목명 확인
        df = pd.read_csv(nstr)
        codes0 = df.Code.to_list()
        codes = [str(x).zfill(6) for x in codes0]
        df_dict = dict()
        for code in codes:
            df_dict[code] = dict()
            df_dict[code]['data'] = pd.DataFrame()

        ## KIS API 자주 사용 시, locking 됨. 테스트 버전은 최대한 재사용 하여 접속양을 줄인다.
        base_path = self.file_manager["system_trade"]["path"]

        if _DEBUG != True :
            ## 현 시점이 장 시작 전이면 초기 데이터를 만들어 놓기 시작합니다.
            now_str = datetime.now().time().strftime("%H%M%S")
            now = datetime.now()
            time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"

            if '090010' <= now_str:
                for code in codes:
                    times = ["090000", now_str]
                    df_fin = self._make_curr_df(code, times)
                    df_dict[code]['data'] = df_fin
                    logger.info(f"장 시작 9시 이후에 시작되었기 때문에 시간({times} 에 대한 종목({code}) 실시간 정보를 미리 준비합니다.)")

                    name = stock.get_market_ticker_name(code)  ## 기본으로 입력해줘야 함
                    stu.file_save(df_fin, file_path=base_path+time_path, file_name=f"{name}_{code}.csv", replace=False)
        else:  ## 최신 데이터로 로드해오기.
            if _DEBUG_NEWDATA:
                now = datetime.now()
                time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"

                for code in codes:
                    times = ["090000", "153000"]
                    df_fin = self._make_curr_df(code, times)
                    df_dict[code]['data'] = df_fin
                    logger.info(f"장 시작 9시 이후에 시작되었기 때문에 시간({times} 에 대한 종목({code}) 실시간 정보를 미리 준비합니다.)")
                    name = stock.get_market_ticker_name(code)  ## 기본으로 입력해줘야 함
                    stu.file_save(df_fin, file_path=base_path + time_path, file_name=f"{name}_{code}.csv", replace=False)
            else:
                dlist = glob(base_path + "*/*/*/*/")  ## 년/월/일/시간/*.csv
                dfin = ''
                for d in dlist:
                    if d > dfin:
                        dfin = d  ## 최신 폴더 찾아내기

                print(f"디버그 모드 사용 중입니다. KIS API 사용 최쇠화를 위해 최신 current 데이터를 csv 로 부터 읽어 옵니다.(경로: {dfin})")
                flist = os.listdir(dfin)

                for code in codes:
                    chk1 = False
                    for f in flist:
                        if code in f:  ## 파일명이 종목 코드 이므로,
                            df =stu.file_load(file_path=dfin, file_name=f, type='csv')
                            df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
                            df_dict[code]["data"] = df
                            chk1 = True
                            break
                    if not chk1:
                        raise ValueError(f"[Debug 모드] 종목코드({code}) 관련 Dataframe 을 찾을 수 없습니다. (경로:{dfin})")

        ## chart 로 표시하기
        cm = tradeStrategy('./config/config.yaml')
        cm.display = 'save'  ## 파일로 차트 저장하기
        day_tlist = self._make_time_list(["090000", "153000"], interval=self.mon_intv)
        times = [0, 0]  ## st, end
        now_time = datetime.now().time().strftime("%H%M%S")
        chart_base_path = self.file_manager["system_trade"]["path"] + "chart/"
        today = datetime.now().date().strftime("%Y%m%d")


        if (now_time > '153000'):
            ## 저장할 경로 확인
            now = datetime.now()
            time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"
            image_path = chart_base_path + time_path
            cm.path = image_path

            for code in codes:
                df_acc = df_dict[code]['data']  ## 이미 앞에서 모두 준비된 상태
                df_acc = df_acc.fillna(0)  ## missing 처리
                df_acc = df_acc.loc[~df_acc.index.duplicated(keep='first')]  # index 중복 제거
                name = stock.get_market_ticker_name(code)  ## 기본으로 입력해줘야 함
                image_name = f"{name}_{code}.png"
                cm.name = image_name
                print(f"차트 (time: {str(now)}) 를 생성합니다. (파일명: {image_name})")
                df_ohlcv = cm.run(code, name, data=df_acc, dates=[today, today], mode='realtime')

                ## 파일로도 저장
                stu.file_save(df_acc, file_path=base_path + time_path, file_name=f"{code}.csv", replace=False)
        else:
            for idx, t in enumerate(day_tlist[:-1]):
                times.append(t)
                times.pop(0)

                ## 저장할 경로 확인
                now = datetime.today()
                time_path = f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/time={now.strftime('%H%M')}/"

                image_path = chart_base_path + time_path
                cm.path = image_path

                if t < now_time:
                    ## 시작 시간이 장 시작 후 일 경우 처리
                    pass
                else:
                    if idx == 0:
                        ## 0900 는 실행 하지 않음
                        pass
                    else:

                        for code in codes:

                            df_fin = self._make_curr_df(code, times)
                            df_acc = df_dict[code]['data']

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
                                try:
                                    print(f"차트 (time: {str(now)}) 를 생성합니다. (파일명: {image_name})")
                                    df_ohlcv = cm.run(code, name, data=df_acc, dates=[today, today], mode='realtime')

                                    ## 다음 사용을 위해 데이터 저장
                                    df_dict[code]['data'] = df_acc

                                    ## csv 파일로도 저장
                                    stu.file_save(df_ohlcv, file_path=base_path + time_path,
                                                  file_name=f"{name}_{code}.csv", replace=False)
                                except Exception as e :
                                    print(e)

                                # 매수 조건이 발생하였는지 확인 (텔레그림..발생)
                                ## 조건1: 채결 강도가 100 을 넘었을 경우, 차트로 알려주기
                                cstrth = float(df_ohlcv.tail(1).ChegyeolStr)
                                close0 = int(df_ohlcv.head(1).Close)
                                close1 = int(df_ohlcv.tail(1).Close)
                                change = self._change_ratio(close1, close0)

                                df_chg = self._bollinger_chegyeol(df_ohlcv) # param 은 기본으로 사용
                                ch_sig = df_chg.tail(self.mon_intv).bolBuy_chegyeol.any()
                                # ch_sig = df_chegyeol.bolBuy_chegyeol.any()
                                # print(name, ch_sig, df_chegyeol.bol_upper_chegyeol.to_list() )
                                if ch_sig :
                                    df_chg = df_chg.tail(self.mon_intv)
                                    idx_list = df_chg[df_chg.bolBuy_chegyeol == True].index.to_list()
                                    del_str = datetime.now().date().strftime("%Y-%m-%d ")
                                    idx_list2 = [str(x) for x in idx_list]
                                    idx_list2 = [x.replace(del_str, '') for x in idx_list2]
                                    _msg = f"현재 ({str(now)}) 종목 ({name}) 의 채결 강도가 급하게 상승한 구간이 최근 10분 내 존재합니다. (시점: {idx_list2}) "
                                    print(_msg)
                                    stu.send_telegram_message(config=self.param_init, message=_msg)
                                    stu.send_telegram_message(config=self.param_init, message=f"현재 등락률은 {change}% 입니다.(장전 갭상은 반영 못함)")
                                    stu.send_telegram_image(config=self.param_init, image_name_path=image_path+image_name)
                            else:
                                print("SSH !!!")


                        ## 실제 시간 확인하여 웨이팅하기
                        while True:
                            real_time = datetime.now().time().strftime("%H%M%S")
                            if real_time >= t:
                                break

                            print(f"현재시간 ({real_time}) 이 목표시간 ({t}) 에 도달하지 못했기 때문에 기다립니다.")
                            time.sleep(30)

        pass

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


if __name__ == '__main__':

    ## 스케쥴러 설정
    # sched = BackgroundScheduler
    # sched.start()

    tr = systemTrade(mode='real')
    tr.run()




    #######
    kis.auth()

    ## 계좌 정보
    # df = kis.get_acct_balance()

    ## 일별 주문 체결 조회
    # kis.get_my_complete(sdt='20220901')

    ## 잔액 조회
    # kis.get_buyable_cash()
