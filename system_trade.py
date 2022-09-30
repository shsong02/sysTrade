import yaml
import requests
import json
import pandas as pd
import os
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

_DEBUG = True

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
        self.score_rule = config["scoreRule"]
        self.keys = config["keyList"]



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
        ## 보안인증키 받기
        url = f"{self.url_base}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {"grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret}
        res = requests.post(url, headers=headers, data=json.dumps(body))
        self.access_token = res.json()["access_token"]

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




if __name__ == '__main__':

    ## 스케쥴러 설정



    code = "064350"

    ## 텔레그램 연동 테스트
    # stu.send_telegram_message(f"test code: {code}")
    # image = './data/theme/market_leader/20201006_20220926/000060_20201006_20220926.png'
    # stu.send_telegram_image(image_name_path=image)

    tr = systemTrade(mode='real')

    ## 스케쥴링 하기 위해서 시간 간격을 생성
    interval = 10  # minutes  (max 30 min.)

    st = timedelta(hours=9)
    now_time = datetime.now().time().strftime("%H%M%S")
    time_list = []
    for i in range(int(60 / interval) * 7):  ## 1분 간격
        if i == 0:
            t = st
        else:
            t = t + timedelta(minutes=interval)
        str_t = f"{t}".replace(':', '').zfill(6)
        if str_t < now_time:
            time_list.append(str_t)
        else:
            time_list.append(now_time)
            break

    times = [0,0]  ## st, end
    df_acc = pd.DataFrame()

    ## chart 로 표시하기
    config_file = './config/config.yaml'
    cm = tradeStrategy(config_file)

    for idx, t in enumerate(time_list):
        times.append(t)
        times.pop(0)

        if idx == 0:
            pass
        else:
            df = tr.get_curr_min_price(code, times)
            df2 =tr.get_curr_min_chegyeol(code, times)

            ## 최종시점 데이터만 가져올수 있음. 실시간으로 가져오도록 해야 함
            df3 = tr.get_curr_member(code, times)

            ## 실시간 미지원으로 확인되어 사용하지 않음..(추후 까지)
            # df4 = tr.get_curr_investor(code, times)

            ## time index 로 join 함
            df_fin = df.join(df2)
            df_fin = df_fin.join(df3)

            # 타입 변환
            cols = ["Low", "High", "Close", "Volume", "Open", "ChegyeolStr"]
            df_fin[cols] = df_fin[cols].apply(pd.to_numeric)

            if len(df_acc) == 0:
                df_acc = df_fin
            else:
                df_acc = pd.concat([df_acc, df_fin])

            if idx > 40 :  ## 시작
                df_acc = df_acc.fillna(0)  ## missing 처리
                df_acc = df_acc.loc[~df_acc.index.duplicated(keep='first')] # index 중복 제거

                name = stock.get_market_ticker_name(code)
                today = datetime.now().date().strftime("%Y%m%d")
                cm.run(code, name, data=df_acc, dates=[today, today], mode='realtime')


    #######
    kis.auth()

    ## 계좌 정보
    # df = kis.get_acct_balance()

    ## 일별 주문 체결 조회
    # kis.get_my_complete(sdt='20220901')

    ## 잔액 조회
    # kis.get_buyable_cash()


    pass
