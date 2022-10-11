import os
import shutil
import logging
import logging.config
import yaml
import numpy as np
import pandas as pd
import telegram
from datetime import datetime, timedelta


###### 설정 파일에서 정보를 읽어서 global 변수로 선언한다.


#### logging 설정
try:
    config_file = "./config/logging.yaml"
    with open(config_file) as f:
        log_config = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print(e)

#############################################

def create_logger():
    logger = logging.getLogger("sysT")

    # 기존 handler 존재 여부 확인 (중복 로깅 방지용)
    if len(logger.handlers) > 0 :
        return logger
    else:
        logging.config.dictConfig(log_config)
        # 핸들러 설정된 인스턴스 다시 생성
        logger = logging.getLogger()

        return logger

def file_save(data, file_path, file_name, type='csv', replace=False):
    ## type check
    try :
        if not type in ['csv', 'ndarray']:
            raise
    except:
        print(f"입력하신 type=({type}) 은 지원하지 않습니다. csv, ndarray 중에하나 선택해 주세요.")
        os.exit()

    ## 폴더 생성
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        else:
            if replace == True:
                shutil.rmtree(file_path)
                os.makedirs(file_path)
            else:
                pass
    except Exception as e:
        raise e

    ## 파일로 저장
    if type == 'csv':
        data.to_csv(file_path + file_name, encoding='utf-8')
    elif type == 'ndarray':
        np.save(file_path + file_name, data)

def file_load(file_path, file_name, type='csv'):
    ## type check
    try :
        if not type in ['csv', 'ndarray']:
            raise
    except:
        print(f"입력하신 type=({type}) 은 지원하지 않습니다. csv, ndarray 중에하나 선택해 주세요.")
        os.exit()


    try:
        if type == 'csv':
            data = pd.read_csv(file_path+file_name, index_col=0)
        elif type == 'ndarray':
            data = np.load(file_path+file_name, allow_pickle=True)
    except Exception as e :
        print(e)

    return data


def period_to_str(period, format="%Y-%m-%d"):
    end_dt = datetime.today()
    end = end_dt.strftime(format)
    st_dt = end_dt - timedelta(days=period)
    st = st_dt.strftime(format)
    return [st, end]


def load_theme_list(path, mode='theme', format="%Y-%m-%d"):
    files = os.listdir(path)
    theme_files = []
    upjong_files = []
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            ## 최근 파일 분리
            name = os.path.splitext(file)[0]
            fmode = name.split("_")[0]
            if fmode == 'theme':
                theme_files.append(file)
            elif fmode == 'upjong':
                upjong_files.append(file)

    ## 테마 파일중 최신 파일을 선택합니다.
    if mode == 'theme':
        target_files = theme_files
    else:
        target_files = upjong_files
    if not len(target_files) == 0:
        for cnt, file in enumerate(target_files):
            name = os.path.splitext(file)[0]
            time = name.split("_")[3]
            dtime = datetime.strptime(time, format)

            if cnt == 0:
                last_time = dtime
                last_file = file
            else:
                if last_time <= dtime:
                    last_time = dtime
                    last_file = file
        df_theme = pd.read_csv(path + last_file, index_col=0)
        print(f"테마/업종별(현재모드:{mode}) 종목리스트 파일을 읽어 dataframe 을 생성합니다. (경로: {path+last_file}")
    else:
        print(f"테마/업종별(현재모드:{mode}) 종목리스트 파일을 찾을 수 없습니다. (경로: {path}")
        df_theme = pd.DataFrame()

    return df_theme


# -----------------------------------------------------------------------------
# - Name : send_telegram_msg
# - Desc : 텔레그램 메세지 전송
# - Input
#   1) message : 메세지
# -----------------------------------------------------------------------------
def send_telegram_message(config, message):
    token  = config["telegram_token"]
    id = config["telegram_id"]

    try:
        # 텔레그램 메세지 발송
        bot = telegram.Bot(token)
        res = bot.sendMessage(chat_id=id, text=message)

        return res
    # ----------------------------------------
    # 모든 함수의 공통 부분(Exception 처리)
    # ----------------------------------------
    except Exception:
        raise

def send_telegram_image(image_name_path, config):
    token  = config["telegram_token"]
    id = config["telegram_id"]

    try:
        # 텔레그램 메세지 발송
        bot = telegram.Bot(token)
        res = bot.send_photo(chat_id=id, photo=open(image_name_path, 'rb'))

        return res

    # ----------------------------------------
    # 모든 함수의 공통 부분(Exception 처리)
    # ----------------------------------------
    except Exception:
        raise
