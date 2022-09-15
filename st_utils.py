import os
import shutil
import logging
import logging.config
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_logger():
    logger = logging.getLogger("sysT")

    # 기존 handler 존재 여부 확인 (중복 로깅 방지용)
    if len(logger.handlers) > 0 :
        return logger
    else:
        #### logging 설정
        try:
            config_file = "./config/logging.yaml"
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)


        logging.config.dictConfig(config)
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




