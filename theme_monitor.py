# -*- coding: utf-8 -*-
import pprint
import shutil
import sys
import requests
from io import BytesIO
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm
import os
import time
import gc

## html
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

## finance
import FinanceDataReader as fdr
from pykrx import stock

## plot
import matplotlib.pyplot as plt
import seaborn as sns

#internal
from news_crawler import newsCrawler
from chart_monitor import chartMonitor
import st_utils as stu




## log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e :
    print(e)

####    로그 생성    #######
logger = stu.create_logger()

## font


class themeCode :
    def __init__(self, config_file):

        ## 설정 파일을 필수적으로 한다.
        try:
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e :
            print (e)

        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")
        pprint.pprint(config)

        # 모드
        self.mode = 'ubjong'   ## theme, ubjong

        # local 설정값
        self.file_save = True
        self.image_save = True
        self.image_save_replace = False
        self.show_image1 = True
        self.show_image2 = True

        self.cutoff = [0, 20]  ## 몇번째 순위부터 몇번째까지 만 디스플레이할지 결정

        ## global 변수 선언
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.score_rule = config["scoreRule"]
        self.keys = config["keyList"]

        pass

    def run(self):
        """테마 코드를 받아 옵니다.

        Args:

        Returns:
            테마 코드를 저장한 파일을 저장합니다.

        """
        ## code 명을 위한 준비
        a = self.file_manager["stock_info"]
        stocks = pd.read_csv(a["path"]+a["name"], index_col=0)


        ## 네이버 테마 코드를 받아 옵니다.
        theme_names = []
        theme_links = []

        if self.mode == 'theme':
            pages = 7
        else:
            pages = 1

        for page in range(1,pages+1):
            print(page)
            if self.mode == 'theme':
                url = f'https://finance.naver.com/sise/theme.naver?page={page}'
            else:
                url = f"https://finance.naver.com/sise/sise_group.naver?type=upjong"
            html = requests.get(url)
            soup = BeautifulSoup(html.content.decode('euc-kr', 'replace'), "html.parser")
            if self.mode == 'theme':
                ## table 가져오기
                table = soup.find('table', {'class': 'type_1 theme'})
                aa = table.find_all('td', {"class":"col_type1"})
                for i in aa :
                    theme_names.append(i.get_text())
                    theme_links.append(f"http://finance.naver.com/{i.find('a')['href']}")
            else:
                table = soup.find('table', {'class': 'type_1'})
                aa = table.find_all('a')
                for i in aa:
                    theme_names.append(i.get_text())
                    theme_links.append(f"http://finance.naver.com/{i['href']}")


        ##테마별 종목 가져오기
        df_accum = pd.DataFrame()  ## 모든 테마의 증감률 누적 저장
        for theme_link, theme_name in zip(theme_links, theme_names):
            logger.info(f"테마명: {theme_name}, 링크: {theme_link} ")
            try:
                with requests.Session() as s:
                    html = s.get(theme_link)
                soup = BeautifulSoup(html.content.decode('euc-kr', 'replace'), "html.parser")
                tables = soup.select('table')
                table_html = str(tables)  ## 테이블 html 정보를 문자열로 변경하기
                table_df_list = pd.read_html(table_html) ## 테이블 정보 읽어 오기
                df_table = table_df_list[2]  ## 종목명, 등락률, 거래대금
                names = df_table["종목명"].to_list()
                conds = df_table["현재가"].to_list()
                names2 = []
                codes = []
                for name, cond in zip(names,conds):
                    if type(name) == str:
                        name = name.replace(" *", "")
                        tmp = stocks[stocks["Name"] == name]
                        if len(tmp) != 0 :
                            code =str(tmp["Symbol"].to_list()[0])
                            if cond > 10000 : ## 현재가가 10,000 이상인 경우에만 집계에 사용한다.
                                names2.append(name)
                                codes.append(code.zfill(6))
                            else:
                                logger.info(f"종목명({name} 의 현재가 ({cond}) 는 10,000 이하이므로 집계에서 제외합니다.")
            except Exception as e:
                print(e)

            ## 여러 코드의 데이터 가져오기
            duration = self.param_init["duration_theme"]
            end_dt = datetime.today()
            end = end_dt.strftime(self.param_init["time_format"])
            st_dt = end_dt - timedelta(days=duration)
            st = st_dt.strftime(self.param_init["time_format"])

            df_list = [fdr.DataReader(code, st, end)[['Close', 'Volume']] for code in codes]
            if len(df_list) == 0 :   ## 종목들이 전부 1만원을 넘지 못해서 하나도 남지 않는 경우
                continue
            else:
                df_data = pd.concat(df_list, axis=1)
            cols = []
            for name in names2:
                cols.append(name+'.close')
                cols.append(name+'.volume')
            df_data.columns = cols

            ## 등락률 계산하기 (거래대금 으로 가중치 주기)
            for name in names2:
                df_data[f"{name}.tr_value"] = round(df_data[f"{name}.volume"] * df_data[f"{name}.close"] / 1000000)
                df_data[f"{name}.diff"] = round((df_data[f"{name}.close"] - df_data[f"{name}.close"].shift()) / df_data[f"{name}.close"].shift()  * 100, 2)

            df_data.dropna(inplace=True)

            ## 최종 등락률
            diff_tot = []
            r_pnts = []
            for idx in df_data.index:
                sum = 0
                diffs = 0
                for name in names2:
                    sum += df_data.at[idx, f"{name}.tr_value"]
                if sum == 0 : sum = 1  ## 거래정지 경우 -> 분모가 0 이 되는 것을 방지

                r_max = 0
                for cnt, name in enumerate(names2):
                    r = round(df_data.at[idx, f"{name}.tr_value"] / sum, 2)
                    d = round(df_data.at[idx, f"{name}.diff"] * r, 2)
                    df_data[f"{name}.ratio"] = r
                    if r > r_max :
                        r_max = r
                        r_pnt = cnt

                    if d > 0:
                        diffs += d
                    else: ## 음수면 두배 가중치 (누적값 원상 보귀 여부 확인 용)
                        diffs += d * 2
                r_pnts.append(r_pnt)
                diff_tot.append(round(diffs,2))


            df_data["avg_range"] = diff_tot
            df_data["accum_avg_range"] = df_data["avg_range"].cumsum()
            df_data["weight_point"] = r_pnts   ## 등락률에 가장 많은 영향을 준 코드 변화 확인
            df_data.dropna(inplace=True)

            ## 한곳에 모으기
            df_accum = pd.concat([df_accum, df_data["accum_avg_range"]], axis=1)
            df_accum.rename(columns={'accum_avg_range': theme_name}, inplace=True)

            if self.show_image1 == True:
                ## 테마별 등락률 그래프 만들기 (종목별 은 아님)
                sns.set(rc={'figure.figsize': (20, 10), 'font.family': 'AppleGothic'})
                p = sns.lineplot(x=df_data.index, y="accum_avg_range", data=df_data, label='증감률 누적')
                p = sns.lineplot(x=df_data.index, y="weight_point", data=df_data, marker=True, label='종목 변경값')
                title = f"테마/업종 이름: {theme_name}"
                p.set_title(title)
                p.set(xlabel='Date', ylabel='증감률 누적')

                plt.xticks(rotation=-45)
                plt.rc('axes', unicode_minus=False)  ## 마이너스 표시 깨짐 해결
                plt.show()


            disp_cols = []
            df_data2 = df_data.copy()
            for col in names2:
                newcol = f"{col}.close"
                disp_cols.append(newcol)
                if df_data2.loc[:,newcol][-1] > 1000000:
                    df_data2[newcol] = df_data2[newcol] / 100  ## 만원 단위로 맞추기 위함
                elif df_data2.loc[:,newcol][-1] > 100000:
                    df_data2[newcol] = df_data2[newcol] / 10  ## 만원 단위로 맞추기 위함

            sns.set(rc={'figure.figsize': (20, 15), 'font.family': 'AppleGothic'})
            title = f"테마 이름: {theme_name}"
            df_data2[disp_cols].plot(title=title+" - 종목별 영향 비율")

            ## 텍스트 삽입
            for posx in df_data2.index:
                for col in names2 :
                    newcol = f"{col}.close"
                    posy = df_data2.at[posx, newcol]

                    newcol2 = f"{col}.ratio"
                    text = df_data2.at[posx, newcol2]

                    newcol3 = f"{col}.diff"
                    text2 = df_data2.at[posx, newcol3]

                    plt.text(posx, posy, text, horizontalalignment='right',
                             verticalalignment='bottom',
                             size='small', color='black', weight='semibold' )
                    if text2 > 0 :
                        color = 'blue'
                    else:
                        color = 'red'
                    plt.text(posx, posy, text2, horizontalalignment='left',
                             verticalalignment='top',
                             size='x-small', color=color, weight='semibold' )

            plt.yscale("log")
            plt.legend(loc='upper left')
            plt.tight_layout()
            if self.image_save == True:
                file_name = f"{theme_name}.png"
                file_name = file_name.replace("/","_")
                if self.mode == 'theme':
                    file_path = self.file_manager["stock_theme"]["path"] + f"theme/img_codes/{st}_{end}/"
                else:
                    file_path = self.file_manager["stock_theme"]["path"] + f"upjong/img_codes/{st}_{end}/"
                ## 폴더 생성
                try:
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    else:
                        if self.image_save_replace == True:
                            shutil.rmtree(file_path)
                            os.makedirs(file_path)
                        else:
                            pass
                except Exception as e:
                    raise e
                plt.savefig(file_path+file_name, dpi=300)
                if self.show_image2 == True:
                    plt.show()
            else:
                plt.show()

            ## 파일저장
            if self.file_save == True:
                file_name = f"{theme_name}.csv"
                file_name = file_name.replace("/","_")
                if self.mode == 'theme':
                    file_path = self.file_manager["stock_theme"]["path"] + f"theme/raw/{st}_{end}/"
                else:
                    file_path = self.file_manager["stock_theme"]["path"] + f"upjong/raw/{st}_{end}/"

                ## 파일로 저장 합니다.
                stu.file_save(df_data, file_path, file_name, replace=False)

            if theme_name == 'PCB(FPCB 등)':  ## for test
                keys = theme_name
                # keys = '유신'
                nc = newsCrawler()
                nc.search_keyword(keys)
                print("SSH")
            # time.sleep(5)

            print(f"테마/업종 ({theme_name}) 종목 리스트: {names2} ")
            print(f"테마/업종 ({theme_name}) 처리를 완료 하였습니다.")

            ## 초기화
            plt.close()
            del [[df_data, df_data2]]
            gc.collect()
            df_data = pd.DataFrame()
            df_data2 = pd.DataFrame()

        ## 증가율이 높은 테마 찾기
        df_accum2 = df_accum.loc[:, ~df_accum.T.duplicated()]
        theme_ord = []
        for col in df_accum2.columns.to_list():
            val= df_accum2[col][-1]
            theme_ord.append((col, val))

        ## sortedj
        theme_ord2 = sorted(theme_ord, key=lambda x: x[1], reverse=True)

        ## 이미지 생성
        cutoff = [0, 20]
        name_list =[]
        for name, score in theme_ord2[cutoff[0]:cutoff[1]]:
            name_list.append(name)

        name_list2 = []
        for name, score in theme_ord2:  ## 파일 저장을 위해 컬럼 순서 변경용
            name_list2.append(name)

        sns.set(rc={'figure.figsize': (20, 15), 'font.family': 'AppleGothic'})
        df_accum2[name_list].plot(title=f" 누적 증감률 상위 {cutoff[0]} ~ {cutoff[1]} 테마/업종")

        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.rc('axes', unicode_minus=False)  ## 마이너스 표시 깨짐 해결

        ## 테마/업종 전체에서 비교하는 이미지 파일 저장 (필수)
        if self.mode == 'theme':
            file_name = f"테마별_증강순위_{st}_{end}.png"
        else:
            file_name = f"업종별_증강순위_{st}_{end}.png"
        file_path = self.file_manager["stock_theme"]["path"]
        ## 폴더 생성
        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            else:
                if self.image_save_replace == True:
                    shutil.rmtree(file_path)
                    os.makedirs(file_path)
                else:
                    pass
        except Exception as e:
            raise e
        plt.savefig(file_path + file_name, dpi=300)
        plt.show()

        ## 파일로 저장 합니다.
        if self.mode == 'theme':
            file_name = f"테마별_증강순위_{st}_{end}.csv"
        else:
            file_name = f"업종별_증강순위_{st}_{end}.csv"
        file_path = self.file_manager["stock_theme"]["path"]
        stu.file_save(df_accum2[name_list2], file_path, file_name, replace=False)

    def search_market_leader(self):
        '''
        주도주 검색:
         - 시장 주도주는 시장의 뭉칫돈이 쏠리는 종목을 뜻함. 단기 이슈나 소재로 인해 급등하는 종목을 제외

         - 환경 : 변동성이 크고 시장의 불확실성이 확대 될때 사용. 하락장에서 영향 줌

         - 우선순위 1. 하락장 에서 하락 비율이 낮음
         - 우선순위 2. 시가총액이 커서 지수에 영향줌.
         - 우선순위 3. (opt.) 하락장에 52주 신고가 달성

         - 예외 1. 계절적 요인으로 인한 하락, 특정 돌발 이슗 인한 인한 하락 은 구분해서 대응

        :return:
        '''



        # duration = self.param_init['duration']
        duration = 60
        end_dt = datetime.today()
        end = end_dt.strftime(self.param_init["time_format"])
        st_dt = end_dt - timedelta(days=duration)
        st = st_dt.strftime(self.param_init["time_format"])

        df_all = stock.get_market_price_change(st.replace("-", ""), end.replace("-", ""))
        df_all.rename(columns={'종목명': 'Name',
                               '시가': 'Open',
                               '종가': 'Close',
                               '거래량': 'Volume',
                               '등락률': 'Change',
                               '변동폭': 'ChangeRatio',
                               '거래대금': 'VolumeCost'
                               }, inplace=True)
        ## 거래대금으로 상위 200개만 선정하고 이중에서 등락률 순으로 정렬
        df_vols = df_all.sort_values(by='VolumeCost', ascending=False)
        df_all = df_vols.head(200).sort_values(by='Change', ascending=False)

        ## 제무재표 스코어 불러오기
        path = self.file_manager["selected_items"]["path"]
        files = os.listdir(path)
        df_finance = pd.read_csv(path+files[0])  ## 파일이 하나밖에 없음

        config_file = './config/config.yaml'
        chart = chartMonitor(config_file)
        for cnt, (code, name) in enumerate(zip(df_all.index, df_all.Name)):
            try:
                fi_score = df_finance[df_finance.Name == name]['total_score'].values[0]
                print(f"code: {code}, name: {name}, score : {fi_score}")
            except:
                print(f"code: {code}, name: {name}, score: None -- 스코어를 찾을 수 없습니다.")
            df = chart.run(code, date=[st, end], data='none')

            if cnt == 5:
                break

        logger.info("Finish!")

if __name__ == "__main__":
    config_file = './config/config.yaml'
    theme = themeCode(config_file)
    theme.search_market_leader()
    # theme.run()