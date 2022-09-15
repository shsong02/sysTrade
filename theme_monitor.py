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

        ## global 변수 선언
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.score_rule = config["scoreRule"]
        self.keys = config["keyList"]

        self.marget_leader_params = config["searchTheme"]["market_leader"]["params"]
        self.market_leader_config = config["searchTheme"]["market_leader"]["config"]

        self.theme_upjong_params = config["searchTheme"]["theme_upjong"]["params"]
        self.theme_upjong_config = config["searchTheme"]["theme_upjong"]["config"]

        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

    def search_theme_upjong(self, mode='theme'):
        """네이버 주식에서 테마/업종을 검색하고, 속한 종목들의 상태를 확인합니다.
        테마 리스트를 저장하여 주도주 검색 시, 조건으로 활용합니다.

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

        if mode == 'theme':
            pages = 7  ## 네이버 주식에서 테마 페이지는 7 개 임. 추후 증가하더라도 뒤쪽순위는 중요도 떨어짐.
        else:
            pages = 1

        logger.info(f"테마/업종(선택:{mode} 에 대한 크롤링을 시작합니다.")
        for page in range(1,pages+1):
            if mode == 'theme':
                url = f'https://finance.naver.com/sise/theme.naver?page={page}'
            else:
                url = f"https://finance.naver.com/sise/sise_group.naver?type=upjong"
            logger.info(f"크롤링 웹 페이지: {url}")
            html = requests.get(url)
            soup = BeautifulSoup(html.content.decode('euc-kr', 'replace'), "html.parser")
            if mode == 'theme':
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

        # summary df 에 저장될 데이터 리스트
        theme_accum = []
        name_accum = []
        code_accum = []
        change_accum =  []
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
                            if cond > self.theme_upjong_params["threshold_close"] : ## 현재가가 10,000 이상인 경우에만 집계에 사용한다.
                                names2.append(name)
                                codes.append(code.zfill(6))

                                theme_accum.append(theme_name)
                                name_accum.append(name) # 최종 summary 용
                                code_accum.append(code.zfill(6)) # 최종 summary 용
                            else:
                                logger.info(f"종목명({name} 의 현재가 ({cond}) 는 10,000 이하이므로 집계에서 제외합니다.")
            except Exception as e:
                print(e)

            ## 여러 코드의 데이터 가져오기
            duration = self.theme_upjong_params["period"]
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
                cols.append(name+'.Close')
                cols.append(name+'.Volume')
            df_data.columns = cols

            ## 등락률 계산하기 (거래대금 으로 가중치 주기)
            for name in names2:
                df_data[f"{name}.VolumeCost"] = round(df_data[f"{name}.Volume"] * df_data[f"{name}.Close"] / 1000000)
                df_data[f"{name}.Change"] = round((df_data[f"{name}.Close"] - df_data[f"{name}.Close"].shift()) / df_data[f"{name}.Close"].shift()  * 100, 2)

                st_close = df_data.iloc[0][f"{name}.Close"]
                end_close = df_data.iloc[-1][f"{name}.Close"]
                change = round((end_close - st_close)/end_close *100, 2)
                change_accum.append(change) # 최종 summary 용

            df_data.dropna(inplace=True)

            ## 최종 등락률
            diff_tot = []
            r_pnts = []
            for idx in df_data.index:
                sum = 0
                diffs = 0
                for name in names2:
                    sum += df_data.at[idx, f"{name}.VolumeCost"]
                if sum == 0 : sum = 1  ## 거래정지 경우 -> 분모가 0 이 되는 것을 방지

                r_max = 0
                for cnt, name in enumerate(names2):
                    r = round(df_data.at[idx, f"{name}.VolumeCost"] / sum, 2)
                    d = round(df_data.at[idx, f"{name}.Change"] * r, 2)
                    df_data[f"{name}.Ratio"] = r
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

            ## 테마/업종 단위 일자별 등락률
            df_accum = pd.concat([df_accum, df_data["accum_avg_range"]], axis=1)
            df_accum.rename(columns={'accum_avg_range': theme_name}, inplace=True)

            # 최종 summary 용
            theme_accum.append(theme_name)
            code_accum.append('ALL')
            name_accum.append('ALL')
            change_accum.append(df_data.iloc[-1]["accum_avg_range"])

            if self.theme_upjong_config["display_theme_chart"] == True:
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
                newcol = f"{col}.Close"
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
                    newcol = f"{col}.Close"
                    posy = df_data2.at[posx, newcol]

                    newcol2 = f"{col}.Ratio"
                    text = df_data2.at[posx, newcol2]

                    newcol3 = f"{col}.Change"
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
            if self.theme_upjong_config["save_stock_chart"] == True:
                file_name = f"{theme_name}.png"
                file_name = file_name.replace("/","_")
                if mode == 'theme':
                    file_path = self.file_manager["stock_theme"]["path"] + f"theme/img_codes/{st}_{end}/"
                else:
                    file_path = self.file_manager["stock_theme"]["path"] + f"upjong/img_codes/{st}_{end}/"
                ## 폴더 생성
                try:
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                except Exception as e:
                    raise e
                plt.savefig(file_path+file_name, dpi=300)
                if self.theme_upjong_config["display_stock_chart"] == True:
                    plt.show()
            else:
                plt.show()

            ## 파일저장
            if self.theme_upjong_config["save_stock_data"] == True:
                file_name = f"{theme_name}.csv"
                file_name = file_name.replace("/","_")
                if mode == 'theme':
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

        ### summary 용 df 생성 ####
        cols = ["Theme", "Name", "Code", "Change"]
        df_summary = pd.DataFrame(columns=cols)
        values = [theme_accum, name_accum, code_accum, change_accum]

        for col, val in zip(cols, values):
            df_summary[col] =  val

        file_name = f"{mode}_stock_list_{st}_{end}.csv"
        file_path = self.file_manager["stock_theme"]["path"]
        ## 폴더 생성
        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        except Exception as e:
            raise e
        df_summary.to_csv(file_path+file_name)


        #########################
        ## 증가율이 높은 테마 찾기
        #########################
        df_accum2 = df_accum.loc[:, ~df_accum.T.duplicated()]
        theme_ord = []
        for col in df_accum2.columns.to_list():
            val= df_accum2[col][-1]
            theme_ord.append((col, val))

        ## sortedj
        theme_ord2 = sorted(theme_ord, key=lambda x: x[1], reverse=True)

        ## 이미지 생성
        cutoff = self.theme_upjong_params["theme_summary_cutoff"]
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
        file_name = f"{mode}_summary_{st}_{end}.png"
        file_path = self.file_manager["stock_theme"]["path"] + f'{mode}/summary/{st}_{end}/'
        ## 폴더 생성
        try:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        except Exception as e:
            raise e
        plt.savefig(file_path + file_name, dpi=300)
        plt.show()

        ## 파일로 저장 합니다.
        file_name = f"{mode}_summary_{st}_{end}.csv"
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
        [st, end] = stu.period_to_str(self.market_leader_config["data_period"])

        ## Search 할 종몬 준비
        df_all = stock.get_market_price_change(st.replace("-", ""), end.replace("-", ""), market='ALL')
        endt = datetime.strptime(end, "%Y-%m-%d")
        for i in range(7):  ## 시장이 쉬는 구간을 피해서 불러오기 위함
            date = endt - timedelta(days=i)
            end_pre = date.strftime(format="%Y%m%d")
            try:
                df_all2 = stock.get_exhaustion_rates_of_foreign_investment(end_pre, market='ALL')['상장주식수']
                break
            except:
                continue

        df_all = df_all.join(df_all2)
        df_all["회전율"] = round(df_all["거래량"]  / df_all["상장주식수"] * 100, 2)

        df_all.rename(columns={'종목명': 'Name',
                               '시가': 'Open',
                               '종가': 'Close',
                               '거래량': 'Volume',
                               '등락률': 'Change',
                               '변동폭': 'ChangeRatio',
                               '거래대금': 'VolumeCost',
                               '상장주식수': 'TotalVolume',
                               '회전율': 'VolumeTurnOver',
                               }, inplace=True)
        path = self.file_manager["selected_items"]["path"]
        files = os.listdir(path)
        ## 제무재표 스코어 불러오기
        df_ref = pd.read_csv(path+files[0])  ## 파일이 하나밖에 없음
        df_ref['code']= df_ref['code'].apply(lambda x: str(x).zfill(6))
        df_ref.set_index(keys=['Name'], inplace=True)


        # 1) 조건1: 종가 최소 값 으로 cut
        thrd_close = self.marget_leader_params["threshold_close"]
        df_close = df_all[df_all.Close > thrd_close]
        df_close.set_index(keys=['Name'], inplace=True)
        logger.info(f"종가 ({thrd_close} 원) 보다 낮은 종믁 제거로,  KOSPI+KOSDAQ ({len(df_all)}) 개 중에 ({len(df_close)}))개 를 선정합니다.")

        # 2) 조건2:  finance score 이상인 경우에만 해당
        df_finance = df_close.join(df_ref, how='left')
        df_finance.dropna(subset=['total_score','code'], inplace=True)
        thrd_score = self.marget_leader_params["threshold_finance_score"]
        df_finance2 = df_finance[df_finance.total_score > thrd_score]
        logger.info(f"재무제표 총점 ({thrd_score}) 기준으로 종목 ({len(df_close)})개 중에 ({len(df_finance2)})개 를 선정합니다.")

        # 3) 조건3:  등락률이 코스피보다 높은 경우에만 해당
        ## 코스피 등락률 보다 높이 상승한 종목 추출
        krx = stock.get_index_price_change(st, end, "KOSPI")
        change_thrd = krx.at["코스피 200", "등락률"]
        if change_thrd < 0 :   #
            change_thrd = 1
        else:
            change_thrd += 1
        df_change = df_finance2[df_finance2.Change > change_thrd]
        logger.info(f"코스피 변동률(최소 +1) ({change_thrd}) 보다 높게 상승한 기준으로 종목 ({len(df_finance2)})개 중에 ({len(df_change)})개 를 선정합니다.")

        # 4) 조건4: 등락률이 너무 높지 않음 (이미 날라간 종목은 관심 없음)
        thrd_maxchg = self.marget_leader_params["threshold_max_change"]
        df_change2 = df_change[df_change.Change < thrd_maxchg]
        logger.info(f"변동률 ({thrd_maxchg}) 보다 낮게 상승 (급한성장은 관심 논외)한 기준으로 종목 ({len(df_change)})개 중에 ({len(df_change2)})개 를 선정합니다.")



        # 5) 조건5: 회전율 상위 순위 순으로 cut
        thrd_volumecost = self.marget_leader_params["volumecost_code_count"]
        df_vcost = df_change2.sort_values(by='VolumeTurnOver', ascending=False).head(thrd_volumecost)
        logger.info(f"거래 회전률이 많은 순으로 상위 ({thrd_volumecost})개 목표로, 종목 ({len(df_change2)})개 중에 ({len(df_vcost)})개 를 선정합니다.\n\n")


        # for 문을 위한 설정 정보들
        config_file = './config/config.yaml'
        chart = chartMonitor(config_file)
        dates = stu.period_to_str(self.market_leader_config["chart_period"], format="%Y%m%d")
        nc = newsCrawler()

        df_fin = df_vcost
        ## 테스트 용
        # df_fin = df_vcost.sort_values(by='Change', ascending=False)
        # df_fin = df_vcost.sort_values(by='VolumeCost', ascending=False)

        ## for test
        df_news = nc.search_keyword('석경에이티')

        for cnt, (code, name) in enumerate(zip(df_fin.code, df_fin.index)):
            if cnt >20:
                print("SSH TEST")

            # logger.info(f"종목 ({name} - {code}) 의 뉴스정보 수집을 시작합니다. ")
            # df_news = nc.search_keyword(name)

            try:
                fi_score = df_fin[df_fin.index == name]['total_score'].values[0]
                vol_cost = df_fin[df_fin.index == name]['VolumeCost'].values[0]
                vol_cost = round(vol_cost / 100000000)  ## 단위 억
                sector = df_fin[df_fin.index == name]['Sector'].values[0]
                industry = df_fin[df_fin.index == name]['Industry'].values[0]
                turnover = df_fin[df_fin.index == name]['VolumeTurnOver'].values[0]
                change = df_fin[df_fin.index == name]['Change'].values[0]
                logger.info(f"code: {code:<8}, name: {name:<10}, 등락률: {change:<8}, "
                            f"거래대금(억원): {vol_cost:<10}, score : {fi_score:<10}, 거래회전률: {turnover:<8}, "
                            f"섹터: {sector:<20}, 산업: {industry:<20}")
            except:
                print(f"code: {code:<8}, name: {name:<15}, score: None -- 스코어를 찾을 수 없습니다.")

            logger.info(f"종목 ({name} - {code}) 의 등락율 차트를 생성합니다.")
            df = chart.run(code, date=dates, data='none')


        logger.info("Finish!")

if __name__ == "__main__":
    config_file = './config/config.yaml'
    theme = themeCode(config_file)
    # theme.search_theme_upjong(mode='theme')
    # theme.search_theme_upjong(mode='upjong')
    theme.search_market_leader()
    # theme.run()