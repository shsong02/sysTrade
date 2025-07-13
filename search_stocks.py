# -*- coding: utf-8 -*-

# pylint: disable=logging-fstring-interpolation

import sys
import requests
import pandas as pd
import yaml
import os
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
from tools.news_crawler import newsCrawler
from trade_strategy import tradeStrategy
from tools import st_utils as stu
# from back_test import backTestCustom  # 필요시에만 import

## warning 무시
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## log 폴더 생성
try:
    if not os.path.exists("./log"):
        os.makedirs("./log")
except Exception as e :
    print(e)

####    로그 생성    #######
logger = stu.create_logger()

## Global
TARGET = ('from_krx', 'from_theme', 'from_upjong', 'from_code', 'from_name')
MODE = ('market_leader', 'theme', 'upjong')


class searchStocks :
    def __init__(self, config_file):

        ## 설정 파일을 필수적으로 한다.
        try:
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e :
            print (e)

        ## global 변수 선언
        self.file_manager = config["data_management"]
        self.init_conifg = config["mainInit"]
        self.trade_config = config["tradeStock"]

        ### 기본 함수 실행 결정
        self.mode = config["searchStock"]["mode"]
        self.target = config["searchStock"]["market_leader"]["target"]

        if not self.mode in MODE:
            logger.error(f"선택된 모드({self.mode})는 지원되지 않습니다. (지원 모드: {MODE})")
            sys.exit()
        if not self.target in TARGET:
            logger.error(f"선택된 타겟({self.target})은 지원되지 않습니다. (지원 모드: {TARGET})")
            sys.exit()

        if self.mode == 'market_leader':
            if self.target == 'from_krx':
                self.params = config["searchStock"]["market_leader"]["from_krx"]["params"]
                self.config = config["searchStock"]["market_leader"]["from_krx"]["config"]
            elif self.target == 'from_theme':
                self.params = config["searchStock"]["market_leader"]["from_theme"]["params"]
                self.config = config["searchStock"]["market_leader"]["from_theme"]["config"]
            elif self.target == 'from_upjong':
                self.params = config["searchStock"]["market_leader"]["from_upjong"]["params"]
                self.config = config["searchStock"]["market_leader"]["from_upjong"]["config"]
            elif self.target == 'from_code':
                self.params = config["searchStock"]["market_leader"]["from_code"]["params"]
                self.config = config["searchStock"]["market_leader"]["from_code"]["config"]
            elif self.target == 'from_name':
                self.params = config["searchStock"]["market_leader"]["from_name"]["params"]
                self.config = config["searchStock"]["market_leader"]["from_name"]["config"]
                logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")
        else:
            self.params = config["searchStock"]["theme_upjong"]["params"]
            self.config = config["searchStock"]["theme_upjong"]["config"]



    def search_theme_upjong(self, mode='theme'):
        """네이버 주식에서 테마/업종을 검색하고, 속한 종목들의 상태를 확인합니다.
        테마 리스트를 저장하여 주도주 검색 시, 조건으로 활용합니다.

        Args:

        Returns:
            테마 코드를 저장한 파일을 저장합니다.

        """
        ## 1) 테마 리스트를 작성하고 테마별 종목코드를 확인합니다. 결과는 파일로 저장합니다.
        krx = fdr.StockListing('KRX')
        stocks = krx.dropna(axis=0, subset=['Sector'])  ## 섹터값없는 코드 삭제 (ETF...)
        stocks.reset_index(drop=True, inplace=True)

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
                            if cond > self.params["threshold_close"] : ## 현재가가 10,000 이상인 경우에만 집계에 사용한다.
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
            duration = self.params["period"]
            end_dt = datetime.today()
            end = end_dt.strftime(self.init_conifg["time_format"])
            st_dt = end_dt - timedelta(days=duration)
            st = st_dt.strftime(self.init_conifg["time_format"])

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
                r_pnt = 0
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

            if self.config["display_theme_chart"] == True:
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
            if self.config["save_stock_chart"] == True:
                file_name = f"{theme_name}.png"
                file_name = file_name.replace("/","_")
                if mode == 'theme':
                    file_path = self.file_manager["search_stocks"]["path"] + f"theme/img_codes/{st}_{end}/"
                else:
                    file_path = self.file_manager["search_stocks"]["path"] + f"upjong/img_codes/{st}_{end}/"
                ## 폴더 생성
                try:
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                except Exception as e:
                    raise e
                plt.savefig(file_path+file_name, dpi=300)
                if self.config["display_stock_chart"] == True:
                    plt.show()
            else:
                plt.show()

            ## 파일저장
            if self.config["save_stock_data"] == True:
                file_name = f"{theme_name}.csv"
                file_name = file_name.replace("/","_")
                if mode == 'theme':
                    file_path = self.file_manager["search_stocks"]["path"] + f"theme/raw/{st}_{end}/"
                else:
                    file_path = self.file_manager["search_stocks"]["path"] + f"upjong/raw/{st}_{end}/"

                ## 파일로 저장 합니다.
                stu.file_save(df_data, file_path, file_name, replace=False)

            if theme_name == 'PCB(FPCB 등)':  ## for test
                keys = theme_name
                # keys = '유신'
                nc = newsCrawler()
                nc.search_keyword(keys)
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

        file_name = f"{mode}_stockList_{st}_{end}.csv"
        file_path = self.file_manager["search_stocks"]["path"]
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
        cutoff = self.params["theme_summary_cutoff"]
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
        file_path = self.file_manager["search_stocks"]["path"] + f'{mode}/summary/{st}_{end}/'
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

        ## 공통
        path = self.file_manager["search_stocks"]["path"]
        format = self.init_conifg["time_format"]
        df_theme = stu.load_theme_list(path, mode='theme', format=format)
        df_upjong = stu.load_theme_list(path, mode='upjong', format=format)

        if self.target == 'from_krx':

            df_fin = self.condition_check()  ## 입력 df 없으면 내부 자동 생성

            ## 테스트 용
            # df_fin = df_vcost.sort_values(by='Change', ascending=False)
            # df_fin = df_vcost.sort_values(by='VolumeCost', ascending=False)
            # df_news = nc.search_keyword('현대코퍼레이션')


        elif self.target == 'from_name':
            name_list = self.config["select_name"].split(',')
            name_list2 = [n.replace(' ', '') for n in name_list]  ## 공백 제거

            df_fin = self.condition_check(name_list=name_list2)

        elif self.target == 'from_theme':


            theme_list = df_theme.Theme.to_list()
            theme_list = list(set(theme_list))

            if self.config["enable_all"] == True:
                df_sel = df_theme.drop_duplicates(subset='Name')
                df_sel = df_sel[df_sel.Name != 'ALL']
                df_sel = df_sel.set_index('Name', drop=True)
            else:
                df_temp = df_theme[df_theme.Name != 'ALL']
                df_temp = df_temp.set_index('Name', drop=True)

                search = self.config["select_theme"]
                sel_theme = []
                ## 테마 존재 여부 찾기
                for th in theme_list:
                    if search in th:
                        logger.error(f"선택한 테마명({search}) 이 테마 ({th})와 일치합니다.")
                        sel_theme.append(th)

                # 테마에 속한 종목 찾기
                df_list = []
                for th in sel_theme:
                    df_seltheme = df_temp[df_temp.Theme == th]
                    df_list.append(df_seltheme)

                if not len(df_list) == 0:
                    df_sel = pd.concat(df_list, axis=1)
                else:
                    logger.error(f"선택한 테마명({sel_theme}) 이 테마 리스트에 존재하지 않습니다.")
                    str = f"테마 리스트:"
                    for cnt, th in enumerate(theme_list):
                        if cnt % 5 == 0 :
                            str = str + "\n"
                        else:
                            str = str + f"{th:<30},"
                    logger.error(str)
                    sys.exit()


            df_sel = df_sel.drop(['Change', 'Code'], axis=1)

            df_fin = self.condition_check(df_sel)



        ####################################
        ####  종목별 차트 그리기 (공통 부분)
        ####################################

        # for 문을 위한 설정 정보들
        config_file = './config/config.yaml'
        strategy = tradeStrategy(config_file)
        dates = stu.period_to_str(self.config["chart_period"], format="%Y%m%d")
        nc = newsCrawler()

        buy_dict = dict()
        buy_name = []
        for cnt, (code, name) in enumerate(zip(df_fin.Symbol, df_fin.index)):
            # if cnt >15:
            #     print("SSH TEST")

            if not len(df_theme) == 0 :
                df_cur_theme = df_theme[df_theme.Name == name]
                cur_themes = df_cur_theme.Theme.to_list()
                for i in cur_themes:
                    cur_stocks = df_theme[df_theme.Theme == i].Name.to_list()
                    if not len(cur_stocks) == 0 :
                        cur_stocks.remove('ALL')
                        cur_stocks.remove(name)
                        # print(cur_stocks)

            if not len(df_upjong) == 0 :
                df_cur_upjong = df_upjong[df_upjong.Name == name]
                cur_upjongs = df_cur_upjong.Theme.to_list()
                for i in cur_upjongs:
                    cur_stocks = df_upjong[df_upjong.Theme == i].Name.to_list()
                    if not len(cur_stocks) == 0 :
                        cur_stocks.remove('ALL')
                        cur_stocks.remove(name)
                        # print(cur_stocks)

            try:
                ####################################
                ####  종목별 Score Print 하기
                ####################################
                fi_score = df_fin[df_fin.index == name]['total_score'].values[0]
                vol_cost = df_fin[df_fin.index == name]['VolumeCost'].values[0]
                vol_cost = round(vol_cost / 100000000)  ## 단위 억
                sector = df_fin[df_fin.index == name]['Sector'].values[0]
                industry = df_fin[df_fin.index == name]['Industry'].values[0]
                turnover = df_fin[df_fin.index == name]['VolumeTurnOver'].values[0]
                change = df_fin[df_fin.index == name]['Change'].values[0]
                change_mid = df_fin[df_fin.index == name]['ChangeMid'].values[0]

                # pylint: disable=W1203
                logger.info(f"Code: {code:<8}, name: {name:<20}, 단기 등락률: {change:<6}, 중기 등락률: {change_mid:<6}, "
                            f"거래대금(억원): {vol_cost:<8}, score : {fi_score:<4}, 거래회전률: {turnover:<8}, "
                            f"섹터: {sector:<20}, 산업: {industry:<20}")
            except Exception:
                print(f"Code: {code:<8}, name: {name:<15}, score: None -- 스코어를 찾을 수 없습니다.")

            # logger.info(f"종목 ({name} - {code}) 의 등락율 차트를 생성합니다.")
            df_ohlcv = strategy.run(code, name=name, dates=dates, data='none', mode='daily')

            ## 한달동안 매수 신호가 발생한 종목 수집하기
            if any(df_ohlcv.tail(self.trade_config["buy_condition"]["codepick_buy_holdday"]).finalBuy.to_list()):
                buy_name.append(name)
                buy_dict[name] = dict()
                buy_dict[name]['code'] = code
                last_d = df_ohlcv[df_ohlcv.finalBuy == True]['finalBuy'].index.to_list()[-1]
                priceEarning = stu.change_ratio(curr=df_ohlcv.Close.iat[-1], prev=df_ohlcv.Close.at[last_d]) # 수익률
                hold_d = datetime.today() - last_d
                buy_dict[name]['buy_day'] = last_d
                buy_dict[name]['buy_hold_day'] = hold_d
                buy_dict[name]['priceEarning'] = priceEarning




        ## 매수 조건이 발생한 종목을 저장 하기
        logger.info(f"\n매수 조건을 만족한 종목을 저장합니다.")
        df_buy = df_fin.loc[buy_name]
        cols = ["Symbol", "total_score", "VolumeCost", "Sector", "Industry", "VolumeTurnOver", "Change", "ChangeMid"]
        df_buy = df_buy.loc[:,cols]

        ## 파일 저장합니다.
        self._make_buy_list(df_buy, buy_dict)


        logger.info("Finish!")

    ######################################################
    ######    Internal Func.
    ######################################################
    def _make_buy_list(self, df, buy_dict):
        # 시작 시점에 시간 체크
        today = datetime.today()

        # 중기, 단기 기간 값 추가, buy 날짜, buy 지연 날짜,
        [st, end] = stu.period_to_str(self.config["change_period"])
        change_period = f"{st} ~ {end}"
        [st, end] = stu.period_to_str(self.config["trend_period"])
        trend_period = f"{st} ~ {end}"

        df["change_period"] = change_period
        df["trend_period"] = trend_period
        df["select_mode"] = self.target
        df["buy_day"] = 0
        df["buy_hold_day"] = 0
        df["priceEarning"] = 0
        for name, value in buy_dict.items():
            df.loc[name, "buy_day"] = value["buy_day"]
            df.loc[name, "buy_hold_day"] = value["buy_hold_day"]
            df.loc[name, "priceEarning"] = value["priceEarning"]
        df.sort_values(by='buy_hold_day', inplace=True)
        for name in df.index:
            last_d = str(df.at[name,'buy_day'])
            hold_d = str(df.at[name,'buy_hold_day'])
            earn = str(df.at[name,'priceEarning'])
            code = str(df.at[name,'Symbol'])
            logger.info(f"종목명 (매수조건 만족): ({code}){name:<20}  -> 매수 신호 날짜: {last_d:<25}, 당일까지 기간: {hold_d:<25} , 당일까지 수익률: {earn}%")

        ## 폴더 생성
        base_path = self.file_manager["monitor_stocks"]["path"]
        path = f"year={today.strftime('%Y')}/month={today.strftime('%m')}/day={today.strftime('%d')}/time={today.strftime('%H%M')}/"

        stu.file_save(df, file_path=base_path+path, file_name=f"buy_list_{today.strftime('%Y%m%d_%H%M%S')}.csv")

        # 차트 이미지 저장.
        ## 매수 종목만 그림 파일 저장합니다.
        st = tradeStrategy('./config/config.yaml')
        st.display = 'save'
        dates = stu.period_to_str(self.config["chart_period"], format="%Y%m%d")
        for name, code in zip(df.index, df.Symbol):
            st.path = base_path+path + "chart/"
            st.name = f"{name}_{code}.png"
            st.run(code, name=name, dates=dates, data='none', mode='daily')


    def condition_check(self, df=pd.DataFrame(), name_list=[],):
        [st, end] = stu.period_to_str(self.config["change_period"])

        ## 등락률 정보
        df_price = stock.get_market_price_change(st.replace("-", ""), end.replace("-", ""), market='ALL')
        endt = datetime.strptime(end, "%Y-%m-%d")
        for i in range(7):  ## 시장이 쉬는 구간을 피해서 불러오기 위함
            date = endt - timedelta(days=i)
            end_pre = date.strftime(format="%Y%m%d")
            try:
                df_price2 = stock.get_exhaustion_rates_of_foreign_investment(end_pre, market='ALL')['상장주식수']
                break
            except:
                continue


        df_price = df_price.join(df_price2)
        df_price["회전율"] = round(df_price["거래량"] / df_price["상장주식수"] * 100, 2)

        df_price.rename(columns={'종목명': 'Name',
                                 '시가': 'Open',
                                 '종가': 'Close',
                                 '거래량': 'Volume',
                                 '등락률': 'Change',
                                 '변동폭': 'ChangeRatio',
                                 '거래대금': 'VolumeCost',
                                 '상장주식수': 'TotalVolume',
                                 '회전율': 'VolumeTurnOver',
                                 }, inplace=True)

        df_price.set_index(keys=['Name'], inplace=True)

        if len(df)  == 0 :
            df = df_price
        else:
            df = df.join(df_price, how='left')

        ## Name list 가 있으면, 선택함.
        if not len(name_list) == 0:
            df = df.loc[name_list,:]

            if len(df) == 0:
                logger.error(f"입력한 종목명들을 찾을 수 없습니다. (종목명: {name_list})")
                sys.exit()
        # 0) 예외 종목 추려놓기
        names = self.params["tracking_stocks"]
        names = names.split(',')
        name2 = list(filter(None, names))
        trck_name = []
        for n in name2:
            trck_name.append(n.replace(' ', ''))  ## 공백제거

        df_extra = df.loc[trck_name]

        # 1) 조건1: 종가 최소 값 으로 cut
        thrd_close = self.params["threshold_close"]
        df_close = df[df.Close > thrd_close]
        logger.info(
            f"종가 ({thrd_close} 원) 보다 낮은 종믁 제거로,  전체 ({len(df)}) 개 중에 ({len(df_close)}))개 를 선정합니다.")


        # 2) 조건2:  finance score 이상인 경우에만 해당
        # 제무 스코어 불러오기
        path = self.file_manager["finance_score"]["path"]
        files = os.listdir(path)
        df_ref = pd.read_csv(path + files[0])  ## 파일이 하나밖에 없음
        df_ref['Symbol'] = df_ref['Symbol'].apply(lambda x: str(x).zfill(6))
        df_ref.set_index(keys=['Name'], inplace=True)

        df_finance = df_close.join(df_ref, how='left')
        df_extra2 = df_extra.join(df_ref, how='left')  ## 추적 주식 관리
        df_finance.dropna(subset=['total_score', 'Symbol'], inplace=True)
        thrd_score = self.params["threshold_finance_score"]
        df_finance2 = df_finance[df_finance.total_score >= thrd_score]
        logger.info(f"재무제표 총점 ({thrd_score}) 기준으로 종목 ({len(df_close)})개 중에 ({len(df_finance2)})개 를 선정합니다.")

        # 3) 조건3: 상승 추세 확인 (중기)
        [chng_st, chng_end] = stu.period_to_str(self.config["trend_period"])
        df_mid_price = stock.get_market_price_change(chng_st, chng_end, market='ALL')
        df_mid_price.rename(columns={'종목명': 'Name',
                                 '시가': 'Open',
                                 '종가': 'Close',
                                 '거래량': 'Volume',
                                 '등락률': 'Change',
                                 '변동폭': 'ChangeRatio',
                                 '거래대금': 'VolumeCost',
                                 }, inplace=True)
        df_mid_price.set_index('Name', drop=True, inplace=True)

        name_list = df_finance2.index.to_list()
        name_extra_list = df_extra2.index.to_list()
        df_mid_price2 = df_mid_price.loc[name_list,:]  ## 상승추세를 확인하기 위해 준비
        df_extra2_2 = df_mid_price.loc[name_extra_list,:]  ## 추가


        thrd_trend = self.params["threshold_min_trend"]
        name_list = df_mid_price2[df_mid_price2.Change > thrd_trend].index.to_list()
        df_trend = df_finance2.loc[name_list,:]  ## 상승추세를 확인하기 위해 준비
        df_trend['ChangeMid'] = df_mid_price2['Change']
        df_extra2['ChangeMid'] = df_extra2_2['Change']
        logger.info(
            f"({self.config['trend_period']})일간  ({thrd_trend})% 보다 높게 상승한 기준으로 종목 ({len(df_finance2)})개 중에 ({len(df_trend)})개 를 선정합니다.")

        # 4) 조건4:  등락률이 코스피보다 높은 경우에만 해당
        ## 코스피 등락률 보다 높이 상승한 종목 추출
        flag = self.params["threshold_min_change"]
        if flag == True:
            krx = stock.get_index_price_change(st, end, "KOSPI")
            change_thrd = krx.at["코스피 200", "등락률"]
            if change_thrd < 0:  #
                change_thrd = 1
            else:
                change_thrd += 1
            df_change = df_trend[df_trend.Change > change_thrd]  ## 등락률 기간이 다르기 때문에
            logger.info(
                f"코스피 변동률(최소 +1) ({change_thrd}) 보다 높게 상승한 기준으로 종목 ({len(df_trend)})개 중에 ({len(df_change)})개 를 선정합니다.")
        else:
            df_change = df_trend

        # 4) 조건4-2: 등락률이 너무 높지 않음 (이미 날라간 종목은 관심 없음)
        thrd_maxchg = self.params["threshold_max_change"]
        df_change2 = df_change[df_change.Change < thrd_maxchg]
        logger.info(
            f"변동률 ({thrd_maxchg}) 보다 낮게 상승 (급한성장은 관심 논외)한 기준으로 종목 ({len(df_change)})개 중에 ({len(df_change2)})개 를 선정합니다.")

        # 5) 조건5: 거래대금이 너무 낮지 않음
        thrd_vcost = self.params["threshold_volumecost"]
        df_vcost = df_change2[df_change2.VolumeCost > thrd_vcost * 100000000]
        logger.info(f"거래대금이 ({thrd_vcost} 억원) 보다 높은 기준으로 종목 ({len(df_change2)})개 중에 ({len(df_vcost)})개 를 선정합니다. ")

        # 6) 조건5: 회전율 상위 순위 순으로 cut
        thrd_shortselling = self.params["shortselling_sort_count"]
        df_shtsell = df_vcost.sort_values(by='VolumeTurnOver', ascending=False).head(thrd_shortselling)
        logger.info(
            f"거래 회전률이 많은 순으로 상위 ({thrd_shortselling})개 목표로, 종목 ({len(df_vcost)})개 중에 ({len(df_shtsell)})개 를 선정합니다.")

        df_out = df_shtsell

        # 7) 추척이 필요한 종목을 추가 한다.
        df_out = pd.concat([df_out, df_extra2], axis=0)
        df_out.drop_duplicates(inplace=True)
        logger.info(
            f"추적이 필요한 종목  ({len(df_extra2)})개를 더하여, 종목 ({len(df_shtsell)})개에서 ({len(df_out)})개 (중복제거)로 더합니다. \n\n")



        return df_out




if __name__ == "__main__":
    config_file = './config/config.yaml'
    theme = searchStocks(config_file)
    # theme.run()