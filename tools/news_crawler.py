## 네이버 뉴스 api
import re
import shutil
import urllib.request
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from bs4 import BeautifulSoup
import time

## multi-processing
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial


### custom
from tools import st_utils as stu

###########     Global var.    #################
################################################

client_id = '2CtLswBVpo1hvrIY9O_5'
client_secret = 'odSWUOS7VY'

## set logger
logger = stu.create_logger()


class newsCrawler:
    def __init__(self):

        ## global
        self.data = pd.DataFrame()
        self.src_text = ''

        ## 외부 인터페이스 용
        self.file_manager = {
            "news":
                {
                    "path": "./data/news/",
                    "name": ""
                },
        }
        ## 설정값
        self.config = {
            "save": True,
            "save_replace": False,
            "trend_display": True
        }

        self.search_detail = {
            "interval": 5, ## 하루 수집되는 뉴스 간격 (모두 수집 시, 오래 걸림)
            "theme": 'economic',    ## total, economic
            "day_maxcnt": 300,
        }

        self.time_format = "%Y%m%d"

        ## limit

    def search_interval(self, dates=[]):

        ## except
        try:
            if not type(dates) == list: raise
        except:
            logger.error(f"arg(date) 는 list 타입이어야 합니다.")
            os.exit()

        try:
            if not len(dates) == 2 : raise
        except:
            logger.error("arg(date)는 start, end_time 으로 구성되어 있어야 합니다. ")

        def date_range(start, end):
            format = self.time_format
            start = datetime.datetime.strptime(start, format)
            end = datetime.datetime.strptime(end, format)
            dates = [(start + datetime.timedelta(days=i)).strftime(format) for i in range((end - start).days + 1)]
            return dates

        date_list = date_range(dates[0], dates[1])

        articles = []
        index_list = []
        for cnt, i in enumerate(range(0, 300, 10)):
            if cnt == 0 :
                pre = i
                cur = i
            else:
                pre = cur
                cur = i
                index_list.append([pre, cur])

        pool = Pool(processes=8)
        for date in date_list :
            df = pool.map(partial(self.search_daily, date), index_list)
            df_day = pd.concat(df)
            df_day = df_day.drop_duplicates()

            articles.append(df_day)
            # pool.close()   # 인스턴스 종료
            # pool.join()  ## pool 들이 수행을 완료하기 까지 대기

            logger.info(f"Date: {date} 기사 수집을 완료 하였습니다. !!")

        print(articles)
        df_article = pd.concat(articles)
        df_article.reset_index(drop=True, inplace=True)

        if self.config["save"] == True:
            file_path= self.file_manager["news"]["path"]
            file_name = self.search_detail["theme"]
            file_name = f"{file_name}_{dates[0]}_{dates[1]}.csv"
            stu.file_save(df_article, file_path,  file_name, replace=self.config["save_replace"])


    def search_daily(self, date='', index_range=[]):
        format = self.time_format

        if date == '':
            now = datetime.datetime.now().strftime(format)
        else:
            try:
                datetime.datetime.strptime(date, format)
            except ValueError:
                raise ValueError("Incorrect data format, should be YYYYMMDD")


        article_list = []  # 기사가 담길 리스트 선언
        if len(index_range) == 0:
            index = 1  # page 이동을 위한 index 설정
        if self.search_detail["theme"] == 'total':
            theme2 = ''
        else:
            theme2 = self.search_detail["theme"]

        for index in range(index_range[0],index_range[1]):
            # page 를 변경하며 한페이지의 전체기사 목록을 가져오기
            url = f'https://news.daum.net/breakingnews/{theme2}?page={index}&regDate={date}'
            html = requests.get(url)
            soup = BeautifulSoup(html.content, "html.parser")
            news_section = soup.find('ul', {'class': 'list_news2 list_allnews'})
            news_list = news_section.find_all('a', {'class': 'link_txt'})
            for cnt, news in enumerate(news_list):
                if cnt % self.search_detail["interval"] == 0 :
                    article_title = news.text
                    article_url = news.get('href')
                    try:
                        article_html = requests.get(article_url)
                        soup = BeautifulSoup(article_html.content, "html.parser")
                        text_list = [text_tag.text for text_tag in soup.find_all('p', {'dmcf-ptype': "general"})]
                        article_content = " ".join(text_list)
                        # 기사 정보를 제목과 본문으로 저장하기 위한 dictionary 형태로 변환
                        article = dict()
                        article['title'] = article_title
                        article["date_time"] = soup.find("meta", {"property": "og:regDate"})["content"].zfill(14)
                        article["date"] = soup.find("meta", {"property": "og:regDate"})["content"][:8]
                        article["time"] = soup.find("meta", {"property": "og:regDate"})["content"][8:].zfill(6)
                        article["author"] = soup.find("meta", {"property": "og:article:author"})["content"]
                        article['content'] = article_content
                    except:
                        continue

                    # print(article)
                    article_list.append(article)  # 기사 담기

                    # 중간중간 잘 진행되는지 여부 확인을 위해 기사 개수가 10배수 일때마다 표시하기
                    c_proc = mp.current_process()
                    if len(article_list) % 30 == 0:
                        _msg = f"[{date}] 누적된 Article 수: {len(article_list)} 개 (new page: {index}) -- PID: {c_proc.pid}, PROC_NAME: {c_proc.name}"
                        logger.info(_msg)
                        time.sleep(0.1)  # 가져오는 작업이 서버의 부담을 줄수있기에 잠깐씩 쉬어준다.

                    ## 하루 수집양을 제한 하는 경우
                    if len(article_list) >= self.search_detail["day_maxcnt"]:
                        _msg = f"[{date}] 누적된 Article 수: {len(article_list)} 개 (new page: {index}) -- PID: {c_proc.pid}, PROC_NAME: {c_proc.name}"
                        logger.info(_msg)
                        df = pd.DataFrame.from_dict(article_list)
                        df['date_time'] = pd.to_datetime(df['date_time'])
                        df.sort_values(by=['date_time'], inplace=True, )
                        df.reset_index(drop=True, inplace=True)
                        return df

            # index += 1

            if len(news_list) < 15:  # page내 기사가 15개 미만이라면 마지막 page로 간주, 반복문 탈출!
                # _msg = f"[{date}] 누적된 Article 수: {len(article_list)} 개 (new page: {index})"
                # logger.info(_msg)
                break

        df = pd.DataFrame.from_dict(article_list)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.sort_values(by=['date_time'], inplace=True, )
        df.reset_index(drop=True, inplace=True)
        return df

    def search_keyword(self, keyword):

        self.src_text = keyword

        node = 'news'  # 크롤링한 대상
        cnt = 0
        jsonResult = []

        jsonResponse = self.getNaverSearch(node,  1, 100)  # [CODE 2]
        total = jsonResponse['total']

        while ((jsonResponse != None) and (jsonResponse['display'] != 0)):
            for post in jsonResponse['items']:
                cnt += 1
                self.getPostData(post, jsonResult, cnt)  # [CODE 3]

            start = jsonResponse['start'] + jsonResponse['display']
            jsonResponse = self.getNaverSearch(node, start, 100)  # [CODE 2]

        # print('전체 검색 : %d 건' % total)

        df = pd.json_normalize(jsonResult)
        df2 = df.sort_values('pDate', ascending=False)


        # print("가져온 데이터 : %d 건" % (cnt))

        ## df 정리
        df2['pDate'] = pd.to_datetime(df2['pDate'])
        df2['Day'] = df2['pDate'].dt.strftime("%Y-%m-%d")

        ## 태그 제거
        def remove_html(sentence):
            sentence = re.sub('(<([^>]+)>)', '', sentence)
            for chk in ['&apos;','&quot;', '<b>', '</b>', 'R&amp;D']:
                sentence = sentence.replace(chk, '')
            return sentence

        df2['title'] = df2['title'].apply(remove_html)
        df2['description'] = df2['description'].apply(remove_html)

        ## column 제거
        df2= df2.drop(['org_link', 'cnt'], axis=1)

        ## column 명 변경
        df2.rename(columns={'Day':'Date'}, inplace=True)
        for col in ['title', 'description', 'link', 'pDate']:
            df2.rename(columns={col: f'news_{col}'}, inplace=True)

        df2.set_index('Date', inplace=True, drop=True)

        ## 파일로 저장
        if self.config["save"] == True:
            file_path= self.file_manager["news"]["path"]
            file_name = self.src_text.replace(' ', '-')
            now = datetime.datetime.now().strftime("%Y-%m-%d")
            file_name = f"{file_name}_{now}.csv"
            stu.file_save(df2, file_path,  file_name, replace=self.config["save_replace"])
            print(f"네이버 뉴스에서 검색어({self.src_text}) 를 검색한 결과를 파일로 저장하였습니다. (파일명: {file_path+file_name})")
        else:
            file_path= self.file_manager["news"]["path"]
            file_name = self.src_text.replace(' ', '-')
            now = datetime.datetime.now().strftime("%Y-%m-%d")
            file_name = f"{file_name}_{now}.csv"

        # PLOTTING
        if self.config["trend_display"] == True:

            df_grp = df2.groupby('Date').count()
            start = df_grp.index.to_list()[0]
            end = df_grp.index.to_list()[-1]
            dt_start = datetime.datetime.strptime(start, "%Y-%m-%d")
            dt_end = datetime.datetime.strptime(end, "%Y-%m-%d")

            if (dt_end - dt_start) < datetime.timedelta(days=30 * 6):
                dt_start_2 = dt_end - datetime.timedelta(days=30 * 6)
            else:
                dt_start_2 = dt_start
            # fig, ax = plt.subplots(figsize=(12, 6))
            sns.set(rc={'figure.figsize': (20, 10), 'font.family': 'AppleGothic'})
            p =sns.barplot(x=df_grp.index, y="news_title", data=df_grp, estimator=sum, ci=None, )
            p.set_title(f"종목 이름: {self.src_text}")
            p.set(xlabel='뉴스 등록 날짜', ylabel='뉴스 Count')

            xticks = df_grp.index.to_list()
            xpos = []
            for i in range(0,len(xticks),5):
                xpos.append(i)
            p.set_xticks(xpos)
            p.set_xticklabels(xticks[::5])

            plt.xticks(rotation=-45)

            plt.show()
            file_name = file_name.replace('.csv', '.png')
            file_path = file_path + 'news_count/'

            ## 폴더 생성
            try:
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                else:
                    if self.config["save_replace"] == True:
                        shutil.rmtree(file_path)
                        os.makedirs(file_path)
                    else:
                        pass
            except Exception as e:
                raise e
            plt.savefig(file_path+file_name, dpi=300)

        # df2.rename(columns={"pDate":"date", "description":"content"}, inplace=True)



        return df2


    # [CODE 1]
    def getRequestUrl(self, url):
        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", client_id)
        req.add_header("X-Naver-Client-Secret", client_secret)

        try:
            response = urllib.request.urlopen(req)
            now = datetime.datetime.now()
            if response.getcode() == 200:
                # print(f"[{now}] Url Request Success: {url}")
                return response.read().decode('utf-8')
        except Exception as e:
            # print(e)
            # print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))
            return None


    # [CODE 2]
    def getNaverSearch(self, node, start, display):
        base = "https://openapi.naver.com/v1/search"
        node = "/%s.json" % node
        parameters = "?query=%s&sort=%s&start=%s&display=%s" % (urllib.parse.quote(self.src_text), 'sim', start, display)

        url = base + node + parameters
        responseDecode = self.getRequestUrl(url)  # [CODE 1]

        if (responseDecode == None):
            return None
        else:
            return json.loads(responseDecode)


    # [CODE 3]
    def getPostData(self, post, jsonResult, cnt):
        title = post['title']
        description = post['description']
        org_link = post['originallink']
        link = post['link']

        pDate = datetime.datetime.strptime(post['pubDate'], '%a, %d %b %Y %H:%M:%S +0900')
        pDate = pDate.strftime('%Y-%m-%d %H:%M:%S')

        jsonResult.append({'cnt': cnt, 'pDate': pDate, 'title': title, 'description': description,
                           'org_link': org_link, 'link': org_link })
        return

    ##########  Internal func    #############
    ##########################################




if __name__ == '__main__':
    words = '에코프로비엠'
    nc = newsCrawler()
    nc.search_keyword(words)
    # nc.search_interval(['20220817', '20220820'])
    # nc.search_daily('20200820', [0,10])

