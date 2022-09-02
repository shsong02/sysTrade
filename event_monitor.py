import os

import numpy as np
import itertools
import pandas as pd
import yaml
import logging
import logging.config
import plotly.graph_objects as go
import math
import umap
import hdbscan
from datetime import datetime
import  re

from tqdm import tqdm
from konlpy.tag import Mecab
from bertopic import BERTopic
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import BertModel, BertTokenizer
import torch

# Then, we perform k-means clustering using sklearn:
from sklearn.cluster import KMeans

## Custom package
from news_crawler import newsCrawler
import st_utils  as stu


## set logger
logger = stu.create_logger()



class eventMonitor:
    def __init__(self):

        ## 외부 인터페이스 용
        self.file_manager = {
            "news":
                {
                    "path": "./data/news/item_search/",
                    "name": ""
                },
            "model_results":
                {
                    "keywords":
                        {
                            "path": "./data/model_results/keywords/",
                            "name": "trigram.dat"
                        }
                },
            "models":
                {
                    "nlp":
                        {
                            "path": "./models/nlp/",
                            "name": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
                        },
                    "nlp-kpfbert":
                        {
                            "path": "./models/kpfbert/",
                            "name": "kpfsbert-base",
                        }
                }
        }

        ## var.
        self.skip_gen_news = True
        self.STEP1_SKIP = False
        self.STEP2_SKIP = False


    def make_news(self, target='title', split=False):
        if self.skip_gen_news != True:
            nc = newsCrawler()

            ## set config
            nc.config["save"] = True
            nc.config["save_replace"] = True  ## 파일명 자동생성되어 하나만 폴더에 저장
            nc.file_manager["news"] = self.file_manager["news"]
            set = {
                "interval": 8,
                "theme": "economic",
                "day_maxcnt": 100,
            }

            # new_tile
            df = nc.search_interval(dates=['20220819','20220821'])
        else:
            file_path = self.file_manager["news"]["path"]
            ## 파일 이름을 알 수 없으므로 확인해서 가져옴
            file_name = os.listdir(file_path)[0]
            df = stu.file_load(file_path, file_name, type='csv')
            logger.info(f"뉴스 정보를 파일로 부터 읽어 왔습니다.(파일명: {file_path + file_name} )")

        ## 제목 하나로 합치기
        docs = []
        if target == '':
            return df
        else:
            for i in range(len(df)):
                if target == 'title':
                    docs.append(f"{str(df.title.iloc[i])}    ")
                else: # content
                    docs.append(f"{str(df.content.iloc[i])}    ")

            if split == True:
                doc = " ".join(docs)
                return doc
            else:
                return docs

    def word2vec(self):
        df = self.make_news(target='', split=False)
        df = df.dropna()

        # 불용어 정의
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

        ## 토큰화
        okt = Okt()

        tokenized_data = []
        for sentence in tqdm(df.content):
            tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
            tokenized_data.append(stopwords_removed_sentence)

        # 리뷰 길이 분포 확인
        print('리뷰의 최대 길이 :', max(len(review) for review in tokenized_data))
        print('리뷰의 평균 길이 :', sum(map(len, tokenized_data)) / len(tokenized_data))
        plt.hist([len(review) for review in tokenized_data], bins=50)
        plt.xlabel('length of samples')
        plt.ylabel('number of samples')
        plt.show()

        model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)

        # 완성된 임베딩 매트릭스의 크기 확인
        model.wv.vectors.shape

        print(model.wv.most_similar("코로나"))

    def news_clustering_kpfbert(self):
        '''
        reference: https://github.com/KPFBERT/kpfSBERT/blob/main/kpfSBERT.ipynb

        :return:
        '''
        ## init
        os.system("ulimit -c unlimited")
        mps_device = torch.device("mps")

        ##set data
        df = self.make_news(target='', split=False)
        df = df.dropna()
        cluster_mode = 'title'


        # kpfSBERT 모델 로딩
        model_path = self.file_manager["models"]["nlp-kpfbert"]["path"]
        model_name = self.file_manager["models"]["nlp-kpfbert"]["name"]

        model =  BertModel.from_pretrained(model_path, add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        word_embedding_model = models.Transformer(model_path)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


        logger.info("Read AllNLI train dataset")
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        train_samples = []
        with open(model_path +'KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv', "rt", encoding="utf-8") as fIn:
            lines = fIn.readlines()
            for line in lines:
                s1, s2, label = line.split('\t')
                label = label2int[label.strip()]
                train_samples.append(InputExample(texts=[s1, s2], label=label))


        train_batch_size = 16
        train_dataset = SentencesDataset(train_samples, model=model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model,
                                        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                        num_labels=len(label2int))

        # Read STSbenchmark dataset and use it as development set
        logger.info("Read STSbenchmark dev dataset")
        dev_samples = []

        with open(model_path+'KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
            lines = fIn.readlines()
            for line in lines:
                s1, s2, score = line.split('\t')
                score = score.strip()
                score = float(score) / 5.0
                dev_samples.append(InputExample(texts=[s1, s2], label=score))

        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                         name='sts-dev')

        num_epochs = 1
        warmup_steps = math.ceil(
            len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=dev_evaluator,
                  epochs=num_epochs,
                  evaluation_steps=1000,
                  warmup_steps=warmup_steps,
                  #           output_path=model_save_path
                  )

        ##################################

        # Corpus with example sentences
        corpus = df.content.to_list()

        corpus_embeddings = model.encode(corpus)


        num_clusters = 50
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i + 1)
            print(cluster)
            print("")

        ######################


        # UMAP 차원축소 실행
        def umap_process(corpus_embeddings, n_components=5):
            umap_embeddings = umap.UMAP(n_neighbors=15,
                                        n_components=n_components,
                                        metric='cosine').fit_transform(corpus_embeddings)
            return umap_embeddings

        # HDBSCAN 실행
        def hdbscan_process(corpus, corpus_embeddings, min_cluster_size=15, min_samples=10, umap=True, n_components=5,
                            method='eom'):

            if umap:
                umap_embeddings = umap_process(corpus_embeddings, n_components)
            else:
                umap_embeddings = corpus_embeddings

            cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                      min_samples=10,
                                      allow_single_cluster=True,
                                      metric='euclidean',
                                      core_dist_n_jobs=1,
                                      # knn_data = Parallel(n_jobs=self.n_jobs, max_nbytes=None) in joblib
                                      cluster_selection_method=method).fit(umap_embeddings)  # eom leaf

            docs_df = pd.DataFrame(corpus, columns=["Doc"])
            docs_df['Topic'] = cluster.labels_
            docs_df['Doc_ID'] = range(len(docs_df))
            docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

            return docs_df, docs_per_topic

        # 카테고리별 클러스터링
        start = datetime.now()
        print('작업 시작시간 : ', start)
        previous = start
        bt_prev = start

        tot_df = pd.DataFrame()

        print(' processing start... with cluster_mode :', cluster_mode)

        category = df.category.unique()

        df_category = []
        for categ in category:
            df_category.append(df[df.category == categ])
        cnt = 0
        rslt = []
        topics = []
        # 순환하며 데이터 만들어 df에 고쳐보자
        for idx, dt in enumerate(df_category):

            corpus = dt[cluster_mode].values.tolist()
            # '[보통공통된꼭지제목]' 형태를 제거해서 클러스터링시 품질을 높인다.
            for i, cp in enumerate(corpus):
                corpus[i] = re.sub(r'\[(.*?)\]', '', cp)
            #     print(corpus[:10])
            corpus_embeddings = model.encode(corpus, show_progress_bar=True)

            docs_df, docs_per_topic = hdbscan_process(corpus, corpus_embeddings,
                                                      umap=False, n_components=15,  # 연산량 줄이기 위해 umap 사용시 True
                                                      method='leaf',
                                                      min_cluster_size=5,
                                                      min_samples=30,
                                                      )
            cnt += len(docs_df)

            rslt.append(docs_df)
            topics.append(docs_per_topic)
            dt['cluster'] = docs_df['Topic'].values.tolist()
            tot_df = pd.concat([tot_df, dt])

            bt = datetime.now()
            print(len(docs_df), 'docs,', len(docs_per_topic) - 1, 'clusters in', category[idx], ', 소요시간 :',
                  bt - bt_prev)
            bt_prev = bt
        now = datetime.now()
        print(' Total docs :', cnt, 'in', len(rslt), 'Categories', ', 소요시간 :', now - previous)
        previous = now

        # cluster update

        df['cluster'] = tot_df['cluster'].astype(str)

        end = datetime.now()
        print('작업 종료시간 : ', end, ', 총 소요시간 :', end - start)

        ####################3

        word_embedding_model = models.Transformer(model_name_or_path=model_path)


        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        logger.info("Read AllNLI train dataset")

        # Corpus with example sentences
        corpus = ['한 남자가 음식을 먹는다.',
                  '한 남자가 빵 한 조각을 먹는다.',
                  '그 여자가 아이를 돌본다.',
                  '한 남자가 말을 탄다.',
                  '한 여자가 바이올린을 연주한다.',
                  '두 남자가 수레를 숲 속으로 밀었다.',
                  '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
                  '원숭이 한 마리가 드럼을 연주한다.',
                  '치타 한 마리가 먹이 뒤에서 달리고 있다.',
                  '한 남자가 파스타를 먹는다.',
                  '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
                  '치타가 들판을 가로 질러 먹이를 쫓는다.']

        corpus_embeddings = model.encode(corpus)

        # Then, we perform k-means clustering using sklearn:
        from sklearn.cluster import KMeans

        num_clusters = 5
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i + 1)
            print(cluster)
            print("")





    def news_clustering(self):
        df = self.make_news(target='', split=False)
        df = df.dropna()

        okt = Okt()  # 형태소 분석기 객체 생성
        noun_list = []
        for content in tqdm(df['content']):
            nouns = okt.nouns(content)  # 명사만 추출하기, 결과값은 명사 리스트
            noun_list.append(nouns)

        df['nouns'] = noun_list

        # 문서를 명사 집합으로 보고 문서 리스트로 치환 (tfidfVectorizer 인풋 형태를 맞추기 위해)
        text = [" ".join(noun) for noun in df['nouns']]

        tfidf_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 5))
        tfidf_vectorizer.fit(text)
        vector = tfidf_vectorizer.transform(text).toarray()

        vector = np.array(vector)  # Normalizer를 이용해 변환된 벡터
        model = DBSCAN(eps=0.3, min_samples=6, metric="cosine")
        # 거리 계산 식으로는 Cosine distance를 이용
        result = model.fit_predict(vector)

        for cluster_num in set(result):
            # -1,0은 노이즈 판별이 났거나 클러스터링이 안된 경우
            if (cluster_num == -1 or cluster_num == 0):
                continue
            else:
                print("cluster num : {}".format(cluster_num))
                temp_df = df[df['result'] == cluster_num]  # cluster num 별로 조회
                for title in temp_df['title']:
                    print(title)  # 제목으로 살펴보자
                print()


    def extract_topic(self):
        docs = self.make_news(target='content', split=False)

        ## 전처리
        docs_prep1 = []
        for line in tqdm(docs):
            if line and not line.replace(' ', '').isdecimal():
                docs_prep1.append(line)

        ## 글이 넓은 짧은 글은 삭제한다.
        docs_prep2 = []
        for line in docs_prep1:
            if len(line) > 30 :
                docs_prep2.append(line)


        custom_tokenizer = CustomTokenizer(Mecab())

        vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

        pretrained_model = "./data/models/nlp/sentence-transformers_xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
        model = BERTopic(embedding_model=pretrained_model,
                        # embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                         vectorizer_model=vectorizer,
                         nr_topics=50,
                         top_n_words=10,
                         calculate_probabilities=True)

        topics, probs = model.fit_transform(docs_prep2)
        fig = model.visualize_topics()
        fig.show()
        model.visualize_distribution(probs[0])
        for i in range(0, 50):
            print(i, '번째 토픽 :', model.get_topic(i))

        pass


    def extract_keywords(self):
        ''' 뉴스에서 키워드를 추출 합니다.
        참고 사이트 : https://wikidocs.net/162079

        :return:
        '''
        '''
        1) 헤드라인 뉴스로 시간대별 뉴스 흐름을 파악합니다.
            - 시간 지정하여 뽑을 수 있나? 또는 1000개 이상 추출 가능한지 확인 필요 (후순위)
        3) 헤드라인 뉴스에서 키워드 추출하고 시간축 빈도 확인
           -  카벌트 사용 →  명사추출 → 유사도 다른 키워드 추출
        4) 뉴스 제목들을 키워드로 classification 하고, 시간축 빈도 확인
        5) 키워드와 섹터 연관성 매칭
        6) 대장주 찾기 및 Selected Items 에서 찾아 내기
        '''

        doc = self.make_news()

        if self.STEP1_SKIP != True:

            ## 명사 추출
            okt = Okt()

            tokenized_doc = okt.pos(doc)
            tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

            print('품사 태깅 10개만 출력 :', tokenized_doc[:10])
            print('명사 추출 :', tokenized_nouns)

            #### 명사들로 조합하여
            n_gram_range = (2, 3)

            count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
            candidates = count.get_feature_names_out()

            print('trigram 개수 :', len(candidates))
            print('trigram 다섯개만 출력 :', candidates[:5])

            file_path = self.file_manager["model_results"]["keywords"]["path"]
            file_name = self.file_manager["model_results"]["keywords"]["name"]
            stu.file_save(candidates, file_path, file_name, type='ndarray', replace=False)
        else:
            file_path = self.file_manager["model_results"]["keywords"]["path"]
            file_name = self.file_manager["model_results"]["keywords"]["name"]
            candidates = stu.file_load(file_path, file_name, type='ndarray')
            logger.info(f"trigram 정보를 파일로 부터 읽어 왔습니다.(파일명: {file_path + file_name} )")

        ## step:
        ## 모델은 없으면 자동으로 다운로드 (1GB ..) .cash 에 저장됨
        model_path = self.file_manager["models"]["nlp"]["path"]
        model_name = self.file_manager["models"]["nlp"]["name"]
        model = SentenceTransformer(model_name_or_path=model_name, cache_folder=model_path)
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)
        top_n = 5
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
        print(keywords)

        words = self._max_sum_sim(candidates, doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10)
        print(words)
        words = self._max_sum_sim(candidates, doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30)
        print(words)

        words = self._mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2)
        print(words)
        words = self._mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)
        print(words)

    ########    Internal func.    #############
    def _max_sum_sim(self, data, doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
        # 문서와 각 키워드들 간의 유사도
        distances = cosine_similarity(doc_embedding, candidate_embeddings)

        # 각 키워드들 간의 유사도
        distances_candidates = cosine_similarity(candidate_embeddings,
                                                 candidate_embeddings)

        # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [data[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]

    def _mmr(self, doc_embedding, candidate_embeddings, words, top_n, diversity):

        # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
        word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

        # 각 키워드들 간의 유사도
        word_similarity = cosine_similarity(candidate_embeddings)

        # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
        # 만약, 2번 문서가 가장 유사도가 높았다면
        # keywords_idx = [2]
        keywords_idx = [np.argmax(word_doc_similarity)]

        # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
        # 만약, 2번 문서가 가장 유사도가 높았다면
        # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
        # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
        for _ in range(top_n - 1):
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # MMR을 계산
            mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # keywords & candidates를 업데이트
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [words[idx] for idx in keywords_idx]

class CustomTokenizer:  ## 혗태소 분류를 위해 사용
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result

if __name__ == '__main__':
    em = eventMonitor()
    # em.extract_keywords()
    # em.extract_topic()
    # em.news_clustering()
    # em.word2vec()
    em.news_clustering_kpfbert()
    ##### 임시 파일 저장











