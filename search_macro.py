"""
ETF 기반 거시경제 분석 모듈

이 모듈은 한국 ETF 시장 데이터를 분석하여 거시경제 동향을 파악하는 기능을 제공합니다.
주요 기능:
    - ETF 시장 데이터 수집 및 분석
    - 섹터별 ETF 성과 분석
    - 시장 지수 분석
    - 테마별 분석
    - 차트 생성 및 저장
    - 시장 동향 요약 리포트 생성

주요 의존성:
    - pykrx: 한국 주식/ETF 데이터 수집
    - pandas: 데이터 처리 및 분석
    - yaml: 설정 파일 관리
    - trade_strategy: 차트 생성

사용 예시:
    sm = searchMacro()
    result = sm.run()
"""

from datetime import datetime, timedelta
import pandas as pd
import yaml
import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from pykrx import stock
import openai

## local file
from tools import st_utils as stu
from trade_strategy import tradeStrategy

# 환경변수 로드
load_dotenv()

####    로그 생성    #######
logger = stu.create_logger()


class searchMacro:
    """
    포괄적인 거시경제 분석을 수행하는 클래스

    이 클래스는 한국 주식/ETF 시장의 데이터를 수집하고 분석하여 거시경제 동향을 파악합니다.
    config.yaml 파일에서 설정을 로드하여 분석 파라미터를 관리합니다.

    주요 속성:
        file_manager (dict): 파일 관리 관련 설정
        param_init (dict): 초기화 파라미터 설정
        macro_config (dict): 거시경제 분석 관련 설정

    설정 파일 구조:
        config.yaml:
            fileControl: 파일 관리 설정
            mainInit: 초기화 파라미터
            searchMacro:
                periods:
                    analysis_months (int): 분석 기간 (개월)
                    chart_months (int): 차트 생성 기간 (개월)
                    comparison_months (int): 비교 분석 기간 (개월)
                analysis_targets:
                    etf (bool): ETF 분석 여부
                    sectors (bool): 섹터별 분석 여부
                    market_indices (bool): 시장 지수 분석 여부
                    themes (bool): 테마별 분석 여부
                data_collection:
                    etf_min_volume (int): ETF 최소 거래량
                    sector_top_count (int): 섹터별 상위 종목 수
                    theme_top_count (int): 테마별 상위 종목 수

    분석 프로세스:
        1. 시장 지수 분석 (KOSPI, KOSDAQ)
        2. ETF 분석 (섹터별, 테마별)
        3. 개별 섹터 주요 종목 분석
        4. 테마별 주요 종목 분석
        5. 거시경제 지표 종합 분석
        6. 리포트 생성 및 저장
    """

    def __init__(self):
        """클래스 초기화 및 설정 로딩"""
        try:
            config_file = './config/config.yaml'
            with open(config_file, encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            logger.error(f"설정 파일 로딩 실패: {e}")
            raise

        logger.info(f"config 파일을 로드 하였습니다. (파일명: {config_file})")

        # 설정 변수 초기화
        self.file_manager = config["fileControl"]
        self.param_init = config["mainInit"]
        self.macro_config = config["searchMacro"]

        # 분석 기간 설정
        self.analysis_months = self.macro_config.get("periods", {}).get("analysis_months", 3)
        self.chart_months = self.macro_config.get("periods", {}).get("chart_months", 12)
        self.comparison_months = self.macro_config.get("periods", {}).get("comparison_months", 6)
        
        # 분석 대상 설정
        self.targets = self.macro_config.get("analysis_targets", {})
        
        # 데이터 수집 설정
        self.data_config = self.macro_config.get("data_collection", {})
        
        # 리포트 설정
        self.report_config = self.macro_config.get("report", {})
        
        # LLM 분석 설정
        self.llm_config = self.macro_config.get("llm_analysis", {})
        
        # OpenAI 클라이언트 초기화
        if self.llm_config.get("enabled", False):
            try:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.openai_client = openai.OpenAI()
                self.llm_model = self.llm_config.get("model", "gpt-4o")
                self.llm_max_tokens = self.llm_config.get("max_tokens", 4000)
                self.llm_temperature = self.llm_config.get("temperature", 0.3)
                logger.info(f"LLM 분석 초기화 완료 - 모델: {self.llm_model}")
            except Exception as e:
                logger.warning(f"LLM 초기화 실패: {e}")
                self.llm_config["enabled"] = False
        
        # 날짜 설정
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.analysis_months * 30)
        self.chart_start_date = self.end_date - timedelta(days=self.chart_months * 30)

    def run(self) -> Dict[str, Any]:
        """
        포괄적인 거시경제 분석을 실행하고 결과를 반환합니다.

        분석 과정:
            1. 시장 지수 분석 (KOSPI, KOSDAQ 등)
            2. ETF 분석 (섹터별, 테마별 분류)
            3. 개별 섹터 주요 종목 분석
            4. 테마별 주요 종목 분석
            5. 거시경제 종합 분석
            6. 리포트 생성 및 저장

        Returns:
            dict: 포괄적인 분석 결과 데이터
                {
                    'analysis_date': str,           # 분석 기준일
                    'period': str,                  # 분석 기간
                    'market_indices': dict,         # 시장 지수 분석
                    'etf_analysis': dict,           # ETF 분석 결과
                    'sector_analysis': dict,        # 섹터별 분석
                    'theme_analysis': dict,         # 테마별 분석
                    'macro_indicators': dict,       # 거시경제 지표
                    'market_sentiment': str,        # 시장 심리
                    'recommendations': list,        # 투자 권고사항
                    'report_path': str             # 생성된 리포트 경로
                }

        오류 발생 시:
            dict: {'error': str} 형태로 오류 메시지 반환
        """
        logger.info("포괄적인 거시경제 분석 시작")
        
        try:
            analysis_result = {
                'analysis_date': self.end_date.strftime("%Y%m%d"),
                'period': f"{self.start_date.strftime('%Y%m%d')} ~ {self.end_date.strftime('%Y%m%d')}",
                'market_indices': {},
                'etf_analysis': {},
                'sector_analysis': {},
                'theme_analysis': {},
                'macro_indicators': {},
                'market_sentiment': 'Neutral',
                'recommendations': [],
                'charts_generated': 0,
                'report_path': ''
            }

            # 1. 시장 지수 분석
            if self.targets.get('market_indices', True):
                logger.info("시장 지수 분석 시작")
                analysis_result['market_indices'] = self._analyze_market_indices()

            # 2. ETF 분석
            if self.targets.get('etf', True):
                logger.info("ETF 분석 시작")
                analysis_result['etf_analysis'] = self._analyze_etfs()

            # 3. 섹터별 분석
            if self.targets.get('sectors', True):
                logger.info("섹터별 분석 시작")
                analysis_result['sector_analysis'] = self._analyze_sectors()

            # 4. 테마별 분석
            if self.targets.get('themes', True):
                logger.info("테마별 분석 시작")
                analysis_result['theme_analysis'] = self._analyze_themes()

            # 5. 거시경제 지표 종합
            analysis_result['macro_indicators'] = self._calculate_macro_indicators(analysis_result)

            # 6. 시장 심리 판단
            analysis_result['market_sentiment'] = self._determine_market_sentiment(analysis_result)

            # 7. 투자 권고사항 생성
            analysis_result['recommendations'] = self._generate_recommendations(analysis_result)

            # 8. LLM 기반 시황 분석
            if self.llm_config.get('enabled', False):
                logger.info("LLM 기반 시황 분석 시작")
                llm_analysis = self._perform_llm_analysis(analysis_result)
                analysis_result['llm_analysis'] = llm_analysis

            # 9. 리포트 생성 및 저장
            if self.report_config.get('generate_summary', True):
                report_path = self._generate_report(analysis_result)
                analysis_result['report_path'] = report_path

            logger.info(f"거시경제 분석 완료 - 시장 심리: {analysis_result['market_sentiment']}")
            return analysis_result

        except Exception as e:
            logger.error(f"거시경제 분석 중 오류: {e}")
            logger.exception("상세 오류:")
            return {"error": str(e)}

    def _analyze_market_indices(self) -> Dict[str, Any]:
        """시장 지수 분석"""
        try:
            indices = {
                'KOSPI': '1001',
                'KOSDAQ': '2001',
                'KRX100': '1003'
            }
            
            result = {}
            start_str = self.start_date.strftime("%Y%m%d")
            end_str = self.end_date.strftime("%Y%m%d")
            
            for name, code in indices.items():
                try:
                    # 지수 데이터 수집
                    df_index = stock.get_index_ohlcv_by_date(start_str, end_str, code)
                    if df_index.empty:
                        logger.warning(f"{name} 지수 데이터가 없습니다.")
                        continue
                        
                    # 데이터 정리
                    df_index.rename(columns={
                        '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close',
                        '거래량': 'Volume', '등락률': 'Change', '변동폭': 'ChangeRatio'
                    }, inplace=True)
                    
                    # 지수 분석
                    current_level = df_index['Close'].iloc[-1]
                    start_level = df_index['Close'].iloc[0]
                    return_rate = ((current_level - start_level) / start_level * 100)
                    
                    # 변동성 계산
                    daily_returns = df_index['Close'].pct_change()
                    volatility = daily_returns.std() * (252 ** 0.5) * 100  # 연간화된 변동성
                    
                    # 추세 판단
                    ma20 = df_index['Close'].rolling(window=20).mean()
                    ma60 = df_index['Close'].rolling(window=60).mean()
                    current_ma20 = ma20.iloc[-1]
                    current_ma60 = ma60.iloc[-1]
                    
                    if current_level > current_ma20 > current_ma60:
                        trend = "상승"
                    elif current_level < current_ma20 < current_ma60:
                        trend = "하락"
                    else:
                        trend = "횡보"
                    
                    result[name] = {
                        'current_level': current_level,
                        'return_rate': return_rate,
                        'volatility': volatility,
                        'trend': trend,
                        'volume': df_index['Volume'].mean(),
                        'data': df_index
                    }
                    
                    logger.info(f"{name} 분석 완료 - 수익률: {return_rate:.2f}%, 추세: {trend}")
                    
                except Exception as e:
                    logger.error(f"{name} 지수 분석 실패: {e}")
                    continue
                    
            return result
            
        except Exception as e:
            logger.error(f"시장 지수 분석 중 오류: {e}")
            return {}

    def _analyze_etfs(self) -> Dict[str, Any]:
        """ETF 분석"""
        try:
            end_str = self.end_date.strftime("%Y%m%d")
            start_str = self.start_date.strftime("%Y%m%d")
            
            # ETF 목록 수집
            tickers = stock.get_etf_ticker_list(end_str)
            names = []
            
            for ticker in tickers:
                try:
                    name = stock.get_etf_ticker_name(ticker)
                    names.append(name)
                except Exception as e:
                    logger.warning(f"ETF {ticker} 이름 조회 실패: {e}")
                    continue
                    
            df_symbol = pd.DataFrame({'Symbol': tickers, 'Name': names})
            df_symbol.set_index('Symbol', inplace=True)
            
            # ETF 가격 변동 데이터 수집 (재시도 로직)
            tries = 0 
            max_tries = 10
            df_etf = None
            
            while tries < max_tries:
                try:
                    df_etf = stock.get_etf_price_change_by_ticker(start_str, end_str)
                    break
                except Exception as e:
                    logger.warning(f"ETF 데이터 로딩 실패 ({tries+1}/{max_tries}): {e}")
                    self.end_date -= timedelta(days=1)
                    end_str = self.end_date.strftime("%Y%m%d")
                    self.start_date -= timedelta(days=1)
                    start_str = self.start_date.strftime("%Y%m%d")
                    tries += 1
                    
            if tries >= max_tries:
                logger.error("ETF 데이터 로딩 실패")
                return {"error": "ETF 데이터 로딩 실패"}
            
            if df_etf is None or df_etf.empty:
                logger.error("ETF 데이터가 없습니다.")
                return {"error": "ETF 데이터가 없습니다."}
            
            # 데이터 정리
            df_etf = df_etf.join(df_symbol)
            df_etf.rename(columns={
                '시가': 'Open', '종가': 'Close', '거래량': 'Volume',
                '등락률': 'Change', '변동폭': 'ChangeRatio', '거래대금': 'VolumeCost'
            }, inplace=True)
            
            # 거래량 필터링
            min_volume = self.data_config.get('etf_min_volume', 1000000)
            df_etf = df_etf[df_etf['VolumeCost'] >= min_volume]
            
            # 수익률 순 정렬
            df_etf.sort_values(by="Change", ascending=False, inplace=True)
            df_positive = df_etf[df_etf.Change >= 0]
            
            # 섹터별 분류
            sector_analysis = self._classify_etf_by_sector(df_positive)
            
            # 상위 성과 ETF
            top_count = min(20, len(df_positive))
            top_performers = df_positive.head(top_count)
            
            # 통계 계산
            total_etfs = len(df_etf)
            positive_etfs = len(df_positive)
            positive_ratio = (positive_etfs / total_etfs * 100) if total_etfs > 0 else 0
            
            result = {
                'total_etfs': total_etfs,
                'positive_etfs': positive_etfs,
                'positive_ratio': round(positive_ratio, 2),
                'average_change': round(df_etf['Change'].mean(), 2),
                'top_performers': [
                    {
                        'name': row['Name'],
                        'code': idx,
                        'change': round(row['Change'], 2),
                        'volume_cost': row['VolumeCost']
                    }
                    for idx, row in top_performers.iterrows()
                ],
                'sector_breakdown': sector_analysis
            }
            
            # 차트 생성
            if self.report_config.get('generate_charts', True):
                charts_count = self._generate_etf_charts(df_positive)
                result['charts_generated'] = charts_count
                
            return result
            
        except Exception as e:
            logger.error(f"ETF 분석 중 오류: {e}")
            logger.exception("상세 오류:")
            return {"error": str(e)}

    def _classify_etf_by_sector(self, df_etf: pd.DataFrame) -> Dict[str, List]:
        """ETF를 섹터별로 분류"""
        try:
            # 섹터 키워드 정의
            sector_keywords = {
                '기술/IT': ['IT', '반도체', '소프트웨어', '인터넷', '게임'],
                '금융': ['은행', '증권', '보험', '금융'],
                '에너지/화학': ['에너지', '화학', '2차전지', '태양광'],
                '바이오/헬스케어': ['바이오', '제약', '헬스케어'],
                '소비재': ['소비재', '유통', '음식료', '패션'],
                '산업재': ['산업재', '기계', '건설', '철강'],
                '원자재': ['원자재', '광물', '귀금속'],
                '부동산': ['리츠', '부동산'],
                '인프라': ['인프라', '통신', '운송'],
                '기타': []
            }
            
            # 섹터별 ETF 분류
            sector_etfs = {sector: [] for sector in sector_keywords.keys()}
            
            for idx, row in df_etf.iterrows():
                etf_name = row['Name']
                classified = False
                
                for sector, keywords in sector_keywords.items():
                    if any(keyword in etf_name for keyword in keywords):
                        sector_etfs[sector].append({
                            'name': etf_name,
                            'code': idx,
                            'change': round(row['Change'], 2),
                            'volume_cost': row['VolumeCost']
                        })
                        classified = True
                        break
                        
                if not classified:
                    sector_etfs['기타'].append({
                        'name': etf_name,
                        'code': idx,
                        'change': round(row['Change'], 2),
                        'volume_cost': row['VolumeCost']
                    })
                    
            # 섹터별 통계 계산
            sector_stats = {}
            for sector, etfs in sector_etfs.items():
                if etfs:
                    changes = [etf['change'] for etf in etfs]
                    sector_stats[sector] = {
                        'count': len(etfs),
                        'average_change': round(sum(changes) / len(changes), 2),
                        'top_etfs': sorted(etfs, key=lambda x: x['change'], reverse=True)[:3]
                    }
                    
            return sector_stats
            
        except Exception as e:
            logger.error(f"ETF 섹터 분류 중 오류: {e}")
            return {}

    def _analyze_sectors(self) -> Dict[str, Any]:
        """섹터별 분석"""
        try:
            end_str = self.end_date.strftime("%Y%m%d")
            start_str = self.start_date.strftime("%Y%m%d")
            
            # 섹터 코드 정의
            sectors = {
                'KRX100': '1003',
                '건설': '1004',
                '금융': '1005',
                '기계': '1006',
                '운수장비': '1007',
                '철강/금속': '1008',
                '에너지/화학': '1009',
                '정보기술': '1010',
                '반도체': '1011',
                '바이오/의료': '1012',
                '통신서비스': '1013',
                '미디어/엔터': '1014',
                '유통/소비재': '1015',
                '전기/전자': '1016',
                '건설/건자재': '1017',
                '증권': '1018',
                '보험': '1019',
                '운송': '1020',
                '유틸리티': '1021'
            }
            
            result = {}
            
            for name, code in sectors.items():
                try:
                    # 섹터 지수 데이터 수집
                    df_sector = stock.get_index_ohlcv_by_date(start_str, end_str, code)
                    if df_sector.empty:
                        logger.warning(f"{name} 섹터 데이터가 없습니다.")
                        continue
                        
                    # 데이터 정리
                    df_sector.rename(columns={
                        '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close',
                        '거래량': 'Volume', '등락률': 'Change', '변동폭': 'ChangeRatio'
                    }, inplace=True)
                    
                    # 섹터 분석
                    current_level = df_sector['Close'].iloc[-1]
                    start_level = df_sector['Close'].iloc[0]
                    return_rate = ((current_level - start_level) / start_level * 100)
                    
                    # 변동성 계산
                    daily_returns = df_sector['Close'].pct_change()
                    volatility = daily_returns.std() * (252 ** 0.5) * 100  # 연간화된 변동성
                    
                    # 추세 판단
                    ma20 = df_sector['Close'].rolling(window=20).mean()
                    ma60 = df_sector['Close'].rolling(window=60).mean()
                    current_ma20 = ma20.iloc[-1]
                    current_ma60 = ma60.iloc[-1]
                    
                    if current_level > current_ma20 > current_ma60:
                        trend = "상승"
                    elif current_level < current_ma20 < current_ma60:
                        trend = "하락"
                    else:
                        trend = "횡보"
                    
                    result[name] = {
                        'current_level': current_level,
                        'return_rate': return_rate,
                        'volatility': volatility,
                        'trend': trend,
                        'volume': df_sector['Volume'].mean(),
                        'data': df_sector
                    }
                    
                    logger.info(f"{name} 섹터 분석 완료 - 수익률: {return_rate:.2f}%, 추세: {trend}")
                    
                except Exception as e:
                    logger.error(f"{name} 섹터 분석 실패: {e}")
                    continue
                    
            # 섹터 순위 계산
            sector_ranks = sorted(
                [(k, v['return_rate']) for k, v in result.items() if 'return_rate' in v],
                key=lambda x: x[1],
                reverse=True
            )
            
            # 섹터 로테이션 분석
            rotation_analysis = self._analyze_sector_rotation(result)
            
            return {
                'sectors': result,
                'rankings': [{'name': k, 'return_rate': v} for k, v in sector_ranks],
                'rotation': rotation_analysis
            }
            
        except Exception as e:
            logger.error(f"섹터 분석 중 오류: {e}")
            return {}

    def _analyze_themes(self) -> Dict[str, Any]:
        """테마별 분석"""
        try:
            end_str = self.end_date.strftime("%Y%m%d")
            start_str = self.start_date.strftime("%Y%m%d")
            
            # 테마 키워드 정의
            theme_keywords = {
                '2차전지': ['2차전지', '리튬', '배터리', '음극재', '양극재', '분리막'],
                '반도체': ['반도체', '파운드리', '메모리', 'DRAM', 'NAND'],
                '인공지능': ['AI', '인공지능', '빅데이터', '클라우드'],
                '메타버스': ['메타버스', 'VR', 'AR', '가상현실'],
                '친환경': ['친환경', '수소', '태양광', '풍력', 'ESG'],
                '바이오': ['바이오', '제약', '백신', '진단키트'],
                '로봇': ['로봇', '자동화', 'RPA'],
                '우주항공': ['우주', '항공', '위성', '드론'],
                '게임': ['게임', '메타버스'],
                '전기차': ['전기차', 'EV', '자율주행']
            }
            
            result = {}
            
            for theme_name, keywords in theme_keywords.items():
                try:
                    # 전체 종목 리스트 수집
                    all_tickers = stock.get_market_ticker_list(end_str, market="ALL")
                    theme_stocks = []
                    
                    # 키워드 기반 종목 필터링
                    for ticker in all_tickers[:200]:  # 성능을 위해 상위 200개만 확인
                        try:
                            stock_name = stock.get_market_ticker_name(ticker)
                            if any(keyword in stock_name for keyword in keywords):
                                theme_stocks.append({
                                    'ticker': ticker,
                                    'name': stock_name
                                })
                        except:
                            continue
                        
                    if not theme_stocks:
                        continue
                        
                    # 테마 종목들의 가격 데이터 수집
                    theme_data = []
                    for stock_info in theme_stocks[:10]:  # 상위 10개 종목만
                        try:
                            df_stock = stock.get_market_ohlcv_by_date(
                                start_str, end_str, stock_info['ticker']
                            )
                            
                            if not df_stock.empty:
                                start_price = df_stock.iloc[0]['종가']
                                end_price = df_stock.iloc[-1]['종가']
                                return_rate = ((end_price - start_price) / start_price * 100)
                                
                                theme_data.append({
                                    'name': stock_info['name'],
                                    'code': stock_info['ticker'],
                                    'return_rate': return_rate,
                                    'current_price': end_price,
                                    'volume': df_stock['거래량'].mean()
                                })
                                
                        except Exception as e:
                            logger.warning(f"테마 종목 {stock_info['name']} 데이터 수집 실패: {e}")
                            continue
                        
                    if theme_data:
                        # 테마 평균 수익률 계산
                        returns = [stock['return_rate'] for stock in theme_data]
                        avg_return = sum(returns) / len(returns)
                        
                        result[theme_name] = {
                            'average_return': round(avg_return, 2),
                            'stock_count': len(theme_data),
                            'top_stocks': sorted(theme_data, key=lambda x: x['return_rate'], reverse=True)[:5]
                        }
                        
                        logger.info(f"{theme_name} 테마 분석 완료 - 평균 수익률: {avg_return:.2f}%")
                        
                except Exception as e:
                    logger.error(f"{theme_name} 테마 분석 실패: {e}")
                    continue
                    
            # 신규 테마 식별
            emerging_themes = self._identify_emerging_themes(result)
            
            return {
                'themes': result,
                'rankings': sorted(
                    [(k, v['average_return']) for k, v in result.items()],
                    key=lambda x: x[1],
                    reverse=True
                ),
                'emerging_themes': emerging_themes
            }
            
        except Exception as e:
            logger.error(f"테마 분석 중 오류: {e}")
            return {}

    def _calculate_macro_indicators(self, analysis_data: Dict) -> Dict[str, Any]:
        """거시경제 지표 계산"""
        try:
            indicators = {}
            
            # ETF 데이터 기반 지표
            if 'etf_analysis' in analysis_data:
                etf_data = analysis_data['etf_analysis']
                indicators['market_breadth'] = etf_data.get('positive_ratio', 0)
                
            # 시장 지수 기반 지표
            if 'market_indices' in analysis_data:
                indices = analysis_data['market_indices']
                if 'KOSPI' in indices and 'KOSDAQ' in indices:
                    kospi_return = indices['KOSPI'].get('return_rate', 0)
                    kosdaq_return = indices['KOSDAQ'].get('return_rate', 0)
                    indicators['market_momentum'] = (kospi_return + kosdaq_return) / 2
                    indicators['growth_vs_value'] = kosdaq_return - kospi_return
                    
            return indicators
            
        except Exception as e:
            logger.error(f"거시경제 지표 계산 실패: {e}")
            return {}

    def _determine_market_sentiment(self, analysis_data: Dict) -> str:
        """시장 심리 판단"""
        try:
            # 기본값
            sentiment = "Neutral"
            
            # 지표 수집
            market_breadth = analysis_data.get('macro_indicators', {}).get('market_breadth', 0)
            market_momentum = analysis_data.get('macro_indicators', {}).get('market_momentum', 0)
            growth_vs_value = analysis_data.get('macro_indicators', {}).get('growth_vs_value', 0)
            
            # 시장 지수 데이터
            indices = analysis_data.get('market_indices', {})
            kospi_data = indices.get('KOSPI', {})
            kosdaq_data = indices.get('KOSDAQ', {})
            
            # 점수 계산
            score = 0
            
            # 시장 참여도 점수 (0~2점)
            if market_breadth >= 60:
                score += 2
            elif market_breadth >= 50:
                score += 1
                
            # 모멘텀 점수 (-2~2점)
            if market_momentum >= 5:
                score += 2
            elif market_momentum >= 2:
                score += 1
            elif market_momentum <= -5:
                score -= 2
            elif market_momentum <= -2:
                score -= 1
                
            # 성장주/가치주 점수 (-1~1점)
            if growth_vs_value >= 3:
                score += 1
            elif growth_vs_value <= -3:
                score -= 1
                
            # 변동성 점수 (-1~1점)
            kospi_vol = kospi_data.get('volatility', 0)
            kosdaq_vol = kosdaq_data.get('volatility', 0)
            avg_vol = (kospi_vol + kosdaq_vol) / 2
            
            if avg_vol >= 30:
                score -= 1
            elif avg_vol <= 15:
                score += 1
                
            # 최종 심리 판단
            if score >= 4:
                sentiment = "매우 긍정적"
            elif score >= 2:
                sentiment = "긍정적"
            elif score >= 0:
                sentiment = "중립적"
            elif score >= -2:
                sentiment = "부정적"
            else:
                sentiment = "매우 부정적"
                
            logger.info(f"시장 심리 판단 완료: {sentiment} (점수: {score})")
            return sentiment
            
        except Exception as e:
            logger.error(f"시장 심리 판단 실패: {e}")
            return "Neutral"

    def _generate_recommendations(self, analysis_data: Dict) -> List[str]:
        """투자 권고사항 생성"""
        try:
            recommendations = []
            
            # 시장 심리
            sentiment = analysis_data.get('market_sentiment', 'Neutral')
            
            # 기본 전략 추천
            if sentiment in ["매우 긍정적", "긍정적"]:
                recommendations.append("시장 상승세가 강해 적극적인 매수 전략 고려")
                recommendations.append("성장주 중심의 포트폴리오 구성 검토")
            elif sentiment in ["매우 부정적", "부정적"]:
                recommendations.append("시장 하락 위험이 높아 보수적인 포지션 유지")
                recommendations.append("가치주 중심의 안정적인 포트폴리오 구성 권장")
            else:
                recommendations.append("중립적인 시장 상황으로 섹터 분산 투자 권장")
                
            # 섹터별 추천
            if 'sector_analysis' in analysis_data:
                sectors = analysis_data['sector_analysis'].get('sectors', {})
                strong_sectors = [
                    (name, data) for name, data in sectors.items()
                    if data.get('trend') == "상승" and data.get('return_rate', 0) > 0
                ]
                
                if strong_sectors:
                    top_sectors = sorted(strong_sectors, key=lambda x: x[1]['return_rate'], reverse=True)[:3]
                    sector_names = [s[0] for s in top_sectors]
                    recommendations.append(f"강세 섹터 주목: {', '.join(sector_names)}")
                    
            # 테마별 추천
            if 'theme_analysis' in analysis_data:
                themes = analysis_data['theme_analysis'].get('themes', {})
                strong_themes = [
                    (name, data) for name, data in themes.items()
                    if data.get('average_return', 0) > 5
                ]
                
                if strong_themes:
                    top_themes = sorted(strong_themes, key=lambda x: x[1]['average_return'], reverse=True)[:3]
                    theme_names = [t[0] for t in top_themes]
                    recommendations.append(f"주목할 테마: {', '.join(theme_names)}")
                    
            # 리스크 관리 추천
            macro_indicators = analysis_data.get('macro_indicators', {})
            market_breadth = macro_indicators.get('market_breadth', 0)
            
            if market_breadth < 40:
                recommendations.append("시장 참여도가 낮아 리스크 관리 강화 필요")
                
            return recommendations
            
        except Exception as e:
            logger.error(f"투자 권고사항 생성 실패: {e}")
            return ["시장 상황을 더 지켜볼 필요가 있습니다."]

    def _generate_etf_charts(self, df_etf: pd.DataFrame) -> int:
        """ETF 차트 생성"""
        try:
            chart_count = 0
            max_charts = 20
            
            # 차트 생성용 설정
            cm = tradeStrategy('./config/config.yaml')
            cm.display = 'save'
            
            chart_start_str = self.chart_start_date.strftime("%Y%m%d")
            chart_end_str = self.end_date.strftime("%Y%m%d")
            
            # 제외할 ETF 패턴
            exclude_patterns = ["200선물", "코스닥", "인버스", "레버리지"]
            
            for code in df_etf.index.to_list():
                if chart_count >= max_charts:
                    break
                    
                code_str = str(code).zfill(6)
                name = df_etf.at[code, "Name"]
                change = round(df_etf.at[code, "Change"], 2)
                
                # 제외 패턴 확인
                if any(pattern in name for pattern in exclude_patterns):
                    continue
                    
                try:
                    # ETF OHLCV 데이터 수집
                    df_ohlcv = stock.get_etf_ohlcv_by_date(chart_start_str, chart_end_str, code_str)
                    if df_ohlcv.empty:
                        continue
                        
                    df_ohlcv.rename(columns={
                        '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close',
                        '거래량': 'Volume', '등락률': 'Change', '변동폭': 'ChangeRatio',
                        '거래대금': 'VolumeCost', '기초지수': 'BaseCost'
                    }, inplace=True)
                    
                    logger.info(f"[수익률: {change}%] ETF ({name}, {code_str}) 차트 생성")
                    
                    # 차트 생성
                    df_chart = cm.run(code_str, name, data=df_ohlcv, 
                                    dates=[chart_start_str, chart_end_str], mode='etf')
                    
                    # 차트 저장 경로 생성
                    from main import create_daily_directories
                    chart_base_path = self.report_config.get('save_path', './data/processed/macro_analysis/')
                    daily_chart_path = create_daily_directories(chart_base_path)
                    
                    chart_file_path = os.path.join(daily_chart_path, f"{code_str}_{name}.csv")
                    df_chart.to_csv(chart_file_path, index=False, encoding='utf-8-sig')
                    
                    chart_count += 1
                    
                except Exception as e:
                    logger.warning(f"ETF {name}({code_str}) 차트 생성 실패: {e}")
                    continue
                    
            logger.info(f"총 {chart_count}개의 ETF 차트가 생성되었습니다.")
            return chart_count
            
        except Exception as e:
            logger.error(f"ETF 차트 생성 실패: {e}")
            return 0

    def _analyze_sector_rotation(self, sector_performance: Dict) -> Dict[str, Any]:
        """섹터 로테이션 분석"""
        try:
            # 섹터 분류
            defensive_sectors = ['유틸리티', '금융', '통신서비스']
            cyclical_sectors = ['소비재', '산업재', '원자재']
            growth_sectors = ['기술/IT', '바이오/헬스케어']
            
            sector_types = {
                'defensive': [],
                'cyclical': [],
                'growth': []
            }
            
            # 섹터별 성과 분류
            for sector_name, data in sector_performance.items():
                if sector_name in defensive_sectors:
                    sector_types['defensive'].append((sector_name, data))
                elif sector_name in cyclical_sectors:
                    sector_types['cyclical'].append((sector_name, data))
                elif sector_name in growth_sectors:
                    sector_types['growth'].append((sector_name, data))
                
            # 섹터 유형별 평균 성과 계산
            type_performance = {}
            for type_name, sectors in sector_types.items():
                if sectors:
                    returns = [data.get('return_rate', 0) for _, data in sectors]
                    type_performance[type_name] = {
                        'average_return': round(sum(returns) / len(returns), 2),
                        'sector_count': len(sectors),
                        'top_sector': max(sectors, key=lambda x: x[1].get('return_rate', 0))[0]
                    }
                
            # 로테이션 단계 판단
            rotation_stage = "불명확"
            
            def_return = type_performance.get('defensive', {}).get('average_return', 0)
            cyc_return = type_performance.get('cyclical', {}).get('average_return', 0)
            gro_return = type_performance.get('growth', {}).get('average_return', 0)
            
            if gro_return > cyc_return > def_return:
                rotation_stage = "성장 국면"
            elif cyc_return > gro_return > def_return:
                rotation_stage = "확장 국면"
            elif def_return > cyc_return and def_return > gro_return:
                rotation_stage = "방어 국면"
                
            return {
                'stage': rotation_stage,
                'type_performance': type_performance
            }
            
        except Exception as e:
            logger.error(f"섹터 로테이션 분석 실패: {e}")
            return {}

    def _identify_emerging_themes(self, theme_performance: Dict) -> List[Dict]:
        """신규 테마 식별"""
        try:
            emerging_themes = []
            
            for theme_name, data in theme_performance.items():
                avg_return = data.get('average_return', 0)
                stock_count = data.get('stock_count', 0)
                top_stocks = data.get('top_stocks', [])
                
                # 신규 테마 판단 기준
                if avg_return > 10 and stock_count >= 3:
                    emerging_themes.append({
                        'name': theme_name,
                        'return_rate': avg_return,
                        'stock_count': stock_count,
                        'representative_stocks': [
                            {'name': stock['name'], 'return_rate': stock['return_rate']}
                            for stock in top_stocks[:3]
                        ]
                    })
                
            # 수익률 기준 정렬
            emerging_themes.sort(key=lambda x: x['return_rate'], reverse=True)
            
            return emerging_themes[:5]  # 상위 5개 신규 테마만 반환
            
        except Exception as e:
            logger.error(f"신규 테마 식별 실패: {e}")
            return []

    def _perform_llm_analysis(self, analysis_data: Dict) -> Dict[str, Any]:
        """LLM 기반 시황 분석"""
        try:
            if not self.llm_config.get("enabled", False):
                logger.info("LLM 분석이 비활성화되어 건너뜁니다")
                return {"llm_enabled": False}
                
            logger.info("LLM 시황 분석 시작")
            
            llm_results = {}
            api_error_count = 0
            max_api_errors = 3  # 최대 3번의 API 에러까지 허용
            
            # 분석 범위 확인
            analysis_scope = self.llm_config.get("analysis_scope", {})
            
            # API 할당량 체크를 위한 간단한 테스트 호출
            try:
                test_prompt = "간단한 테스트입니다. '테스트 완료'라고 답해주세요."
                test_result = self._call_llm(test_prompt, 'api_test')
                if 'error' in test_result:
                    error_msg = str(test_result.get('error', ''))
                    if 'insufficient_quota' in error_msg or '429' in error_msg:
                        logger.warning("OpenAI API 할당량 초과 - LLM 분석을 건너뜁니다")
                        return {
                            'llm_results': {},
                            'total_tokens_used': 0,
                            'analysis_timestamp': datetime.now().isoformat(),
                            'api_quota_exceeded': True,
                            'message': 'OpenAI API 할당량 초과로 인해 LLM 분석을 건너뛰었습니다.'
                        }
            except Exception as e:
                logger.warning(f"API 테스트 실패 - LLM 분석을 건너뜁니다: {e}")
                return {
                    'llm_results': {},
                    'total_tokens_used': 0,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'api_error': True,
                    'message': f'API 연결 실패로 인해 LLM 분석을 건너뛰었습니다: {e}'
                }
                
            # 1. 전체 시장 개관
            if analysis_scope.get("market_overview", True) and api_error_count < max_api_errors:
                market_prompt = self._create_market_overview_prompt(analysis_data)
                result = self._call_llm(market_prompt, "market_overview")
                llm_results['market_overview'] = result
                if 'error' in result:
                    api_error_count += 1
                
            # 2. 섹터별 상세 분석 (API 에러가 적으면 계속)
            if analysis_scope.get("sector_analysis", True) and api_error_count < max_api_errors:
                sector_prompt = self._create_sector_analysis_prompt(analysis_data)
                result = self._call_llm(sector_prompt, "sector_analysis")
                llm_results['sector_analysis'] = result
                if 'error' in result:
                    api_error_count += 1
                
            # 3. ETF 트렌드 분석
            if analysis_scope.get("etf_trends", True) and api_error_count < max_api_errors:
                etf_prompt = self._create_etf_trends_prompt(analysis_data)
                result = self._call_llm(etf_prompt, "etf_trends")
                llm_results['etf_trends'] = result
                if 'error' in result:
                    api_error_count += 1
                
            # API 에러가 많으면 나머지 분석 건너뛰기
            if api_error_count >= max_api_errors:
                logger.warning(f"API 에러가 {api_error_count}회 발생하여 나머지 LLM 분석을 건너뜁니다")
            else:
                # 4. 테마별 분석
                if analysis_scope.get("theme_analysis", True):
                    theme_prompt = self._create_theme_analysis_prompt(analysis_data)
                    result = self._call_llm(theme_prompt, "theme_analysis")
                    llm_results['theme_analysis'] = result
                    if 'error' in result:
                        api_error_count += 1
                    
                # 5. 기술적 지표 분석
                if analysis_scope.get("technical_indicators", True) and api_error_count < max_api_errors:
                    technical_prompt = self._create_technical_analysis_prompt(analysis_data)
                    result = self._call_llm(technical_prompt, "technical_indicators")
                    llm_results['technical_indicators'] = result
                    if 'error' in result:
                        api_error_count += 1
                    
                # 6. 시장 심리 분석
                if analysis_scope.get("sentiment_analysis", True) and api_error_count < max_api_errors:
                    sentiment_prompt = self._create_sentiment_analysis_prompt(analysis_data)
                    result = self._call_llm(sentiment_prompt, "sentiment_analysis")
                    llm_results['sentiment_analysis'] = result
                    if 'error' in result:
                        api_error_count += 1
                    
                # 7. 종합 분석 (가장 중요하므로 마지막에)
                if api_error_count < max_api_errors:
                    comprehensive_prompt = self._create_comprehensive_analysis_prompt(analysis_data, llm_results)
                    result = self._call_llm(comprehensive_prompt, "comprehensive_analysis")
                    llm_results['comprehensive_analysis'] = result
                
            # 토큰 사용량 계산
            total_tokens = sum(
                result.get('tokens_used', 0)
                for result in llm_results.values()
                if isinstance(result, dict)
            )
            
            return {
                'llm_results': llm_results,
                'total_tokens_used': total_tokens,
                'analysis_timestamp': datetime.now().isoformat(),
                'api_errors': api_error_count
            }
            
        except Exception as e:
            logger.error(f"LLM 분석 중 오류: {e}")
            logger.exception("상세 오류:")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }

    def _create_market_overview_prompt(self, data: Dict) -> str:
        """전체 시장 개관 프롬프트 생성"""
        market_indices = data.get('market_indices', {})
        etf_analysis = data.get('etf_analysis', {})
        
        prompt = f"""
한국 주식시장 전체 시황을 분석해주세요.

## 분석 기간
- 기간: {data.get('period', 'N/A')}
- 분석일: {data.get('analysis_date', 'N/A')}

## 주요 지수 현황
"""
        
        for index_name, index_data in market_indices.items():
            prompt += f"- {index_name}: {index_data.get('return_rate', 0):+.2f}% (변동성: {index_data.get('volatility', 0):.2f}%)\n"
        
        prompt += f"""
## ETF 시장 현황
- 전체 ETF 수: {etf_analysis.get('total_etfs', 0)}개
- 상승 ETF 비율: {etf_analysis.get('positive_ratio', 0):.1f}%

다음 관점에서 분석해주세요:
1. 전체 시장의 방향성과 강도
2. 주요 지수 간 상관관계 및 의미
3. 시장 참여자들의 심리 상태
4. 현재 시장 사이클에서의 위치
5. 단기/중기 시장 전망

분석 결과를 한국어로 상세하고 전문적으로 작성해주세요.
"""
        return prompt

    def _create_sector_analysis_prompt(self, data: Dict) -> str:
        """섹터별 분석 프롬프트 생성"""
        sector_data = data.get('sector_analysis', {})
        top_sectors = sector_data.get('top_sectors', [])
        sector_rotation = sector_data.get('sector_rotation', {})
        
        prompt = f"""
한국 주식시장의 섹터별 상세 분석을 수행해주세요.

## 상위 성과 섹터 (최근 {self.analysis_months}개월)
"""
        
        for i, sector in enumerate(top_sectors[:10], 1):
            prompt += f"{i}. {sector.get('name', 'N/A')}: {sector.get('return_rate', 0):+.2f}% (변동성: {sector.get('volatility', 0):.2f}%)\n"
        
        prompt += f"""
## 섹터 로테이션 현황
- 로테이션 패턴: {sector_rotation.get('rotation_pattern', 'N/A')}
- 강세 섹터: {len(sector_rotation.get('strong_sectors', []))}개
- 약세 섹터: {len(sector_rotation.get('weak_sectors', []))}개

다음 관점에서 분석해주세요:
1. 섹터별 성과의 배경과 원인 분석
2. 섹터 로테이션의 의미와 시사점
3. 경제 사이클과 섹터 성과의 연관성
4. 향후 주목해야 할 섹터와 그 이유
5. 섹터별 투자 전략 및 리스크 요인

각 섹터의 특성과 현재 시장 환경을 고려하여 전문적으로 분석해주세요.
"""
        return prompt

    def _create_etf_trends_prompt(self, data: Dict) -> str:
        """ETF 트렌드 분석 프롬프트 생성"""
        etf_data = data.get('etf_analysis', {})
        sector_breakdown = etf_data.get('sector_breakdown', {})
        top_performers = etf_data.get('top_performers', [])
        
        prompt = f"""
한국 ETF 시장의 트렌드를 분석해주세요.

## ETF 시장 현황
- 전체 분석 ETF: {etf_data.get('total_etfs', 0)}개
- 상승 ETF: {etf_data.get('positive_etfs', 0)}개 ({etf_data.get('positive_ratio', 0):.1f}%)

## 상위 성과 ETF
"""
        
        for i, etf in enumerate(top_performers[:10], 1):
            prompt += f"{i}. {etf.get('name', 'N/A')} ({etf.get('code', 'N/A')}): {etf.get('change', 0):+.2f}%\n"
        
        prompt += f"""
## 섹터별 ETF 현황
"""
        
        for sector, etfs in sector_breakdown.items():
            if etfs:
                prompt += f"- {sector}: {len(etfs)}개 ETF\n"
        
        prompt += f"""
다음 관점에서 분석해주세요:
1. ETF 시장의 전반적인 트렌드와 특징
2. 섹터별 ETF 성과의 의미와 투자자 선호도
3. 테마/섹터별 자금 흐름 분석
4. ETF를 통해 본 투자자 심리와 시장 전망
5. 주목할 만한 ETF와 투자 기회

ETF 시장의 특성과 최근 동향을 반영하여 분석해주세요.
"""
        return prompt

    def _create_theme_analysis_prompt(self, data: Dict) -> str:
        """테마별 분석 프롬프트 생성"""
        theme_data = data.get('theme_analysis', {})
        hot_themes = theme_data.get('hot_themes', [])
        emerging_themes = theme_data.get('emerging_themes', [])
        
        prompt = f"""
한국 주식시장의 테마별 트렌드를 분석해주세요.

## 핫 테마 (상승률 5% 이상)
"""
        
        for i, theme in enumerate(hot_themes, 1):
            prompt += f"{i}. {theme.get('name', 'N/A')}: {theme.get('return', 0):+.2f}% (관련 종목 {theme.get('stock_count', 0)}개)\n"
        
        prompt += f"""
## 신흥 테마
"""
        
        for i, theme in enumerate(emerging_themes, 1):
            prompt += f"{i}. {theme.get('name', 'N/A')}: {theme.get('return', 0):+.2f}% (잠재력: {theme.get('potential', 'N/A')})\n"
        
        prompt += f"""
다음 관점에서 분석해주세요:
1. 현재 시장을 주도하는 핫 테마의 배경과 지속성
2. 신흥 테마의 성장 가능성과 투자 기회
3. 테마별 밸류에이션과 리스크 수준
4. 글로벌 트렌드와 한국 시장의 연관성
5. 향후 주목해야 할 테마와 투자 전략

각 테마의 펀더멘털과 시장 환경을 종합적으로 고려하여 분석해주세요.
"""
        return prompt

    def _create_technical_analysis_prompt(self, data: Dict) -> str:
        """기술적 지표 분석 프롬프트 생성"""
        market_indices = data.get('market_indices', {})
        macro_indicators = data.get('macro_indicators', {})
        
        prompt = f"""
한국 주식시장의 기술적 지표를 분석해주세요.

## 주요 지수 기술적 현황
"""
        
        for index_name, index_data in market_indices.items():
            prompt += f"""
- {index_name}:
  * 수익률: {index_data.get('return_rate', 0):+.2f}%
  * 변동성: {index_data.get('volatility', 0):.2f}%
  * 현재 수준: {index_data.get('current_level', 0):,.0f}
  * 트렌드: {index_data.get('trend', 'N/A')}
"""
        
        prompt += f"""
## 거시 지표
- 시장 폭: {macro_indicators.get('market_breadth', 0):.1f}%
- 시장 모멘텀: {macro_indicators.get('market_momentum', 0):+.2f}%
- 성장주 vs 가치주: {macro_indicators.get('growth_vs_value', 0):+.2f}%

다음 관점에서 기술적 분석을 수행해주세요:
1. 주요 지수의 기술적 패턴과 지지/저항 수준
2. 시장 폭과 모멘텀 지표의 해석
3. 변동성 수준과 시장 리스크 평가
4. 단기/중기 기술적 전망
5. 기술적 관점에서의 매매 전략

차트 패턴, 이동평균, 거래량 등을 종합적으로 고려하여 분석해주세요.
"""
        return prompt

    def _create_sentiment_analysis_prompt(self, data: Dict) -> str:
        """시장 심리 분석 프롬프트 생성"""
        market_sentiment = data.get('market_sentiment', 'Neutral')
        etf_data = data.get('etf_analysis', {})
        sector_data = data.get('sector_analysis', {})
        
        prompt = f"""
한국 주식시장의 투자 심리를 분석해주세요.

## 현재 시장 심리
- 전체 심리: {market_sentiment}
- ETF 상승 비율: {etf_data.get('positive_ratio', 0):.1f}%
- 섹터 브레드스: {sector_data.get('sector_rotation', {}).get('sector_breadth', 0)}

## 심리 지표
- 상승 ETF 수: {etf_data.get('positive_etfs', 0)}개 / {etf_data.get('total_etfs', 0)}개
- 강세 섹터 수: {len(sector_data.get('sector_rotation', {}).get('strong_sectors', []))}개
- 약세 섹터 수: {len(sector_data.get('sector_rotation', {}).get('weak_sectors', []))}개

다음 관점에서 시장 심리를 분석해주세요:
1. 현재 투자자 심리의 특징과 배경
2. 시장 참여자별(개인/기관/외국인) 동향 추정
3. 공포/탐욕 지수와 시장 과열/침체 신호
4. 심리적 지지/저항 수준과 변곡점
5. 심리 변화가 시장에 미칠 영향

행동경제학적 관점과 시장 데이터를 종합하여 분석해주세요.
"""
        return prompt

    def _create_comprehensive_analysis_prompt(self, data: Dict, llm_results: Dict) -> str:
        """종합 분석 프롬프트 생성"""
        prompt = f"""
앞서 분석한 모든 내용을 종합하여 한국 주식시장에 대한 최종 투자 전략을 제시해주세요.

## 분석 요약
- 분석 기간: {data.get('period', 'N/A')}
- 시장 심리: {data.get('market_sentiment', 'N/A')}
- 분석 범위: 시장지수, ETF, 섹터, 테마, 기술적 지표, 시장 심리

## 종합 분석 요청사항

### 1. 시장 현황 종합 평가
- 현재 시장 상황의 핵심 특징
- 주요 리스크와 기회 요인
- 시장 사이클에서의 위치

### 2. 투자 전략 제안
- 단기 (1-4주) 투자 전략
- 중기 (1-3개월) 투자 전략  
- 장기 (3-6개월) 투자 전략

### 3. 섹터/테마별 투자 우선순위
- 1순위 투자 대상과 근거
- 2순위 투자 대상과 근거
- 회피해야 할 섹터/테마

### 4. 리스크 관리 방안
- 주요 리스크 요인 식별
- 포트폴리오 리스크 관리 전략
- 손절/익절 기준 제시

### 5. 시장 모니터링 포인트
- 주요 모니터링 지표
- 투자 전략 수정이 필요한 시그널
- 다음 분석까지 주목할 이벤트

전문적이고 실용적인 투자 가이드를 한국어로 작성해주세요.
각 전략에는 구체적인 근거와 실행 방안을 포함해주세요.
"""
        return prompt

    def _call_llm(self, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """LLM API 호출"""
        try:
            logger.debug(f"{analysis_type} LLM 분석 시작")
            
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 한국 주식시장 전문 애널리스트입니다. 데이터를 바탕으로 정확하고 실용적인 시장 분석을 제공해주세요."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature
            )
            
            analysis_result = {
                'analysis_type': analysis_type,
                'content': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"{analysis_type} LLM 분석 완료 (토큰: {analysis_result['tokens_used']})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"{analysis_type} LLM 분석 실패: {e}")
            return {
                'analysis_type': analysis_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_report(self, analysis_data: Dict) -> str:
        """분석 리포트 생성 및 저장"""
        try:
            # 리포트 저장 경로 생성
            from main import create_daily_directories
            report_base_path = self.report_config.get('save_path', './data/processed/macro_analysis/')
            daily_report_path = create_daily_directories(report_base_path)
            
            # 리포트 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"macro_analysis_report_{timestamp}.json"
            report_filepath = os.path.join(daily_report_path, report_filename)
            
            # JSON 리포트 저장
            with open(report_filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                
            # 텍스트 요약 리포트 생성
            summary_filename = f"macro_analysis_summary_{timestamp}.txt"
            summary_filepath = os.path.join(daily_report_path, summary_filename)
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("📊 한국 주식시장 거시경제 분석 리포트\n")
                f.write("=" * 100 + "\n")
                f.write(f"🗓️ 분석 일시: {analysis_data['analysis_date']}\n")
                f.write(f"📅 분석 기간: {analysis_data['period']}\n")
                f.write(f"💭 시장 심리: {analysis_data['market_sentiment']}\n")
                f.write("\n")
                
                # 시장 지수 요약
                if analysis_data.get('market_indices'):
                    f.write("📈 시장 지수 현황\n")
                    f.write("-" * 50 + "\n")
                    for name, data in analysis_data['market_indices'].items():
                        trend_emoji = "🔴" if data.get('return_rate', 0) < 0 else "🟢"
                        f.write(f"{trend_emoji} {name}: {data.get('return_rate', 0):+.2f}% "
                               f"(변동성: {data.get('volatility', 0):.2f}%)\n")
                    f.write("\n")
                
                # ETF 분석 요약
                if analysis_data.get('etf_analysis'):
                    etf_data = analysis_data['etf_analysis']
                    f.write("🏛️ ETF 시장 현황\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"📊 전체 ETF 수: {etf_data.get('total_etfs', 0)}개\n")
                    f.write(f"📈 상승 ETF 수: {etf_data.get('positive_etfs', 0)}개\n")
                    f.write(f"📊 상승 비율: {etf_data.get('positive_ratio', 0):.1f}%\n")
                    f.write("\n")
                
                # 섹터 분석 요약
                if analysis_data.get('sector_analysis'):
                    sector_data = analysis_data['sector_analysis']
                    f.write("🏭 섹터 분석 현황\n")
                    f.write("-" * 50 + "\n")
                    top_sectors = sector_data.get('top_sectors', [])[:5]
                    for i, sector in enumerate(top_sectors, 1):
                        f.write(f"{i}. {sector.get('name', 'N/A')}: {sector.get('return_rate', 0):+.2f}%\n")
                    f.write("\n")
                
                # 테마 분석 요약
                if analysis_data.get('theme_analysis'):
                    theme_data = analysis_data['theme_analysis']
                    hot_themes = theme_data.get('hot_themes', [])
                    if hot_themes:
                        f.write("🔥 핫 테마 현황\n")
                        f.write("-" * 50 + "\n")
                        for i, theme in enumerate(hot_themes[:5], 1):
                            f.write(f"{i}. {theme.get('name', 'N/A')}: {theme.get('return', 0):+.2f}%\n")
                        f.write("\n")
                
                # LLM 분석 결과
                if analysis_data.get('llm_analysis'):
                    llm_data = analysis_data['llm_analysis']
                    f.write("🤖 AI 기반 시황 분석\n")
                    f.write("=" * 100 + "\n")
                    
                    # 각 LLM 분석 결과를 섹션별로 작성
                    for analysis_type, analysis_result in llm_data.items():
                        if 'content' in analysis_result:
                            section_titles = {
                                'market_overview': '📊 시장 전체 개관',
                                'sector_analysis': '🏭 섹터별 상세 분석',
                                'etf_trends': '🏛️ ETF 트렌드 분석',
                                'theme_analysis': '🔥 테마별 분석',
                                'technical_indicators': '📈 기술적 지표 분석',
                                'sentiment_analysis': '💭 시장 심리 분석',
                                'comprehensive_analysis': '🎯 종합 분석 및 투자 전략'
                            }
                            
                            title = section_titles.get(analysis_type, analysis_type.upper())
                            f.write(f"\n{title}\n")
                            f.write("-" * 80 + "\n")
                            f.write(analysis_result['content'])
                            f.write("\n\n")
                            
                            # 토큰 사용량 정보
                            tokens = analysis_result.get('tokens_used', 0)
                            if tokens > 0:
                                f.write(f"💡 분석 토큰 사용량: {tokens:,}개\n")
                            f.write("\n")
                
                # 기존 투자 권고사항
                if analysis_data.get('recommendations'):
                    f.write("💡 기본 투자 권고사항\n")
                    f.write("-" * 50 + "\n")
                    for i, rec in enumerate(analysis_data['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                # 리포트 메타데이터
                f.write("=" * 100 + "\n")
                f.write("📋 리포트 메타데이터\n")
                f.write("-" * 50 + "\n")
                f.write(f"🕐 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"📊 분석된 ETF 수: {analysis_data.get('etf_analysis', {}).get('total_etfs', 0)}개\n")
                f.write(f"🏭 분석된 섹터 수: {analysis_data.get('sector_analysis', {}).get('total_sectors_analyzed', 0)}개\n")
                f.write(f"🔥 분석된 테마 수: {analysis_data.get('theme_analysis', {}).get('total_themes_analyzed', 0)}개\n")
                f.write(f"📈 생성된 차트 수: {analysis_data.get('etf_analysis', {}).get('charts_generated', 0)}개\n")
                
                # LLM 분석 메타데이터
                if analysis_data.get('llm_analysis'):
                    total_tokens = sum(
                        result.get('tokens_used', 0) 
                        for result in analysis_data['llm_analysis'].values() 
                        if isinstance(result, dict)
                    )
                    f.write(f"🤖 LLM 총 토큰 사용량: {total_tokens:,}개\n")
                    f.write(f"🧠 사용된 AI 모델: {self.llm_model}\n")
                
                f.write("=" * 100 + "\n")
                f.write("✅ 리포트 생성 완료\n")
                f.write("=" * 100 + "\n")
                
            logger.info(f"거시경제 분석 리포트 저장 완료: {report_filepath}")
            return report_filepath
            
        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            return ""


if __name__ == "__main__":
    sm = searchMacro()
    result = sm.run()
    
    if 'error' not in result:
        print("거시경제 분석 완료!")
        print(f"시장 심리: {result['market_sentiment']}")
        print(f"리포트 경로: {result.get('report_path', 'N/A')}")
    else:
        print(f"분석 실패: {result['error']}")