#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST (System Trading) v0.1 - 메인 실행 시스템
스윙 매매 기반 한국 주식 자동매매 시스템

Author: ST Development Team
Created: 2024-12-19
"""

import os
import sys
import asyncio
import argparse
import signal
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import uvicorn
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 로컬 모듈 import
from tools import st_utils as stu
from system_trade import systemTrade
from trade_strategy import tradeStrategy
from search_stocks import searchStocks
from search_macro import searchMacro
from finance_score import financeScore

# 로거 설정
from tools.custom_logger import configure_pykrx_logging
logger = stu.create_logger()
configure_pykrx_logging()

class STSystemManager:
    """
    ST 자동매매 시스템 메인 관리자
    
    주요 기능:
    - 시스템 초기화 및 설정 검증
    - 모드별 실행 (백테스팅, 실시간 거래, API 서버)
    - 모듈 간 의존성 관리
    - 안전한 종료 처리
    """
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        self.config_path = config_path
        self.config: Optional[Dict[str, Any]] = None
        self.running = False
        self.modules = {}
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """시스템 종료 시그널 처리"""
        logger.info(f"종료 시그널 수신 (Signal: {signum}). 안전하게 종료합니다...")
        self.running = False
        
    def load_config(self) -> bool:
        """설정 파일 로딩 및 검증"""
        try:
            logger.info("설정 파일 로딩 시작...")
            if not os.path.exists(self.config_path):
                logger.error(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
                return False
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
                
            logger.info(f"설정 파일 로딩 완료: {self.config_path}")
            logger.debug(f"로드된 설정: {self.config.keys()}")
            return self._validate_config()
            
        except yaml.YAMLError as e:
            logger.error(f"YAML 파싱 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"설정 파일 로딩 실패: {str(e)}")
            logger.exception("상세 오류:")
            return False
            
    def _validate_config(self) -> bool:
        """설정 파일 필수 항목 검증"""
        required_sections = [
            'mainInit', 'tradeStock', 'searchStock', 
            'searchMacro', 'scoreRule', 'fileControl'
        ]
        
        logger.info("설정 파일 검증 시작...")
        
        # 필수 섹션 검증
        for section in required_sections:
            if section not in self.config:
                logger.error(f"필수 설정 섹션 누락: {section}")
                return False
            else:
                logger.debug(f"설정 섹션 확인: {section}")
                
        # API 키 검증
        kis_config_path = './config/kisdev_vi.yaml'
        if not os.path.exists(kis_config_path):
            logger.error(f"한국투자증권 API 설정 파일 누락: {kis_config_path}")
            return False
            
        logger.info("설정 파일 검증 완료")
        return True
        
    def initialize_modules(self) -> bool:
        """시스템 모듈 초기화"""
        try:
            logger.info("시스템 모듈 초기화 시작...")
            
            # 1. 거래 시스템 초기화
            logger.info("거래 시스템 초기화 중...")
            try:
                self.modules['trade_system'] = systemTrade(
                    mode=self.config['tradeStock']['scheduler']['mode']
                )
                logger.debug("거래 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"거래 시스템 초기화 실패: {str(e)}")
                raise
            
            # 2. 전략 분석 모듈 초기화
            logger.info("전략 분석 모듈 초기화 중...")
            try:
                self.modules['strategy'] = tradeStrategy(self.config_path)
                logger.debug("전략 분석 모듈 초기화 완료")
            except Exception as e:
                logger.error(f"전략 분석 모듈 초기화 실패: {str(e)}")
                raise
            
            # 3. 종목 검색 모듈 초기화
            logger.info("종목 검색 모듈 초기화 중...")
            try:
                self.modules['stock_search'] = searchStocks(self.config_path)
                logger.debug("종목 검색 모듈 초기화 완료")
            except Exception as e:
                logger.error(f"종목 검색 모듈 초기화 실패: {str(e)}")
                raise
            
            # 4. 거시경제 분석 모듈 초기화
            logger.info("거시경제 분석 모듈 초기화 중...")
            try:
                self.modules['macro_search'] = searchMacro()
                logger.debug("거시경제 분석 모듈 초기화 완료")
            except Exception as e:
                logger.error(f"거시경제 분석 모듈 초기화 실패: {str(e)}")
                raise
            
            # 5. 재무 점수 모듈 초기화
            logger.info("재무 점수 모듈 초기화 중...")
            try:
                self.modules['finance_score'] = financeScore(self.config_path)
                logger.debug("재무 점수 모듈 초기화 완료")
            except Exception as e:
                logger.error(f"재무 점수 모듈 초기화 실패: {str(e)}")
                raise
            
            logger.info("모든 모듈 초기화 완료")
            return True
            
        except Exception as e:
            logger.error("모듈 초기화 중 치명적 오류 발생")
            logger.exception("상세 오류:")
            return False
            
    def run_backtest_mode(self):
        """백테스팅 모드 실행"""
        logger.info("=" * 60)
        logger.info("=== 백테스팅 모드 시작 ===")
        logger.info("=" * 60)
        
        try:
            # 1. 종목 스크리닝
            logger.info("[1/3] 종목 스크리닝 실행 시작")
            stock_search = self.modules['stock_search']
            
            logger.info("시장 주도주 검색 중...")
            try:
                stock_search.search_market_leader()
                logger.debug("시장 주도주 검색 완료")
            except Exception as e:
                logger.error(f"시장 주도주 검색 실패: {str(e)}")
                raise
            
            logger.info("테마/업종 검색 중...")
            try:
                stock_search.search_theme_upjong(mode=self.config['tradeStock']['scheduler']['mode'])
                logger.debug("테마/업종 검색 완료")
            except Exception as e:
                logger.error(f"테마/업종 검색 실패: {str(e)}")
                raise
            
            # 2. 재무 점수 계산
            logger.info("[2/3] 재무 점수 계산 시작")
            finance_score = self.modules['finance_score']
            try:
                finance_score.run()
                logger.debug("재무 점수 계산 완료")
            except Exception as e:
                logger.error(f"재무 점수 계산 실패: {str(e)}")
                raise
            
            # 3. 전략 백테스팅
            logger.info("[3/3] 전략 백테스팅 실행")
            # TODO: 백테스팅 로직 구현
            logger.warning("백테스팅 로직이 아직 구현되지 않았습니다")
            
            logger.info("=" * 60)
            logger.info("백테스팅 완료")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error("백테스팅 실행 중 치명적 오류 발생")
            logger.exception("상세 오류:")
            raise
            
    async def run_trading_mode(self):
        """실시간 거래 모드 실행"""
        logger.info("=== 실시간 거래 모드 시작 ===")
        
        try:
            # 새로운 거래 엔진 사용
            from core.engines.trading_engine import TradingEngine, TradingEngineConfig
            
            # 거래 엔진 설정
            trading_config = TradingEngineConfig(
                max_positions=self.config['trading']['max_positions'],
                position_size_pct=self.config['trading']['position_size_pct'],
                stop_loss_pct=self.config['trading']['stop_loss_pct'],
                take_profit_pct=self.config['trading']['take_profit_pct'],
                daily_loss_limit_pct=self.config['trading']['daily_loss_limit_pct'],
                check_interval_seconds=self.config['trading']['check_interval_seconds'],
                market_open_time=self.config['trading']['market_open_time'],
                market_close_time=self.config['trading']['market_close_time'],
                strategy_name=self.config.get('trading', {}).get('strategy_name', 'Combined')
            )
            
            # 거래 엔진 생성
            trading_engine = TradingEngine(trading_config)
            
            # 거래 대상 종목 선정 (종목 발굴 결과 사용)
            candidates = self._select_final_candidates()
            symbols = [c['code'] for c in candidates[:10]]  # 상위 10개 종목
            
            if not symbols:
                logger.warning("거래할 종목이 없습니다. 종목 발굴을 먼저 실행해주세요.")
                return
            
            logger.info(f"거래 대상 종목: {symbols}")
            
            # 거래 시작
            await trading_engine.start_trading(symbols)
                
        except Exception as e:
            logger.error(f"실시간 거래 실행 중 오류: {e}")
            
    async def run_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """FastAPI 서버 모드 실행"""
        logger.info(f"=== API 서버 모드 시작 (http://{host}:{port}) ===")
        
        try:
            # FastAPI 앱 import (추후 구현)
            # from api.main import app
            
            config = uvicorn.Config(
                "api.main:app",
                host=host,
                port=port,
                log_level="info",
                reload=False
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except ImportError:
            logger.warning("FastAPI 서버가 아직 구현되지 않았습니다. 추후 구현 예정")
        except Exception as e:
            logger.error(f"API 서버 실행 중 오류: {e}")
            
    def run_analysis_mode(self):
        """분석 모드 실행 (종목 검색, 거시경제 분석)"""
        logger.info("=== 분석 모드 시작 ===")
        
        try:
            # 1. 거시경제 분석
            logger.info("1단계: 거시경제 분석")
            macro_search = self.modules['macro_search']
            macro_search.run()
            
            # 2. 종목 스크리닝
            logger.info("2단계: 종목 스크리닝")
            stock_search = self.modules['stock_search']
            stock_search.search_market_leader()
            stock_search.search_theme_upjong(mode=self.config['tradeStock']['scheduler']['mode'])
            
            # 3. 재무 점수 계산
            logger.info("3단계: 재무 점수 계산")
            finance_score = self.modules['finance_score']
            finance_score.run()
            
            logger.info("분석 완료")
            
        except Exception as e:
            logger.error(f"분석 실행 중 오류: {e}")
            
    def run_discovery_mode(self):
        """종목 발굴 모드 실행 (통합 종목 발굴 프로세스)"""
        logger.info("=" * 60)
        logger.info("=== 종목 발굴 모드 시작 ===")
        logger.info("=" * 60)
        
        # Discovery 설정 로드
        discovery_config = self.config.get('discovery', {})
        steps_config = discovery_config.get('steps', {})
        general_config = discovery_config.get('general', {})
        continue_on_failure = general_config.get('continue_on_step_failure', False)
        
        # 활성화된 단계 확인
        enabled_steps = []
        for step_name, enabled in steps_config.items():
            if enabled:
                enabled_steps.append(step_name)
        
        logger.info(f"활성화된 단계: {enabled_steps}")
        
        # 결과 저장용 변수
        macro_result = None
        candidates = []
        
        try:
            # 1. 거시경제 상황 분석
            if steps_config.get('macro_analysis', True):
                logger.info("[1/5] 포괄적인 거시경제 분석 시작")
                try:
                    macro_search = self.modules['macro_search']
                    macro_result = macro_search.run()
                    
                    if 'error' not in macro_result:
                        logger.info(f"거시경제 분석 완료 - 시장 심리: {macro_result.get('market_sentiment', 'Unknown')}")
                        logger.info(f"분석 리포트: {macro_result.get('report_path', 'N/A')}")
                        logger.debug(f"ETF 긍정 비율: {macro_result.get('etf_analysis', {}).get('positive_ratio', 0):.1f}%")
                    else:
                        logger.error(f"거시경제 분석 실패: {macro_result['error']}")
                        if not continue_on_failure:
                            raise Exception(macro_result['error'])
                        
                except Exception as e:
                    logger.error(f"거시경제 분석 실패: {str(e)}")
                    if not continue_on_failure:
                        raise
                    logger.warning("continue_on_step_failure=True로 다음 단계 진행")
            else:
                logger.info("[1/5] 거시경제 분석 단계 건너뜀 (disabled)")
            
            # 2. 재무제표 기반 종목 스크리닝
            if steps_config.get('finance_screening', True):
                logger.info("[2/5] 재무제표 기반 종목 스크리닝 시작")
                try:
                    finance_score = self.modules['finance_score']
                    finance_score.run()
                    logger.debug("재무제표 스크리닝 완료")
                except Exception as e:
                    logger.error(f"재무제표 스크리닝 실패: {str(e)}")
                    if not continue_on_failure:
                        raise
                    logger.warning("continue_on_step_failure=True로 다음 단계 진행")
            else:
                logger.info("[2/5] 재무제표 스크리닝 단계 건너뜀 (disabled)")
            
            # 3. 테마/업종별 종목 검색
            if steps_config.get('theme_sector_search', True):
                logger.info("[3/5] 테마/업종별 종목 검색 시작")
                try:
                    stock_search = self.modules['stock_search']
                    search_options = discovery_config.get('theme_sector_search', {}).get('search_options', {})
                    
                    # 시장 주도주 검색
                    if search_options.get('market_leader', True):
                        logger.info("시장 주도주 검색 중...")
                        try:
                            stock_search.search_market_leader()
                            logger.debug("시장 주도주 검색 완료")
                        except Exception as e:
                            logger.error(f"시장 주도주 검색 실패: {str(e)}")
                            if not continue_on_failure:
                                raise
                    else:
                        logger.info("시장 주도주 검색 건너뜀 (disabled)")
                    
                    # 테마/업종 검색
                    if search_options.get('theme_upjong', True):
                        logger.info("테마/업종 검색 중...")
                        try:
                            stock_search.search_theme_upjong(mode=self.config['tradeStock']['scheduler']['mode'])
                            logger.debug("테마/업종 검색 완료")
                        except Exception as e:
                            logger.error(f"테마/업종 검색 실패: {str(e)}")
                            if not continue_on_failure:
                                raise
                    else:
                        logger.info("테마/업종 검색 건너뜀 (disabled)")
                        
                except Exception as e:
                    logger.error(f"테마/업종별 종목 검색 실패: {str(e)}")
                    if not continue_on_failure:
                        raise
                    logger.warning("continue_on_step_failure=True로 다음 단계 진행")
            else:
                logger.info("[3/5] 테마/업종별 종목 검색 단계 건너뜀 (disabled)")
            
            # 4. 최종 투자 후보 종목 선정
            if steps_config.get('candidate_selection', True):
                logger.info("[4/5] 최종 투자 후보 종목 선정 시작")
                try:
                    candidates = self._select_final_candidates()
                    max_candidates = discovery_config.get('candidate_selection', {}).get('max_candidates', 50)
                    
                    if len(candidates) > max_candidates:
                        candidates = candidates[:max_candidates]
                        logger.info(f"후보 종목을 최대 {max_candidates}개로 제한")
                    
                    logger.info(f"후보 종목 선정 완료: 총 {len(candidates)}개 종목")
                    if candidates:
                        logger.debug(f"상위 5개 후보: {[c['name'] for c in candidates[:5]]}")
                except Exception as e:
                    logger.error(f"후보 종목 선정 실패: {str(e)}")
                    if not continue_on_failure:
                        raise
                    logger.warning("continue_on_step_failure=True로 다음 단계 진행")
                    candidates = []  # 빈 리스트로 초기화
            else:
                logger.info("[4/5] 최종 투자 후보 종목 선정 단계 건너뜀 (disabled)")
            
            # 5. 결과 저장 및 리포트 생성
            if steps_config.get('report_generation', True):
                logger.info("[5/5] 결과 저장 및 리포트 생성 시작")
                try:
                    self._generate_discovery_report(candidates, macro_result)
                    logger.debug("리포트 생성 완료")
                except Exception as e:
                    logger.error(f"리포트 생성 실패: {str(e)}")
                    # 리포트 생성은 기본적으로 skip_on_error=True
                    skip_on_error = discovery_config.get('report_generation', {}).get('skip_on_error', True)
                    if not skip_on_error and not continue_on_failure:
                        raise
                    logger.warning("리포트 생성 실패하지만 계속 진행")
            else:
                logger.info("[5/5] 결과 저장 및 리포트 생성 단계 건너뜀 (disabled)")
            
            logger.info("=" * 60)
            logger.info(f"종목 발굴 완료 - 총 {len(candidates)}개 후보 종목 선정")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error("종목 발굴 실행 중 치명적 오류 발생")
            logger.exception("상세 오류:")
            raise
            
    def _select_final_candidates(self) -> list:
        """최종 투자 후보 종목 선정"""
        try:
            import pandas as pd
            import os
            from glob import glob
            
            # 재무 점수 데이터 로드
            finance_path = self.config['fileControl']['finance_score']['path']
            finance_files = glob(finance_path + "*.csv")
            if not finance_files:
                logger.warning("재무 점수 데이터를 찾을 수 없습니다.")
                return []
                
            # 가장 최신 재무 점수 파일 로드
            latest_finance_file = max(finance_files, key=os.path.getctime)
            df_finance = pd.read_csv(latest_finance_file)
            
            # 매수 조건 만족 종목 데이터 로드
            monitor_path = self.config['fileControl']['monitor_stocks']['path']
            buy_files = glob(monitor_path + "*/*/*/*/buy_list_*.csv")
            if not buy_files:
                logger.warning("매수 조건 만족 종목 데이터를 찾을 수 없습니다.")
                return []
                
            # 가장 최신 매수 리스트 파일 로드
            latest_buy_file = max(buy_files, key=os.path.getctime)
            df_buy = pd.read_csv(latest_buy_file)
            
            # 최종 후보 선정 로직
            candidates = []
            for _, row in df_buy.iterrows():
                candidate = {
                    'name': row.get('Name', ''),
                    'code': row.get('Symbol', ''),
                    'finance_score': row.get('total_score', 0),
                    'volume_cost': row.get('VolumeCost', 0),
                    'change': row.get('Change', 0),
                    'sector': row.get('Sector', ''),
                    'industry': row.get('Industry', ''),
                    'buy_signal_date': row.get('buy_day', ''),
                    'holding_days': row.get('buy_hold_day', ''),
                    'price_earning': row.get('priceEarning', 0)
                }
                candidates.append(candidate)
                
            return candidates
            
        except Exception as e:
            logger.error(f"최종 후보 선정 중 오류: {e}")
            return []
            
    def _generate_discovery_report(self, candidates: list, macro_result: dict = None):
        """종목 발굴 리포트 생성"""
        try:
            from datetime import datetime
            import json
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'macro_analysis': macro_result or {},
                'total_candidates': len(candidates),
                'candidates': candidates,
                'summary': {
                    'high_score_count': len([c for c in candidates if c.get('finance_score', 0) >= 80]),
                    'avg_finance_score': sum(c.get('finance_score', 0) for c in candidates) / len(candidates) if candidates else 0,
                    'sectors': list(set(c.get('sector', '') for c in candidates if c.get('sector'))),
                    'top_performers': sorted(candidates, key=lambda x: x.get('price_earning', 0), reverse=True)[:5]
                }
            }
            
            # 리포트 저장
            report_path = f"./data/discovery_reports/"
            os.makedirs(report_path, exist_ok=True)
            
            report_filename = f"discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path + report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            logger.info(f"종목 발굴 리포트 저장완료: {report_path + report_filename}")
            
            # 요약 정보 로그 출력
            logger.info("=== 종목 발굴 결과 요약 ===")
            logger.info(f"총 후보 종목: {report['total_candidates']}개")
            logger.info(f"고득점(80점 이상) 종목: {report['summary']['high_score_count']}개")
            logger.info(f"평균 재무 점수: {report['summary']['avg_finance_score']:.1f}점")
            logger.info(f"주요 섹터: {', '.join(report['summary']['sectors'][:5])}")
            
        except Exception as e:
            logger.error(f"리포트 생성 중 오류: {e}")

def create_directories():
    """필요한 디렉토리 생성 - DB 저장을 고려한 구조"""
    # 기본 디렉토리 구조
    base_directories = [
        "./log", 
        "./data", 
        "./models", 
        "./models/nlp"
    ]
    
    # 데이터베이스 스키마를 고려한 데이터 디렉토리 구조
    data_directories = [
        # 원시 데이터 (Raw Data)
        "./data/raw/stock_prices",           # 주식 가격 데이터
        "./data/raw/market_data",            # 시장 데이터
        "./data/raw/financial_statements",   # 재무제표 데이터
        "./data/raw/news",                   # 뉴스 데이터
        "./data/raw/economic_indicators",    # 경제 지표
        
        # 처리된 데이터 (Processed Data)
        "./data/processed/stock_analysis",   # 주식 분석 결과
        "./data/processed/sector_analysis",  # 섹터 분석 결과
        "./data/processed/macro_analysis",   # 거시경제 분석 결과
        "./data/processed/strategy_signals", # 전략 신호
        "./data/processed/risk_metrics",     # 리스크 지표
        
        # 분석 결과 (Analytics)
        "./data/analytics/backtest_results", # 백테스팅 결과
        "./data/analytics/performance",      # 성과 분석
        "./data/analytics/reports",          # 리포트
        "./data/analytics/charts",           # 차트 이미지
        
        # 거래 데이터 (Trading)
        "./data/trading/positions",          # 포지션 데이터
        "./data/trading/orders",             # 주문 데이터
        "./data/trading/transactions",       # 거래 내역
        "./data/trading/portfolio",          # 포트폴리오 상태
        
        # 설정 및 메타데이터 (Configuration & Metadata)
        "./data/config/strategies",          # 전략 설정
        "./data/config/parameters",          # 파라미터 설정
        "./data/metadata/symbols",           # 종목 메타데이터
        "./data/metadata/sectors",           # 섹터 정보
        "./data/metadata/themes",            # 테마 정보
        
        # 임시 및 캐시 (Temporary & Cache)
        "./data/cache/market_data",          # 시장 데이터 캐시
        "./data/cache/calculations",         # 계산 결과 캐시
        "./data/temp/downloads",             # 임시 다운로드
        "./data/temp/processing",            # 처리 중 데이터
        
        # 백업 (Backup)
        "./data/backup/daily",               # 일별 백업
        "./data/backup/weekly",              # 주별 백업
        "./data/backup/monthly",             # 월별 백업
        
        # 기존 호환성을 위한 디렉토리
        "./data/search_stocks",
        "./data/monitor_stocks", 
        "./data/finance_score",
        "./data/system_trade",
        "./data/discovery_reports"
    ]
    
    all_directories = base_directories + data_directories
    
    logger.info("시스템 디렉토리 구조 확인 중...")
    created_count = 0
    for directory in all_directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.debug(f"디렉토리 생성: {directory}")
                created_count += 1
            else:
                logger.debug(f"디렉토리 확인: {directory}")
        except Exception as e:
            logger.error(f"디렉토리 생성 실패 ({directory}): {str(e)}")
            raise
    
    if created_count > 0:
        logger.info(f"총 {created_count}개의 새 디렉토리가 생성되었습니다.")
    else:
        logger.info("모든 시스템 디렉토리가 정상입니다.")
        
def create_daily_directories(base_path: str, date_str: str = None) -> str:
    """
    날짜별 디렉토리 구조 생성
    
    Args:
        base_path: 기본 경로
        date_str: 날짜 문자열 (YYYYMMDD), None이면 오늘 날짜
        
    Returns:
        생성된 날짜별 디렉토리 경로
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    # 년/월/일 구조로 디렉토리 생성
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    daily_path = os.path.join(base_path, year, month, day)
    
    try:
        os.makedirs(daily_path, exist_ok=True)
        logger.debug(f"날짜별 디렉토리 생성/확인: {daily_path}")
        return daily_path
    except Exception as e:
        logger.error(f"날짜별 디렉토리 생성 실패 ({daily_path}): {str(e)}")
        raise

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='ST 자동매매 시스템')
    parser.add_argument(
        '--mode', 
        choices=['backtest', 'trading', 'api', 'analysis', 'discovery'],
        default='api',
        help='실행 모드 선택'
    )
    parser.add_argument(
        '--config', 
        default='./config/config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='API 서버 호스트 (api 모드에서만 사용)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='API 서버 포트 (api 모드에서만 사용)'
    )
    
    args = parser.parse_args()
    
    # 시스템 시작 로그
    logger.info("=" * 80)
    logger.info("ST (System Trading) v0.1 시작")
    logger.info("-" * 80)
    logger.info(f"실행 모드: {args.mode}")
    logger.info(f"설정 파일: {args.config}")
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"실행 환경: Python {sys.version}")
    logger.info(f"작업 디렉토리: {os.getcwd()}")
    logger.info("=" * 80)
    
    try:
        # 필요한 디렉토리 생성
        create_directories()
        
        # 시스템 매니저 초기화
        logger.info("시스템 매니저 초기화 중...")
        system_manager = STSystemManager(args.config)
        
        # 설정 로딩
        if not system_manager.load_config():
            logger.error("시스템 초기화 실패: 설정 파일 문제")
            sys.exit(1)
            
        # 모듈 초기화
        if not system_manager.initialize_modules():
            logger.error("시스템 초기화 실패: 모듈 초기화 문제")
            sys.exit(1)
            
        # 모드별 실행
        logger.info(f"{args.mode} 모드로 시스템을 시작합니다...")
        
        try:
            if args.mode == 'backtest':
                system_manager.run_backtest_mode()
                
            elif args.mode == 'trading':
                asyncio.run(system_manager.run_trading_mode())
                
            elif args.mode == 'api':
                asyncio.run(system_manager.run_api_server(args.host, args.port))
                
            elif args.mode == 'analysis':
                system_manager.run_analysis_mode()
                
            elif args.mode == 'discovery':
                system_manager.run_discovery_mode()
                
        except KeyboardInterrupt:
            logger.warning("사용자에 의해 시스템이 중단되었습니다.")
        except Exception as e:
            logger.error("시스템 실행 중 예상치 못한 오류가 발생했습니다.")
            logger.exception("상세 오류:")
            sys.exit(1)
            
    except Exception as e:
        logger.critical("시스템 초기화 중 치명적 오류가 발생했습니다.")
        logger.exception("상세 오류:")
        sys.exit(1)
        
    finally:
        logger.info("=" * 80)
        logger.info("ST 시스템 종료")
        logger.info(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()
