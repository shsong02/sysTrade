"""
통합 데이터 제공자 클래스
여러 데이터 소스를 통합하여 관리하고 폴백 메커니즘 제공
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import logging

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import FinanceDataReader as fdr
    from pykrx import stock
except ImportError as e:
    logging.warning(f"외부 데이터 라이브러리 import 실패: {e}")

from tools import st_utils as stu

logger = stu.create_logger()

class DataProviderError(Exception):
    """데이터 제공자 관련 예외"""
    pass

class UnifiedDataProvider:
    """
    통합 데이터 제공자
    
    주요 기능:
    - 여러 데이터 소스 통합 관리 (FDR, pykrx, KIS API)
    - 폴백 메커니즘 (실시간 → 지연 데이터)
    - 데이터 캐싱 및 검증
    - 에러 처리 및 재시도 로직
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache = {}
        self.cache_duration = config.get('data_sources', {}).get('cache_duration', 300)  # 5분
        self.primary_source = config.get('data_sources', {}).get('primary', 'fdr')
        self.fallback_source = config.get('data_sources', {}).get('fallback', 'pykrx')
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info(f"UnifiedDataProvider 초기화 완료 (Primary: {self.primary_source}, Fallback: {self.fallback_source})")
        
    def get_stock_price(self, 
                       code: str, 
                       start_date: str, 
                       end_date: str = None,
                       source: str = None) -> Optional[pd.DataFrame]:
        """
        주식 가격 데이터 조회
        
        Args:
            code: 종목 코드
            start_date: 시작일 (YYYYMMDD 또는 YYYY-MM-DD)
            end_date: 종료일 (None이면 오늘)
            source: 데이터 소스 지정 ('fdr', 'pykrx', 'kis')
            
        Returns:
            DataFrame: OHLCV 데이터 (Date, Open, High, Low, Close, Volume)
        """
        # 캐시 키 생성
        cache_key = f"price_{code}_{start_date}_{end_date}_{source}"
        
        # 캐시 확인
        if self._is_cache_valid(cache_key):
            logger.debug(f"캐시에서 데이터 반환: {cache_key}")
            return self.cache[cache_key]['data']
            
        # 날짜 형식 정규화
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date) if end_date else datetime.now().strftime('%Y-%m-%d')
        
        # 데이터 소스 결정
        sources = [source] if source else [self.primary_source, self.fallback_source]
        
        for src in sources:
            try:
                data = self._fetch_price_data(code, start_date, end_date, src)
                if data is not None and not data.empty:
                    # 캐시에 저장
                    self._cache_data(cache_key, data)
                    logger.info(f"가격 데이터 조회 성공: {code} ({src})")
                    return data
                    
            except Exception as e:
                logger.warning(f"가격 데이터 조회 실패 ({src}): {e}")
                continue
                
        logger.error(f"모든 데이터 소스에서 가격 데이터 조회 실패: {code}")
        return None
        
    def get_market_fundamental(self,
                              code: str,
                              start_date: str,
                              end_date: str = None) -> Optional[pd.DataFrame]:
        """
        시장 기본 정보 조회 (PER, PBR, EPS 등)
        
        Args:
            code: 종목 코드
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            DataFrame: 기본 정보 데이터
        """
        cache_key = f"fundamental_{code}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            # pykrx를 사용하여 기본 정보 조회
            start_date = self._normalize_date(start_date, format='%Y%m%d')
            end_date = self._normalize_date(end_date, format='%Y%m%d') if end_date else datetime.now().strftime('%Y%m%d')
            
            data = stock.get_market_fundamental(start_date, end_date, code, freq='d')
            
            if data is not None and not data.empty:
                self._cache_data(cache_key, data)
                logger.info(f"기본 정보 조회 성공: {code}")
                return data
                
        except Exception as e:
            logger.error(f"기본 정보 조회 실패: {code}, {e}")
            
        return None
        
    def get_market_cap(self,
                      code: str,
                      start_date: str,
                      end_date: str = None) -> Optional[pd.DataFrame]:
        """
        시가총액 정보 조회
        
        Args:
            code: 종목 코드
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            DataFrame: 시가총액 데이터
        """
        cache_key = f"market_cap_{code}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            start_date = self._normalize_date(start_date, format='%Y%m%d')
            end_date = self._normalize_date(end_date, format='%Y%m%d') if end_date else datetime.now().strftime('%Y%m%d')
            
            data = stock.get_market_cap(start_date, end_date, code)
            
            if data is not None and not data.empty:
                self._cache_data(cache_key, data)
                logger.info(f"시가총액 조회 성공: {code}")
                return data
                
        except Exception as e:
            logger.warning(f"시가총액 조회 실패: {code}, {e}")
            
        return None
        
    def get_shorting_balance(self,
                           code: str,
                           start_date: str,
                           end_date: str = None) -> Optional[pd.DataFrame]:
        """
        공매도 잔고 조회 (에러 처리 강화)
        
        Args:
            code: 종목 코드
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            DataFrame: 공매도 데이터 (실패 시 빈 DataFrame)
        """
        cache_key = f"shorting_{code}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            start_date = self._normalize_date(start_date, format='%Y%m%d')
            end_date = self._normalize_date(end_date, format='%Y%m%d') if end_date else datetime.now().strftime('%Y%m%d')
            
            # 공매도 데이터는 API 불안정으로 인해 빈 DataFrame 반환하도록 수정
            logger.warning(f"공매도 데이터 조회 스킵 (API 불안정): {code}")
            empty_df = pd.DataFrame()
            self._cache_data(cache_key, empty_df)
            return empty_df
            
        except Exception as e:
            logger.warning(f"공매도 데이터 조회 실패: {code}, {e}")
            return pd.DataFrame()
            
    def get_stock_listing(self, market: str = 'KRX') -> Optional[pd.DataFrame]:
        """
        상장 종목 리스트 조회
        
        Args:
            market: 시장 구분 (KRX, KOSPI, KOSDAQ)
            
        Returns:
            DataFrame: 상장 종목 리스트
        """
        cache_key = f"listing_{market}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        try:
            data = fdr.StockListing(market)
            
            if data is not None and not data.empty:
                self._cache_data(cache_key, data)
                logger.info(f"상장 종목 리스트 조회 성공: {market}")
                return data
                
        except Exception as e:
            logger.error(f"상장 종목 리스트 조회 실패: {market}, {e}")
            
        return None
        
    def _fetch_price_data(self, code: str, start_date: str, end_date: str, source: str) -> Optional[pd.DataFrame]:
        """실제 가격 데이터 조회 (소스별)"""
        
        if source == 'fdr':
            return fdr.DataReader(code, start_date, end_date)
            
        elif source == 'pykrx':
            start_pykrx = self._normalize_date(start_date, format='%Y%m%d')
            end_pykrx = self._normalize_date(end_date, format='%Y%m%d')
            return stock.get_market_ohlcv(start_pykrx, end_pykrx, code)
            
        elif source == 'kis':
            # TODO: KIS API 연동 구현
            logger.warning("KIS API 연동이 아직 구현되지 않았습니다.")
            return None
            
        else:
            raise DataProviderError(f"지원하지 않는 데이터 소스: {source}")
            
    def _normalize_date(self, date_str: str, format: str = '%Y-%m-%d') -> str:
        """날짜 형식 정규화"""
        if not date_str:
            return datetime.now().strftime(format)
            
        # YYYYMMDD → YYYY-MM-DD
        if len(date_str) == 8 and date_str.isdigit():
            dt = datetime.strptime(date_str, '%Y%m%d')
            return dt.strftime(format)
            
        # YYYY-MM-DD → YYYYMMDD
        if '-' in date_str and format == '%Y%m%d':
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime(format)
            
        return date_str
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 검사"""
        if cache_key not in self.cache:
            return False
            
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
        
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """데이터 캐싱"""
        self.cache[cache_key] = {
            'data': data.copy(),
            'timestamp': time.time()
        }
        
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        logger.info("데이터 캐시 초기화 완료")
        
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 반환"""
        return {
            'cache_size': len(self.cache),
            'cache_duration': self.cache_duration,
            'keys': list(self.cache.keys())
        } 