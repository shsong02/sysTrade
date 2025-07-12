"""
pykrx 데이터 제공자
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    from pykrx import stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

class PyKRXDataProvider:
    """
    pykrx 데이터 제공자
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        pykrx 데이터 제공자 초기화
        
        Args:
            config: 설정 정보
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if not PYKRX_AVAILABLE:
            self.logger.warning("pykrx가 설치되지 않음")
        
    def get_stock_data(self, 
                      symbol: str, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        주식 데이터 조회
        
        Args:
            symbol: 종목 코드
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            
        Returns:
            DataFrame: 주가 데이터 (OHLCV)
        """
        if not PYKRX_AVAILABLE:
            return None
            
        try:
            # 기본 기간 설정
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            else:
                start_date = start_date.replace('-', '')
                
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            else:
                end_date = end_date.replace('-', '')
            
            # pykrx로 데이터 조회
            df = stock.get_market_ohlcv_by_date(start_date, end_date, symbol)
            
            if df is None or df.empty:
                return None
                
            # 컬럼명 영어로 변경
            column_mapping = {
                '시가': 'open',
                '고가': 'high', 
                '저가': 'low',
                '종가': 'close',
                '거래량': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 필요한 컬럼만 선택
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if not available_cols:
                self.logger.warning(f"필요한 컬럼이 없음: {df.columns.tolist()}")
                return None
                
            df = df[available_cols]
            
            # Volume이 없으면 0으로 채움
            if 'volume' not in df.columns:
                df['volume'] = 0
                
            self.logger.info(f"pykrx 데이터 조회 성공: {symbol}, {len(df)}일")
            return df
            
        except Exception as e:
            self.logger.error(f"pykrx 데이터 조회 실패 {symbol}: {e}")
            return None
            
    def get_market_data(self) -> Optional[Dict[str, Any]]:
        """
        시장 지수 데이터 조회
        
        Returns:
            Dict: 시장 지수 정보
        """
        if not PYKRX_AVAILABLE:
            return None
            
        try:
            today = datetime.now().strftime('%Y%m%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            
            # KOSPI 지수
            kospi_df = stock.get_index_ohlcv_by_date(yesterday, today, "1001")  # KOSPI
            kosdaq_df = stock.get_index_ohlcv_by_date(yesterday, today, "2001")  # KOSDAQ
            
            results = {}
            
            if kospi_df is not None and not kospi_df.empty:
                latest = kospi_df.iloc[-1]
                prev = kospi_df.iloc[-2] if len(kospi_df) > 1 else latest
                
                results['KOSPI'] = {
                    'current': float(latest['종가']),
                    'change': float(latest['종가'] - prev['종가']),
                    'change_pct': float((latest['종가'] - prev['종가']) / prev['종가'] * 100),
                    'volume': int(latest.get('거래량', 0)),
                    'date': latest.name.strftime('%Y-%m-%d')
                }
                
            if kosdaq_df is not None and not kosdaq_df.empty:
                latest = kosdaq_df.iloc[-1]
                prev = kosdaq_df.iloc[-2] if len(kosdaq_df) > 1 else latest
                
                results['KOSDAQ'] = {
                    'current': float(latest['종가']),
                    'change': float(latest['종가'] - prev['종가']),
                    'change_pct': float((latest['종가'] - prev['종가']) / prev['종가'] * 100),
                    'volume': int(latest.get('거래량', 0)),
                    'date': latest.name.strftime('%Y-%m-%d')
                }
                
            return results
            
        except Exception as e:
            self.logger.error(f"pykrx 시장 데이터 조회 실패: {e}")
            return None
            
    def is_available(self) -> bool:
        """
        데이터 제공자 사용 가능 여부
        
        Returns:
            bool: 사용 가능 여부
        """
        return PYKRX_AVAILABLE