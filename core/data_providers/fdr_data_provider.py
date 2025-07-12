"""
FinanceDataReader 데이터 제공자
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

class FDRDataProvider:
    """
    FinanceDataReader 데이터 제공자
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        FDR 데이터 제공자 초기화
        
        Args:
            config: 설정 정보
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if not FDR_AVAILABLE:
            self.logger.warning("FinanceDataReader가 설치되지 않음")
        
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
        if not FDR_AVAILABLE:
            return None
            
        try:
            # 기본 기간 설정
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # FDR로 데이터 조회
            df = fdr.DataReader(symbol, start_date, end_date)
            
            if df is None or df.empty:
                return None
                
            # 컬럼명 표준화 (소문자)
            df.columns = [col.lower() for col in df.columns]
            
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
                
            self.logger.info(f"FDR 데이터 조회 성공: {symbol}, {len(df)}일")
            return df
            
        except Exception as e:
            self.logger.error(f"FDR 데이터 조회 실패 {symbol}: {e}")
            return None
            
    def get_market_data(self) -> Optional[Dict[str, Any]]:
        """
        시장 지수 데이터 조회
        
        Returns:
            Dict: 시장 지수 정보
        """
        if not FDR_AVAILABLE:
            return None
            
        try:
            indices = {
                'KOSPI': 'KS11',
                'KOSDAQ': 'KQ11',
                'KOSPI200': 'KS200'
            }
            
            results = {}
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            for name, symbol in indices.items():
                try:
                    df = fdr.DataReader(symbol, start_date, end_date)
                    if df is not None and not df.empty:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else latest
                        
                        results[name] = {
                            'current': float(latest['Close']),
                            'change': float(latest['Close'] - prev['Close']),
                            'change_pct': float((latest['Close'] - prev['Close']) / prev['Close'] * 100),
                            'volume': int(latest.get('Volume', 0)),
                            'date': latest.name.strftime('%Y-%m-%d')
                        }
                except Exception as e:
                    self.logger.error(f"지수 데이터 조회 실패 {name}: {e}")
                    continue
                    
            return results
            
        except Exception as e:
            self.logger.error(f"FDR 시장 데이터 조회 실패: {e}")
            return None
            
    def is_available(self) -> bool:
        """
        데이터 제공자 사용 가능 여부
        
        Returns:
            bool: 사용 가능 여부
        """
        return FDR_AVAILABLE