"""
기본 전략 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class BaseStrategy(ABC):
    """
    모든 전략의 기본 클래스
    """
    
    def __init__(self, name: str, **params):
        """
        전략 초기화
        
        Args:
            name: 전략 이름
            **params: 전략 파라미터
        """
        self.name = name
        self.params = params
        
    @abstractmethod
    def generate_signals(self, 
                        current_data: Dict[str, pd.Series],
                        historical_data: Dict[str, pd.DataFrame],
                        current_date: datetime,
                        positions: Dict[str, Any],
                        **kwargs) -> List[Dict]:
        """
        거래 신호 생성
        
        Args:
            current_data: 현재 시점 데이터
            historical_data: 과거 데이터
            current_date: 현재 날짜
            positions: 현재 포지션
            
        Returns:
            List[Dict]: 거래 신호 리스트
        """
        pass
        
    def calculate_position_size(self, 
                               symbol: str, 
                               price: float, 
                               available_capital: float,
                               risk_per_trade: float = 0.02) -> int:
        """
        포지션 크기 계산
        
        Args:
            symbol: 종목 코드
            price: 현재 가격
            available_capital: 사용 가능 자본
            risk_per_trade: 거래당 리스크 비율
            
        Returns:
            int: 매수 수량
        """
        max_risk_amount = available_capital * risk_per_trade
        quantity = int(max_risk_amount / price)
        return max(1, quantity)  # 최소 1주
        
    def get_description(self) -> str:
        """전략 설명 반환"""
        return f"{self.name} 전략"