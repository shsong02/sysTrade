"""
샘플 트레이딩 전략들
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy

class BollingerBandStrategy(BaseStrategy):
    """
    볼린저 밴드 전략
    """
    
    def __init__(self, window: int = 20, std_dev: float = 2.0, **kwargs):
        super().__init__("Bollinger Band", window=window, std_dev=std_dev, **kwargs)
        
    def generate_signals(self, 
                        current_data: Dict[str, pd.Series],
                        historical_data: Dict[str, pd.DataFrame],
                        current_date: datetime,
                        positions: Dict[str, Any],
                        **kwargs) -> List[Dict]:
        
        signals = []
        
        for symbol, current_row in current_data.items():
            if symbol not in historical_data:
                continue
                
            df = historical_data[symbol]
            
            # 현재 날짜까지의 데이터만 사용
            mask = df.index <= current_date
            df_filtered = df.loc[mask]
            
            if len(df_filtered) < self.params['window']:
                continue
                
            # 볼린저 밴드 계산
            window = self.params['window']
            std_dev = self.params['std_dev']
            
            close_prices = df_filtered['close']
            sma = close_prices.rolling(window=window).mean()
            std = close_prices.rolling(window=window).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = current_row['close']
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            # 신호 생성
            if current_price <= current_lower and symbol not in positions:
                # 하단 밴드 터치 시 매수
                quantity = self.calculate_position_size(
                    symbol, current_price, kwargs.get('available_capital', 1000000)
                )
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'reason': f'볼린저 하단 터치 (가격: {current_price:,.0f}, 하단: {current_lower:,.0f})'
                })
                
            elif current_price >= current_upper and symbol in positions:
                # 상단 밴드 터치 시 매도
                signals.append({
                    'symbol': symbol,
                    'action': 'close',
                    'quantity': 0,
                    'reason': f'볼린저 상단 터치 (가격: {current_price:,.0f}, 상단: {current_upper:,.0f})'
                })
                
        return signals


class RSIStrategy(BaseStrategy):
    """
    RSI 전략
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        super().__init__("RSI", period=period, oversold=oversold, overbought=overbought, **kwargs)
        
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def generate_signals(self, 
                        current_data: Dict[str, pd.Series],
                        historical_data: Dict[str, pd.DataFrame],
                        current_date: datetime,
                        positions: Dict[str, Any],
                        **kwargs) -> List[Dict]:
        
        signals = []
        
        for symbol, current_row in current_data.items():
            if symbol not in historical_data:
                continue
                
            df = historical_data[symbol]
            
            # 현재 날짜까지의 데이터만 사용
            mask = df.index <= current_date
            df_filtered = df.loc[mask]
            
            if len(df_filtered) < self.params['period'] + 1:
                continue
                
            # RSI 계산
            rsi = self.calculate_rsi(df_filtered['close'], self.params['period'])
            current_rsi = rsi.iloc[-1]
            
            # 신호 생성
            if current_rsi <= self.params['oversold'] and symbol not in positions:
                # 과매도 시 매수
                quantity = self.calculate_position_size(
                    symbol, current_row['close'], kwargs.get('available_capital', 1000000)
                )
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'reason': f'RSI 과매도 (RSI: {current_rsi:.1f})'
                })
                
            elif current_rsi >= self.params['overbought'] and symbol in positions:
                # 과매수 시 매도
                signals.append({
                    'symbol': symbol,
                    'action': 'close',
                    'quantity': 0,
                    'reason': f'RSI 과매수 (RSI: {current_rsi:.1f})'
                })
                
        return signals


class MovingAverageCrossStrategy(BaseStrategy):
    """
    이동평균선 교차 전략
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20, **kwargs):
        super().__init__("MA Cross", short_window=short_window, long_window=long_window, **kwargs)
        
    def generate_signals(self, 
                        current_data: Dict[str, pd.Series],
                        historical_data: Dict[str, pd.DataFrame],
                        current_date: datetime,
                        positions: Dict[str, Any],
                        **kwargs) -> List[Dict]:
        
        signals = []
        
        for symbol, current_row in current_data.items():
            if symbol not in historical_data:
                continue
                
            df = historical_data[symbol]
            
            # 현재 날짜까지의 데이터만 사용
            mask = df.index <= current_date
            df_filtered = df.loc[mask]
            
            if len(df_filtered) < self.params['long_window']:
                continue
                
            # 이동평균 계산
            short_ma = df_filtered['close'].rolling(window=self.params['short_window']).mean()
            long_ma = df_filtered['close'].rolling(window=self.params['long_window']).mean()
            
            if len(short_ma) < 2 or len(long_ma) < 2:
                continue
                
            # 골든크로스/데드크로스 확인
            prev_short = short_ma.iloc[-2]
            prev_long = long_ma.iloc[-2]
            curr_short = short_ma.iloc[-1]
            curr_long = long_ma.iloc[-1]
            
            # 골든크로스 (단기 MA가 장기 MA를 상향 돌파)
            if prev_short <= prev_long and curr_short > curr_long and symbol not in positions:
                quantity = self.calculate_position_size(
                    symbol, current_row['close'], kwargs.get('available_capital', 1000000)
                )
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'reason': f'골든크로스 (단기MA: {curr_short:.0f}, 장기MA: {curr_long:.0f})'
                })
                
            # 데드크로스 (단기 MA가 장기 MA를 하향 돌파)
            elif prev_short >= prev_long and curr_short < curr_long and symbol in positions:
                signals.append({
                    'symbol': symbol,
                    'action': 'close',
                    'quantity': 0,
                    'reason': f'데드크로스 (단기MA: {curr_short:.0f}, 장기MA: {curr_long:.0f})'
                })
                
        return signals