"""
실시간 거래 전략 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

@dataclass
class TradingSignal:
    """거래 신호 데이터 클래스"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: int
    price: float
    confidence: float  # 0.0 ~ 1.0
    reason: str
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Position:
    """포지션 데이터 클래스"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0

class BaseTradingStrategy(ABC):
    """
    실시간 거래 전략의 기본 클래스
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        전략 초기화
        
        Args:
            name: 전략 이름
            config: 전략 설정
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")
        self.positions: Dict[str, Position] = {}
        self.last_update = None
        
    @abstractmethod
    def analyze_market_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        시장 데이터 분석
        
        Args:
            data: {symbol: DataFrame} 형태의 시장 데이터
            
        Returns:
            Dict: 분석 결과
        """
        pass
        
    @abstractmethod
    def generate_signals(self, 
                        market_data: Dict[str, pd.DataFrame],
                        positions: Dict[str, Position]) -> List[TradingSignal]:
        """
        거래 신호 생성
        
        Args:
            market_data: 시장 데이터
            positions: 현재 포지션
            
        Returns:
            List[TradingSignal]: 거래 신호 리스트
        """
        pass
        
    @abstractmethod
    def should_stop_loss(self, position: Position, current_data: pd.Series) -> bool:
        """
        손절매 조건 확인
        
        Args:
            position: 포지션 정보
            current_data: 현재 시장 데이터
            
        Returns:
            bool: 손절매 여부
        """
        pass
        
    @abstractmethod
    def should_take_profit(self, position: Position, current_data: pd.Series) -> bool:
        """
        익절 조건 확인
        
        Args:
            position: 포지션 정보
            current_data: 현재 시장 데이터
            
        Returns:
            bool: 익절 여부
        """
        pass
        
    def update_positions(self, market_data: Dict[str, pd.DataFrame]):
        """포지션 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in market_data and len(market_data[symbol]) > 0:
                current_price = market_data[symbol]['Close'].iloc[-1]
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                
    def check_risk_management(self, positions: Dict[str, Position]) -> List[TradingSignal]:
        """리스크 관리 체크"""
        signals = []
        
        for symbol, position in positions.items():
            # 손절매 체크
            if position.stop_loss and position.current_price <= position.stop_loss:
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=position.current_price,
                    confidence=1.0,
                    reason='Stop Loss',
                    timestamp=datetime.now(),
                )
                signals.append(signal)
                
            # 익절 체크
            elif position.take_profit and position.current_price >= position.take_profit:
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=position.current_price,
                    confidence=1.0,
                    reason='Take Profit',
                    timestamp=datetime.now(),
                )
                signals.append(signal)
                
        return signals
        
    def calculate_position_size(self, 
                               symbol: str, 
                               price: float, 
                               available_capital: float) -> int:
        """포지션 크기 계산"""
        position_size_pct = self.config.get('position_size_pct', 5.0) / 100
        max_position_value = available_capital * position_size_pct
        quantity = int(max_position_value / price)
        return max(1, quantity)
        
    def get_strategy_status(self) -> Dict[str, Any]:
        """전략 상태 반환"""
        return {
            'name': self.name,
            'active_positions': len(self.positions),
            'total_unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'last_update': self.last_update,
            'config': self.config
        }


class BollingerBandTradingStrategy(BaseTradingStrategy):
    """볼린저 밴드 기반 거래 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BollingerBand", config)
        self.window = config.get('bb_window', 20)
        self.std_dev = config.get('bb_std', 2.0)
        
    def analyze_market_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """볼린저 밴드 분석"""
        analysis = {}
        
        for symbol, df in data.items():
            if len(df) < self.window:
                continue
                
            # 볼린저 밴드 계산
            df['BB_Middle'] = df['Close'].rolling(window=self.window).mean()
            df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(window=self.window).std() * self.std_dev)
            df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(window=self.window).std() * self.std_dev)
            
            # 현재 위치 분석
            current_price = df['Close'].iloc[-1]
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            bb_middle = df['BB_Middle'].iloc[-1]
            
            # BB 위치 비율 (0: 하단, 0.5: 중간, 1: 상단)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            analysis[symbol] = {
                'bb_position': bb_position,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'current_price': current_price,
                'oversold': bb_position < 0.1,  # 하단 10% 이하
                'overbought': bb_position > 0.9  # 상단 10% 이상
            }
            
        return analysis
        
    def generate_signals(self, 
                        market_data: Dict[str, pd.DataFrame],
                        positions: Dict[str, Position]) -> List[TradingSignal]:
        """볼린저 밴드 기반 신호 생성"""
        signals = []
        analysis = self.analyze_market_data(market_data)
        
        for symbol, data in analysis.items():
            # 매수 신호: 하단 밴드 근처에서 반등
            if data['oversold'] and symbol not in positions:
                signal = TradingSignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=0,  # 나중에 계산
                    price=data['current_price'],
                    confidence=0.8,
                    reason='Bollinger Band Oversold',
                    timestamp=datetime.now(),
                    stop_loss=data['current_price'] * 0.95,  # 5% 손절
                    take_profit=data['bb_middle']  # 중간선까지 익절
                )
                signals.append(signal)
                
            # 매도 신호: 상단 밴드 근처
            elif data['overbought'] and symbol in positions:
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=positions[symbol].quantity,
                    price=data['current_price'],
                    confidence=0.8,
                    reason='Bollinger Band Overbought',
                    timestamp=datetime.now()
                )
                signals.append(signal)
                
        return signals
        
    def should_stop_loss(self, position: Position, current_data: pd.Series) -> bool:
        """손절매 조건"""
        if position.stop_loss:
            return current_data['Close'] <= position.stop_loss
        return False
        
    def should_take_profit(self, position: Position, current_data: pd.Series) -> bool:
        """익절 조건"""
        if position.take_profit:
            return current_data['Close'] >= position.take_profit
        return False


class RSITradingStrategy(BaseTradingStrategy):
    """RSI 기반 거래 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI", config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_level = config.get('rsi_oversold', 30)
        self.overbought_level = config.get('rsi_overbought', 70)
        
    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI 계산"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def analyze_market_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """RSI 분석"""
        analysis = {}
        
        for symbol, df in data.items():
            if len(df) < self.rsi_period + 1:
                continue
                
            rsi = self.calculate_rsi(df)
            current_rsi = rsi.iloc[-1]
            
            analysis[symbol] = {
                'rsi': current_rsi,
                'oversold': current_rsi < self.oversold_level,
                'overbought': current_rsi > self.overbought_level,
                'current_price': df['Close'].iloc[-1]
            }
            
        return analysis
        
    def generate_signals(self, 
                        market_data: Dict[str, pd.DataFrame],
                        positions: Dict[str, Position]) -> List[TradingSignal]:
        """RSI 기반 신호 생성"""
        signals = []
        analysis = self.analyze_market_data(market_data)
        
        for symbol, data in analysis.items():
            # 매수 신호: RSI 과매도
            if data['oversold'] and symbol not in positions:
                signal = TradingSignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=0,
                    price=data['current_price'],
                    confidence=0.7,
                    reason=f'RSI Oversold ({data["rsi"]:.1f})',
                    timestamp=datetime.now(),
                    stop_loss=data['current_price'] * 0.95,
                    take_profit=data['current_price'] * 1.1
                )
                signals.append(signal)
                
            # 매도 신호: RSI 과매수
            elif data['overbought'] and symbol in positions:
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=positions[symbol].quantity,
                    price=data['current_price'],
                    confidence=0.7,
                    reason=f'RSI Overbought ({data["rsi"]:.1f})',
                    timestamp=datetime.now()
                )
                signals.append(signal)
                
        return signals
        
    def should_stop_loss(self, position: Position, current_data: pd.Series) -> bool:
        """손절매 조건"""
        if position.stop_loss:
            return current_data['Close'] <= position.stop_loss
        return False
        
    def should_take_profit(self, position: Position, current_data: pd.Series) -> bool:
        """익절 조건"""
        if position.take_profit:
            return current_data['Close'] >= position.take_profit
        return False


class CombinedTradingStrategy(BaseTradingStrategy):
    """복합 지표 거래 전략 (볼린저 밴드 + RSI + 거래량)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Combined", config)
        self.bb_strategy = BollingerBandTradingStrategy(config)
        self.rsi_strategy = RSITradingStrategy(config)
        self.volume_threshold = config.get('volume_threshold', 1.5)  # 평균 거래량 대비
        
    def analyze_market_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """복합 분석"""
        bb_analysis = self.bb_strategy.analyze_market_data(data)
        rsi_analysis = self.rsi_strategy.analyze_market_data(data)
        
        combined_analysis = {}
        
        for symbol in data.keys():
            if symbol in bb_analysis and symbol in rsi_analysis:
                df = data[symbol]
                
                # 거래량 분석
                avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = df['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                combined_analysis[symbol] = {
                    **bb_analysis[symbol],
                    **rsi_analysis[symbol],
                    'volume_ratio': volume_ratio,
                    'high_volume': volume_ratio > self.volume_threshold
                }
                
        return combined_analysis
        
    def generate_signals(self, 
                        market_data: Dict[str, pd.DataFrame],
                        positions: Dict[str, Position]) -> List[TradingSignal]:
        """복합 신호 생성"""
        signals = []
        analysis = self.analyze_market_data(market_data)
        
        for symbol, data in analysis.items():
            # 강한 매수 신호: BB 과매도 + RSI 과매도 + 높은 거래량
            if (data['oversold'] and data.get('oversold', False) and 
                data['high_volume'] and symbol not in positions):
                
                signal = TradingSignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=0,
                    price=data['current_price'],
                    confidence=0.9,
                    reason='Combined: BB+RSI Oversold + High Volume',
                    timestamp=datetime.now(),
                    stop_loss=data['current_price'] * 0.95,
                    take_profit=data['current_price'] * 1.15
                )
                signals.append(signal)
                
            # 강한 매도 신호: BB 과매수 + RSI 과매수
            elif (data['overbought'] and data.get('overbought', False) and 
                  symbol in positions):
                
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=positions[symbol].quantity,
                    price=data['current_price'],
                    confidence=0.9,
                    reason='Combined: BB+RSI Overbought',
                    timestamp=datetime.now()
                )
                signals.append(signal)
                
        return signals
        
    def should_stop_loss(self, position: Position, current_data: pd.Series) -> bool:
        """손절매 조건"""
        return self.bb_strategy.should_stop_loss(position, current_data)
        
    def should_take_profit(self, position: Position, current_data: pd.Series) -> bool:
        """익절 조건"""
        return self.bb_strategy.should_take_profit(position, current_data) 