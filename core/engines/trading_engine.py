"""
실시간 거래 엔진
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict

from ..strategies.trading_strategy import (
    BaseTradingStrategy, TradingSignal, Position,
    BollingerBandTradingStrategy, RSITradingStrategy, CombinedTradingStrategy
)

@dataclass
class TradingEngineConfig:
    """거래 엔진 설정"""
    max_positions: int = 20
    position_size_pct: float = 5.0
    stop_loss_pct: float = -5.0
    take_profit_pct: float = 10.0
    daily_loss_limit_pct: float = -10.0
    check_interval_seconds: int = 10
    market_open_time: str = "09:00"
    market_close_time: str = "15:30"
    strategy_name: str = "Combined"
    
class TradingEngine:
    """
    실시간 거래 엔진
    """
    
    def __init__(self, config: TradingEngineConfig, api_client=None):
        """
        거래 엔진 초기화
        
        Args:
            config: 거래 엔진 설정
            api_client: API 클라이언트 (한국투자증권 등)
        """
        self.config = config
        self.api_client = api_client
        self.logger = logging.getLogger("trading_engine")
        
        # 포지션 및 상태 관리
        self.positions: Dict[str, Position] = {}
        self.available_capital = 0.0
        self.daily_pnl = 0.0
        self.is_running = False
        
        # 전략 초기화
        self.strategy = self._initialize_strategy()
        
        # 데이터 저장
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.signal_history: List[TradingSignal] = []
        self.trade_history: List[Dict] = []
        
    def _initialize_strategy(self) -> BaseTradingStrategy:
        """전략 초기화"""
        strategy_config = asdict(self.config)
        
        if self.config.strategy_name == "BollingerBand":
            return BollingerBandTradingStrategy(strategy_config)
        elif self.config.strategy_name == "RSI":
            return RSITradingStrategy(strategy_config)
        elif self.config.strategy_name == "Combined":
            return CombinedTradingStrategy(strategy_config)
        else:
            self.logger.warning(f"Unknown strategy: {self.config.strategy_name}, using Combined")
            return CombinedTradingStrategy(strategy_config)
            
    async def start_trading(self, symbols: List[str]):
        """
        거래 시작
        
        Args:
            symbols: 거래할 종목 리스트
        """
        self.logger.info("거래 엔진 시작")
        self.is_running = True
        
        try:
            while self.is_running:
                # 시장 시간 체크
                if not self._is_market_open():
                    self.logger.info("장 시간이 아닙니다. 대기 중...")
                    await asyncio.sleep(60)  # 1분 대기
                    continue
                
                # 1. 시장 데이터 업데이트
                await self._update_market_data(symbols)
                
                # 2. 포지션 업데이트
                self._update_positions()
                
                # 3. 리스크 관리 체크
                risk_signals = self._check_risk_management()
                
                # 4. 전략 신호 생성
                strategy_signals = self.strategy.generate_signals(
                    self.market_data, self.positions
                )
                
                # 5. 신호 실행
                all_signals = risk_signals + strategy_signals
                for signal in all_signals:
                    await self._execute_signal(signal)
                
                # 6. 상태 로깅
                self._log_status()
                
                # 7. 다음 체크까지 대기
                await asyncio.sleep(self.config.check_interval_seconds)
                
        except Exception as e:
            self.logger.error(f"거래 엔진 오류: {e}")
        finally:
            self.logger.info("거래 엔진 종료")
            
    def stop_trading(self):
        """거래 중지"""
        self.is_running = False
        self.logger.info("거래 엔진 중지 요청")
        
    def _is_market_open(self) -> bool:
        """장 시간 체크"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # 주말 체크
        if now.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
            
        # 시간 체크
        return self.config.market_open_time <= current_time <= self.config.market_close_time
        
    async def _update_market_data(self, symbols: List[str]):
        """시장 데이터 업데이트"""
        try:
            if self.api_client:
                # API를 통한 실시간 데이터 수집
                for symbol in symbols:
                    # 실제 구현 시 한국투자증권 API 호출
                    # data = await self.api_client.get_current_price(symbol)
                    # self.market_data[symbol] = data
                    pass
            else:
                # 테스트용 더미 데이터
                self._generate_dummy_data(symbols)
                
        except Exception as e:
            self.logger.error(f"시장 데이터 업데이트 오류: {e}")
            
    def _generate_dummy_data(self, symbols: List[str]):
        """테스트용 더미 데이터 생성"""
        import numpy as np
        
        for symbol in symbols:
            if symbol not in self.market_data:
                # 초기 데이터 생성
                dates = pd.date_range(start='2024-01-01', end='2024-12-19', freq='D')
                prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
                volumes = np.random.randint(1000, 10000, len(dates))
                
                self.market_data[symbol] = pd.DataFrame({
                    'Open': prices * (1 + np.random.randn(len(dates)) * 0.01),
                    'High': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02),
                    'Low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02),
                    'Close': prices,
                    'Volume': volumes
                }, index=dates)
            else:
                # 새로운 데이터 포인트 추가
                last_price = self.market_data[symbol]['Close'].iloc[-1]
                new_price = last_price * (1 + np.random.randn() * 0.02)
                new_volume = np.random.randint(1000, 10000)
                
                new_row = pd.DataFrame({
                    'Open': [new_price * (1 + np.random.randn() * 0.01)],
                    'High': [new_price * (1 + abs(np.random.randn()) * 0.02)],
                    'Low': [new_price * (1 - abs(np.random.randn()) * 0.02)],
                    'Close': [new_price],
                    'Volume': [new_volume]
                }, index=[datetime.now()])
                
                self.market_data[symbol] = pd.concat([
                    self.market_data[symbol].tail(100),  # 최근 100개 데이터만 유지
                    new_row
                ])
                
    def _update_positions(self):
        """포지션 업데이트"""
        self.strategy.update_positions(self.market_data)
        
        # 일일 손익 계산
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        # 실현 손익은 거래 내역에서 계산 (여기서는 단순화)
        self.daily_pnl = total_unrealized_pnl
        
    def _check_risk_management(self) -> List[TradingSignal]:
        """리스크 관리 체크"""
        signals = []
        
        # 일일 손실 한도 체크
        if self.daily_pnl / self.available_capital < self.config.daily_loss_limit_pct / 100:
            self.logger.warning("일일 손실 한도 도달, 모든 포지션 청산")
            for symbol, position in self.positions.items():
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    price=position.current_price,
                    confidence=1.0,
                    reason='Daily Loss Limit',
                    timestamp=datetime.now()
                )
                signals.append(signal)
                
        # 개별 포지션 리스크 체크
        risk_signals = self.strategy.check_risk_management(self.positions)
        signals.extend(risk_signals)
        
        return signals
        
    async def _execute_signal(self, signal: TradingSignal):
        """신호 실행"""
        try:
            self.logger.info(f"신호 실행: {signal.action} {signal.symbol} {signal.quantity}주 @ {signal.price}")
            
            if signal.action == 'BUY':
                await self._execute_buy(signal)
            elif signal.action == 'SELL':
                await self._execute_sell(signal)
                
            # 신호 히스토리에 저장
            self.signal_history.append(signal)
            
        except Exception as e:
            self.logger.error(f"신호 실행 오류: {e}")
            
    async def _execute_buy(self, signal: TradingSignal):
        """매수 실행"""
        # 포지션 수 제한 체크
        if len(self.positions) >= self.config.max_positions:
            self.logger.warning(f"최대 포지션 수({self.config.max_positions}) 도달")
            return
            
        # 포지션 크기 계산
        if signal.quantity == 0:
            signal.quantity = self.strategy.calculate_position_size(
                signal.symbol, signal.price, self.available_capital
            )
            
        # 실제 매수 실행 (API 호출)
        if self.api_client:
            # success = await self.api_client.buy_order(signal.symbol, signal.quantity, signal.price)
            # if not success:
            #     return
            pass
            
        # 포지션 생성
        position = Position(
            symbol=signal.symbol,
            quantity=signal.quantity,
            entry_price=signal.price,
            current_price=signal.price,
            entry_date=signal.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        self.positions[signal.symbol] = position
        
        # 거래 히스토리에 저장
        trade_record = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': 'BUY',
            'quantity': signal.quantity,
            'price': signal.price,
            'reason': signal.reason
        }
        self.trade_history.append(trade_record)
        
        self.logger.info(f"매수 완료: {signal.symbol} {signal.quantity}주 @ {signal.price}")
        
    async def _execute_sell(self, signal: TradingSignal):
        """매도 실행"""
        if signal.symbol not in self.positions:
            self.logger.warning(f"매도할 포지션이 없습니다: {signal.symbol}")
            return
            
        position = self.positions[signal.symbol]
        
        # 실제 매도 실행 (API 호출)
        if self.api_client:
            # success = await self.api_client.sell_order(signal.symbol, signal.quantity, signal.price)
            # if not success:
            #     return
            pass
            
        # 실현 손익 계산
        realized_pnl = (signal.price - position.entry_price) * signal.quantity
        
        # 포지션 제거
        del self.positions[signal.symbol]
        
        # 거래 히스토리에 저장
        trade_record = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': 'SELL',
            'quantity': signal.quantity,
            'price': signal.price,
            'realized_pnl': realized_pnl,
            'reason': signal.reason
        }
        self.trade_history.append(trade_record)
        
        self.logger.info(f"매도 완료: {signal.symbol} {signal.quantity}주 @ {signal.price}, 손익: {realized_pnl:,.0f}원")
        
    def _log_status(self):
        """상태 로깅"""
        total_positions = len(self.positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        if total_positions > 0:
            self.logger.info(
                f"포지션: {total_positions}개, "
                f"미실현 손익: {total_unrealized_pnl:,.0f}원, "
                f"일일 손익: {self.daily_pnl:,.0f}원"
            )
            
    def get_portfolio_status(self) -> Dict[str, Any]:
        """포트폴리오 상태 반환"""
        total_value = sum(p.current_price * p.quantity for p in self.positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        return {
            'timestamp': datetime.now(),
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'available_capital': self.available_capital,
            'positions': [
                {
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'unrealized_pnl': p.unrealized_pnl,
                    'entry_date': p.entry_date.isoformat()
                }
                for p in self.positions.values()
            ],
            'strategy_status': self.strategy.get_strategy_status()
        }
        
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """거래 히스토리 반환"""
        return self.trade_history[-limit:]
        
    def get_signal_history(self, limit: int = 100) -> List[Dict]:
        """신호 히스토리 반환"""
        return [
            {
                'timestamp': s.timestamp.isoformat(),
                'symbol': s.symbol,
                'action': s.action,
                'quantity': s.quantity,
                'price': s.price,
                'confidence': s.confidence,
                'reason': s.reason
            }
            for s in self.signal_history[-limit:]
        ]
        
    async def force_close_all_positions(self):
        """모든 포지션 강제 청산"""
        self.logger.warning("모든 포지션 강제 청산 시작")
        
        for symbol, position in list(self.positions.items()):
            signal = TradingSignal(
                symbol=symbol,
                action='SELL',
                quantity=position.quantity,
                price=position.current_price,
                confidence=1.0,
                reason='Force Close',
                timestamp=datetime.now()
            )
            await self._execute_sell(signal)
            
        self.logger.info("모든 포지션 강제 청산 완료")


class TradingEngineManager:
    """거래 엔진 관리자"""
    
    def __init__(self):
        self.engines: Dict[str, TradingEngine] = {}
        self.logger = logging.getLogger("trading_engine_manager")
        
    def create_engine(self, 
                     name: str, 
                     config: TradingEngineConfig, 
                     api_client=None) -> TradingEngine:
        """거래 엔진 생성"""
        engine = TradingEngine(config, api_client)
        self.engines[name] = engine
        self.logger.info(f"거래 엔진 생성: {name}")
        return engine
        
    def get_engine(self, name: str) -> Optional[TradingEngine]:
        """거래 엔진 조회"""
        return self.engines.get(name)
        
    async def start_all_engines(self, symbols: List[str]):
        """모든 엔진 시작"""
        tasks = []
        for name, engine in self.engines.items():
            task = asyncio.create_task(engine.start_trading(symbols))
            tasks.append(task)
            self.logger.info(f"거래 엔진 시작: {name}")
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    def stop_all_engines(self):
        """모든 엔진 중지"""
        for name, engine in self.engines.items():
            engine.stop_trading()
            self.logger.info(f"거래 엔진 중지: {name}")
            
    def get_all_status(self) -> Dict[str, Any]:
        """모든 엔진 상태 반환"""
        return {
            name: engine.get_portfolio_status()
            for name, engine in self.engines.items()
        }