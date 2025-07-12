"""
새로운 백테스팅 엔진
- 다양한 전략 지원
- 상세한 성과 분석
- 리스크 관리 통합
- 웹 시각화 지원
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import json

@dataclass
class Trade:
    """거래 정보"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    side: str = 'long'  # 'long' or 'short'
    entry_reason: str = ''
    exit_reason: str = ''
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: int
    avg_price: float
    side: str = 'long'
    entry_date: datetime = field(default_factory=datetime.now)
    unrealized_pnl: float = 0.0
    
@dataclass
class BacktestResult:
    """백테스트 결과"""
    trades: List[Trade] = field(default_factory=list)
    portfolio_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, Any] = field(default_factory=dict)
    drawdowns: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """
    새로운 백테스팅 엔진
    """
    
    def __init__(self, initial_capital: float = 10_000_000, commission: float = 0.0015):
        """
        백테스트 엔진 초기화
        
        Args:
            initial_capital: 초기 자본금
            commission: 수수료율 (기본 0.15%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.logger = logging.getLogger(__name__)
        
        # 백테스트 상태
        self.reset()
        
    def reset(self):
        """백테스트 상태 초기화"""
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.current_date: Optional[datetime] = None
        
    def run_backtest(self, 
                    data: Dict[str, pd.DataFrame], 
                    strategy_func,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    **strategy_params) -> BacktestResult:
        """
        백테스트 실행
        
        Args:
            data: {symbol: DataFrame} 형태의 가격 데이터
            strategy_func: 전략 함수
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            **strategy_params: 전략 파라미터
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        self.logger.info("백테스트 시작")
        self.reset()
        
        # 날짜 범위 설정
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            all_dates = [d for d in all_dates if d >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            all_dates = [d for d in all_dates if d <= end_dt]
            
        self.logger.info(f"백테스트 기간: {all_dates[0]} ~ {all_dates[-1]}")
        
        # 일별 백테스트 실행
        for date in all_dates:
            self.current_date = date
            
            # 현재 날짜의 데이터 준비
            current_data = {}
            for symbol, df in data.items():
                if date in df.index:
                    current_data[symbol] = df.loc[date]
                    
            if not current_data:
                continue
                
            # 포지션 업데이트 (미실현 손익 계산)
            self._update_positions(current_data)
            
            # 전략 실행
            signals = strategy_func(
                current_data=current_data,
                historical_data=data,
                current_date=date,
                positions=self.positions.copy(),
                **strategy_params
            )
            
            # 신호 처리
            if signals:
                self._process_signals(signals, current_data)
                
            # 포트폴리오 상태 기록
            self._record_portfolio_state(date, current_data)
            
        # 남은 포지션 정리
        self._close_all_positions(current_data)
        
        # 결과 생성
        result = self._generate_result()
        self.logger.info("백테스트 완료")
        
        return result
        
    def _update_positions(self, current_data: Dict[str, pd.Series]):
        """포지션 미실현 손익 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol]['close']
                if position.side == 'long':
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.avg_price - current_price) * position.quantity
                    
    def _process_signals(self, signals: List[Dict], current_data: Dict[str, pd.Series]):
        """거래 신호 처리"""
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']  # 'buy', 'sell', 'close'
            quantity = signal.get('quantity', 0)
            reason = signal.get('reason', '')
            
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close']
            
            if action == 'buy':
                self._execute_buy(symbol, quantity, current_price, reason)
            elif action == 'sell':
                self._execute_sell(symbol, quantity, current_price, reason)
            elif action == 'close':
                self._close_position(symbol, current_price, reason)
                
    def _execute_buy(self, symbol: str, quantity: int, price: float, reason: str):
        """매수 실행"""
        if quantity <= 0:
            return
            
        total_cost = quantity * price * (1 + self.commission)
        
        if total_cost > self.current_capital:
            # 자본 부족 시 가능한 만큼만 매수
            quantity = int(self.current_capital / (price * (1 + self.commission)))
            if quantity <= 0:
                return
            total_cost = quantity * price * (1 + self.commission)
            
        self.current_capital -= total_cost
        
        if symbol in self.positions:
            # 기존 포지션에 추가
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            total_cost_basis = pos.avg_price * pos.quantity + price * quantity
            pos.avg_price = total_cost_basis / total_quantity
            pos.quantity = total_quantity
        else:
            # 새 포지션 생성
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                side='long',
                entry_date=self.current_date
            )
            
        self.logger.debug(f"매수: {symbol} {quantity}주 @ {price:,.0f}")
        
    def _execute_sell(self, symbol: str, quantity: int, price: float, reason: str):
        """매도 실행"""
        if symbol not in self.positions or quantity <= 0:
            return
            
        pos = self.positions[symbol]
        sell_quantity = min(quantity, pos.quantity)
        
        proceeds = sell_quantity * price * (1 - self.commission)
        self.current_capital += proceeds
        
        # 거래 기록
        pnl = (price - pos.avg_price) * sell_quantity
        pnl_after_commission = proceeds - (sell_quantity * pos.avg_price)
        
        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=self.current_date,
            entry_price=pos.avg_price,
            exit_price=price,
            quantity=sell_quantity,
            side='long',
            entry_reason='',
            exit_reason=reason,
            pnl=pnl_after_commission,
            pnl_pct=(pnl_after_commission / (sell_quantity * pos.avg_price)) * 100
        )
        self.trades.append(trade)
        
        # 포지션 업데이트
        pos.quantity -= sell_quantity
        if pos.quantity <= 0:
            del self.positions[symbol]
            
        self.logger.debug(f"매도: {symbol} {sell_quantity}주 @ {price:,.0f}")
        
    def _close_position(self, symbol: str, price: float, reason: str):
        """포지션 전체 청산"""
        if symbol in self.positions:
            quantity = self.positions[symbol].quantity
            self._execute_sell(symbol, quantity, price, reason)
            
    def _close_all_positions(self, current_data: Dict[str, pd.Series]):
        """모든 포지션 청산"""
        for symbol in list(self.positions.keys()):
            if symbol in current_data:
                price = current_data[symbol]['close']
                self._close_position(symbol, price, 'backtest_end')
                
    def _record_portfolio_state(self, date: datetime, current_data: Dict[str, pd.Series]):
        """포트폴리오 상태 기록"""
        total_value = self.current_capital
        
        # 포지션 가치 계산
        position_values = {}
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol]['close']
                value = position.quantity * current_price
                total_value += value
                position_values[symbol] = {
                    'quantity': position.quantity,
                    'price': current_price,
                    'value': value,
                    'unrealized_pnl': position.unrealized_pnl
                }
                
        self.portfolio_history.append({
            'date': date,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions': position_values.copy(),
            'returns': (total_value / self.initial_capital - 1) * 100
        })
        
    def _generate_result(self) -> BacktestResult:
        """백테스트 결과 생성"""
        # 포트폴리오 DataFrame 생성
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index('date', inplace=True)
            
        # 수익률 계산
        returns = portfolio_df['returns'] if not portfolio_df.empty else pd.Series()
        
        # 드로우다운 계산
        if not portfolio_df.empty:
            peak = portfolio_df['total_value'].expanding().max()
            drawdown = (portfolio_df['total_value'] / peak - 1) * 100
        else:
            drawdown = pd.Series()
            
        # 성과 지표 계산
        metrics = self._calculate_metrics(portfolio_df, returns, drawdown)
        
        return BacktestResult(
            trades=self.trades,
            portfolio_values=portfolio_df,
            metrics=metrics,
            drawdowns=drawdown,
            returns=returns
        )
        
    def _calculate_metrics(self, portfolio_df: pd.DataFrame, returns: pd.Series, drawdown: pd.Series) -> Dict[str, Any]:
        """성과 지표 계산"""
        if portfolio_df.empty or len(self.trades) == 0:
            return {}
            
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # 거래 통계
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 리스크 지표
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # 샤프 비율 (일간 수익률 기준)
        if len(returns) > 1:
            daily_returns = returns.pct_change().dropna()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'start_date': portfolio_df.index[0] if not portfolio_df.empty else None,
            'end_date': portfolio_df.index[-1] if not portfolio_df.empty else None
        }
        
    def export_results(self, result: BacktestResult, output_path: str):
        """결과를 JSON 파일로 내보내기"""
        export_data = {
            'metrics': result.metrics,
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date.isoformat() if t.entry_date else None,
                    'exit_date': t.exit_date.isoformat() if t.exit_date else None,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'entry_reason': t.entry_reason,
                    'exit_reason': t.exit_reason
                }
                for t in result.trades
            ],
            'portfolio_values': result.portfolio_values.to_dict('records') if not result.portfolio_values.empty else []
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
        self.logger.info(f"백테스트 결과 저장: {output_path}")