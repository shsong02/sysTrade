#!/usr/bin/env python3
"""
백테스팅 엔진 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engines.backtest_engine import BacktestEngine
from core.strategies.sample_strategies import BollingerBandStrategy, RSIStrategy
from core.data_providers.unified_data_provider import UnifiedDataProvider

def create_sample_data():
    """샘플 데이터 생성"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # 삼성전자 모의 데이터
    np.random.seed(42)
    base_price = 70000
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    samsung_data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 10000000) for _ in prices]
    }, index=dates)
    
    # SK하이닉스 모의 데이터
    np.random.seed(123)
    base_price = 90000
    returns = np.random.normal(0.0005, 0.025, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sk_data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(500000, 5000000) for _ in prices]
    }, index=dates)
    
    return {
        '005930': samsung_data,
        '000660': sk_data
    }

def test_bollinger_strategy():
    """볼린저 밴드 전략 테스트"""
    print("=" * 50)
    print("볼린저 밴드 전략 백테스트")
    print("=" * 50)
    
    # 샘플 데이터 생성
    data = create_sample_data()
    
    # 백테스트 엔진 초기화
    engine = BacktestEngine(initial_capital=10_000_000, commission=0.0015)
    
    # 볼린저 밴드 전략
    strategy = BollingerBandStrategy(window=20, std_dev=2.0)
    
    # 전략 함수 래퍼
    def strategy_wrapper(**kwargs):
        return strategy.generate_signals(**kwargs)
    
    # 백테스트 실행
    result = engine.run_backtest(
        data=data,
        strategy_func=strategy_wrapper,
        start_date='2023-01-01',
        end_date='2023-12-31',
        available_capital=10_000_000
    )
    
    # 결과 출력
    print(f"총 수익률: {result.metrics['total_return']:.2f}%")
    print(f"최종 자산: {result.metrics['final_value']:,.0f}원")
    print(f"총 거래 수: {result.metrics['total_trades']}회")
    print(f"승률: {result.metrics['win_rate']:.1f}%")
    print(f"최대 낙폭: {result.metrics['max_drawdown']:.2f}%")
    print(f"샤프 비율: {result.metrics['sharpe_ratio']:.2f}")
    
    if result.trades:
        print(f"\n최근 5개 거래:")
        for trade in result.trades[-5:]:
            print(f"  {trade.symbol}: {trade.entry_date.strftime('%Y-%m-%d')} ~ {trade.exit_date.strftime('%Y-%m-%d')}, "
                  f"수익률: {trade.pnl_pct:.2f}%, 사유: {trade.exit_reason}")
    
    return result

def test_rsi_strategy():
    """RSI 전략 테스트"""
    print("\n" + "=" * 50)
    print("RSI 전략 백테스트")
    print("=" * 50)
    
    # 샘플 데이터 생성
    data = create_sample_data()
    
    # 백테스트 엔진 초기화
    engine = BacktestEngine(initial_capital=10_000_000, commission=0.0015)
    
    # RSI 전략
    strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    
    # 전략 함수 래퍼
    def strategy_wrapper(**kwargs):
        return strategy.generate_signals(**kwargs)
    
    # 백테스트 실행
    result = engine.run_backtest(
        data=data,
        strategy_func=strategy_wrapper,
        start_date='2023-01-01',
        end_date='2023-12-31',
        available_capital=10_000_000
    )
    
    # 결과 출력
    print(f"총 수익률: {result.metrics['total_return']:.2f}%")
    print(f"최종 자산: {result.metrics['final_value']:,.0f}원")
    print(f"총 거래 수: {result.metrics['total_trades']}회")
    print(f"승률: {result.metrics['win_rate']:.1f}%")
    print(f"최대 낙폭: {result.metrics['max_drawdown']:.2f}%")
    print(f"샤프 비율: {result.metrics['sharpe_ratio']:.2f}")
    
    if result.trades:
        print(f"\n최근 5개 거래:")
        for trade in result.trades[-5:]:
            print(f"  {trade.symbol}: {trade.entry_date.strftime('%Y-%m-%d')} ~ {trade.exit_date.strftime('%Y-%m-%d')}, "
                  f"수익률: {trade.pnl_pct:.2f}%, 사유: {trade.exit_reason}")
    
    return result

def test_real_data():
    """실제 데이터로 테스트 (데이터 제공자 사용)"""
    print("\n" + "=" * 50)
    print("실제 데이터 백테스트")
    print("=" * 50)
    
    try:
        # 데이터 제공자 초기화
        data_provider = UnifiedDataProvider()
        
        # 실제 데이터 수집
        symbols = ['005930', '000660']  # 삼성전자, SK하이닉스
        data = {}
        
        for symbol in symbols:
            try:
                df = data_provider.get_stock_data(
                    symbol=symbol,
                    start_date='2023-06-01',
                    end_date='2023-12-31'
                )
                if df is not None and not df.empty:
                    data[symbol] = df
                    print(f"{symbol} 데이터 수집 완료: {len(df)}일")
            except Exception as e:
                print(f"{symbol} 데이터 수집 실패: {e}")
        
        if not data:
            print("실제 데이터 수집 실패, 샘플 데이터로 대체")
            return test_bollinger_strategy()
        
        # 백테스트 실행
        engine = BacktestEngine(initial_capital=10_000_000, commission=0.0015)
        strategy = BollingerBandStrategy(window=20, std_dev=2.0)
        
        def strategy_wrapper(**kwargs):
            return strategy.generate_signals(**kwargs)
        
        result = engine.run_backtest(
            data=data,
            strategy_func=strategy_wrapper,
            start_date='2023-06-01',
            end_date='2023-12-31',
            available_capital=10_000_000
        )
        
        # 결과 출력
        print(f"총 수익률: {result.metrics['total_return']:.2f}%")
        print(f"최종 자산: {result.metrics['final_value']:,.0f}원")
        print(f"총 거래 수: {result.metrics['total_trades']}회")
        print(f"승률: {result.metrics['win_rate']:.1f}%")
        print(f"최대 낙폭: {result.metrics['max_drawdown']:.2f}%")
        print(f"샤프 비율: {result.metrics['sharpe_ratio']:.2f}")
        
        return result
        
    except Exception as e:
        print(f"실제 데이터 백테스트 실패: {e}")
        print("샘플 데이터로 대체")
        return test_bollinger_strategy()

if __name__ == "__main__":
    print("백테스팅 엔진 테스트 시작")
    print("=" * 50)
    
    # 각 전략 테스트
    bollinger_result = test_bollinger_strategy()
    rsi_result = test_rsi_strategy()
    real_result = test_real_data()
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)