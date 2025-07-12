"""
백테스트 관련 API 라우트
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.engines.backtest_engine import BacktestEngine, BacktestResult
from core.strategies.sample_strategies import BollingerBandStrategy, RSIStrategy, MovingAverageCrossStrategy
from core.data_providers.unified_data_provider import UnifiedDataProvider

router = APIRouter()

class BacktestRequest(BaseModel):
    """백테스트 요청 모델"""
    symbols: List[str]
    strategy: str
    strategy_params: Dict[str, Any] = {}
    start_date: str
    end_date: str
    initial_capital: float = 10_000_000
    commission: float = 0.0015

class BacktestResponse(BaseModel):
    """백테스트 응답 모델"""
    status: str
    task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# 백테스트 결과 저장소 (실제로는 Redis나 DB 사용)
backtest_results: Dict[str, Any] = {}
running_tasks: Dict[str, str] = {}

@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    백테스트 실행
    """
    try:
        # 태스크 ID 생성
        task_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        running_tasks[task_id] = "running"
        
        # 백그라운드에서 백테스트 실행
        background_tasks.add_task(
            execute_backtest_task,
            task_id,
            request
        )
        
        return BacktestResponse(
            status="started",
            task_id=task_id
        )
        
    except Exception as e:
        return BacktestResponse(
            status="error",
            error=str(e)
        )

@router.get("/result/{task_id}")
async def get_backtest_result(task_id: str):
    """
    백테스트 결과 조회
    """
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    status = running_tasks[task_id]
    
    if status == "running":
        return {"status": "running", "progress": "백테스트 실행 중..."}
    elif status == "completed":
        result = backtest_results.get(task_id)
        return {"status": "completed", "result": result}
    elif status == "error":
        error = backtest_results.get(task_id, {}).get("error", "Unknown error")
        return {"status": "error", "error": error}
    else:
        return {"status": "unknown"}

@router.get("/list")
async def list_backtests():
    """
    백테스트 목록 조회
    """
    results = []
    for task_id, status in running_tasks.items():
        result_data = backtest_results.get(task_id, {})
        results.append({
            "task_id": task_id,
            "status": status,
            "created_at": result_data.get("created_at"),
            "strategy": result_data.get("strategy"),
            "symbols": result_data.get("symbols")
        })
    
    return {"results": results}

@router.delete("/result/{task_id}")
async def delete_backtest_result(task_id: str):
    """
    백테스트 결과 삭제
    """
    if task_id in running_tasks:
        del running_tasks[task_id]
    if task_id in backtest_results:
        del backtest_results[task_id]
    
    return {"status": "deleted"}

async def execute_backtest_task(task_id: str, request: BacktestRequest):
    """
    백테스트 실행 태스크
    """
    try:
        # 데이터 제공자 초기화
        data_provider = UnifiedDataProvider()
        
        # 데이터 수집
        data = {}
        for symbol in request.symbols:
            try:
                df = data_provider.get_stock_data(
                    symbol=symbol,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                if df is not None and not df.empty:
                    data[symbol] = df
            except Exception as e:
                print(f"데이터 수집 실패 {symbol}: {e}")
                continue
        
        if not data:
            running_tasks[task_id] = "error"
            backtest_results[task_id] = {"error": "데이터 수집 실패"}
            return
        
        # 전략 선택
        strategy_map = {
            "bollinger": BollingerBandStrategy,
            "rsi": RSIStrategy,
            "ma_cross": MovingAverageCrossStrategy
        }
        
        strategy_class = strategy_map.get(request.strategy)
        if not strategy_class:
            running_tasks[task_id] = "error"
            backtest_results[task_id] = {"error": f"알 수 없는 전략: {request.strategy}"}
            return
        
        strategy = strategy_class(**request.strategy_params)
        
        # 백테스트 엔진 초기화 및 실행
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        # 전략 함수 래퍼
        def strategy_wrapper(**kwargs):
            return strategy.generate_signals(**kwargs)
        
        result = engine.run_backtest(
            data=data,
            strategy_func=strategy_wrapper,
            start_date=request.start_date,
            end_date=request.end_date,
            available_capital=request.initial_capital
        )
        
        # 결과 저장
        result_data = {
            "created_at": datetime.now().isoformat(),
            "strategy": request.strategy,
            "symbols": request.symbols,
            "metrics": result.metrics,
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                    "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason
                }
                for t in result.trades
            ],
            "portfolio_values": result.portfolio_values.to_dict('records') if not result.portfolio_values.empty else []
        }
        
        backtest_results[task_id] = result_data
        running_tasks[task_id] = "completed"
        
    except Exception as e:
        running_tasks[task_id] = "error"
        backtest_results[task_id] = {"error": str(e)}

@router.get("/strategies")
async def get_available_strategies():
    """
    사용 가능한 전략 목록
    """
    return {
        "strategies": [
            {
                "id": "bollinger",
                "name": "볼린저 밴드",
                "description": "볼린저 밴드 상/하단 터치 전략",
                "params": {
                    "window": {"type": "int", "default": 20, "min": 5, "max": 50},
                    "std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0}
                }
            },
            {
                "id": "rsi",
                "name": "RSI",
                "description": "RSI 과매수/과매도 전략",
                "params": {
                    "period": {"type": "int", "default": 14, "min": 5, "max": 30},
                    "oversold": {"type": "float", "default": 30, "min": 10, "max": 40},
                    "overbought": {"type": "float", "default": 70, "min": 60, "max": 90}
                }
            },
            {
                "id": "ma_cross",
                "name": "이동평균 교차",
                "description": "단기/장기 이동평균 교차 전략",
                "params": {
                    "short_window": {"type": "int", "default": 5, "min": 3, "max": 20},
                    "long_window": {"type": "int", "default": 20, "min": 10, "max": 50}
                }
            }
        ]
    }