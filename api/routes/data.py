"""
데이터 관련 API 라우트
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data_providers.unified_data_provider import UnifiedDataProvider

router = APIRouter()

class StockDataRequest(BaseModel):
    """주식 데이터 요청 모델"""
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = "1y"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

@router.get("/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = "1y"
):
    """
    주식 데이터 조회
    """
    try:
        data_provider = UnifiedDataProvider()
        
        # 기간 설정
        if not start_date and period:
            end_dt = datetime.now()
            if period == "1d":
                start_dt = end_dt - timedelta(days=1)
            elif period == "5d":
                start_dt = end_dt - timedelta(days=5)
            elif period == "1mo":
                start_dt = end_dt - timedelta(days=30)
            elif period == "3mo":
                start_dt = end_dt - timedelta(days=90)
            elif period == "6mo":
                start_dt = end_dt - timedelta(days=180)
            elif period == "1y":
                start_dt = end_dt - timedelta(days=365)
            elif period == "2y":
                start_dt = end_dt - timedelta(days=730)
            elif period == "5y":
                start_dt = end_dt - timedelta(days=1825)
            else:
                start_dt = end_dt - timedelta(days=365)
                
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")
        
        df = data_provider.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"데이터를 찾을 수 없습니다: {symbol}")
        
        # DataFrame을 JSON으로 변환
        data = df.reset_index().to_dict('records')
        
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "data": data,
            "count": len(data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 조회 실패: {str(e)}")

@router.get("/search/{query}")
async def search_stocks(query: str, limit: int = 10):
    """
    종목 검색
    """
    try:
        data_provider = UnifiedDataProvider()
        
        # 간단한 종목 검색 (실제로는 종목 마스터 DB에서 검색)
        # 여기서는 샘플 데이터 반환
        sample_stocks = [
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI"},
            {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI"},
            {"symbol": "035420", "name": "NAVER", "market": "KOSPI"},
            {"symbol": "051910", "name": "LG화학", "market": "KOSPI"},
            {"symbol": "006400", "name": "삼성SDI", "market": "KOSPI"},
            {"symbol": "207940", "name": "삼성바이오로직스", "market": "KOSPI"},
            {"symbol": "068270", "name": "셀트리온", "market": "KOSPI"},
            {"symbol": "035720", "name": "카카오", "market": "KOSPI"},
            {"symbol": "028260", "name": "삼성물산", "market": "KOSPI"},
            {"symbol": "012330", "name": "현대모비스", "market": "KOSPI"}
        ]
        
        # 쿼리에 맞는 종목 필터링
        filtered_stocks = [
            stock for stock in sample_stocks
            if query.lower() in stock["name"].lower() or query in stock["symbol"]
        ]
        
        return {
            "query": query,
            "results": filtered_stocks[:limit]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@router.get("/market/indices")
async def get_market_indices():
    """
    주요 지수 정보
    """
    try:
        data_provider = UnifiedDataProvider()
        
        indices = ["KS11", "KQ11", "KS200"]  # KOSPI, KOSDAQ, KOSPI200
        results = {}
        
        for index in indices:
            try:
                df = data_provider.get_stock_data(
                    symbol=index,
                    start_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    results[index] = {
                        "current": float(latest['close']),
                        "change": float(latest['close'] - prev['close']),
                        "change_pct": float((latest['close'] - prev['close']) / prev['close'] * 100),
                        "volume": int(latest.get('volume', 0)),
                        "date": latest.name.strftime("%Y-%m-%d")
                    }
            except Exception as e:
                print(f"지수 데이터 조회 실패 {index}: {e}")
                continue
        
        return {"indices": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지수 조회 실패: {str(e)}")

@router.get("/market/top-movers")
async def get_top_movers():
    """
    상승/하락 상위 종목
    """
    try:
        # 실제로는 실시간 데이터에서 조회
        # 여기서는 샘플 데이터 반환
        sample_data = {
            "gainers": [
                {"symbol": "123456", "name": "상승종목1", "price": 50000, "change": 2500, "change_pct": 5.26},
                {"symbol": "234567", "name": "상승종목2", "price": 30000, "change": 1200, "change_pct": 4.17},
                {"symbol": "345678", "name": "상승종목3", "price": 80000, "change": 3000, "change_pct": 3.90}
            ],
            "losers": [
                {"symbol": "987654", "name": "하락종목1", "price": 25000, "change": -1500, "change_pct": -5.66},
                {"symbol": "876543", "name": "하락종목2", "price": 40000, "change": -1800, "change_pct": -4.31},
                {"symbol": "765432", "name": "하락종목3", "price": 60000, "change": -2200, "change_pct": -3.54}
            ],
            "most_active": [
                {"symbol": "005930", "name": "삼성전자", "volume": 15000000},
                {"symbol": "000660", "name": "SK하이닉스", "volume": 8500000},
                {"symbol": "035420", "name": "NAVER", "volume": 6200000}
            ]
        }
        
        return sample_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상위 종목 조회 실패: {str(e)}")