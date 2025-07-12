"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .routes import backtest, data, strategy, discovery
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ST Trading System API",
    description="스윙 트레이딩 자동매매 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# 라우터 등록
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(strategy.router, prefix="/api/strategy", tags=["strategy"])
app.include_router(discovery.router, prefix="/api/discovery", tags=["discovery"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """메인 대시보드 페이지"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    """백테스트 페이지"""
    return templates.TemplateResponse("backtest.html", {"request": request})

@app.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request):
    """전략 관리 페이지"""
    return templates.TemplateResponse("strategy.html", {"request": request})

@app.get("/discovery", response_class=HTMLResponse)
async def discovery_page(request: Request):
    """종목 발굴 페이지"""
    return templates.TemplateResponse("discovery.html", {"request": request})

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "message": "ST Trading System API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)