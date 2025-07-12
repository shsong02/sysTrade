"""
종목 발굴 API 라우트
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from glob import glob

router = APIRouter()

@router.get("/status")
async def get_discovery_status():
    """종목 발굴 상태 조회"""
    try:
        # 최신 리포트 파일 찾기
        report_path = "./data/discovery_reports/"
        if not os.path.exists(report_path):
            return {"status": "no_reports", "message": "아직 종목 발굴을 실행하지 않았습니다."}
            
        report_files = glob(report_path + "discovery_report_*.json")
        if not report_files:
            return {"status": "no_reports", "message": "아직 종목 발굴을 실행하지 않았습니다."}
            
        # 가장 최신 파일 로드
        latest_file = max(report_files, key=os.path.getctime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        return {
            "status": "completed",
            "report": report,
            "file_path": latest_file
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 조회 실패: {str(e)}")

@router.post("/run")
async def run_discovery(background_tasks: BackgroundTasks):
    """종목 발굴 실행"""
    try:
        # 백그라운드에서 종목 발굴 실행
        background_tasks.add_task(_run_discovery_background)
        
        return {
            "status": "started",
            "message": "종목 발굴을 시작했습니다. 완료까지 몇 분 소요됩니다."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"종목 발굴 실행 실패: {str(e)}")

async def _run_discovery_background():
    """백그라운드에서 종목 발굴 실행"""
    try:
        import sys
        import os
        
        # 프로젝트 루트 경로 추가
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        from main import STSystemManager
        
        # 시스템 매니저 초기화
        system_manager = STSystemManager("./config/config.yaml")
        
        # 설정 로딩
        if not system_manager.load_config():
            raise Exception("설정 파일 로딩 실패")
            
        # 모듈 초기화
        if not system_manager.initialize_modules():
            raise Exception("모듈 초기화 실패")
            
        # 종목 발굴 실행
        system_manager.run_discovery_mode()
        
    except Exception as e:
        print(f"백그라운드 종목 발굴 실행 오류: {e}")

@router.get("/reports")
async def get_discovery_reports(limit: int = 10):
    """종목 발굴 리포트 목록 조회"""
    try:
        report_path = "./data/discovery_reports/"
        if not os.path.exists(report_path):
            return {"reports": []}
            
        report_files = glob(report_path + "discovery_report_*.json")
        
        reports = []
        for file_path in sorted(report_files, key=os.path.getctime, reverse=True)[:limit]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    
                # 요약 정보만 포함
                summary = {
                    "file_name": os.path.basename(file_path),
                    "timestamp": report.get("timestamp"),
                    "total_candidates": report.get("total_candidates", 0),
                    "high_score_count": report.get("summary", {}).get("high_score_count", 0),
                    "avg_finance_score": report.get("summary", {}).get("avg_finance_score", 0),
                    "market_sentiment": report.get("macro_analysis", {}).get("market_sentiment", "Unknown")
                }
                reports.append(summary)
                
            except Exception as e:
                print(f"리포트 파일 읽기 오류 ({file_path}): {e}")
                continue
                
        return {"reports": reports}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 목록 조회 실패: {str(e)}")

@router.get("/report/{file_name}")
async def get_discovery_report(file_name: str):
    """특정 종목 발굴 리포트 조회"""
    try:
        report_path = f"./data/discovery_reports/{file_name}"
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="리포트 파일을 찾을 수 없습니다.")
            
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        return {"report": report}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="리포트 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 조회 실패: {str(e)}")

@router.get("/candidates")
async def get_top_candidates(limit: int = 20):
    """상위 투자 후보 종목 조회"""
    try:
        # 최신 리포트에서 후보 종목 추출
        report_path = "./data/discovery_reports/"
        if not os.path.exists(report_path):
            return {"candidates": []}
            
        report_files = glob(report_path + "discovery_report_*.json")
        if not report_files:
            return {"candidates": []}
            
        # 가장 최신 파일 로드
        latest_file = max(report_files, key=os.path.getctime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        candidates = report.get("candidates", [])
        
        # 재무 점수 기준으로 정렬
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get("finance_score", 0), 
            reverse=True
        )
        
        return {
            "candidates": sorted_candidates[:limit],
            "timestamp": report.get("timestamp"),
            "total_count": len(candidates)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"후보 종목 조회 실패: {str(e)}")

@router.get("/macro-analysis")
async def get_macro_analysis():
    """최신 거시경제 분석 결과 조회"""
    try:
        # 최신 리포트에서 거시경제 분석 추출
        report_path = "./data/discovery_reports/"
        if not os.path.exists(report_path):
            return {"analysis": None}
            
        report_files = glob(report_path + "discovery_report_*.json")
        if not report_files:
            return {"analysis": None}
            
        # 가장 최신 파일 로드
        latest_file = max(report_files, key=os.path.getctime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        macro_analysis = report.get("macro_analysis", {})
        
        return {
            "analysis": macro_analysis,
            "timestamp": report.get("timestamp")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"거시경제 분석 조회 실패: {str(e)}")

@router.get("/sectors")
async def get_sector_analysis():
    """섹터별 분석 결과 조회"""
    try:
        # 최신 리포트에서 섹터 분석 추출
        report_path = "./data/discovery_reports/"
        if not os.path.exists(report_path):
            return {"sectors": {}}
            
        report_files = glob(report_path + "discovery_report_*.json")
        if not report_files:
            return {"sectors": {}}
            
        # 가장 최신 파일 로드
        latest_file = max(report_files, key=os.path.getctime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        # 후보 종목들의 섹터별 분석
        candidates = report.get("candidates", [])
        sector_analysis = {}
        
        for candidate in candidates:
            sector = candidate.get("sector", "기타")
            if sector not in sector_analysis:
                sector_analysis[sector] = {
                    "count": 0,
                    "avg_finance_score": 0,
                    "total_volume_cost": 0,
                    "candidates": []
                }
                
            sector_analysis[sector]["count"] += 1
            sector_analysis[sector]["total_volume_cost"] += candidate.get("volume_cost", 0)
            sector_analysis[sector]["candidates"].append(candidate)
            
        # 평균 점수 계산
        for sector, data in sector_analysis.items():
            if data["count"] > 0:
                data["avg_finance_score"] = sum(
                    c.get("finance_score", 0) for c in data["candidates"]
                ) / data["count"]
                
        return {
            "sectors": sector_analysis,
            "timestamp": report.get("timestamp")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"섹터 분석 조회 실패: {str(e)}")

@router.delete("/reports/{file_name}")
async def delete_discovery_report(file_name: str):
    """종목 발굴 리포트 삭제"""
    try:
        report_path = f"./data/discovery_reports/{file_name}"
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="리포트 파일을 찾을 수 없습니다.")
            
        os.remove(report_path)
        
        return {"message": f"리포트 {file_name}이 삭제되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 삭제 실패: {str(e)}") 