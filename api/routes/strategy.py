"""
전략 관련 API 라우트
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime

router = APIRouter()

class StrategyConfig(BaseModel):
    """전략 설정 모델"""
    name: str
    strategy_type: str
    params: Dict[str, Any]
    enabled: bool = True
    description: Optional[str] = None

@router.get("/list")
async def get_strategies():
    """
    사용 가능한 전략 목록 조회
    """
    strategies = [
        {
            "id": "bollinger",
            "name": "볼린저 밴드",
            "description": "볼린저 밴드 상/하단 터치 시 매매",
            "type": "mean_reversion",
            "params": {
                "window": {"type": "int", "default": 20, "min": 5, "max": 50, "description": "이동평균 기간"},
                "std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "description": "표준편차 배수"}
            },
            "risk_level": "medium"
        },
        {
            "id": "rsi",
            "name": "RSI",
            "description": "RSI 과매수/과매도 구간 매매",
            "type": "momentum",
            "params": {
                "period": {"type": "int", "default": 14, "min": 5, "max": 30, "description": "RSI 계산 기간"},
                "oversold": {"type": "float", "default": 30, "min": 10, "max": 40, "description": "과매도 기준"},
                "overbought": {"type": "float", "default": 70, "min": 60, "max": 90, "description": "과매수 기준"}
            },
            "risk_level": "low"
        },
        {
            "id": "ma_cross",
            "name": "이동평균 교차",
            "description": "단기/장기 이동평균 교차 매매",
            "type": "trend_following",
            "params": {
                "short_window": {"type": "int", "default": 5, "min": 3, "max": 20, "description": "단기 이동평균 기간"},
                "long_window": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "장기 이동평균 기간"}
            },
            "risk_level": "medium"
        }
    ]
    
    return {"strategies": strategies}

@router.get("/{strategy_id}")
async def get_strategy_detail(strategy_id: str):
    """
    특정 전략의 상세 정보 조회
    """
    strategies = await get_strategies()
    strategy = next((s for s in strategies["strategies"] if s["id"] == strategy_id), None)
    
    if not strategy:
        raise HTTPException(status_code=404, detail="전략을 찾을 수 없습니다")
    
    return strategy

@router.post("/validate")
async def validate_strategy_params(config: StrategyConfig):
    """
    전략 파라미터 유효성 검증
    """
    try:
        # 전략 목록에서 해당 전략 찾기
        strategies = await get_strategies()
        strategy = next((s for s in strategies["strategies"] if s["id"] == config.strategy_type), None)
        
        if not strategy:
            return {"valid": False, "errors": ["알 수 없는 전략 타입"]}
        
        errors = []
        
        # 파라미터 검증
        for param_name, param_config in strategy["params"].items():
            if param_name not in config.params:
                if "default" not in param_config:
                    errors.append(f"필수 파라미터 누락: {param_name}")
                continue
            
            value = config.params[param_name]
            param_type = param_config["type"]
            
            # 타입 검증
            if param_type == "int" and not isinstance(value, int):
                errors.append(f"{param_name}은(는) 정수여야 합니다")
            elif param_type == "float" and not isinstance(value, (int, float)):
                errors.append(f"{param_name}은(는) 숫자여야 합니다")
            
            # 범위 검증
            if "min" in param_config and value < param_config["min"]:
                errors.append(f"{param_name}은(는) {param_config['min']} 이상이어야 합니다")
            if "max" in param_config and value > param_config["max"]:
                errors.append(f"{param_name}은(는) {param_config['max']} 이하여야 합니다")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
        
    except Exception as e:
        return {"valid": False, "errors": [f"검증 중 오류 발생: {str(e)}"]}

@router.get("/performance/{strategy_id}")
async def get_strategy_performance(strategy_id: str, period: str = "1y"):
    """
    전략 성과 통계 조회 (모의 데이터)
    """
    # 실제로는 백테스트 결과나 실거래 성과에서 조회
    # 여기서는 샘플 데이터 반환
    sample_performance = {
        "bollinger": {
            "total_return": 15.8,
            "win_rate": 68.5,
            "max_drawdown": -8.2,
            "sharpe_ratio": 1.24,
            "total_trades": 156,
            "avg_holding_days": 3.2
        },
        "rsi": {
            "total_return": 12.3,
            "win_rate": 72.1,
            "max_drawdown": -6.5,
            "sharpe_ratio": 1.15,
            "total_trades": 203,
            "avg_holding_days": 2.8
        },
        "ma_cross": {
            "total_return": 18.6,
            "win_rate": 65.2,
            "max_drawdown": -11.3,
            "sharpe_ratio": 1.31,
            "total_trades": 89,
            "avg_holding_days": 5.7
        }
    }
    
    performance = sample_performance.get(strategy_id)
    if not performance:
        raise HTTPException(status_code=404, detail="전략 성과 데이터를 찾을 수 없습니다")
    
    return {
        "strategy_id": strategy_id,
        "period": period,
        "performance": performance
    }

@router.post("/save")
async def save_strategy_config(config: StrategyConfig):
    """
    전략 설정 저장
    """
    try:
        # 실제로는 데이터베이스에 저장
        # 여기서는 파일에 저장하는 예시
        import json
        import os
        
        config_dir = "config/strategies"
        os.makedirs(config_dir, exist_ok=True)
        
        # 파일명 생성 (특수문자 제거)
        safe_name = "".join(c for c in config.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name.replace(' ', '_')}.json"
        filepath = os.path.join(config_dir, filename)
        
        # 설정 저장
        config_data = {
            "name": config.name,
            "strategy_type": config.strategy_type,
            "params": config.params,
            "enabled": config.enabled,
            "description": config.description,
            "created_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "message": "전략이 저장되었습니다",
            "config_id": safe_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전략 저장 실패: {str(e)}")

@router.get("/saved")
async def get_saved_strategies():
    """
    저장된 전략 목록 조회
    """
    try:
        import json
        import os
        from datetime import datetime
        
        config_dir = "config/strategies"
        if not os.path.exists(config_dir):
            return {"strategies": []}
        
        saved_strategies = []
        for filename in os.listdir(config_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(config_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    saved_strategies.append(config_data)
                except Exception as e:
                    print(f"전략 파일 로드 실패 {filename}: {e}")
                    continue
        
        # 생성일 기준 정렬
        saved_strategies.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {"strategies": saved_strategies}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장된 전략 조회 실패: {str(e)}")

@router.delete("/saved/{config_name}")
async def delete_saved_strategy(config_name: str):
    """
    저장된 전략 삭제
    """
    try:
        import os
        
        config_dir = "config/strategies"
        filename = f"{config_name}.json"
        filepath = os.path.join(config_dir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return {"status": "success", "message": "전략이 삭제되었습니다"}
        else:
            raise HTTPException(status_code=404, detail="전략을 찾을 수 없습니다")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전략 삭제 실패: {str(e)}")

@router.get("/recommendations")
async def get_strategy_recommendations():
    """
    현재 시장 상황에 맞는 전략 추천
    """
    # 실제로는 시장 상황 분석 후 추천
    # 여기서는 샘플 추천 반환
    recommendations = [
        {
            "strategy_id": "rsi",
            "reason": "현재 횡보장에서 RSI 전략이 효과적",
            "confidence": 85,
            "expected_return": "월 2-4%",
            "risk_level": "낮음"
        },
        {
            "strategy_id": "bollinger",
            "reason": "변동성이 높은 구간에서 볼린저 밴드 효과적",
            "confidence": 72,
            "expected_return": "월 3-6%",
            "risk_level": "중간"
        }
    ]
    
    return {"recommendations": recommendations}