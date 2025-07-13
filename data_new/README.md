# ST 시스템 데이터 구조 v2.0

## 🎯 프로세스 기반 최적화된 데이터 구조

main.py의 실행 프로세스를 기반으로 설계된 직관적이고 효율적인 데이터 관리 구조입니다.

## 📁 디렉토리 구조

```
data_new/
├── 1_discovery/           # 종목 발굴 단계 (Discovery Mode)
│   ├── macro_analysis/    # 거시경제 분석 결과
│   ├── stock_screening/   # 종목 스크리닝 결과
│   ├── finance_scores/    # 재무제표 점수 계산 결과
│   ├── candidates/        # 최종 투자 후보 종목
│   └── reports/          # 종목 발굴 리포트
│
├── 2_backtest/           # 백테스팅 단계 (Backtest Mode)
│   ├── strategies/       # 전략 설정 및 정의
│   ├── results/         # 백테스팅 결과 데이터
│   └── reports/         # 백테스팅 성과 리포트
│
├── 3_trading/           # 실시간 거래 단계 (Trading Mode)
│   ├── positions/       # 현재 포지션 정보
│   ├── orders/         # 주문 내역 및 체결 정보
│   ├── performance/    # 실시간 성과 데이터
│   └── logs/           # 거래 실행 로그
│
├── 4_shared/           # 공통 데이터 (모든 모드에서 사용)
│   ├── market_data/    # 시장 데이터 (가격, 거래량 등)
│   ├── reference/      # 기준 데이터 (종목 코드, 업종 정보 등)
│   ├── cache/         # 임시 캐시 데이터
│   └── temp/          # 임시 파일
│
└── 5_backup/          # 백업 데이터
```

## 🔄 프로세스별 데이터 플로우

### 1. Discovery Mode (종목 발굴)
1. **거시경제 분석** → `1_discovery/macro_analysis/`
2. **종목 스크리닝** → `1_discovery/stock_screening/`
3. **재무점수 계산** → `1_discovery/finance_scores/`
4. **후보 종목 선정** → `1_discovery/candidates/`
5. **리포트 생성** → `1_discovery/reports/`

### 2. Backtest Mode (백테스팅)
1. **전략 설정** → `2_backtest/strategies/`
2. **백테스팅 실행** → `2_backtest/results/`
3. **성과 분석** → `2_backtest/reports/`

### 3. Trading Mode (실시간 거래)
1. **포지션 관리** → `3_trading/positions/`
2. **주문 실행** → `3_trading/orders/`
3. **성과 모니터링** → `3_trading/performance/`
4. **로그 기록** → `3_trading/logs/`

## 📝 파일 명명 규칙

### 날짜 기반 파일명
- 형식: `{기능}_{YYYYMMDD}_{HHMMSS}.{확장자}`
- 예시: `macro_analysis_20241219_143022.json`

### 프로세스 기반 파일명
- 형식: `{단계}_{내용}_{날짜}.{확장자}`
- 예시: `discovery_candidates_20241219.csv`

## 🔧 주요 개선사항

1. **프로세스 가시성**: 번호 기반 폴더명으로 실행 순서 명확화
2. **중복 제거**: 유사한 목적의 폴더 통합 (analytics → 각 단계별 분산)
3. **역할 명확화**: 각 폴더의 역할과 데이터 타입 명확 정의
4. **확장성**: 새로운 기능 추가 시 기존 구조 유지하며 확장 가능
5. **유지보수성**: 데이터 생명주기 관리 용이

## 📊 데이터 보존 정책

- **Discovery 데이터**: 30일 보존
- **Backtest 결과**: 90일 보존  
- **Trading 로그**: 1년 보존
- **Shared 데이터**: 영구 보존 (캐시/temp 제외)
- **Backup 데이터**: 설정에 따라 관리

## 🔧 경로 사용 방법

### base_path 기준 설정
모든 경로는 `data_management.base_path`를 기준으로 한 상대 경로로 설정됩니다.

```yaml
data_management:
  base_path: "./data_new/"  # 기준 경로
  discovery:
    macro_analysis: "1_discovery/macro_analysis/"  # 상대 경로
```

### 코드에서 경로 사용
```python
# main.py의 get_data_path 함수 사용
from main import get_data_path

# 거시경제 분석 결과 저장 경로
macro_path = get_data_path('discovery.macro_analysis')
# 결과: ./data_new/1_discovery/macro_analysis/

# 파일 제어 설정 경로
report_path = get_data_path('fileControl.discovery.reports')
# 결과: ./data_new/1_discovery/reports/
```

### 경로 변경 시 장점
1. **중앙 집중 관리**: base_path만 변경하면 모든 경로 자동 업데이트
2. **환경별 설정**: 개발/운영 환경에 따라 base_path만 변경
3. **코드 간소화**: 중복된 경로 문자열 제거

## 🚀 마이그레이션 가이드

기존 `data/` 폴더에서 새로운 구조로 마이그레이션할 때:

1. 각 파일의 용도와 생성 프로세스 확인
2. 해당하는 새로운 폴더로 이동
3. 파일명을 새로운 규칙에 맞게 변경
4. config.yaml의 경로 설정 업데이트

### 자동 마이그레이션
```bash
python migrate_data_structure.py
```