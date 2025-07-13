# ST Trading System v0.1

## 🚀 새로운 기능: LLM 기반 시황 분석

### 📊 포괄적인 거시경제 분석
- **시장 지수 분석**: KOSPI, KOSDAQ, KRX100 등 주요 지수 동향
- **ETF 트렌드 분석**: 섹터별/테마별 ETF 성과 및 자금 흐름
- **섹터 로테이션**: 21개 주요 업종별 성과 및 로테이션 패턴
- **테마 분석**: AI/반도체, 2차전지, 바이오 등 10개 핫 테마 추적
- **🤖 AI 기반 시황 분석**: OpenAI o3-mini 모델을 활용한 전문적 시장 분석

### 🔧 환경 설정

#### 1. OpenAI API 키 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# LLM Analysis Configuration (선택사항)
OPENAI_MODEL=o3-mini
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.3
```

#### 2. 필요 패키지 설치
```bash
pip install openai python-dotenv tiktoken
```

#### 3. LLM 분석 설정 (config.yaml)
```yaml
searchMacro:
  llm_analysis:
    enabled: true                     # LLM 분석 활성화
    model: "o3-mini"                 # OpenAI 모델명
    analysis_scope:
      market_overview: true          # 전체 시장 개관
      sector_analysis: true          # 섹터별 상세 분석
      etf_trends: true              # ETF 트렌드 분석
      theme_analysis: true          # 테마별 분석
      technical_indicators: true     # 기술적 지표 분석
      sentiment_analysis: true       # 시장 심리 분석
```

### 📋 LLM 분석 결과
- **📊 시장 전체 개관**: 전반적인 시장 방향성과 투자자 심리
- **🏭 섹터별 상세 분석**: 업종별 성과 배경과 투자 전략
- **🏛️ ETF 트렌드 분석**: ETF를 통한 자금 흐름과 투자 선호도

## 🔍 종목 발굴 시스템 (Discovery Mode)

### 📋 5단계 종목 발굴 프로세스
1. **[1/5] 거시경제 상황 분석**: 시장 전반적 흐름과 투자 환경 파악
2. **[2/5] 재무제표 기반 종목 스크리닝**: 기업의 재무 건전성 평가
3. **[3/5] 테마/업종별 종목 검색**: 시장 주도주와 테마주 발굴
4. **[4/5] 최종 투자 후보 종목 선정**: 종합적 분석을 통한 후보 선별
5. **[5/5] 결과 저장 및 리포트 생성**: 분석 결과 정리 및 보고서 작성

### ⚙️ Discovery 단계별 설정

`config.yaml`에서 각 단계를 개별적으로 활성화/비활성화할 수 있습니다:

```yaml
discovery:
  # 종목 발굴 단계별 활성화 설정
  steps:
    macro_analysis: true             # [1/5] 거시경제 상황 분석
    finance_screening: true          # [2/5] 재무제표 기반 종목 스크리닝
    theme_sector_search: true        # [3/5] 테마/업종별 종목 검색
    candidate_selection: true        # [4/5] 최종 투자 후보 종목 선정
    report_generation: true          # [5/5] 결과 저장 및 리포트 생성
    
  # 각 단계별 세부 설정
  macro_analysis:
    enabled: true
    skip_on_error: false            # 에러 발생 시 다음 단계 진행 여부
    timeout_minutes: 30             # 타임아웃 (분)
    
  theme_sector_search:
    enabled: true
    search_options:
      market_leader: true           # 시장 주도주 검색
      theme_upjong: true           # 테마/업종 검색
      
  candidate_selection:
    enabled: true
    max_candidates: 50              # 최대 후보 종목 수
    
  # 전체 Discovery 설정
  general:
    continue_on_step_failure: false  # 단계 실패 시 전체 중단 여부
    max_total_runtime_minutes: 120  # 전체 최대 실행 시간
```

### 🎛️ 주요 설정 옵션

#### 단계별 활성화/비활성화
- 각 단계를 `true`/`false`로 개별 제어
- 필요한 단계만 선택적으로 실행 가능

#### 에러 처리 옵션
- `skip_on_error`: 개별 단계에서 에러 발생 시 다음 단계 진행 여부
- `continue_on_step_failure`: 전체적으로 단계 실패 시 중단 여부

#### 성능 제어
- `timeout_minutes`: 각 단계별 최대 실행 시간
- `max_candidates`: 후보 종목 수 제한
- `max_total_runtime_minutes`: 전체 최대 실행 시간

### 🏃‍♂️ 실행 방법
```bash
python main.py --mode discovery
```

### 📊 출력 결과
- 단계별 실행 상태와 결과 로그
- 활성화된 단계 목록 표시
- 에러 발생 시 상세 정보와 대응 방안
- 최종 후보 종목 리스트와 분석 리포트
- **🔥 테마별 분석**: 핫 테마의 지속성과 신흥 테마 발굴
- **📈 기술적 지표 분석**: 차트 패턴과 기술적 전망
- **💭 시장 심리 분석**: 투자자 심리와 행동경제학적 관점
- **🎯 종합 투자 전략**: 단기/중기/장기 투자 전략 및 리스크 관리

### 📁 리포트 저장 구조
```
./data/processed/macro_analysis/YYYYMMDD/
├── macro_analysis_report_YYYYMMDD_HHMMSS.json    # 상세 데이터
├── macro_analysis_summary_YYYYMMDD_HHMMSS.txt    # AI 분석 리포트
└── charts/                                        # ETF 차트들
    ├── 122630_KODEX_레버리지.csv
    └── ...
```

# 📈 ST (System Trading) v0.1

한국투자증권 API를 활용한 종합 주식 자동 거래 시스템

## 🎯 프로젝트 개요

ST_ver0.1은 한국 주식 시장에서 완전 자동화된 거래를 수행하는 시스템입니다. 재무 분석, 기술적 분석, 뉴스 감성 분석을 통합하여 체계적인 투자 전략을 실행합니다.

### 주요 기능
- 🔄 **실시간 자동 거래**: 한국투자증권 API 연동
- 📊 **종목 스크리닝**: 재무제표 기반 종목 평가
- 📈 **기술적 분석**: 볼린저 밴드, RSI, MACD 등 다양한 지표
- 📰 **뉴스 분석**: 시장 동향 파악을 위한 뉴스 크롤링
- 🧪 **백테스팅**: 전략 검증 및 성과 분석
- 📱 **텔레그램 알림**: 거래 상황 실시간 알림

## 🛠️ 시스템 구조

```
ST_ver0.1/
├── system_trade.py         # 메인 거래 시스템
├── trade_strategy.py       # 거래 전략 및 기술적 분석
├── finance_score.py        # 재무 점수 평가
├── search_stocks.py        # 종목 검색 및 스크리닝
├── search_macro.py         # 거시 경제 분석
├── back_test.py           # 백테스팅 시스템
├── config/
│   ├── config.yaml        # 메인 설정 파일
│   ├── kisdev_vi.yaml     # 한국투자증권 API 설정
│   └── logging.yaml       # 로깅 설정
├── tools/
│   ├── news_crawler.py    # 뉴스 크롤링
│   ├── market_condition.py # 시장 상황 분석
│   ├── st_utils.py        # 유틸리티 함수
│   └── custom_logger.py   # 로깅 설정
└── src/kis/               # 한국투자증권 API 관련 파일
```

## 🐍 가상환경 및 패키지 설치

### 1. Python 3.10 설치 (M1/M2/M3 사용자는 arm64 네이티브 권장)
- [python.org](https://www.python.org/downloads/macos/)에서 3.10 버전 arm64 설치

### 2. uv 설치 (최신 패키지 매니저)
```bash
pip install uv
```

### 3. uv 기반 가상환경 생성 및 패키지 설치
```bash
# (기존 venv/venv_new 폴더가 있다면 삭제)
uv venv .venv --python=3.10

# 가상환경 활성화 (zsh/bash)
source .venv/bin/activate

# pip 최신화 및 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Apple Silicon) 아키텍처 혼동 주의
- 반드시 arm64 네이티브 Python으로 가상환경을 만들고, arm64 wheel이 설치되는지 확인하세요.
- x86_64와 arm64가 섞이면 numpy/pandas import 에러가 발생할 수 있습니다.

## 📋 사전 준비사항

### 1. 필수 계정 및 API 키
- **한국투자증권 계좌**: 실거래 또는 모의거래 계좌
- **한국투자증권 API 키**: 앱키(APP_KEY), 앱시크릿(APP_SECRET)
- **네이버 뉴스 API 키**: 뉴스 크롤링용
- **텔레그램 봇 토큰**: 알림 기능용

### 2. 시스템 요구사항
- Python 3.8+
- macOS (chromedriver-mac-arm64 포함)
- Chrome 브라우저
- 8GB+ RAM 권장

### 3. 필수 Python 패키지

> requirements.txt 파일을 사용하여 일괄 설치하세요. (아래 개별 설치 명령어는 삭제)

## ⚙️ 설정 방법

### 1. 한국투자증권 API 설정
`config/kisdev_vi.yaml` 파일을 다음과 같이 설정:

```yaml
# 모의거래 설정
vps: "https://openapivts.koreainvestment.com:29443"
paper_app: "YOUR_PAPER_APP_KEY"
paper_sec: "YOUR_PAPER_SECRET_KEY"
my_paper_stock: "YOUR_PAPER_ACCOUNT_NUMBER"

# 실거래 설정
prod: "https://openapi.koreainvestment.com:9443"
my_app: "YOUR_REAL_APP_KEY"
my_sec: "YOUR_REAL_SECRET_KEY"
my_acct_stock: "YOUR_REAL_ACCOUNT_NUMBER"
```

### 2. 메인 설정 파일 수정
`config/config.yaml`에서 주요 설정을 조정:

```yaml
tradeStock:
  scheduler:
    mode: "virtual"  # real, virtual, backfill
    interval: 5      # 5분 간격
    target: "all"    # all, stock, kospi
  
  buy_condition:
    codepick_buy_holdday: 40
    timepick_trend_period: "ma60"
    
  sell_condition:
    default_profit_change: 10  # 10% 목표 수익률
    default_holding_days: 30   # 30일 최대 보유
```

### 3. 텔레그램 봇 설정
`config/config.yaml`에서 텔레그램 설정:

```yaml
mainInit:
  telegram_token: "YOUR_BOT_TOKEN"
  telegram_id: "YOUR_CHAT_ID"
```

## 🚀 실행 방법

### 0. 가상환경 활성화
```bash
source .venv/bin/activate
```

### 1. 새로운 웹 기반 시스템 실행 (권장)

#### 1.1 FastAPI 서버 실행
```bash
# 웹 서버 실행 (백그라운드)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 1.2 웹 인터페이스 접속
- **대시보드**: http://localhost:8000/
  - 실시간 포트폴리오 현황
  - 주요 지표 모니터링
  - 활성 포지션 및 거래 내역

- **백테스팅**: http://localhost:8000/backtest
  - 전략 선택 및 파라미터 설정
  - 실시간 백테스트 실행
  - 성과 분석 및 시각화

- **전략 관리**: http://localhost:8000/strategy
  - 전략 파라미터 동적 조정
  - 전략 저장 및 로드
  - 실시간 미리보기

#### 1.3 백테스팅 엔진 단독 테스트
```bash
# 백테스팅 엔진 테스트
python tools/test_backtest.py
```

### 2. 기존 스크립트 실행 (레거시)

#### 2.1 종목 검색 및 분석
```bash
# 재무제표 기반 종목 스크리닝
python search_stocks.py

# 거시경제 분석
python search_macro.py

# 종목 재무 점수 계산
python finance_score.py
```

#### 2.2 자동 거래 시스템 실행
```bash
# 메인 거래 시스템 실행
python system_trade.py
```

#### 2.3 백테스팅 실행
```bash
# 전략 백테스팅
python back_test.py
```

#### 2.4 뉴스 분석
```bash
# 뉴스 크롤링 및 분석
python -c "
from tools.news_crawler import newsCrawler
nc = newsCrawler()
nc.search_interval(['20240101', '20240131'])
"
```

### 3. 통합 시스템 실행
```bash
# 메인 시스템 매니저 실행
python main.py

# 실행 모드 선택:
# 1. backtest - 백테스팅 모드
# 2. trading - 실시간 거래 모드  
# 3. api - API 서버 모드
# 4. analysis - 분석 모드
```

## 📊 핵심 알고리즘

### 1. 종목 선별 기준
- **재무 점수**: 70점 이상 (매출액 증가율, 영업이익률, ROE 기반)
- **최소 주가**: 5,000원 이상
- **최소 거래대금**: 5억원 이상
- **기술적 지표**: 볼린저 밴드, RSI, MACD 종합 판단

### 2. 매수 조건
- 재무 점수 기준 통과
- 볼린저 밴드 하단 터치 (과매도)
- RSI 30 이하
- 거래량 급증 (평균 대비 1.5배 이상)
- 기관/외국인 순매수

### 3. 매도 조건
- 목표 수익률 달성 (기본 10%)
- 손절매 기준 (-5%)
- 최대 보유 기간 초과 (30일)
- 기술적 매도 신호 발생

### 4. 리스크 관리
- **포지션 크기**: 계좌 자금의 10% 이내
- **분산 투자**: 최대 5개 종목 동시 보유
- **손절매**: 자동 손절매 시스템
- **일일 거래 한도**: 계좌 자금의 20% 이내

## 📈 성과 분석 및 모니터링

### 1. 로그 확인
```bash
# 시스템 로그 확인
tail -f log/systemTrade.log

# 거래 내역 확인
ls -la data/system_trade/
```

### 2. 백테스팅 결과
- 연간 수익률
- 샤프 비율
- 최대 낙폭 (MDD)
- 승률 및 손익비

### 3. 실시간 모니터링
- 텔레그램 알림을 통한 실시간 거래 상황 확인
- 웹 브라우저를 통한 차트 시각화
- 일일/주간/월간 성과 리포트

## 🔧 고급 설정

### 1. 거래 전략 커스터마이징
`trade_strategy.py`에서 다음 함수들을 수정:
- `_bollinger()`: 볼린저 밴드 설정
- `_rsi()`: RSI 지표 설정
- `_macd()`: MACD 지표 설정

### 2. 종목 스크리닝 기준 변경
`config/config.yaml`에서 스크리닝 기준 조정:
```yaml
searchStock:
  market_leader:
    from_krx:
      params:
        threshold_close: 5000
        threshold_finance_score: 70
        threshold_min_trend: 5
        threshold_volumecost: 5
```

### 3. 뉴스 분석 설정
`tools/news_crawler.py`에서 뉴스 수집 설정:
- 수집 간격
- 키워드 필터링
- 감성 분석 모델 변경

## 📝 주의사항

### 1. 투자 위험
- 이 시스템은 교육 및 연구 목적으로 제작되었습니다
- 실제 투자 시 발생하는 손실에 대해서는 사용자 본인이 책임집니다
- 충분한 백테스팅과 모의거래 후 실거래에 적용하시기 바랍니다

### 2. API 사용 제한
- 한국투자증권 API 호출 제한을 준수해야 합니다
- 과도한 API 호출 시 계정이 제한될 수 있습니다

### 3. 시장 상황 고려
- 시장 상황에 따라 전략의 성과가 달라질 수 있습니다
- 정기적인 전략 검토와 업데이트가 필요합니다

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/새기능`)
3. 변경사항을 커밋합니다 (`git commit -am '새 기능 추가'`)
4. 브랜치에 푸시합니다 (`git push origin feature/새기능`)
5. Pull Request를 생성합니다

## 📞 지원

- 이슈 리포트: [GitHub Issues](https://github.com/your-repo/issues)
- 문의사항: 이메일 또는 텔레그램

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🔄 업데이트 내역

### v0.1.2 (2024-12-19) ✨ **최신**
- **새로운 백테스팅 엔진**: 완전히 새로 구축된 백테스팅 시스템
- **웹 기반 시각화**: FastAPI + Bootstrap 기반 웹 인터페이스
- **전략 관리 시스템**: 동적 파라미터 조정 및 실시간 미리보기
- **다양한 전략 지원**: 볼린저밴드, RSI, 이동평균교차 전략
- **통합 데이터 제공자**: FinanceDataReader, pykrx 통합
- **실시간 성과 분석**: 수익률, 승률, 샤프비율, 최대낙폭 등

### v0.1.1 (기존)
- 기본 자동 거래 시스템 구축
- 한국투자증권 API 연동
- 재무 분석 및 기술적 분석 구현
- 뉴스 크롤링 및 분석 기능
- 기본 백테스팅 시스템
- 텔레그램 알림 기능

### 📁 **새로운 디렉토리 구조**
```
ST_ver0.1/
├── core/                  # 🆕 핵심 엔진
│   ├── engines/          # 백테스팅/트레이딩 엔진
│   ├── strategies/       # 전략 모듈
│   ├── data_providers/   # 데이터 제공자
│   └── risk_management/  # 리스크 관리
├── api/                   # 🆕 FastAPI 백엔드
│   ├── main.py          # 메인 앱
│   └── routes/          # API 라우트
├── frontend/              # 🆕 웹 인터페이스
│   ├── templates/       # HTML 템플릿
│   └── static/          # 정적 파일
├── config/
│   ├── strategies/      # 🆕 저장된 전략 설정
│   └── config.yaml      # 확장된 설정
└── tools/
    └── test_backtest.py # 🆕 테스트 스크립트
```

---

**⚠️ 면책조항**: 이 소프트웨어는 "있는 그대로" 제공되며, 투자 손실에 대한 어떠한 보증도 하지 않습니다. 실제 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다. 