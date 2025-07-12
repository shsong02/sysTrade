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
```bash
pip install pandas numpy
pip install pykrx FinanceDataReader
pip install selenium beautifulsoup4 requests
pip install matplotlib seaborn mplfinance
pip install scikit-learn torch
pip install apscheduler
pip install python-telegram-bot
pip install backtesting
pip install bertopic sentence-transformers
pip install konlpy
pip install PyYAML
```

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

### 1. 종목 검색 및 분석
```bash
# 재무제표 기반 종목 스크리닝
python search_stocks.py

# 거시경제 분석
python search_macro.py

# 종목 재무 점수 계산
python finance_score.py
```

### 2. 자동 거래 시스템 실행
```bash
# 메인 거래 시스템 실행
python system_trade.py
```

### 3. 백테스팅 실행
```bash
# 전략 백테스팅
python back_test.py
```

### 4. 뉴스 분석
```bash
# 뉴스 크롤링 및 분석
python -c "
from tools.news_crawler import newsCrawler
nc = newsCrawler()
nc.search_interval(['20240101', '20240131'])
"
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

### v0.1 (현재)
- 기본 자동 거래 시스템 구축
- 한국투자증권 API 연동
- 재무 분석 및 기술적 분석 구현
- 뉴스 크롤링 및 분석 기능
- 백테스팅 시스템 구축
- 텔레그램 알림 기능

---

**⚠️ 면책조항**: 이 소프트웨어는 "있는 그대로" 제공되며, 투자 손실에 대한 어떠한 보증도 하지 않습니다. 실제 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다. 