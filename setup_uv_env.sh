#!/bin/bash

# ST_ver0.1 프로젝트 uv 기반 가상환경 설정 스크립트

echo "🚀 ST_ver0.1 프로젝트 uv 기반 가상환경 설정을 시작합니다..."

# 1. uv 설치 확인 및 설치
echo "📦 uv 설치 확인 중..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv가 설치되지 않았습니다. 설치를 진행합니다..."
    
    # macOS의 경우
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "🍺 Homebrew를 사용하여 uv 설치 중..."
            brew install uv
        else
            echo "📥 curl을 사용하여 uv 설치 중..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
    # Linux의 경우
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "📥 curl을 사용하여 uv 설치 중..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        echo "❌ 지원되지 않는 운영체제입니다. 수동으로 uv를 설치해주세요."
        echo "   https://github.com/astral-sh/uv#installation"
        exit 1
    fi
else
    echo "✅ uv가 이미 설치되어 있습니다."
fi

# 2. Python 버전 확인
echo "🐍 Python 버전 확인 중..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "현재 Python 버전: $PYTHON_VERSION"

# 3. 가상환경 생성
echo "🔧 가상환경 생성 중..."
uv venv venv --python 3.9

# 4. 가상환경 활성화
echo "🔌 가상환경 활성화 중..."
source venv/bin/activate

# 5. 패키지 설치
echo "📦 패키지 설치 중... (uv는 매우 빠릅니다!)"
uv pip install -r requirements.txt

# 6. 추가 설정
echo "⚙️  추가 설정 중..."

# mecab-ko-dic 설치 (한국어 형태소 분석기)
echo "🔤 한국어 형태소 분석기 설치 중..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        brew install mecab mecab-ko mecab-ko-dic
    else
        echo "⚠️  Homebrew가 없습니다. mecab 수동 설치가 필요합니다."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux에서는 mecab을 수동으로 설치해야 합니다."
    echo "Ubuntu/Debian: sudo apt-get install mecab mecab-ko-dic"
    echo "CentOS/RHEL: sudo yum install mecab mecab-ko-dic"
fi

# 7. 디렉토리 생성
echo "📁 필요한 디렉토리 생성 중..."
mkdir -p data/{search_stocks,monitor_stocks,finance_score,system_trade,news,models,reference,model_results/keywords}
mkdir -p log
mkdir -p models/{nlp,kpfbert}

# 8. 권한 설정
echo "🔐 실행 권한 설정 중..."
chmod +x *.py

echo ""
echo "🎉 설정이 완료되었습니다!"
echo ""
echo "📋 다음 단계:"
echo "1. 가상환경 활성화: source venv/bin/activate"
echo "2. config/kisdev_vi.yaml 파일에 API 키 설정"
echo "3. config/config.yaml 파일에 텔레그램 봇 설정"
echo "4. 프로그램 실행: python system_trade.py"
echo ""
echo "💡 도움말:"
echo "- uv 명령어: uv --help"
echo "- 패키지 추가 설치: uv pip install <package_name>"
echo "- 패키지 목록 확인: uv pip list"
echo "- 가상환경 비활성화: deactivate"
echo ""
echo "⚠️  주의사항:"
echo "- 실거래 전에 반드시 모의거래로 테스트하세요"
echo "- API 키는 절대 공개하지 마세요"
echo "- 충분한 백테스팅 후 실거래에 적용하세요" 