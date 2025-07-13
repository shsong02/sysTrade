#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 구조 마이그레이션 스크립트
기존 data/ 폴더의 파일들을 새로운 프로세스 기반 구조로 이동

Author: ST Development Team
Created: 2024-12-19
"""

import os
import shutil
import glob
import yaml
from datetime import datetime
from pathlib import Path

def load_config():
    """설정 파일 로드"""
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_old_structure():
    """기존 데이터 구조 분석"""
    old_data_path = "./data/"
    if not os.path.exists(old_data_path):
        print("기존 data/ 폴더가 존재하지 않습니다.")
        return {}
    
    analysis = {}
    for root, dirs, files in os.walk(old_data_path):
        if files:
            rel_path = os.path.relpath(root, old_data_path)
            analysis[rel_path] = {
                'files': files,
                'count': len(files),
                'total_size': sum(os.path.getsize(os.path.join(root, f)) for f in files)
            }
    
    return analysis

def create_migration_mapping(config):
    """마이그레이션 매핑 정의 (base_path 기준)"""
    base_path = config.get('data_management', {}).get('base_path', './data_new/')
    
    # 상대 경로 매핑
    relative_mapping = {
        # 기존 경로 -> 새로운 상대 경로 매핑
        'search_stocks': '1_discovery/stock_screening',
        'monitor_stocks': '1_discovery/stock_screening', 
        'finance_score': '1_discovery/finance_scores',
        'system_trade': '3_trading/logs',
        'news': '4_shared/market_data',
        'models': '2_backtest/strategies',
        'model_results': '2_backtest/results',
        'reference': '4_shared/reference',
        'analytics': '1_discovery/reports',
        'processed': '4_shared/market_data',
        'metadata': '4_shared/reference',
        'cache': '4_shared/cache',
        'temp': '4_shared/temp',
        'backup': '5_backup',
        'trading': '3_trading'
    }
    
    # base_path와 결합하여 절대 경로 생성
    absolute_mapping = {}
    for old_key, relative_path in relative_mapping.items():
        absolute_mapping[old_key] = os.path.join(base_path, relative_path)
    
    return absolute_mapping

def migrate_files(dry_run=True):
    """파일 마이그레이션 실행"""
    config = load_config()
    old_analysis = analyze_old_structure()
    mapping = create_migration_mapping(config)
    
    print("=" * 60)
    print("데이터 구조 마이그레이션 시작")
    print("=" * 60)
    
    if dry_run:
        print("🔍 DRY RUN 모드 - 실제 이동은 하지 않고 계획만 출력합니다")
        print()
    
    migration_plan = []
    total_files = 0
    total_size = 0
    
    for old_path, info in old_analysis.items():
        # 매핑 찾기
        new_path = None
        for old_key, new_key in mapping.items():
            if old_key in old_path or old_path.startswith(old_key):
                new_path = f"./data_new/{new_key}/"
                break
        
        if not new_path:
            # 매핑되지 않은 경우 shared/temp로 이동 (base_path 기준)
            base_path = config.get('data_management', {}).get('base_path', './data_new/')
            new_path = os.path.join(base_path, "4_shared/temp/")
            print(f"⚠️  매핑되지 않은 경로: {old_path} -> {new_path}")
        
        old_full_path = f"./data/{old_path}"
        migration_plan.append({
            'old_path': old_full_path,
            'new_path': new_path,
            'files': info['files'],
            'count': info['count'],
            'size': info['total_size']
        })
        
        total_files += info['count']
        total_size += info['total_size']
        
        print(f"📁 {old_full_path}")
        print(f"   → {new_path}")
        print(f"   📄 파일 수: {info['count']}, 크기: {info['total_size']:,} bytes")
        print()
    
    print("=" * 60)
    print(f"마이그레이션 요약:")
    print(f"총 파일 수: {total_files:,}")
    print(f"총 크기: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print("=" * 60)
    
    if not dry_run:
        print("🚀 실제 마이그레이션을 시작합니다...")
        
        success_count = 0
        error_count = 0
        
        for plan in migration_plan:
            try:
                # 대상 디렉토리 생성
                os.makedirs(plan['new_path'], exist_ok=True)
                
                # 파일들 이동
                for file in plan['files']:
                    old_file = os.path.join(plan['old_path'], file)
                    new_file = os.path.join(plan['new_path'], file)
                    
                    # 파일명 중복 처리
                    if os.path.exists(new_file):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        name, ext = os.path.splitext(file)
                        new_file = os.path.join(plan['new_path'], f"{name}_{timestamp}{ext}")
                    
                    shutil.move(old_file, new_file)
                    success_count += 1
                    print(f"✅ 이동 완료: {file}")
                    
            except Exception as e:
                print(f"❌ 이동 실패: {plan['old_path']} - {str(e)}")
                error_count += 1
        
        print("=" * 60)
        print(f"마이그레이션 완료!")
        print(f"성공: {success_count} 파일")
        print(f"실패: {error_count} 파일")
        print("=" * 60)
        
        # 빈 디렉토리 정리
        cleanup_empty_directories("./data/")
    
    return migration_plan

def cleanup_empty_directories(path):
    """빈 디렉토리 정리"""
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # 빈 디렉토리인 경우
                        os.rmdir(dir_path)
                        print(f"🗑️  빈 디렉토리 삭제: {dir_path}")
                except OSError:
                    pass  # 삭제할 수 없는 경우 무시
    except Exception as e:
        print(f"디렉토리 정리 중 오류: {str(e)}")

def create_backup():
    """기존 데이터 백업"""
    if not os.path.exists("./data/"):
        print("백업할 데이터가 없습니다.")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"data_backup_{timestamp}"
    
    try:
        shutil.copytree("./data/", f"./backup/{backup_name}")
        print(f"✅ 백업 완료: ./backup/{backup_name}")
        return backup_name
    except Exception as e:
        print(f"❌ 백업 실패: {str(e)}")
        return None

def main():
    """메인 실행 함수"""
    print("🔄 ST 시스템 데이터 구조 마이그레이션 도구")
    print()
    
    # 1. 기존 구조 분석
    print("1️⃣ 기존 데이터 구조 분석 중...")
    old_analysis = analyze_old_structure()
    
    if not old_analysis:
        print("마이그레이션할 데이터가 없습니다.")
        return
    
    print(f"분석 완료: {len(old_analysis)}개 디렉토리 발견")
    print()
    
    # 2. 마이그레이션 계획 확인 (Dry Run)
    print("2️⃣ 마이그레이션 계획 생성 중...")
    migration_plan = migrate_files(dry_run=True)
    print()
    
    # 3. 사용자 확인
    while True:
        choice = input("마이그레이션을 진행하시겠습니까? (y/n/b): ").lower().strip()
        
        if choice == 'y':
            # 백업 생성 여부 확인
            backup_choice = input("기존 데이터를 백업하시겠습니까? (y/n): ").lower().strip()
            
            if backup_choice == 'y':
                print("3️⃣ 기존 데이터 백업 중...")
                backup_name = create_backup()
                if not backup_name:
                    print("백업 실패. 마이그레이션을 중단합니다.")
                    return
            
            # 실제 마이그레이션 실행
            print("4️⃣ 마이그레이션 실행 중...")
            migrate_files(dry_run=False)
            break
            
        elif choice == 'n':
            print("마이그레이션을 취소했습니다.")
            break
            
        elif choice == 'b':
            print("3️⃣ 백업만 생성 중...")
            create_backup()
            break
            
        else:
            print("y, n, 또는 b를 입력해주세요.")

if __name__ == "__main__":
    main()