from finance_score import financeScore
import pandas as pd

def test_all_stocks():
    # financeScore 인스턴스 생성
    fs = financeScore('./config/config.yaml')
    
    # 전체 종목 분석 실행
    result_df = fs.run()
    
    if result_df is not None:
        print("\n=== 재무 데이터 분석 결과 ===")
        print(f"총 분석 종목 수: {len(result_df)}")
        
        # 점수 분포 확인
        print("\n점수 분포:")
        score_ranges = [(80, float('inf')), (60, 80), (40, 60), (20, 40), (0, 20), (float('-inf'), 0)]
        for start, end in score_ranges:
            count = len(result_df[result_df['total_score'].between(start, end)])
            print(f"{start}~{end if end != float('inf') else '∞'}: {count}개 종목")
        
        # 상위 10개 종목 출력
        print("\n상위 10개 종목:")
        top_10 = result_df.head(10)
        for idx, row in top_10.iterrows():
            print(f"\n{idx} (총점: {row['total_score']:.1f})")
            for col in row.index:
                if col.endswith('_list'):
                    print(f"{col}: {row[col]}")
                elif col.endswith('_score'):
                    print(f"{col}: {row[col]:.1f}")
        
        # 결과 저장
        print("\n저장된 파일 경로:")
        print(result_df.index.name)
    else:
        print("분석 실패")

if __name__ == "__main__":
    test_all_stocks() 