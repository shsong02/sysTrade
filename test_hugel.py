from finance_score import financeScore

def test_hugel():
    # financeScore 객체 생성
    fs = financeScore('config/config.yaml')
    
    # 휴젤 테스트 (코드: 145020)
    result = fs.finance_state(['145020', '휴젤'], mode='quarter')
    
    # 결과 출력
    print("\n=== 휴젤 재무 데이터 ===")
    print(f"매출액증가율: {result.loc['휴젤', '매출액증가율_list']}")
    print(f"영업이익증가율: {result.loc['휴젤', '영업이익증가율_list']}")
    print(f"영업이익률: {result.loc['휴젤', '영업이익률_list']}")
    print(f"ROE: {result.loc['휴젤', 'ROE_list']}")
    print(f"부채비율: {result.loc['휴젤', '부채비율_list']}")
    print(f"총점: {result.loc['휴젤', 'total_score']}")

if __name__ == '__main__':
    test_hugel() 