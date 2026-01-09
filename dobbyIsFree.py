import json
import requests

ENV = "paper"  # "paper" (모의), "real" (실전)

APP_KEY = "PSCAO26koElyDyx1QvBkDJZjkYLG1RsqR3O3"
APP_SECRET = "JUOgQSNziZ69eYT12qGzxR81+mlzfU+6qTm+XK/Mqx4UhBw2+empdYAjH/OAGZrlC3v3zlC9HpuCZyAvCCSYQrs55dg/L8E+XZ/Dz2HdnLs2GDWS2t5AVgomVeBEnT1EBy0RcIGcOVqIRtJgtVIdgMJ24BFaWIobNh4M+5EAmqTqQXOPJqs="

if ENV == "paper":
    BASE_URL = "https://openapivts.koreainvestment.com:29443"
    BALANCE_TR_ID = "VTTC8434R"
else:
    BASE_URL = "https://openapi.koreainvestment.com:9443"
    BALANCE_TR_ID = "TTTC8434R"

CANO = "50154249"       # 계좌 앞 8자리 (본인 걸로 교체)
ACNT_PRDT_CD = "01"     # 보통 01

def get_access_token() -> str:

    url = f"{BASE_URL}/oauth2/tokenP"

    headers = {
        "content-type" : "application/json"
    }

    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
    }

    print("[1] 토큰 발급 요청 중...")
    resp = requests.post(url, headers=headers, data=json.dumps(body))

    if resp.status_code != 200:
        print(f"HTTP 오류: {resp.status_code}")
        print(resp.text)
        raise SystemExit("토큰 발급 HTTP 오류로 종료")
    
    data = resp.json()
    # 보통 access_token 키에 토큰이 들어있음
    access_token = data.get("access_token")

    if not access_token:
        print("응답 JSON:", json.dumps(data, ensure_ascii=False, indent=2))
        raise SystemExit("access_token이 응답에 없습니다. (앱키/시스템 설정 확인 필요)")

    print("[1] 토큰 발급 성공")
    return access_token

#========================
#  현재가 가져오기
#========================
def get_current_price(stock_code: str, access_token: str) -> dict:

    path = "/uapi/domestic-stock/v1/quotations/inquire-price"
    url = f"{BASE_URL}{path}"

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010100",  # [국내주식] 주식현재가 시세 (실전 기준)
        "custtype": "P",           # 개인(P), 법인(B)
    }

    params = {
        # J : 주식, ETF 등 (문서는 'FID_COND_MRKT_DIV_CODE')
        "FID_COND_MRKT_DIV_CODE": "J",
        # 종목 코드 (삼성전자: 005930, 현대차: 005380 등)
        "FID_INPUT_ISCD": stock_code,
    }

    print(f"[2] 현재가 조회 요청 중... (종목코드: {stock_code})")
    resp = requests.get(url, headers=headers, params=params)

    if resp.status_code != 200:
        print(f"HTTP 오류: {resp.status_code}")
        print(resp.text)
        raise SystemExit("현재가 조회 HTTP 오류로 종료")

    data = resp.json()

    # KIS 공통 응답 형식: rt_cd == "0" 이면 성공
    rt_cd = data.get("rt_cd")
    if rt_cd != "0":
        print("API 에러 응답:", json.dumps(data, ensure_ascii=False, indent=2))
        raise SystemExit("현재가 조회 API 에러(rt_cd != '0')")

    # 정상 처리
    return data


def get_stock_balance(access_token: str) -> dict:

    path = "/uapi/domestic-stock/v1/trading/inquire-balance"
    url = f"{BASE_URL}{path}"

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": BALANCE_TR_ID,  # 모의/실전 분기
        "custtype": "P",         # 개인
    }

    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "AFHR_FLPR_YN": "N",        # 시간외 단일가 여부 (일반: N)
        "OFL_YN": "",               # 오프라인 여부
        "INQR_DVSN": "02",          # 조회구분 (02: 종목별 합산)
        "UNPR_DVSN": "01",          # 단가구분 (01: 기본)
        "FUND_STTL_ICLD_YN": "N",   # 펀드결제분 포함 여부
        "FNCG_AMT_AUTO_RDPT_YN": "N",  # 융자금액자동상환여부
        "PRCS_DVSN": "00",          # 처리구분
        "CTX_AREA_FK100": "",       # 연속조회검색조건
        "CTX_AREA_NK100": "",       # 연속조회키
    }

    print("[3] 잔고 조회 요청 중...")
    resp = requests.get(url, headers=headers, params=params)

    if resp.status_code != 200:
        print(f"HTTP 오류: {resp.status_code}")
        print(resp.text)
        raise SystemExit("잔고 조회 HTTP 오류로 종료")

    data = resp.json()
    if data.get("rt_cd") != "0":
        print("API 에러 응답:", json.dumps(data, ensure_ascii=False, indent=2))
        raise SystemExit("잔고 조회 API 에러(rt_cd != '0')")

    return data

def main():
    # 1) 토큰 발급
    token = get_access_token()
    print(f"발급받은 access_token (일부만 표시): {token[:20]}...")

    # 2) 삼성전자 현재가 조회
    response = get_current_price("005930", token)

    #print("\n[응답 전체 JSON]")
    #print(json.dumps(response, ensure_ascii=False, indent=2))

    # 3) 현재가만 따로 뽑아서 보기
    output = response.get("output", {})

    print("\n[디버깅] output 전체")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("[디버깅] 키 목록:", list(output.keys()))


    current_price = output.get("stck_prpr")  # 주식 현재가 (문자열)
    print("\n[정리된 정보]")
    print(f"종목명      : {output.get('hts_kor_isnm')}")
    print(f"단축코드    : {output.get('stck_shrn_iscd')}")
    print(f"현재가      : {current_price}")
    print(f"전일대비    : {output.get('prdy_vrss')} ({output.get('prdy_ctrt')}%)")

    # 3) 잔고 조회
    balance = get_stock_balance(token)

    print("\n[잔고 조회 응답 전체 JSON]")
    print(json.dumps(balance, ensure_ascii=False, indent=2))

    # 4) 잔고 요약 출력 (output1 리스트 기준)
    holdings = balance.get("output1", [])
    print("\n[보유 종목 요약]")
    if not holdings:
        print("보유 중인 주식이 없습니다.")
    else:
        for item in holdings:
            pdno = item.get("pdno")           # 종목코드
            name = item.get("prdt_name")      # 종목명
            qty = item.get("hldg_qty")        # 보유수량
            buy_amt = item.get("pchs_amt")    # 매수금액
            eval_amt = item.get("evlu_amt")   # 평가금액
            pl_amt = item.get("evlu_pfls_amt")  # 평가손익
            pl_rt = item.get("evlu_pfls_rt")    # 평가손익률

            print(f"- {name}({pdno}): 수량 {qty}, 평가손익 {pl_amt} ({pl_rt}%), 평가금액 {eval_amt}")

if __name__ == "__main__":
    main()

