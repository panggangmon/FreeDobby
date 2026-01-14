# KIS API config
# For security, it's better to load secrets from environment variables rather than hardcoding them in the file.
# However, for now, we'll stick to the current method.

# 트레이딩 환경: "paper" (모의), "real" (실전)
ENV = "paper"

# KIS API 키 (본인 것으로 교체)
APP_KEY = "PSCAO26koElyDyx1QvBkDJZjkYLG1RsqR3O3"
APP_SECRET = "JUOgQSNziZ69eYT12qGzxR81+mlzfU+6qTm+XK/Mqx4UhBw2+empdYAjH/OAGZrlC3v3zlC9HpuCZyAvCCSYQrs55dg/L8E+XZ/Dz2HdnLs2GDWS2t5AVgomVeBEnT1EBy0RcIGcOVqIRtJgtVIdgMJ24BFaWIobNh4M+5EAmqTqQXOPJqs="

# 계좌번호 (본인 것으로 교체)
CANO = "50154249"
ACNT_PRDT_CD = "01"

# 로그 파일이 저장될 디렉토리
LOG_DIR = "logs"

# 멀티 종목 
UNIVERSE = [
    "005930", # 삼성전자
    "066570", # LG전자
    "000660", # SK하이닉스
    "000810", # 삼성화재
    "005380", # 현대차
    "138040", # 메리츠금유지주
    "207940", # 삼성바이오로직스
    "086790", # 하나금융지주
    "373220", # LG에너지솔루션
    "010620", # 현대미포
    "329180", # HD현대중공업
    "035720", # 카카오
    "012450", # 한화에어로스페이스
    "011200", # HMM
    "402340", # sk스퀘어
    "018260", # 삼성에스디에스
    "034020", # 두산에너빌리티
    "030200", # KT
    "000270", # 기아
    "010130", # 고려아연
    "068270", # 셀트리온
    "034730", # SK
    "105560", # KB금융
    "051910", # LG 화학
    "028260", # 삼성물산
    "015760", # 한국전력
    "042660", # 한화오션
    "259960", # 크래프톤
    "035420", # 네이버
    "003490", # 대한항공
    "055550", # 신한지주
    "267250", # HD현대
    "012330", # 현대모비스
    "096770", # SK이노베이션
    "009540", # 한국조선해양
    "316140", # 우리금융지주
    "032830", # 삼성생명
    "010950", # S-OIL
    "005490", # posco 홀딩스
    "009150", # 삼성전기
    "006400", # 삼성 sdi
    "042700", # 한미반도체
    "017670", # sk 텔레콤
    "241560", # 두산밥켓
    "003550", # LG
    "051900", # LG생활건강
    "003670", # 포스코퓨처엠
    "024110", # 기업은행
    "033780", # KT&G
    "086280", #현대글로비스

]

#데이터 수집 기간
HISTORY_YEARS = 5

#라벨 : n일 뒤 수익 예측
HORIZON = 5

# 라벨 임계치(거래비용+슬리피지 버퍼) 0.2% 부터 시작 권장
FEE_BUFFER = 0.002

# 종목별 DATASET 이 너무 짧으면 제외
MIN_DATASET_ROWS = 600

# KIS API 호출 간격(레이트리밋 완화)
REQUEST_SLEEP_SEC = 0.35

TOP_K = 3


# rebalance cadence
REBALANCE_EVERY_N_DAYS = 1
REBALANCE_BAND = 0.10
MIN_ORDER_VALUE = 10000

# regime filters (초기엔 보수적으로)
MIN_PRED_RET = 0.002
MAX_ATR_PCT = 0.05
MAX_RVOL_20 = 0.06
MIN_DOLLAR_VOL = 3e9

# execution switches (중요)
EXECUTE_PAPER_ORDERS = False  # 4주 동안은 False 권장 (로그로만 검증)
EXECUTE_REAL_ORDERS = False