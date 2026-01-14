import os
import csv
from datetime import datetime, timedelta
import config
from kis_client import KISClient
import time

#data 디렉토리 있는지 확인
def create_data_dir_if_not_exists():
    
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory.")

#data row 행 출력, 헤더 제외
def get_csv_row_count(file_path: str) -> int:
    
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # header 1줄 제외
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0

def collect_and_save_daily_history(client: KISClient, stock_code: str, years: int = 2):
    """
    Collects N years of daily historical data for a stock and saves it to a CSV file.
    
    Args:
        client: An initialized KISClient instance.
        stock_code: The stock code to fetch data for.
        years: The number of years of historical data to fetch.
    """
    create_data_dir_if_not_exists()

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')

    print(f"Fetching daily history for {stock_code} from {start_date_str} to {end_date_str}...")
    
    # Fetch data using the client
    daily_prices = client.get_historical_daily_price(
        stock_code=stock_code,
        start_date=start_date_str,
        end_date=end_date_str
    )

    #날짜기준 정렬
    daily_prices = sorted(
        daily_prices,
        key=lambda x: x.get("stck_bsop_date", "")
    )
    #중복 날짜 제거
    dedup = {}
    for p in daily_prices:
        d = p.get("stck_bsop_date")
        if d:
            dedup[d] = p
            
    daily_prices = [dedup[k] for k in sorted(dedup.keys())]

    if not daily_prices:
        print(f"No data fetched for {stock_code}. Exiting.")
        return

    # Define CSV file path and header
    file_path = f"data/{stock_code}_daily.csv"
    header = [
        'date', 'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Write to CSV
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for price_data in daily_prices:
                row = [
                    price_data.get('stck_bsop_date'), # 날짜
                    price_data.get('stck_oprc'),      # 시가
                    price_data.get('stck_hgpr'),      # 고가
                    price_data.get('stck_lwpr'),      # 저가
                    price_data.get('stck_clpr'),      # 종가
                    price_data.get('acml_vol'),       # 거래량
                ]
                writer.writerow(row)
        
        print(f"Successfully saved data to {file_path}")

    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

if __name__ == '__main__':
    create_data_dir_if_not_exists()

    kis_client = KISClient(
        env=config.ENV,
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        cano=config.CANO,
        acnt_prdt_cd=config.ACNT_PRDT_CD
    )

    years = getattr(config, "HISTORY_YEARS", 5)

    for stock_code in config.UNIVERSE:
        file_path = f"data/{stock_code}_daily.csv"
        existing_rows = get_csv_row_count(file_path)

        # (권장) 대충 5년이면 원천 1200행 전후.
        # 너무 빡빡하게 잡지 말고, 600행 이상이면 우선 스킵(재실행/resume 목적).
        if existing_rows >= 600:
            print(f"[SKIP] {stock_code}: already exists ({existing_rows} rows) -> {file_path}")
            continue

        try:
            print(f"[FETCH] {stock_code} (years={years})")
            collect_and_save_daily_history(kis_client, stock_code, years=years)
        except Exception as e:
            print(f"[ERROR] {stock_code}: {e}")
        finally:
            time.sleep(getattr(config, "REQUEST_SLEEP_SEC", 0.35))