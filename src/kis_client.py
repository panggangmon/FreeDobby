import json
import logging
import time
import requests
from datetime import datetime, timedelta

class KISClient:
    def __init__(self, env="paper", app_key="", app_secret="", cano="", acnt_prdt_cd="", logger=None):
        if env not in ["paper", "real"]:
            raise ValueError("env must be 'paper' or 'real'")

        self.env = env
        self.app_key = app_key
        self.app_secret = app_secret
        self.cano = cano
        self.acnt_prdt_cd = acnt_prdt_cd
        self.base_url = self._get_base_url()
        self.access_token = None
        self.request_timeout = 10
        self.max_retries = 3
        self.backoff_seconds = 1.0
        self.logger = logger or logging.getLogger(__name__)

    def _request_with_retry(self, method, url, *, headers=None, params=None, data=None):
        retry_statuses = {429, 500, 502, 503, 504}
        attempt = 0
        last_exc = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    data=data,
                    timeout=self.request_timeout,
                )
                if resp.status_code in retry_statuses and attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed without a response")

    def _get_base_url(self):
        if self.env == "paper":
            return "https://openapivts.koreainvestment.com:29443"
        else:
            return "https://openapi.koreainvestment.com:9443"

    def get_access_token(self) -> str:
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        self.logger.info("[1] Requesting access token...")
        resp = self._request_with_retry("POST", url, headers=headers, data=json.dumps(body))

        if resp.status_code != 200:
            raise requests.HTTPError(f"Token issuance HTTP error: {resp.status_code} - {resp.text}")

        data = resp.json()
        access_token = data.get("access_token")

        if not access_token:
            err_desc = json.dumps(data, ensure_ascii=False, indent=2)
            raise ValueError(f"access_token not in response. Check app_key/secret. Response: {err_desc}")

        self.logger.info("[1] Access token issued successfully.")
        self.access_token = f"Bearer {access_token}"
        return self.access_token

    def get_current_price(self, stock_code: str) -> dict:
        if not self.access_token:
            self.get_access_token()

        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{self.base_url}{path}"

        headers = {
            "Content-Type": "application/json",
            "authorization": self.access_token,
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": "FHKST01010100",
            "custtype": "P",
        }

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }

        self.logger.info(f"[2] Requesting current price... (ticker: {stock_code})")
        resp = self._request_with_retry("GET", url, headers=headers, params=params)

        if resp.status_code != 200:
            raise requests.HTTPError(f"Current price HTTP error: {resp.status_code} - {resp.text}")

        data = resp.json()
        if data.get("rt_cd") != "0":
            err_desc = json.dumps(data, ensure_ascii=False, indent=2)
            raise ValueError(f"Current price API error(rt_cd != '0'). Response: {err_desc}")

        return data

    def get_stock_balance(self) -> dict:
        if not self.access_token:
            self.get_access_token()

        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{self.base_url}{path}"
        
        balance_tr_id = "VTTC8434R" if self.env == "paper" else "TTTC8434R"

        headers = {
            "Content-Type": "application/json",
            "authorization": self.access_token,
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": balance_tr_id,
            "custtype": "P",
        }

        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        self.logger.info("[3] Requesting account balance...")
        resp = self._request_with_retry("GET", url, headers=headers, params=params)

        if resp.status_code != 200:
            raise requests.HTTPError(f"Balance check HTTP error: {resp.status_code} - {resp.text}")

        data = resp.json()
        if data.get("rt_cd") != "0":
            err_desc = json.dumps(data, ensure_ascii=False, indent=2)
            raise ValueError(f"Balance check API error(rt_cd != '0'). Response: {err_desc}")

        return data

    def get_historical_daily_price(self, stock_code: str, start_date: str, end_date: str) -> list:
        if not self.access_token:
            self.get_access_token()

        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{self.base_url}{path}"
        
        headers = {
            "Content-Type": "application/json",
            "authorization": self.access_token,
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": "FHKST03010100",
            "custtype": "P",
        }

        all_prices = []
        # KIS API limitation: max 100 days per request. Loop to get all data.
        current_date = datetime.strptime(end_date, "%Y%m%d")
        start_datetime = datetime.strptime(start_date, "%Y%m%d")

        while current_date >= start_datetime:
            loop_start_date = max(current_date - timedelta(days=99), start_datetime)
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_DATE_1": loop_start_date.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": current_date.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",
            }
            
            self.logger.info(f"[4] Requesting historical prices... ({loop_start_date.strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')})")
            resp = self._request_with_retry("GET", url, headers=headers, params=params)

            if resp.status_code != 200:
                raise requests.HTTPError(f"Hist. price HTTP error: {resp.status_code} - {resp.text}")

            data = resp.json()
            if data.get("rt_cd") != "0":
                # For this specific API, an rt_cd of '1' can mean no data in the range, which is not a fatal error.
                if data.get("msg1") and "조회 결과가 없습니다" in data.get("msg1"):
                    price_list = []
                else:
                    err_desc = json.dumps(data, ensure_ascii=False, indent=2)
                    raise ValueError(f"Hist. price API error(rt_cd != '0'). Response: {err_desc}")
            else:
                price_list = data.get("output2", [])
            if price_list:
                all_prices.extend(price_list)
            
            # Move to the next chunk
            current_date = loop_start_date - timedelta(days=1)
            time.sleep(0.2)

        # Sort by date ascending
        all_prices.sort(key=lambda x: x['stck_bsop_date'])
        return all_prices

    def _send_order(self, stock_code: str, quantity: int, price: int, order_type: str) -> dict:
        """Helper function to send buy or sell orders."""
        if not self.access_token:
            self.get_access_token()

        path = "/uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.base_url}{path}"

        if self.env == 'paper':
            tr_id = "VTTC0802U" if order_type == "buy" else "VTTC0801U" # 모의투자 매수/매도
        else:
            tr_id = "TTTC0802U" if order_type == "buy" else "TTTC0801U" # 실전 매수/매도

        headers = {
            "Content-Type": "application/json",
            "authorization": self.access_token,
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": stock_code,
            "ORD_DVSN": "01",  # 01: 지정가, 02: 시장가
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        }

        action_name = "매수" if order_type == "buy" else "매도"
        self.logger.info(f"[5] Requesting {order_type} order... ({stock_code}, {quantity} @ {price})")
        resp = self._request_with_retry("POST", url, headers=headers, data=json.dumps(body))

        if resp.status_code != 200:
            self.logger.error(f"Order HTTP error: {resp.status_code} - {resp.text}")
            return {"success": False, "message": f"HTTP 오류: {resp.status_code}", "response": resp.text}

        data = resp.json()
        if data.get("rt_cd") != "0":
            self.logger.warning(f"Order API error: {data.get('msg1')}")
            return {"success": False, "message": data.get('msg1'), "response": data}
        else:
            self.logger.info(f"Order successful: {action_name} {stock_code}")
            return {"success": True, "message": "주문 성공", "response": data}


    def buy(self, stock_code: str, quantity: int, price: int) -> dict:
        """주식을 지정가에 매수하는 주문을 전송합니다."""
        return self._send_order(stock_code, quantity, price, "buy")

    def sell(self, stock_code: str, quantity: int, price: int) -> dict:
        """주식을 지정가에 매도하는 주문을 전송합니다."""
        return self._send_order(stock_code, quantity, price, "sell")
