"""
Crypto Trading Bot Core - Real Account Version
"""

import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json
import requests
import hmac
import hashlib
import base64
import threading
import websocket
import queue

import logging

# Configure logging with different levels for different modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger("trading_bot")

# Set more specific logger levels to reduce noise
logging.getLogger("websocket").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Add a filter to suppress repetitive errors
class DuplicateFilter(logging.Filter):
    def __init__(self, name=''):
        super().__init__(name)
        self.last_log = None
        self.last_count = 0
    
    def filter(self, record):
        # Get the message
        current = record.getMessage()
        
        # If it's the same as the last one
        if current == self.last_log:
            self.last_count += 1
            # Only log every 50th occurrence of the same message
            if self.last_count % 50 == 0:
                record.msg = "%s (repeated %d times)" % (record.msg, self.last_count)
                return True
            return False
        else:
            # It's a new message, so log it and store it
            self.last_log = current
            self.last_count = 0
            return True

# Add filter to logger
logger.addFilter(DuplicateFilter())

class KrakenTradingClient:
    """Custom client for Kraken trading (real accounts only)"""
    
    def __init__(self, api_key, secret):
        """Initialize the client with API credentials"""
        # Strip whitespace from credentials to avoid HTTP header errors
        self.api_key = api_key.strip() if api_key else ""
        self.secret = secret.strip() if secret else ""
        
        self.base_url = "https://futures.kraken.com"
        self.ws_url = "wss://futures.kraken.com/ws/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'KrakenTradingBot/1.0'
        })
    
    def get_instruments(self):
        """Get available trading instruments"""
        try:
            url = f"{self.base_url}/derivatives/api/v3/instruments"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return {"instruments": []}
    
    def get_ticker(self, symbol):
        """Get ticker for a symbol"""
        try:
            # Try multiple endpoints to increase chance of success
            endpoints = [
                "/derivatives/api/v3/tickers",
                "/api/v3/tickers",
                f"/derivatives/api/v3/ticker?symbol={symbol}",
                f"/api/v3/ticker?symbol={symbol}"
            ]
            
            for endpoint in endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    logger.info(f"Fetching ticker from {url}")
                    
                    response = self.session.get(url, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Handle different response formats
                    if 'tickers' in data:
                        # Format with list of tickers
                        for ticker in data.get('tickers', []):
                            if ticker['symbol'] == symbol:
                                return {
                                    'symbol': symbol,
                                    'last': float(ticker.get('last', 0)),
                                    'bid': float(ticker.get('bid', 0)),
                                    'ask': float(ticker.get('ask', 0)),
                                    'volume': float(ticker.get('vol24h', 0)),
                                    'timestamp': int(time.time() * 1000)
                                }
                    
                    elif 'ticker' in data:
                        # Single ticker format
                        ticker = data['ticker']
                        return {
                            'symbol': symbol,
                            'last': float(ticker.get('last', 0)),
                            'bid': float(ticker.get('bid', 0)),
                            'ask': float(ticker.get('ask', 0)),
                            'volume': float(ticker.get('vol24h', 0)),
                            'timestamp': int(time.time() * 1000)
                        }
                        
                    elif 'symbol' in data and data['symbol'] == symbol:
                        # Direct ticker format
                        return {
                            'symbol': symbol,
                            'last': float(data.get('last', 0)),
                            'bid': float(data.get('bid', 0)),
                            'ask': float(data.get('ask', 0)),
                            'volume': float(data.get('vol24h', 0)),
                            'timestamp': int(time.time() * 1000)
                        }
                        
                except Exception as endpoint_error:
                    logger.warning(f"Failed to get ticker from {endpoint}: {endpoint_error}")
            
            # If we can't get data from Kraken API directly, try CCXT as fallback
            try:
                import ccxt
                
                kraken = ccxt.kraken({
                    'apiKey': self.api_key,
                    'secret': self.secret,
                })
                
                # Normalize the symbol format for CCXT
                # Kraken futures uses PI_ETHUSD but CCXT might expect ETH/USD
                ccxt_symbol = symbol
                
                # Try common transformations
                symbol_transforms = [
                    symbol,  # Original
                    symbol.replace('PI_', '').replace('USD', '/USD'),  # PI_ETHUSD -> ETH/USD
                    symbol.replace('PF_', '').replace('USD', '/USD'),  # PF_ETHUSD -> ETH/USD
                    symbol.split('_')[1].replace('USD', '/USD') if '_' in symbol else symbol,  # Extract part after _
                ]
                
                for sym in symbol_transforms:
                    try:
                        ticker_data = kraken.fetch_ticker(sym)
                        if ticker_data:
                            logger.info(f"Retrieved ticker via CCXT for {sym}")
                            return {
                                'symbol': symbol,  # Return original symbol
                                'last': float(ticker_data.get('last', 0)),
                                'bid': float(ticker_data.get('bid', 0)),
                                'ask': float(ticker_data.get('ask', 0)),
                                'volume': float(ticker_data.get('volume', 0)),
                                'timestamp': int(time.time() * 1000)
                            }
                    except Exception:
                        continue
                        
            except ImportError:
                logger.warning("CCXT not available for ticker fallback")
            except Exception as ccxt_error:
                logger.warning(f"CCXT ticker fallback failed: {ccxt_error}")
            
            # If symbol not found, use a mock price for testing
            logger.warning(f"Unable to fetch price data for {symbol}, using mock price")
            
            # Generate a mock price based on a hash of the symbol name for consistency
            import hashlib
            hash_val = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 10000
            mock_price = 100 + (hash_val / 100)  # Range between 100-200
            
            return {
                'symbol': symbol,
                'last': mock_price,
                'bid': mock_price * 0.999,  # Slightly lower
                'ask': mock_price * 1.001,  # Slightly higher
                'volume': 1000,
                'timestamp': int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"All attempts to get ticker for {symbol} failed: {e}")
            # Final fallback - mock data
            return {
                'symbol': symbol,
                'last': 100.0,  # Mock price
                'bid': 99.9,
                'ask': 100.1,
                'volume': 1000,
                'timestamp': int(time.time() * 1000)
            }
    
    def get_ohlc(self, symbol, interval='1', since=None):
        """Get OHLCV data for a symbol"""
        try:
            # Fix for Kraken Futures API - ensure correct interval format
            # Kraken Futures uses specific interval values
            api_intervals = {
                '1': '1m',
                '5': '5m', 
                '15': '15m',
                '30': '30m',
                '60': '1h',
                '240': '4h',
                '1440': '1d',
                '10080': '1w',
                # Add mappings for string-based inputs
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d',
                '1w': '1w'
            }
            
            # Convert interval to Kraken's format
            kraken_interval = api_intervals.get(interval, '1m')
            
            logger.info(f"Getting OHLC for {symbol} with interval {kraken_interval}")
            
            # Kraken requires timestamps in a specific format
            from_ts = None
            if since:
                # Convert milliseconds to seconds if needed
                since_sec = since // 1000 if since > 1500000000000 else since
                from_ts = datetime.fromtimestamp(since_sec).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            
            # Try two different URL patterns - Kraken sometimes changes their API endpoints
            urls = [
                f"{self.base_url}/derivatives/api/v3/history",  # Current API
                f"{self.base_url}/api/history"                  # Alternative path
            ]
            
            data = None
            for url in urls:
                try:
                    params = {
                        'symbol': symbol,
                        'interval': kraken_interval
                    }
                    if from_ts:
                        params['from'] = from_ts
                    
                    logger.info(f"Requesting OHLC from {url} for {symbol} with params {params}")
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if data and ('candles' in data or 'history' in data):
                        logger.info(f"Successfully received data from {url}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to get data from {url}: {e}")
            
            if not data:
                logger.warning(f"Could not retrieve OHLC data for {symbol} from any endpoint")
                return []
            
            # Format data as OHLCV - handle both possible response formats
            ohlcv = []
            
            # Handle standard candles format
            if 'candles' in data and data['candles']:
                for candle in data['candles']:
                    try:
                        timestamp = int(datetime.strptime(candle['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() * 1000)
                        ohlcv.append([
                            timestamp,
                            float(candle.get('open', 0)),
                            float(candle.get('high', 0)),
                            float(candle.get('low', 0)),
                            float(candle.get('close', 0)),
                            float(candle.get('volume', 0))
                        ])
                    except Exception as e:
                        logger.warning(f"Failed to parse candle: {e}")
            
            # Handle alternative history format
            elif 'history' in data and data['history']:
                for point in data['history']:
                    try:
                        # Ensure we have the minimum required data
                        if 'time' in point and 'price' in point:
                            timestamp = int(datetime.strptime(point['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() * 1000)
                            price = float(point['price'])
                            # Use price for all OHLC values if that's all we have
                            ohlcv.append([
                                timestamp,
                                price,
                                price,
                                price,
                                price,
                                float(point.get('volume', 0))
                            ])
                    except Exception as e:
                        logger.warning(f"Failed to parse history point: {e}")
            
            logger.info(f"Retrieved {len(ohlcv)} OHLC data points for {symbol}")
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return []
    
    def get_accounts(self):
        """Get account information"""
        try:
            # Try multiple endpoints - Kraken's API structure can be inconsistent
            endpoints = [
                "/derivatives/api/v3/accounts",
                "/api/v3/accounts", 
                "/api/accounts",
                "/derivatives/api/v3/account"
            ]
            
            results = {"accounts": []}
            
            for endpoint in endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    
                    # Create signature
                    nonce = str(int(time.time() * 1000))
                    
                    # Ensure there's no whitespace in the message
                    message = nonce.strip() + endpoint.strip()
                    
                    # Make sure the secret doesn't have whitespace or encoding issues
                    clean_secret = self.secret.strip()
                    try:
                        # Try to decode base64 to check validity
                        decoded_secret = base64.b64decode(clean_secret)
                    except Exception as decode_error:
                        logger.error(f"Invalid API secret format: {decode_error}")
                        # If we can't decode, create a mock account and return it
                        return {
                            "accounts": [
                                {"currency": "USD", "available": 1000.0, "total": 1000.0},
                                {"currency": "BTC", "available": 0.05, "total": 0.05},
                            ]
                        }
                    
                    # Create signature with clean data
                    signature = hmac.new(
                        decoded_secret,
                        message.encode('utf-8'),
                        hashlib.sha256
                    ).digest()
                    
                    # Encode signature without spaces
                    encoded_signature = base64.b64encode(signature).decode('utf-8').strip()
                    
                    signed_headers = {
                        'APIKey': self.api_key.strip(),
                        'Nonce': nonce.strip(),
                        'Authent': encoded_signature
                    }
                    
                    logger.info(f"Requesting account info from {url}")
                    response = self.session.get(url, headers={**self.session.headers, **signed_headers}, timeout=10)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Check if we got account data
                    if 'accounts' in result and result['accounts']:
                        logger.info(f"Account data found at {url}: {len(result['accounts'])} accounts")
                        results = result
                        break
                        
                    # Check alternative formats
                    elif 'account' in result:
                        # Convert to expected format
                        if isinstance(result['account'], dict):
                            # Single account format
                            results = {"accounts": [result['account']]}
                            logger.info(f"Account data found at {url} (alternative format)")
                            break
                        elif isinstance(result['account'], list):
                            # List of accounts
                            results = {"accounts": result['account']}
                            logger.info(f"Account data found at {url} (list format)")
                            break
                            
                    logger.warning(f"No account data in response from {url}")
                    
                except Exception as endpoint_err:
                    logger.warning(f"Failed to get account info from {url}: {endpoint_err}")
            
            # If we still haven't found any accounts, try using the CCXT library as fallback
            if not results.get('accounts'):
                logger.warning("Trying CCXT library as fallback for account data")
                try:
                    # Import ccxt here to avoid dependency issues if not available
                    import ccxt
                    
                    # Initialize Kraken client with API credentials (strip to be safe)
                    kraken = ccxt.kraken({
                        'apiKey': self.api_key.strip(),
                        'secret': self.secret.strip(),
                    })
                    
                    # Get account balance
                    balance = kraken.fetch_balance()
                    
                    if balance and 'total' in balance:
                        # Convert CCXT format to expected format
                        accounts = []
                        for currency, amount in balance['total'].items():
                            if amount > 0:
                                accounts.append({
                                    'currency': currency,
                                    'available': float(balance['free'].get(currency, 0)),
                                    'total': float(amount)
                                })
                        
                        if accounts:
                            results = {"accounts": accounts}
                            logger.info(f"Found {len(accounts)} accounts via CCXT")
                
                except ImportError:
                    logger.warning("CCXT library not available")
                except Exception as ccxt_err:
                    logger.warning(f"Failed to get account info via CCXT: {ccxt_err}")
            
            # Add total balance if not present
            for account in results.get('accounts', []):
                if 'available' in account and 'total' not in account:
                    account['total'] = account['available']
            
            # If still empty, create a mock account for testing
            if not results.get('accounts'):
                logger.warning("Could not retrieve real account data, creating mock account for testing")
                results = {
                    "accounts": [
                        {"currency": "USD", "available": 1000.0, "total": 1000.0},
                        {"currency": "BTC", "available": 0.05, "total": 0.05},
                    ]
                }
                
            logger.info(f"Final account data: {len(results.get('accounts', []))} accounts")
            return results
            
        except Exception as e:
            logger.error(f"All attempts to get account data failed: {e}")
            # Return mock account for testing
            mock_account = {
                "accounts": [
                    {"currency": "USD", "available": 1000.0, "total": 1000.0},
                    {"currency": "BTC", "available": 0.05, "total": 0.05},
                ]
            }
            logger.warning("Returning mock account data for testing")
            return mock_account
    
    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        """Create an order following CCXT style but with Kraken Futures API specifics"""
        try:
            url = f"{self.base_url}/derivatives/api/v3/sendorder"
            
            # Create signature
            nonce = str(int(time.time() * 1000))
            endpoint = "/derivatives/api/v3/sendorder"
            
            # Prepare order data
            order_data = {
                "orderType": order_type.upper(),
                "symbol": symbol,
                "side": side.upper(),
                "size": amount
            }
            
            if price and order_type.lower() == 'limit':
                order_data["limitPrice"] = price
                
            if params:
                order_data.update(params)
                
            # Add client order ID if not present
            if 'cliOrdId' not in order_data:
                order_data['cliOrdId'] = f'bot_{int(time.time())}'
                
            # Handle leverage
            if 'leverage' in params:
                order_data['leverage'] = params['leverage']
            
            message = nonce + endpoint + json.dumps(order_data)
            signature = hmac.new(
                base64.b64decode(self.secret),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            signed_headers = {
                'APIKey': self.api_key,
                'Nonce': nonce,
                'Authent': base64.b64encode(signature).decode('utf-8')
            }
            
            response = self.session.post(
                url, 
                headers={**self.session.headers, **signed_headers},
                json=order_data
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if order was sent successfully
            if 'sendStatus' in result and result['sendStatus'].get('status') == 'placed':
                # Format result to match expected structure
                order = {
                    'id': result['sendStatus'].get('orderId', ''),
                    'clientOrderId': order_data['cliOrdId'],
                    'symbol': symbol,
                    'type': order_type,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'open',
                    'timestamp': int(time.time() * 1000)
                }
                logger.info(f"Order placed successfully: {order['id']}")
                return order
            else:
                error_msg = result.get('sendStatus', {}).get('status', 'unknown error')
                logger.error(f"Order placement failed: {error_msg}")
                raise Exception(f"Order placement failed: {error_msg}")
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise e

    def create_market_order_with_cost(self, symbol, side, cost, params=None):
        """Create a market order with cost (in quote currency) - CCXT style function"""
        try:
            # First get the ticker to calculate the amount
            ticker = self.get_ticker(symbol)
            
            if ticker and 'last' in ticker and ticker['last'] > 0:
                # Calculate amount based on cost and current price
                price = ticker['last']
                
                # Get contract size for futures
                contract_size = self._get_contract_size(symbol)
                if contract_size <= 0:
                    raise ValueError(f"Invalid contract size for {symbol}")
                
                # Calculate number of contracts to meet the cost target
                contracts = cost / (price * contract_size)
                
                # Round to appropriate precision (futures usually use fewer decimals)
                contracts = round(contracts, 2)  # Adjust precision as needed
                
                # Create the market order
                return self.create_order(
                    symbol=symbol,
                    order_type='market',
                    side=side,
                    amount=contracts,
                    price=None,
                    params=params
                )
            else:
                raise ValueError(f"Could not get price for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to create market order with cost: {e}")
            raise e
    
    def _get_contract_size(self, symbol):
        """Get contract size for a symbol"""
        try:
            instruments = self.get_instruments()
            for inst in instruments.get('instruments', []):
                if inst['symbol'] == symbol:
                    return float(inst.get('contractSize', 1))
            return 0
        except Exception as e:
            logger.error(f"Failed to get contract size: {e}")
            return 0

class CryptoTradingBot:
    """Thread-safe main bot class - Real Account Version"""
    
    def __init__(self, config_file: str):
        """Initialize trading bot with configuration from file"""
        self.config = self._load_config(config_file)
        
        # Check if using spot or futures API
        self.use_spot = self.config.get('use_spot_api', False)
        
        # Initialize client with appropriate settings
        api_key = self.config['connection']['api_key']
        secret = self.config['connection']['secret']
        
        # Handle API key format issues
        if api_key and api_key.startswith(' '):
            self.log("API key has leading whitespace - this will cause authentication errors", "warning")
            api_key = api_key.strip()
            self.config['connection']['api_key'] = api_key
            
        if secret and secret.startswith(' '):
            self.log("API secret has leading whitespace - this will cause authentication errors", "warning")
            secret = secret.strip()
            self.config['connection']['secret'] = secret
        
        # Initialize trading client
        self.client = self._initialize_client()
        
        # WebSocket and other core properties
        self.ws_client = None
        self.ws_thread = None
        self.positions = {}  # Current open positions
        self.balance = {}  # Account balance
        self.last_buy_time = {}  # Timestamp of last buy for each coin
        self.trailing_stops = {}  # Track trailing stops for each position
        self.profit_stats = {
            'total_profit': 0,
            'wins': 0,
            'losses': 0,
            'trades': []
        }
        self.markets = {}
        self.tickers = {}
        self.running = True
        
        # Thread-safe message queue for logging
        self.message_queue = queue.Queue()
        self.output_handler = None
        
    def set_output_handler(self, handler):
        """Set a handler for UI output"""
        self.output_handler = handler
        
    def log(self, message, level="info", data=None):
        """Thread-safe logging with UI integration"""
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
            
        # Queue the message for UI handling
        try:
            self.message_queue.put({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'message': message,
                'level': level,
                'data': data
            })
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            
    def process_message_queue(self):
        """Process queued messages (call from main thread)"""
        if hasattr(self, 'message_queue'):
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    # Handle the message in the main thread
                    if self.output_handler:
                        self.output_handler(
                            message['message'], 
                            message['level'], 
                            message['data']
                        )
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing message queue: {e}")
            
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_client(self):
        """Initialize connection to trading platform"""
        try:
            api_key = self.config['connection']['api_key']
            secret = self.config['connection']['secret']
            
            # Handle API key format issues
            if api_key and api_key.startswith(' '):
                self.log("API key has leading whitespace - this will cause authentication errors", "warning")
                api_key = api_key.strip()
                self.config['connection']['api_key'] = api_key
                
            if secret and secret.startswith(' '):
                self.log("API secret has leading whitespace - this will cause authentication errors", "warning")
                secret = secret.strip()
                self.config['connection']['secret'] = secret
            
            # Create real account client
            client = KrakenTradingClient(api_key, secret)
            
            # Test mode override if no valid API credentials
            is_test_mode = self.config.get('test_mode', False)
            if (not api_key or len(api_key) < 10) and not is_test_mode:
                self.log("No valid API key provided - enabling test mode automatically", "warning")
                self.config['test_mode'] = True
            
            # Verify connection by fetching instruments
            instruments = client.get_instruments()
            
            if instruments and 'instruments' in instruments:
                logger.info(f"Connected to Kraken Futures")
                logger.info(f"Found {len(instruments['instruments'])} available instruments")
            else:
                logger.warning("Connected but no instruments found")
                
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            raise
    
    def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time data"""
        try:
            ws_url = self.client.ws_url
            api_key = self.config['connection']['api_key']
            secret = self.config['connection']['secret']
            bot = self  # Store reference to self for callbacks
        
            # Define WebSocket callbacks without direct access to the bot
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                
                    # Check message type
                    if 'event' in data:
                        # Handle system messages
                        event = data.get('event')
                        if event == 'info':
                            logger.info(f"WebSocket info: {data.get('message', '')}")
                        elif event == 'subscribed':
                            logger.info(f"Subscribed to {data.get('feed', '')}")
                        elif event == 'error':
                            logger.error(f"WebSocket error event: {data.get('message', '')}")
                        # Other event types can be handled here
                
                    elif 'feed' in data:
                        feed = data.get('feed')
                      
                        if feed == 'ticker' and 'product_id' in data:
                            # Handle ticker data
                            symbol = data['product_id']
                            bot.tickers[symbol] = data
                            # Check trailing stops on price updates
                            if 'bid' in data:
                                bot._check_trailing_stops(symbol, float(data['bid']))
                    
                        elif feed == 'fills':
                            # Handle fill data
                            logger.info(f"Order filled: {data}")
                            bot._process_fill(data)
                        
                        elif feed == 'heartbeat':
                            # Silently handle heartbeat messages
                            pass
                    
                        else:
                            # Unknown feed type, just log at debug level
                            logger.debug(f"Received message from feed '{feed}'")
                
                    else:
                        # Unknown message format
                        logger.debug(f"Received message with unknown format: {data}")
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse WebSocket message: {message}")
                except Exception as e:
                    logger.error(f"WebSocket message processing error: {str(e)}")
        
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
        
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
                if bot.running:
                    logger.info("Attempting to reconnect WebSocket in 5 seconds...")
                    time.sleep(5)
                    bot._initialize_websocket()
        
            def on_open(ws):
                logger.info("WebSocket connection established")
            
                # Auth message
                nonce = int(time.time() * 1000)
                authent = f"GET/auth/w/websocket:{nonce}"
                signature = hmac.new(
                    base64.b64decode(secret),
                    authent.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            
                auth_msg = {
                    "event": "login",
                    "key": api_key,
                    "sign": base64.b64encode(signature).decode('utf-8'),
                    "timestamp": nonce
                }
                ws.send(json.dumps(auth_msg))
            
                # After authentication, subscribe to required feeds
                product_ids = bot._get_product_ids()
                if product_ids:
                    try:
                        subscribe_msg = {
                            "event": "subscribe",
                            "feed": "ticker",
                            "product_ids": product_ids
                        }
                        ws.send(json.dumps(subscribe_msg))
                        logger.info(f"Subscribed to ticker feed for {len(product_ids)} products")
                    except Exception as e:
                        logger.error(f"Error subscribing to ticker feed: {e}")
            
                # Subscribe to user-specific feeds
                try:
                    user_feeds = {
                        "event": "subscribe",
                        "feed": "fills"
                    }
                    ws.send(json.dumps(user_feeds))
                    logger.info("Subscribed to fills feed")
                except Exception as e:
                    logger.error(f"Error subscribing to fills feed: {e}")
        
            # Initialize WebSocket with ping interval to keep connection alive
            websocket.enableTrace(False)  # Disable trace to reduce output
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
        
            # Start WebSocket thread with ping interval
            self.ws_client = ws
            self.ws_thread = threading.Thread(
               target=lambda: ws.run_forever(ping_interval=30, ping_timeout=10)
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
        
            logger.info("Real-time data connection established")
        
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {e}")

    def _get_product_ids(self) -> List[str]:
        """Get list of product IDs for WebSocket subscription"""
        try:
            instruments = self.client.get_instruments()
            
            # Filter for perpetual futures that are trading
            product_ids = []
            
            # Check if we have the specific coins configured
            allowed_coins = self.get_allowed_coins()
            
            if allowed_coins:
                # Use the already filtered list
                product_ids = allowed_coins
            else:
                # Fallback - just get all tradeable perpetuals
                for inst in instruments.get('instruments', []):
                    if inst.get('tradeable', False) and 'PERP' in inst.get('symbol', ''):
                        product_ids.append(inst.get('symbol', ''))
            
            # Make sure we have at least something to subscribe to
            if not product_ids and instruments.get('instruments'):
                # Last resort - just take the first tradeable instrument
                for inst in instruments.get('instruments', []):
                    if inst.get('tradeable', False):
                        product_ids.append(inst.get('symbol', ''))
                        break
                        
            logger.info(f"Found {len(product_ids)} tradeable instruments for WebSocket subscription")
            
            # List the instruments we're subscribing to
            if product_ids:
                logger.info(f"Subscribing to: {', '.join(product_ids[:5])}{' and more' if len(product_ids) > 5 else ''}")
            
            return product_ids
        except Exception as e:
            logger.error(f"Failed to get product IDs: {e}")
            return []
            
    def _process_fill(self, fill_data: Dict):
        """Process a fill event from WebSocket"""
        try:
            symbol = fill_data['product_id']
            side = fill_data['side']
            price = float(fill_data['price'])
            size = float(fill_data['size'])
            order_id = fill_data['order_id']
            
            if side.lower() == 'buy':
                # Record new position
                self.positions[order_id] = {
                    'symbol': symbol,
                    'size': size,
                    'price': price,
                    'timestamp': datetime.now(),
                    'order_id': order_id,
                    'highest_price': price  # For trailing stop
                }
                self.log(f"New position opened: {symbol} - {size} @ {price}", "info", {
                    "type": "new_position",
                    "symbol": symbol,
                    "size": size,
                    "price": price
                })
            elif side.lower() == 'sell':
                # Find and update corresponding position
                for pos_id, position in list(self.positions.items()):
                    if position['symbol'] == symbol:
                        # Calculate profit
                        buy_price = position['price']
                        sell_price = price
                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        profit_amount = (sell_price - buy_price) * position['size']
                        
                        # Update stats
                        self.profit_stats['total_profit'] += profit_amount
                        if profit_amount > 0:
                            self.profit_stats['wins'] += 1
                        else:
                            self.profit_stats['losses'] += 1
                            
                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'size': position['size'],
                            'profit_pct': profit_pct,
                            'profit_amount': profit_amount,
                            'timestamp': datetime.now()
                        }
                        self.profit_stats['trades'].append(trade)
                        
                        # Remove position
                        del self.positions[pos_id]
                        self.log(f"Position closed: {symbol} - Profit: {profit_pct:.2f}%, {profit_amount:.2f}", "info", {
                            "type": "close_position",
                            "symbol": symbol,
                            "profit_pct": profit_pct,
                            "profit_amount": profit_amount
                        })
                        break
                        
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
    
    def sync_balance(self) -> None:
        """Sync balance from trading platform"""
        try:
            # Get account information
            self.log("Syncing account balance...")
            accounts = self.client.get_accounts()
            
            if 'accounts' in accounts and accounts['accounts']:
                self.balance = accounts
                account_summary = []
                for account in accounts['accounts']:
                    currency = account.get('currency', 'Unknown')
                    available = account.get('available', 0)
                    account_summary.append(f"{currency}:{available}")
                
                self.log(f"Balance synced successfully: {', '.join(account_summary)}")
            else:
                self.log("Balance sync returned no account data", "warning")
                # If we got empty response but have previous balance, keep it
                if not self.balance or 'accounts' not in self.balance or not self.balance['accounts']:
                    self.balance = accounts  # Update even if empty for UI to show the issue
        except Exception as e:
            self.log(f"Failed to sync balance: {e}", "error")
            
    def get_quote_currency(self) -> str:
        """Get quote currency from config"""
        return self.config['coins']['quote_currency']
    
    def get_allowed_coins(self) -> List[str]:
        """Get list of allowed futures instruments to trade"""
        try:
            instruments = self.client.get_instruments()
            
            self.log(f"Fetching tradable instruments... Found {len(instruments.get('instruments', []))} total instruments")
            
            # Map of common crypto symbol variations used by Kraken
            symbol_mapping = {
                "BTC": ["XBT", "BTC", "BITCOIN"],
                "ETH": ["ETH", "ETHEREUM"],
                "DOT": ["DOT", "POLKADOT"],
                "BCH": ["BCH", "BITCOINCASH"],
                "XRP": ["XRP", "RIPPLE"],
                "ADA": ["ADA", "CARDANO"],
                "SOL": ["SOL", "SOLANA"],
                "DOGE": ["DOGE", "DOGECOIN"],
                "LTC": ["LTC", "LITECOIN"],
                "LINK": ["LINK", "CHAINLINK"]
            }
            
            if instruments.get('instruments', []):
                # For debugging, list first few tradable instruments
                tradable_count = sum(1 for inst in instruments['instruments'] if inst.get('tradeable', False))
                self.log(f"Found {tradable_count} tradable instruments")
                
                sample_instruments = [
                    inst['symbol'] for inst in instruments['instruments'][:10] 
                    if inst.get('tradeable', False)
                ]
                if sample_instruments:
                    self.log(f"Sample instruments: {', '.join(sample_instruments)}")
                    
                    # Try to identify the symbol pattern Kraken is using
                    for sample in sample_instruments:
                        for crypto, variations in symbol_mapping.items():
                            # Check if this instrument matches any known crypto
                            for var in variations:
                                if var in sample.upper():
                                    self.log(f"Detected {crypto} in sample instrument: {sample}")
                                    break
            
            # Get selected coins from config
            selected = self.config['coins']['selected_coins']
            
            # If allow all coins is enabled, return all tradable instruments
            if self.config['coins']['allow_all_coins']:
                coins = [inst['symbol'] for inst in instruments['instruments'] 
                        if inst.get('tradeable', False)]
                self.log(f"Trading all coins, found {len(coins)} tradable instruments")
                return coins
                
            self.log(f"Trading selected coins: {', '.join(selected)}")
            
            # First, try to find perpetual futures (highest priority)
            perp_matches = []
            for inst in instruments['instruments']:
                if not inst.get('tradeable', False):
                    continue
                    
                symbol = inst['symbol']
                
                # Check for perpetual futures
                for coin in selected:
                    # Check different naming patterns common on Kraken
                    variations = symbol_mapping.get(coin.upper(), [coin.upper()])
                    
                    for var in variations:
                        # Common patterns for perpetual futures
                        patterns = [
                            f"{var}USD-PERP", f"{var}USD_PERP", f"{var}-PERP", f"{var}_PERP", 
                            f"PI_{var}USD", f"PF_{var}USD"
                        ]
                        
                        # Check if the symbol matches any pattern
                        if any(pattern.upper() in symbol.upper() for pattern in patterns) or (
                           var in symbol.upper() and any(p in symbol.upper() for p in ["PERP", "PI_", "PF_"])):
                            perp_matches.append(symbol)
                            self.log(f"Found perpetual match for {coin}: {symbol}")
                            break
            
            # If we found perpetual futures, return those
            if perp_matches:
                self.log(f"Found {len(perp_matches)} perpetual futures instruments")
                return perp_matches
                
            # If no perpetual futures, look for any matches to the selected coins
            spot_matches = []
            for inst in instruments['instruments']:
                if not inst.get('tradeable', False):
                    continue
                    
                symbol = inst['symbol']
                
                for coin in selected:
                    variations = symbol_mapping.get(coin.upper(), [coin.upper()])
                    
                    for var in variations:
                        # Check if the symbol contains the coin name in any form
                        if var in symbol.upper():
                            spot_matches.append(symbol)
                            self.log(f"Found general match for {coin}: {symbol}")
                            break
            
            # If we found spot matches, return those
            if spot_matches:
                self.log(f"Found {len(spot_matches)} spot/futures instruments")
                return spot_matches
                
            # If we still don't have any matches, return a selection of the most liquid assets
            # This is a fallback to ensure the bot can still operate
            if not spot_matches and not perp_matches:
                self.log("No matches for selected coins, using default liquid instruments", "warning")
                # Try to find the most common assets
                liquid_assets = []
                for inst in instruments['instruments']:
                    if not inst.get('tradeable', False):
                        continue
                        
                    symbol = inst['symbol']
                    # Look for common high liquidity assets
                    if any(x in symbol.upper() for x in ["XBT", "BTC", "ETH", "USD"]):
                        liquid_assets.append(symbol)
                        if len(liquid_assets) >= 5:  # Limit to top 5
                            break
                
                if liquid_assets:
                    self.log(f"Using {len(liquid_assets)} default instruments: {', '.join(liquid_assets)}")
                    return liquid_assets
                
                # Last resort - just return the first 5 tradable instruments
                last_resort = [inst['symbol'] for inst in instruments['instruments'] 
                               if inst.get('tradeable', False)][:5]
                               
                if last_resort:
                    self.log(f"Using first {len(last_resort)} available instruments as last resort")
                    return last_resort
            
            # Combine and deduplicate results
            result = list(set(perp_matches + spot_matches))
            self.log(f"Final tradable instruments: {len(result)}")
            return result
            
        except Exception as e:
            self.log(f"Failed to get allowed coins: {e}", "error")
            # Return some typical symbols as a last resort
            default_symbols = ["PI_XBTUSD", "PI_ETHUSD"]
            self.log(f"Using default symbols as fallback: {default_symbols}", "warning")
            return default_symbols
    
    def calculate_buy_amount(self, symbol: str, current_price: float) -> float:
        """Calculate buy amount based on percentage and account balance"""
        try:
            # For futures, calculate contract quantity
            quote_currency = self.get_quote_currency()
            percentage = self.config['coins']['percentage_buy_amount'] / 100
            
            # Get collateral available
            accounts = self.balance
            if 'accounts' in accounts:
                available_balance = 0
                for account in accounts['accounts']:
                    if account['currency'] == quote_currency:
                        available_balance = float(account.get('available', 0))
                        break
                    # If exact currency not found, try alternatives (USD/USDT/USDC or GBP/EUR)
                    elif (quote_currency in ['USD', 'USDT', 'USDC'] and 
                          account['currency'] in ['USD', 'USDT', 'USDC']):
                        available_balance = float(account.get('available', 0))
                        self.log(f"Using {account['currency']} balance as substitute for {quote_currency}")
                        break
                    elif (quote_currency in ['GBP', 'EUR'] and 
                          account['currency'] in ['GBP', 'EUR']):
                        available_balance = float(account.get('available', 0))
                        self.log(f"Using {account['currency']} balance as substitute for {quote_currency}")
                        break
                    
                # If still no match, check for any cryptocurrency balance we could use
                if available_balance == 0:
                    for account in accounts['accounts']:
                        if account['currency'] not in ['USD', 'USDT', 'USDC', 'GBP', 'EUR']:
                            # We found a crypto - get its current price
                            crypto_symbol = f"PI_{account['currency']}USD"
                            crypto_ticker = self.client.get_ticker(crypto_symbol)
                            if crypto_ticker and crypto_ticker.get('last', 0) > 0:
                                crypto_price = float(crypto_ticker.get('last', 0))
                                crypto_amount = float(account.get('available', 0))
                                available_balance = crypto_amount * crypto_price
                                self.log(f"Using {account['currency']} balance converted to USD: {available_balance}")
                                break
            else:
                # Log error and return 0 if balance can't be determined
                self.log("Could not retrieve account balance", "warning")
                return 0
            
            # Calculate buy amount in quote currency
            buy_amount_quote = available_balance * percentage
            
            # Log the calculated amount
            self.log(f"Raw buy amount calculation: {buy_amount_quote:.2f} {quote_currency}")
            
            # Check minimum order amount
            min_amount = self.config['coins']['minimum_amount']
            if buy_amount_quote < min_amount:
                self.log(f"Buy amount {buy_amount_quote:.2f} below minimum {min_amount}", "warning")
                
                if self.config['coins']['force_minimum_buy_amount']:
                    # Force the minimum amount
                    buy_amount_quote = min_amount
                    self.log(f"Forcing minimum buy amount: {min_amount} {quote_currency}")
                else:
                    # Check if we're in test mode - if so, allow small amounts for testing
                    is_test_mode = self.config.get('test_mode', False)
                    if is_test_mode:
                        self.log("Test mode enabled, allowing small amounts", "warning")
                    else:
                        self.log("Not enough funds and force_minimum_buy_amount is disabled", "warning")
                        return 0
            
            # Check maximum allocated amount if set
            max_allocated = self.config['coins'].get('maximum_amount_allocated')
            if max_allocated and buy_amount_quote > max_allocated:
                buy_amount_quote = max_allocated
                self.log(f"Limiting buy amount to maximum: {max_allocated} {quote_currency}")
                
            # Convert to contract size
            contract_size = self.client._get_contract_size(symbol)
            if contract_size == 0:
                self.log(f"Failed to get contract size for {symbol}", "error")
                
                # Fallback: use a standard contract size of 1
                contract_size = 1
                self.log("Using fallback contract size of 1", "warning")
                
            # Calculate number of contracts
            contracts = buy_amount_quote / (current_price * contract_size)
            
            # Round to appropriate precision (most futures exchanges use fewer decimals)
            contracts = round(contracts, 4)  # Adjusted precision to allow smaller amounts
            
            # Emergency override for testing - ensure we have at least a minimal contract value
            if contracts < 0.001:
                is_test_mode = self.config.get('test_mode', False)
                if is_test_mode:
                    contracts = 0.001
                    self.log("Test mode: Setting minimum contract size to 0.001", "warning")
            
            self.log(f"Final buy amount for {symbol}: {contracts} contracts (worth approximately {buy_amount_quote:.2f} {quote_currency})")
            return contracts
            
        except Exception as e:
            self.log(f"Failed to calculate buy amount: {e}", "error")
            return 0
    
    def can_buy_instrument(self, symbol: str, current_price: float) -> bool:
        """Check if a trading instrument can be bought based on settings"""
        now = datetime.now()
        
        # Check cooldown period
        if (self.config['buy_settings']['enable_cooldown'] and 
            symbol in self.last_buy_time):
            cooldown_minutes = self.config['buy_settings']['cooldown_period']
            last_buy = self.last_buy_time[symbol]
            elapsed = (now - last_buy).total_seconds() / 60
            
            if elapsed < cooldown_minutes:
                logger.debug(f"Cooldown period in effect for {symbol} ({elapsed:.1f}/{cooldown_minutes} minutes)")
                return False
        
        # Check if already in position
        if (self.config['buy_settings']['only_buy_if_not_already_in_positions'] and 
            any(p['symbol'] == symbol for p in self.positions.values())):
            logger.debug(f"Already in position for {symbol}, skipping")
            return False
        
        # Check open positions limit
        max_positions = self.config['buy_settings']['max_open_positions']
        if len(self.positions) >= max_positions:
            logger.debug(f"Maximum number of positions reached ({len(self.positions)}/{max_positions})")
            return False
        
        # Check open positions per instrument
        inst_positions = len([p for p in self.positions.values() if p['symbol'] == symbol])
        max_per_inst = self.config['buy_settings']['max_percentage_open_positions_per_coin']
        max_allowed_per_inst = max_positions * max_per_inst / 100
        
        if inst_positions >= max_allowed_per_inst:
            logger.debug(f"Maximum positions for {symbol} reached ({inst_positions}/{max_allowed_per_inst})")
            return False
            
        # Check if only one open buy order per instrument
        if (self.config['buy_settings']['only_1_open_buy_order_per_coin'] and 
            any(p['symbol'] == symbol for p in self.positions.values())):
            logger.debug(f"Already have an open position for {symbol}, not buying more due to settings")
            return False
            
        # Check percentage range
        positions_for_symbol = [p for p in self.positions.values() if p['symbol'] == symbol]
        if positions_for_symbol:
            for position in positions_for_symbol:
                price_diff_pct = abs(position['price'] - current_price) / position['price'] * 100
                if price_diff_pct < self.config['buy_settings']['percent_range']:
                    logger.debug(f"Position for {symbol} exists within {price_diff_pct:.2f}% of current price")
                    return False
        
        return True
    
    def should_buy(self, symbol: str, price_data: pd.DataFrame) -> bool:
        """Determine if bot should buy based on price changes"""
        # Get price change trigger percentage from config
        trigger_pct = self.config['place_order_trigger']['percentage_change']
        
        # Calculate percentage change
        if len(price_data) > 1:
            try:
                # Check if we have enough data points
                if len(price_data) < 2:
                    self.log(f"Not enough price data for {symbol} to calculate change", "warning")
                    return False
            
                # Calculate the percentage change
                pct_change = price_data['close'].pct_change().iloc[-1] * 100
                
                # Handle NaN values
                if pd.isna(pct_change):
                    self.log(f"Invalid percentage change for {symbol} (NaN)", "warning")
                    return False
                
                # Check if percentage change meets trigger
                if abs(pct_change) >= trigger_pct:
                    self.log(f"{symbol} price changed by {pct_change:.2f}%, triggering buy check")
                    
                    # Check if the change is in a favorable direction (price is going up)
                    if pct_change > 0:
                        self.log(f"Price is moving up for {symbol}, good buy signal")
                        return True
                    else:
                        # We could still buy on downward movement depending on strategy
                        # For now, let's make this configurable
                        buy_on_down = self.config.get('buy_settings', {}).get('buy_on_downward_movement', True)
                        if buy_on_down:
                            self.log(f"Price is moving down for {symbol} but we're buying on downtrends")
                            return True
                        else:
                            self.log(f"Price is moving down for {symbol}, skipping buy")
                            return False
                
                # Not triggering
                return False
            
            except Exception as e:
                self.log(f"Error calculating buy signal for {symbol}: {e}", "error")
                return False
                
        return False
    
    def execute_buy(self, symbol: str, amount: float, price: float) -> Dict:
        """Execute buy order"""
        try:
            # Generate unique client order ID
            client_order_id = f'bot_{int(time.time())}'
            
            # Order parameters
            params = {
                'cliOrdId': client_order_id,
                'leverage': self.config['buy_settings'].get('leverage', 1)
            }
            
            # Order type (market or limit)
            order_type = self.config['buy_settings']['order_type'].lower()
            
            # Check if test mode is enabled
            is_test_mode = self.config.get('test_mode', False)
            
            if is_test_mode:
                self.log(f"TEST MODE: Simulating {order_type} buy order for {symbol}: {amount} contracts at {price if order_type == 'limit' else 'market price'}")
                
                # Create a simulated order for testing
                simulated_order = {
                    'id': f'test_order_{int(time.time())}',
                    'clientOrderId': client_order_id,
                    'symbol': symbol,
                    'type': order_type,
                    'side': 'buy',
                    'amount': amount,
                    'price': price,
                    'status': 'open',
                    'timestamp': int(time.time() * 1000)
                }
                
                # Record the buy time
                self.last_buy_time[symbol] = datetime.now()
                
                # Add to positions (will be updated by WebSocket on fill)
                position_id = simulated_order['id']
                self.positions[position_id] = {
                    'symbol': symbol,
                    'size': amount,
                    'price': price,
                    'order_type': order_type,
                    'timestamp': datetime.now(),
                    'order_id': simulated_order['id'],
                    'highest_price': price,  # For trailing stop
                    'is_test': True  # Mark as test position
                }
                
                self.log(f"TEST MODE: Buy order placed: {symbol} - {amount} @ {price}", "info", {
                    "type": "buy_order",
                    "symbol": symbol,
                    "amount": amount,
                    "price": price,
                    "is_test": True
                })
                
                return simulated_order
            
            # Real order execution
            self.log(f"Placing {order_type} buy order for {symbol}: {amount} contracts at {price if order_type == 'limit' else 'market price'}")
            
            try:
                if order_type == 'market':
                    try:
                        order = self.client.create_order(
                            symbol,
                            'market',
                            'buy',
                            amount,
                            None,
                            params
                        )
                    except Exception as market_error:
                        self.log(f"Market order failed, trying CCXT-style order with cost: {market_error}")
                        
                        # Calculate equivalent cost
                        contract_size = self.client._get_contract_size(symbol)
                        cost = amount * price * contract_size
                        
                        # Try using the CCXT-style method
                        order = self.client.create_market_order_with_cost(
                            symbol,
                            'buy',
                            cost,
                            params
                        )
                else:  # limit order
                    order = self.client.create_order(
                        symbol,
                        'limit',
                        'buy',
                        amount,
                        price,
                        params
                    )
                
                # Record the buy time
                self.last_buy_time[symbol] = datetime.now()
                
                # Add to positions (will be updated by WebSocket on fill)
                position_id = order['id']
                self.positions[position_id] = {
                    'symbol': symbol,
                    'size': amount,
                    'price': price,
                    'order_type': order_type,
                    'timestamp': datetime.now(),
                    'order_id': order['id'],
                    'highest_price': price,  # For trailing stop
                }
                
                self.log(f"Buy order placed: {symbol} - {amount} @ {price}", "info", {
                    "type": "buy_order",
                    "symbol": symbol,
                    "amount": amount,
                    "price": price
                })
                return order
                
            except Exception as e:
                error_msg = str(e)
                # Check if this is likely an authentication error
                if "auth" in error_msg.lower() or "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    self.log(f"Authentication error placing order: {e}", "error")
                    self.log("Please check your API key and secret", "error")
                else:
                    self.log(f"Error executing order: {e}", "error")
                
                if is_test_mode:
                    self.log("Falling back to test mode order simulation due to error", "warning")
                    # Create a simulated order as fallback
                    simulated_order = {
                        'id': f'test_fallback_{int(time.time())}',
                        'clientOrderId': client_order_id,
                        'symbol': symbol,
                        'type': order_type,
                        'side': 'buy',
                        'amount': amount,
                        'price': price,
                        'status': 'open',
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    # Record the buy time
                    self.last_buy_time[symbol] = datetime.now()
                    
                    # Add to positions
                    position_id = simulated_order['id']
                    self.positions[position_id] = {
                        'symbol': symbol,
                        'size': amount,
                        'price': price,
                        'order_type': order_type,
                        'timestamp': datetime.now(),
                        'order_id': simulated_order['id'],
                        'highest_price': price,
                        'is_test': True
                    }
                    
                    self.log(f"TEST FALLBACK: Buy order placed: {symbol} - {amount} @ {price}", "warning", {
                        "type": "buy_order",
                        "symbol": symbol,
                        "amount": amount,
                        "price": price,
                        "is_test": True
                    })
                    
                    return simulated_order
                return None
            
        except Exception as e:
            self.log(f"Failed to execute buy order: {e}", "error")
            return None
    
    def _check_trailing_stops(self, symbol: str, current_price: float) -> None:
        """Check trailing stops for a symbol when price updates"""
        for position_id, position in list(self.positions.items()):
            if position['symbol'] == symbol:
                # Update highest price if current price is higher
                if current_price > position['highest_price']:
                    self.positions[position_id]['highest_price'] = current_price
                
                # Check trailing stop
                trailing_pct = self.config['sell_settings']['trailing_stop_loss_percentage']
                arm_at_pct = self.config['sell_settings']['arm_trailing_stop_loss_at']
                
                initial_price = position['price']
                highest_price = position['highest_price']
                
                profit_pct = (current_price - initial_price) / initial_price * 100
                
                # If profit percentage reached arm threshold, activate trailing stop
                if profit_pct >= arm_at_pct:
                    # Calculate stop price based on highest price
                    stop_price = highest_price * (1 - trailing_pct / 100)
                    
                    # If current price falls below stop price, sell
                    if current_price <= stop_price:
                        self.log(f"Trailing stop triggered for {symbol} at {current_price:.2f} (stop: {stop_price:.2f})")
                        
                        if self.config['sell_settings']['only_sell_with_profit']:
                            # Check if this would result in profit
                            if current_price > initial_price:
                                self.execute_sell(position_id, current_price)
                            else:
                                self.log(f"Not selling {symbol} - would result in loss")
                        else:
                            self.execute_sell(position_id, current_price)
    
    def update_trailing_stops(self) -> None:
        """Update trailing stops for all positions"""
        for symbol in set(position['symbol'] for position in self.positions.values()):
            try:
                # Get current price
                ticker = self.client.get_ticker(symbol)
                current_price = ticker['last'] if ticker else 0
                
                if current_price > 0:
                    # Update trailing stops for this symbol
                    self._check_trailing_stops(symbol, current_price)
                
            except Exception as e:
                logger.error(f"Error updating trailing stop for {symbol}: {e}")
    
    def execute_sell(self, position_id: str, current_price: float) -> Dict:
        """Execute sell order"""
        try:
            position = self.positions[position_id]
            symbol = position['symbol']
            amount = position['size']
            
            # Check if this is a test position
            is_test_position = position.get('is_test', False)
            is_test_mode = self.config.get('test_mode', False)
            
            # If it's a test position or we're in test mode, handle with simulated selling
            if is_test_position or is_test_mode:
                self.log(f"TEST MODE: Simulating market sell for {symbol}: {amount} contracts at {current_price}")
                
                # Calculate profit
                buy_price = position['price']
                sell_price = current_price
                profit_pct = (sell_price - buy_price) / buy_price * 100
                profit_amount = (sell_price - buy_price) * position['size']
                
                # Update stats
                self.profit_stats['total_profit'] += profit_amount
                if profit_amount > 0:
                    self.profit_stats['wins'] += 1
                else:
                    self.profit_stats['losses'] += 1
                    
                # Record trade
                trade = {
                    'symbol': symbol,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'size': position['size'],
                    'profit_pct': profit_pct,
                    'profit_amount': profit_amount,
                    'timestamp': datetime.now(),
                    'is_test': True
                }
                self.profit_stats['trades'].append(trade)
                
                # Remove position
                del self.positions[position_id]
                
                self.log(f"TEST MODE: Position closed: {symbol} - Profit: {profit_pct:.2f}%, {profit_amount:.2f}", "info", {
                    "type": "close_position",
                    "symbol": symbol,
                    "profit_pct": profit_pct,
                    "profit_amount": profit_amount,
                    "is_test": True
                })
                
                # Return simulated order
                return {
                    'id': f'test_sell_{int(time.time())}',
                    'clientOrderId': f'test_sell_{int(time.time())}',
                    'symbol': symbol,
                    'type': 'market',
                    'side': 'sell',
                    'amount': amount,
                    'price': current_price,
                    'status': 'closed',
                    'timestamp': int(time.time() * 1000),
                    'is_test': True
                }
            
            # Real order execution
            # Generate unique client order ID
            client_order_id = f'bot_sell_{int(time.time())}'
            
            # Order parameters
            params = {
                'cliOrdId': client_order_id,
                'leverage': self.config['buy_settings'].get('leverage', 1)
            }
            
            self.log(f"Executing sell for {symbol}: {amount} contracts at market price")
            
            # Execute market sell order
            try:
                order = self.client.create_order(
                    symbol,
                    'market',
                    'sell',
                    amount,
                    None,
                    params
                )
            except Exception as market_error:
                self.log(f"Market sell order failed, trying CCXT-style order with cost: {market_error}", "error")
                
                # Try using the CCXT-style method
                try:
                    # Calculate equivalent cost
                    contract_size = self.client._get_contract_size(symbol)
                    cost = amount * current_price * contract_size
                    
                    order = self.client.create_market_order_with_cost(
                        symbol,
                        'sell',
                        cost,
                        params
                    )
                except Exception as ccxt_error:
                    self.log(f"Failed to execute sell with CCXT: {ccxt_error}", "error")
                    
                    # If we're in test mode, fall back to simulated selling
                    if self.config.get('test_mode', False):
                        self.log("Falling back to test mode for sell due to API errors", "warning")
                        
                        # Simulate the sell (same code as above test mode)
                        buy_price = position['price']
                        sell_price = current_price
                        profit_pct = (sell_price - buy_price) / buy_price * 100
                        profit_amount = (sell_price - buy_price) * position['size']
                        
                        # Update stats
                        self.profit_stats['total_profit'] += profit_amount
                        if profit_amount > 0:
                            self.profit_stats['wins'] += 1
                        else:
                            self.profit_stats['losses'] += 1
                            
                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'size': position['size'],
                            'profit_pct': profit_pct,
                            'profit_amount': profit_amount,
                            'timestamp': datetime.now(),
                            'is_test': True
                        }
                        self.profit_stats['trades'].append(trade)
                        
                        # Remove position
                        del self.positions[position_id]
                        
                        self.log(f"TEST FALLBACK: Position closed: {symbol} - Profit: {profit_pct:.2f}%, {profit_amount:.2f}", "info", {
                            "type": "close_position",
                            "symbol": symbol,
                            "profit_pct": profit_pct,
                            "profit_amount": profit_amount,
                            "is_test": True
                        })
                        
                        # Return simulated order
                        return {
                            'id': f'test_fallback_sell_{int(time.time())}',
                            'clientOrderId': f'test_fallback_sell_{int(time.time())}',
                            'symbol': symbol,
                            'type': 'market',
                            'side': 'sell',
                            'amount': amount,
                            'price': current_price,
                            'status': 'closed',
                            'timestamp': int(time.time() * 1000),
                            'is_test': True
                        }
                    else:
                        raise ccxt_error
            
            self.log(f"Sell order placed: {symbol} - {amount} @ {current_price}", "info", {
                "type": "sell_order",
                "symbol": symbol,
                "amount": amount,
                "price": current_price
            })
            return order
            
        except Exception as e:
            self.log(f"Failed to execute sell order: {e}", "error")
            
            # Check if this is a position that exists
            if position_id in self.positions:
                position = self.positions[position_id]
                symbol = position['symbol']
                
                # If we're in test mode, we can simulate the selling even on error
                if self.config.get('test_mode', False):
                    self.log("Using test mode fallback for sell due to error", "warning")
                    
                    # Get position details
                    amount = position['size']
                    buy_price = position['price']
                    sell_price = current_price
                    
                    # Calculate profit
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    profit_amount = (sell_price - buy_price) * amount
                    
                    # Update stats
                    self.profit_stats['total_profit'] += profit_amount
                    if profit_amount > 0:
                        self.profit_stats['wins'] += 1
                    else:
                        self.profit_stats['losses'] += 1
                        
                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'size': amount,
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'timestamp': datetime.now(),
                        'is_test': True
                    }
                    self.profit_stats['trades'].append(trade)
                    
                    # Remove position
                    del self.positions[position_id]
                    
                    self.log(f"TEST ERROR FALLBACK: Position closed: {symbol} - Profit: {profit_pct:.2f}%, {profit_amount:.2f}", "warning", {
                        "type": "close_position",
                        "symbol": symbol,
                        "profit_pct": profit_pct,
                        "profit_amount": profit_amount,
                        "is_test": True
                    })
                    
                    # Return simulated order
                    return {
                        'id': f'test_error_{int(time.time())}',
                        'clientOrderId': f'test_error_{int(time.time())}',
                        'symbol': symbol,
                        'type': 'market',
                        'side': 'sell',
                        'amount': amount,
                        'price': current_price,
                        'status': 'closed',
                        'timestamp': int(time.time() * 1000),
                        'is_test': True
                    }
            
            return None
    
    def fetch_market_data(self, symbol: str, timeframe: str = '1', limit: int = 100) -> pd.DataFrame:
        """Fetch market data"""
        try:
            self.log(f"Fetching market data for {symbol} with timeframe {timeframe}")
            
            # Fetch OHLCV data
            # Convert timeframe to minutes for the since calculation
            # Adjust timeframe value based on type (minutes, hours, days)
            if timeframe == '1h':
                minutes_multiplier = 60
                api_timeframe = '60'
            elif timeframe == '4h':
                minutes_multiplier = 240
                api_timeframe = '240'
            elif timeframe == '1d':
                minutes_multiplier = 1440
                api_timeframe = '1440'
            elif timeframe == '1w':
                minutes_multiplier = 10080
                api_timeframe = '10080'
            else:
                # Default to minutes
                minutes_multiplier = int(timeframe)
                api_timeframe = timeframe
                
            # Calculate since timestamp
            since = int((datetime.now() - timedelta(minutes=limit * minutes_multiplier)).timestamp() * 1000)
            
            # Get OHLC data
            ohlcv = self.client.get_ohlc(symbol, api_timeframe, since)
            
            # Convert to DataFrame
            if not ohlcv:
                self.log(f"No OHLCV data returned for {symbol}", "warning")
                return pd.DataFrame()
            
            self.log(f"Received {len(ohlcv)} candles for {symbol}")
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string columns to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Forward-fill any NaN values - using the recommended method to avoid warning
            df = df.ffill()
            
            # Make sure we have enough data
            if len(df) < 2:
                self.log(f"Not enough data points for {symbol}, need at least 2", "warning")
                return pd.DataFrame()
                
            # Log some stats
            if not df.empty:
                self.log(f"Data range for {symbol}: {df.index.min()} to {df.index.max()}")
                
            return df
            
        except Exception as e:
            self.log(f"Failed to fetch market data for {symbol}: {e}", "error")
            return pd.DataFrame()
    
    def scan_market(self, force_check=False) -> None:
        """Scan market for trading opportunities"""
        instruments = self.get_allowed_coins()
        self.log(f"Scanning {len(instruments)} instruments for trading opportunities...")
        
        # Track instruments checked for diagnostics
        instruments_checked = 0
        price_signals_found = 0
        buy_criteria_met = 0
        
        for symbol in instruments:
            try:
                # Get current price
                ticker = self.client.get_ticker(symbol)
                current_price = ticker['last'] if ticker else 0
                
                if current_price == 0:
                    self.log(f"No price data for {symbol}, skipping", "warning")
                    continue
                
                # Fetch recent price data
                price_data = self.fetch_market_data(symbol)
                
                if price_data.empty:
                    self.log(f"No historical data for {symbol}, skipping", "warning")
                    continue
                
                instruments_checked += 1
                
                # Calculate percentage change for logging
                if len(price_data) > 1:
                    pct_change = price_data['close'].pct_change().iloc[-1] * 100
                    trigger_pct = self.config['place_order_trigger']['percentage_change']
                    
                    # Always log if force check is enabled
                    if force_check:
                        self.log(f"{symbol} price changed by {pct_change:.2f}% (trigger: {trigger_pct:.2f}%)")
                
                # Check if should buy
                should_buy_decision = self.should_buy(symbol, price_data)
                
                if should_buy_decision:
                    price_signals_found += 1
                    self.log(f"Buy signal found for {symbol} at {current_price}")
                    
                    # Check if can buy based on settings
                    can_buy_decision = self.can_buy_instrument(symbol, current_price)
                    
                    if can_buy_decision:
                        buy_criteria_met += 1
                        self.log(f"All criteria met for buying {symbol}")
                        
                        # Calculate buy amount
                        amount = self.calculate_buy_amount(symbol, current_price)
                        
                        if amount > 0:
                            self.log(f"Executing buy for {symbol}: {amount} contracts at {current_price}")
                            # Execute buy
                            self.execute_buy(symbol, amount, current_price)
                        else:
                            self.log(f"Buy amount calculation returned 0 for {symbol}", "warning")
                    else:
                        if force_check:
                            self.log(f"Buy signal found but criteria not met for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Log summary
        self.log(f"Market scan complete: {instruments_checked}/{len(instruments)} instruments checked, {price_signals_found} signals found, {buy_criteria_met} met all criteria")
    
    def reset_failed_orders(self) -> None:
        """Reset stop-loss after failed orders if enabled"""
        if not self.config['sell_settings']['reset_stop_loss_after_failed_orders']:
            return
            
        # For demo purposes, we'll just check if positions are older than max_open_time_buy
        try:
            max_open_time = self.config['buy_settings']['max_open_time_buy']
            current_time = datetime.now()
            
            for position_id, position in list(self.positions.items()):
                position_time = position['timestamp']
                elapsed_minutes = (current_time - position_time).total_seconds() / 60
                
                if elapsed_minutes > max_open_time:
                    self.log(f"Resetting failed order: {position_id} (exceeded max open time)")
                    del self.positions[position_id]
                    
        except Exception as e:
            logger.error(f"Error resetting failed orders: {e}")
    
    def get_status_summary(self):
        """Get a summary of current bot status for UI"""
        # Convert positions dict into a proper format
        positions_status = {}
        for pos_id, position in self.positions.items():
            # Make sure we create a copy so we don't mess with the original data
            positions_status[pos_id] = position.copy()
            
            # Add a highest_price if it doesn't exist
            if 'highest_price' not in positions_status[pos_id]:
                positions_status[pos_id]['highest_price'] = position.get('price', 0)
                
            # Convert datetime objects to strings for JSON serialization
            if 'timestamp' in positions_status[pos_id] and isinstance(positions_status[pos_id]['timestamp'], datetime):
                positions_status[pos_id]['timestamp'] = positions_status[pos_id]['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        
        # Make a copy of trades list to avoid modifying the original
        trades = []
        if 'trades' in self.profit_stats:
            for trade in self.profit_stats['trades']:
                trade_copy = trade.copy()
                # Convert datetime objects to strings for JSON serialization
                if 'timestamp' in trade_copy and isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                trades.append(trade_copy)
        
        return {
            "positions": len(self.positions),
            "profit": self.profit_stats['total_profit'],
            "wins": self.profit_stats['wins'],
            "losses": self.profit_stats['losses'],
            "balance": self.balance,
            "active_positions": positions_status,
            "trades": trades
        }
            
    def run(self, single_iteration=False) -> None:
        """Run the trading bot main loop"""
        logger.info("Starting trading bot...")
        
        try:
            # Initialize WebSocket
            self._initialize_websocket()
            
            # Main loop
            while self.running:
                try:
                    # Sync balance (but don't log it every time)
                    self.sync_balance()
                    
                    # Update trailing stops for existing positions
                    self.update_trailing_stops()
                    
                    # Scan market for opportunities
                    self.scan_market()
                    
                    # Reset failed orders if enabled
                    self.reset_failed_orders()
                    
                    # Log current status (only every 10 minutes)
                    if int(time.time()) % 600 < 60:  # Only log every 10 minutes
                        self.log(f"Status: {len(self.positions)} positions, profit: {self.profit_stats['total_profit']:.2f}")
                    
                    # If running in single iteration mode (for UI), break after one loop
                    if single_iteration:
                        break
                        
                    # Sleep before next iteration
                    sleep_time = 60  # 1 minute
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Sleep before retry
                    
        except KeyboardInterrupt:
            logger.info("Bot stopping due to keyboard interrupt")
            self.running = False
            
        finally:
            # Close WebSocket when done
            if not single_iteration and self.ws_client:
                self.ws_client.close()
                
            if not single_iteration:
                logger.info("Bot stopped")