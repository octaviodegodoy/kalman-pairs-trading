"""
realtime_trader.py

Real-time trading implementation with broker integration for live pairs trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import time
import threading
import logging
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrokerInterface:
    """
    Abstract base class for broker integration.
    """
    
    def connect(self, credentials: Dict):
        """Connect to broker API."""
        raise NotImplementedError
    
    def disconnect(self):
        """Disconnect from broker."""
        raise NotImplementedError
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        raise NotImplementedError
    
    def get_positions(self) -> Dict:
        """Get current positions."""
        raise NotImplementedError
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for symbol."""
        raise NotImplementedError
    
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market'
    ) -> Dict:
        """Place order."""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str):
        """Cancel order."""
        raise NotImplementedError


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker integration. 
    """
    
    def __init__(self):
        """Initialize Alpaca broker."""
        self.api = None
        self.connected = False
        
    def connect(self, credentials:  Dict):
        """
        Connect to Alpaca API. 
        
        Args:
            credentials: Dictionary with 'api_key' and 'api_secret'
        """
        try:
            import alpaca_trade_api as tradeapi
            
            self.api = tradeapi.REST(
                credentials['api_key'],
                credentials['api_secret'],
                credentials. get('base_url', 'https://paper-api.alpaca.markets'),
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.connected = True
            
            logger.info(f"Connected to Alpaca.  Account status: {account. status}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Alpaca."""
        self.api = None
        self.connected = False
        logger.info("Disconnected from Alpaca")
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self. connected:
            raise ConnectionError("Not connected to broker")
        
        account = self.api.get_account()
        
        return {
            'equity':  float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'status': account.status
        }
    
    def get_positions(self) -> Dict:
        """Get current positions."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        positions = self.api.list_positions()
        
        position_dict = {}
        for pos in positions:
            position_dict[pos.symbol] = {
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'side': pos.side
            }
        
        return position_dict
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            quote = self.api.get_latest_trade(symbol)
            
            return {
                'symbol': symbol,
                'price': float(quote.price),
                'size':  int(quote.size),
                'timestamp': quote.timestamp
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def place_order(
        self,
        symbol: str,
        qty:  float,
        side: str,
        order_type: str = 'market',
        time_in_force: str = 'day'
    ) -> Dict:
        """Place order."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        try: 
            order = self.api. submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            
            logger.info(f"Order placed: {side} {qty} {symbol} @ {order_type}")
            
            return {
                'order_id': order.id,
                'symbol': symbol,
                'qty': float(order.qty),
                'side':  order.side,
                'type':  order.type,
                'status': order.status
            }
            
        except Exception as e: 
            logger.error(f"Error placing order: {e}")
            raise
    
    def cancel_order(self, order_id:  str):
        """Cancel order."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise


class RealtimeTrader:
    """
    Real-time pairs trading system.
    """
    
    def __init__(
        self,
        broker: BrokerInterface,
        strategy,
        risk_manager
    ):
        """
        Initialize real-time trader.
        
        Args:
            broker: Broker interface instance
            strategy: Trading strategy instance
            risk_manager: Risk manager instance
        """
        self. broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager
        
        self.is_running = False
        self. positions = {}
        self.price_history = {}
        self.spread_history = pd.Series(dtype=float)
        
        self.update_interval = 60  # seconds
        self.price_queue = Queue()
        
    def connect(self, credentials: Dict):
        """Connect to broker."""
        self.broker.connect(credentials)
        logger.info("Trader connected to broker")
    
    def disconnect(self):
        """Disconnect from broker."""
        self.broker.disconnect()
        logger.info("Trader disconnected from broker")
    
    def start_trading(
        self,
        pairs: List[Tuple[str, str]],
        update_interval: int = 60
    ):
        """
        Start live trading.
        
        Args:
            pairs: List of (asset_a, asset_b) tuples
            update_interval: Update frequency in seconds
        """
        self.pairs = pairs
        self.update_interval = update_interval
        self.is_running = True
        
        logger.info(f"Starting live trading for {len(pairs)} pair(s)")
        
        # Start price update thread
        price_thread = threading.Thread(target=self._price_update_loop)
        price_thread. daemon = True
        price_thread.start()
        
        # Start trading loop
        self._trading_loop()
    
    def stop_trading(self):
        """Stop live trading."""
        self.is_running = False
        logger.info("Stopping live trading")
    
    def _price_update_loop(self):
        """Continuously update prices."""
        while self.is_running:
            for asset_a, asset_b in self.pairs:
                try:
                    # Get quotes
                    quote_a = self.broker.get_quote(asset_a)
                    quote_b = self.broker.get_quote(asset_b)
                    
                    if quote_a and quote_b: 
                        timestamp = datetime.now()
                        
                        # Update price history
                        if asset_a not in self.price_history:
                            self. price_history[asset_a] = pd.Series(dtype=float)
                        if asset_b not in self.price_history:
                            self.price_history[asset_b] = pd.Series(dtype=float)
                        
                        self.price_history[asset_a][timestamp] = quote_a['price']
                        self.price_history[asset_b][timestamp] = quote_b['price']
                        
                        # Limit history size
                        max_history = 1000
                        if len(self.price_history[asset_a]) > max_history:
                            self. price_history[asset_a] = self.price_history[asset_a]. iloc[-max_history:]
                        if len(self.price_history[asset_b]) > max_history:
                            self. price_history[asset_b] = self.price_history[asset_b].iloc[-max_history:]
                        
                        # Put in queue for trading loop
                        self.price_queue.put({
                            'pair': (asset_a, asset_b),
                            'price_a': quote_a['price'],
                            'price_b':  quote_b['price'],
                            'timestamp': timestamp
                        })
                
                except Exception as e:
                    logger.error(f"Error updating prices for {asset_a}/{asset_b}: {e}")
            
            time.sleep(self.update_interval)
    
    def _trading_loop(self):
        """Main trading loop."""
        while self.is_running:
            try:
                # Get latest prices from queue
                if not self.price_queue.empty():
                    price_data = self.price_queue.get()
                    
                    asset_a, asset_b = price_data['pair']
                    price_a = price_data['price_a']
                    price_b = price_data['price_b']
                    
                    # Get price series
                    if len(self.price_history[asset_a]) < 50:
                        logger.info(f"Waiting for more price history...  ({len(self.price_history[asset_a])} observations)")
                        time.sleep(1)
                        continue
                    
                    prices_a = self.price_history[asset_a]
                    prices_b = self.price_history[asset_b]
                    
                    # Generate signal
                    signal, info = self.strategy.generate_signal(
                        price_a, price_b, self.spread_history
                    )
                    
                    # Update spread history
                    self.spread_history[price_data['timestamp']] = info['spread']
                    
                    # Apply risk management
                    account_info = self.broker.get_account_info()
                    current_positions = self.broker.get_positions()
                    
                    risk_check = self._check_risk_limits(
                        signal, info, account_info, current_positions
                    )
                    
                    if not risk_check['approved']:
                        logger.warning(f"Trade blocked by risk management: {risk_check['reason']}")
                        continue
                    
                    # Execute trades if signal changed
                    current_position = self.positions.get((asset_a, asset_b), 0)
                    
                    if signal != current_position:
                        self._execute_pair_trade(
                            asset_a, asset_b,
                            signal, info,
                            account_info
                        )
                        
                        self.positions[(asset_a, asset_b)] = signal
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger. info("Trading interrupted by user")
                self.stop_trading()
                break
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def _check_risk_limits(
        self,
        signal: int,
        info: Dict,
        account_info: Dict,
        current_positions: Dict
    ) -> Dict:
        """
        Check risk limits before trading.
        
        Returns:
            Dictionary with 'approved' boolean and 'reason' string
        """
        # Check account status
        if account_info['status'] != 'ACTIVE':
            return {'approved': False, 'reason':  'Account not active'}
        
        # Check available capital
        min_capital = 1000
        if account_info['cash'] < min_capital:
            return {'approved':  False, 'reason': 'Insufficient capital'}
        
        # Check maximum positions
        max_positions = 10
        if len(current_positions) >= max_positions and signal != 0:
            return {'approved': False, 'reason': 'Maximum positions reached'}
        
        # Check z-score extreme
        if abs(info. get('zscore', 0)) > 5:
            return {'approved': False, 'reason': 'Z-score too extreme'}
        
        return {'approved': True, 'reason': 'All checks passed'}
    
    def _execute_pair_trade(
        self,
        asset_a: str,
        asset_b: str,
        signal: int,
        info: Dict,
        account_info: Dict
    ):
        """
        Execute pair trade.
        
        Args:
            asset_a, asset_b: Asset symbols
            signal: Trading signal (1, -1, 0)
            info: Signal information including beta
            account_info: Account information
        """
        try:
            # Get current prices
            quote_a = self. broker.get_quote(asset_a)
            quote_b = self.broker.get_quote(asset_b)
            
            price_a = quote_a['price']
            price_b = quote_b['price']
            beta = info['beta']
            
            # Calculate position sizes
            allocation = account_info['portfolio_value'] * 0.1  # 10% per pair
            
            if signal == 0: 
                # Close positions
                current_positions = self.broker.get_positions()
                
                if asset_a in current_positions: 
                    qty_a = abs(current_positions[asset_a]['qty'])
                    side_a = 'sell' if current_positions[asset_a]['side'] == 'long' else 'buy'
                    self. broker.place_order(asset_a, qty_a, side_a)
                
                if asset_b in current_positions:
                    qty_b = abs(current_positions[asset_b]['qty'])
                    side_b = 'sell' if current_positions[asset_b]['side'] == 'long' else 'buy'
                    self.broker.place_order(asset_b, qty_b, side_b)
                
                logger.info(f"Closed pair position: {asset_a}/{asset_b}")
            
            elif signal == 1:
                # Long spread:  buy A, sell B
                qty_a = int(allocation / price_a)
                qty_b = int((allocation * beta) / price_b)
                
                self.broker.place_order(asset_a, qty_a, 'buy')
                self.broker.place_order(asset_b, qty_b, 'sell')
                
                logger.info(f"Opened LONG spread: Buy {qty_a} {asset_a}, Sell {qty_b} {asset_b}")
            
            elif signal == -1:
                # Short spread: sell A, buy B
                qty_a = int(allocation / price_a)
                qty_b = int((allocation * beta) / price_b)
                
                self. broker.place_order(asset_a, qty_a, 'sell')
                self.broker. place_order(asset_b, qty_b, 'buy')
                
                logger.info(f"Opened SHORT spread:  Sell {qty_a} {asset_a}, Buy {qty_b} {asset_b}")
        
        except Exception as e: 
            logger.error(f"Error executing pair trade: {e}")
            raise
    
    def get_live_metrics(self) -> Dict:
        """Get current trading metrics."""
        account_info = self.broker.get_account_info()
        positions = self.broker.get_positions()
        
        # Calculate total P&L
        total_unrealized_pl = sum(
            pos['unrealized_pl'] for pos in positions.values()
        )
        
        return {
            'timestamp': datetime.now(),
            'account_equity': account_info['equity'],
            'cash': account_info['cash'],
            'num_positions': len(positions),
            'total_unrealized_pl': total_unrealized_pl,
            'positions': positions
        }
    
    def print_status(self):
        """Print current trading status."""
        metrics = self.get_live_metrics()
        
        print("\n" + "=" * 80)
        print("LIVE TRADING STATUS")
        print("=" * 80)
        print(f"Time: {metrics['timestamp']}")
        print(f"Account Equity: ${metrics['account_equity']: ,.2f}")
        print(f"Cash: ${metrics['cash']:,.2f}")
        print(f"Active Positions: {metrics['num_positions']}")
        print(f"Unrealized P&L: ${metrics['total_unrealized_pl']: ,.2f}")
        
        if metrics['positions']:
            print("\nPositions:")
            for symbol, pos in metrics['positions'].items():
                print(f"  {symbol}: {pos['qty']: +. 2f} @ ${pos['avg_entry_price']:.2f} "
                      f"(Current: ${pos['current_price']:.2f}, "
                      f"P&L: ${pos['unrealized_pl']: +.2f})")
        
        print("=" * 80)


# Example usage
if __name__ == "__main__": 
    from . pairs_strategy import PairsTradingStrategy
    from .risk_management import RiskManager
    
    # Initialize components
    broker = AlpacaBroker()
    
    strategy = PairsTradingStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        delta=1e-4
    )
    
    risk_manager = RiskManager()
    
    # Create trader
    trader = RealtimeTrader(broker, strategy, risk_manager)
    
    # Connect (use paper trading credentials)
    credentials = {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'base_url': 'https://paper-api.alpaca.markets'  # Paper trading
    }
    
    try:
        trader.connect(credentials)
        
        # Start trading
        pairs = [('GLD', 'GDX')]
        trader.start_trading(pairs, update_interval=60)
        
    except KeyboardInterrupt:
        print("\nStopping trader...")
        trader.stop_trading()
        trader.disconnect()