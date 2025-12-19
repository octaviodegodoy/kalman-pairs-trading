"""
realtime_trading_demo.py

Demo of real-time trading setup (using paper trading).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime_trader import AlpacaBroker, RealtimeTrader
from src.pairs_strategy import PairsTradingStrategy
from src.risk_management import RiskManager
import time


def main():
    """Run real-time trading demo."""
    
    print("=" * 80)
    print("REAL-TIME TRADING DEMO (PAPER TRADING)")
    print("=" * 80)
    
    print("\n⚠️  NOTE: This demo requires valid Alpaca paper trading credentials.")
    print("Get free credentials at:  https://alpaca.markets/")
    
    # Get credentials
    api_key = input("\nEnter Alpaca API Key (or press Enter to skip): ").strip()
    
    if not api_key: 
        print("\nSkipping live trading demo.  To run:")
        print("1. Get Alpaca paper trading credentials")
        print("2. Set credentials in this script")
        print("3. Run the script again")
        return
    
    api_secret = input("Enter Alpaca API Secret:  ").strip()
    
    # Initialize components
    print("\n1. Initializing components...")
    
    broker = AlpacaBroker()
    
    strategy = PairsTradingStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=4.0,
        delta=1e-4
    )
    
    risk_manager = RiskManager(
        max_position_size=0.1,
        max_drawdown_limit=0.15
    )
    
    trader = RealtimeTrader(broker, strategy, risk_manager)
    
    # Connect to broker
    print("\n2. Connecting to Alpaca...")
    
    credentials = {
        'api_key': api_key,
        'api_secret': api_secret,
        'base_url': 'https://paper-api.alpaca.markets'
    }
    
    try:
        trader.connect(credentials)
        
        # Get account info
        account_info = broker.get_account_info()
        print(f"\n   Account Status: {account_info['status']}")
        print(f"   Equity: ${account_info['equity']: ,.2f}")
        print(f"   Cash: ${account_info['cash']: ,.2f}")
        
        # Define pairs to trade
        pairs = [('GLD', 'GDX')]
        
        print(f"\n3. Starting trading for {len(pairs)} pair(s)...")
        print("   Press Ctrl+C to stop\n")
        
        # Start trading (will run until interrupted)
        trader.start_trading(pairs, update_interval=60)
        
    except KeyboardInterrupt:
        print("\n\n4. Stopping trader...")
        trader.stop_trading()
        trader.disconnect()
        print("   Trader stopped successfully")
        
    except Exception as e: 
        print(f"\nError:  {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()