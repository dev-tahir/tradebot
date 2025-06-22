import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Tuple
import json
import threading
import os
import sys
from collections import defaultdict
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
import sqlite3

class TradingDatabase:
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for trade tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                amount REAL,
                entry_price REAL,
                exit_price REAL,
                pnl_pct REAL,
                pnl_usdt REAL,
                exit_reason TEXT,
                duration_minutes INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl_usdt REAL,
                total_pnl_pct REAL,
                max_drawdown REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data: Dict):
        """Save completed trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, side, amount, entry_price, 
                              exit_price, pnl_pct, pnl_usdt, exit_reason, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['timestamp'],
            trade_data['symbol'],
            trade_data['side'],
            trade_data['amount'],
            trade_data['entry_price'],
            trade_data['exit_price'],
            trade_data['pnl_pct'],
            trade_data['pnl_usdt'],
            trade_data['exit_reason'],
            trade_data['duration_minutes']
        ))
        
        conn.commit()
        conn.close()
    
    def get_trades(self, days: int = 30) -> pd.DataFrame:
        """Get trades from last N days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM trades 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

class BinanceTradingBot:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialize the trading bot with CLI support"""
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': testnet,
            'enableRateLimit': True,
        })
        
        # Trading parameters
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        self.timeframe = '1m'  # Changed from '5m'
        self.risk_per_trade = 0.35
        self.max_positions = 3
        
        # Strategy parameters
        self.rsi_oversold = 40  # Changed from 30
        self.rsi_overbought = 60  # Changed from 70
        self.bb_period = 20
        self.volume_threshold = 1.2  # Changed from 1.5
        
        # Tracking
        self.positions = {}
        self.balance = {}
        self.initial_balance = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.daily_pnl = defaultdict(float)
        self.hourly_pnl = defaultdict(float)
        self.max_drawdown = 0
        self.peak_balance = 0
        
        # Database
        self.db = TradingDatabase()
        
        # Console for rich output
        self.console = Console()
        
        # Control flags
        self.running = False
        self.cli_mode = False
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_balance(self) -> Dict:
        """Get account balance with detailed crypto holdings"""
        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance
            
            # Get all non-zero balances
            self.crypto_holdings = {}
            
            for currency, amounts in balance.items():
                if currency in ['info', 'free', 'used', 'total']:  # Skip metadata
                    continue
                
                # Handle different balance data structures
                if isinstance(amounts, dict):
                    # Standard CCXT format
                    total = amounts.get('total', 0)
                    free = amounts.get('free', 0) 
                    used = amounts.get('used', 0)
                elif isinstance(amounts, (int, float)):
                    # Simple number format - treat as total available
                    total = float(amounts)
                    free = float(amounts)
                    used = 0
                elif isinstance(amounts, str):
                    # String number format
                    try:
                        total = float(amounts)
                        free = float(amounts)
                        used = 0
                    except ValueError:
                        continue
                else:
                    self.logger.warning(f"Unknown balance format for {currency}: {type(amounts)} - {amounts}")
                    continue
                
                # Only include currencies with balance > 0.001 (to avoid dust)
                if total > 0.001:
                    self.crypto_holdings[currency] = {
                        'total': total,
                        'free': free,
                        'used': used
                    }
            
            # Calculate total portfolio value in USDT
            self.portfolio_value_usdt = 0
            
            for currency, amounts in self.crypto_holdings.items():
                if currency == 'USDT':
                    self.portfolio_value_usdt += amounts['total']
                else:
                    try:
                        # Get current price in USDT
                        symbol = f"{currency}/USDT"
                        markets = self.exchange.load_markets()
                        if symbol in markets:
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = ticker['last']
                            value_usdt = amounts['total'] * price
                            self.portfolio_value_usdt += value_usdt
                            
                            # Store price for display
                            self.crypto_holdings[currency]['price_usdt'] = price
                            self.crypto_holdings[currency]['value_usdt'] = value_usdt
                    except Exception as price_error:
                        # If can't get price, set to 0
                        self.logger.warning(f"Could not get price for {currency}: {price_error}")
                        self.crypto_holdings[currency]['price_usdt'] = 0
                        self.crypto_holdings[currency]['value_usdt'] = 0
            
            # Track peak balance for drawdown calculation
            if self.portfolio_value_usdt > self.peak_balance:
                self.peak_balance = self.portfolio_value_usdt
            
            # Calculate drawdown
            if self.peak_balance > 0:
                drawdown = (self.peak_balance - self.portfolio_value_usdt) / self.peak_balance * 100
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
            
            return balance
            
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {}
        
    def get_market_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data and calculate indicators"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['bb_upper'] = ta.volatility.BollingerBands(df['close'], window=self.bb_period).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close'], window=self.bb_period).bollinger_lband()
            df['bb_middle'] = ta.volatility.BollingerBands(df['close'], window=self.bb_period).bollinger_mavg()
            df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_change_pct'] = df['close'].pct_change() * 100
            
            return df
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()

    def identify_opportunity(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Identify trading opportunities using multiple strategies"""
        if len(df) < 30:
            return 'hold', 0
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        confidence = 0
        
        # Strategy 1: RSI Mean Reversion with Volume Confirmation
        if (latest['rsi'] < self.rsi_oversold and 
            latest['volume_ratio'] > self.volume_threshold and
            latest['close'] < latest['bb_lower']):
            signals.append('buy')
            confidence += 0.3
            
        elif (latest['rsi'] > self.rsi_overbought and 
              latest['volume_ratio'] > self.volume_threshold and
              latest['close'] > latest['bb_upper']):
            signals.append('sell')
            confidence += 0.3
        
        # Strategy 2: EMA Crossover with Momentum
        if (latest['ema_fast'] > latest['ema_slow'] and 
            prev['ema_fast'] <= prev['ema_slow'] and
            latest['volume_ratio'] > 1.2):
            signals.append('buy')
            confidence += 0.25
            
        elif (latest['ema_fast'] < latest['ema_slow'] and 
              prev['ema_fast'] >= prev['ema_slow'] and
              latest['volume_ratio'] > 1.2):
            signals.append('sell')
            confidence += 0.25
        
        # Strategy 3: Breakout Detection
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        
        if (latest['close'] > recent_high * 1.002 and
            latest['volume_ratio'] > 2.0):
            signals.append('buy')
            confidence += 0.2
            
        elif (latest['close'] < recent_low * 0.998 and
              latest['volume_ratio'] > 2.0):
            signals.append('sell')
            confidence += 0.2
        
        # Strategy 4: Quick Scalping on Strong Momentum
        if abs(latest['price_change_pct']) > 1.0:
            if (latest['price_change_pct'] > 0 and 
                latest['volume_ratio'] > 2.5):
                signals.append('buy')
                confidence += 0.25
            elif (latest['price_change_pct'] < 0 and 
                  latest['volume_ratio'] > 2.5):
                signals.append('sell')
                confidence += 0.25
        
        # Determine final signal
        if len(signals) == 0:
            return 'hold', 0
        
        buy_votes = signals.count('buy')
        sell_votes = signals.count('sell')
        
        if buy_votes > sell_votes:
            return 'buy', confidence
        elif sell_votes > buy_votes:
            return 'sell', confidence
        else:
            return 'hold', 0

    def calculate_position_size(self, symbol: str, signal: str, price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            balance = self.get_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < 10:
                return 0
            
            risk_amount = usdt_balance * self.risk_per_trade
            position_size = risk_amount / price
            
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    def place_order(self, symbol: str, side: str, amount: float, price: float = None) -> Dict:
        """Place market or limit order"""
        try:
            if side == 'buy':
                if price:
                    order = self.exchange.create_limit_buy_order(symbol, amount, price)
                else:
                    order = self.exchange.create_market_buy_order(symbol, amount)
            else:
                if price:
                    order = self.exchange.create_limit_sell_order(symbol, amount, price)
                else:
                    order = self.exchange.create_market_sell_order(symbol, amount)
            
            self.logger.info(f"Order placed: {side} {amount:.6f} {symbol} at {price or 'market'}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {}

    def manage_position(self, symbol: str, df: pd.DataFrame):
        """Manage existing positions with dynamic stop-loss and take-profit"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = df.iloc[-1]['close']
        entry_price = position['entry_price']
        side = position['side']
        amount = position['amount']
        
        # Calculate P&L
        if side == 'buy':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_usdt = (current_price - entry_price) * amount
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            pnl_usdt = (entry_price - current_price) * amount
        
        # Update position with current P&L
        position['current_pnl_pct'] = pnl_pct
        position['current_pnl_usdt'] = pnl_usdt
        position['current_price'] = current_price
        
        # Dynamic exit conditions
        should_exit = False
        exit_reason = ""
        
        # Take profit at 0.5% 0.2
        if pnl_pct >= 0.005:
            should_exit = True
            exit_reason = "Take profit"

        # Stop loss at -0.3%
        elif pnl_pct <= -0.3:
            should_exit = True
            exit_reason = "Stop loss"

        # Time-based exit at 5 minutes
        elif time.time() - position['timestamp'] > 900000:
            should_exit = True
            exit_reason = "Time-based exit"
        
        # Signal reversal
        signal, confidence = self.identify_opportunity(df)
        if ((side == 'buy' and signal == 'sell' and confidence > 0.3) or 
            (side == 'sell' and signal == 'buy' and confidence > 0.3)):
            should_exit = True
            exit_reason = "Signal reversal"
    
        
        if should_exit:
            # Close position
            close_side = 'sell' if side == 'buy' else 'buy'
            order = self.place_order(symbol, close_side, amount)
            
            if order:
                # Update statistics
                self.total_trades += 1
                if pnl_pct > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.total_pnl += pnl_usdt
                
                # Track daily and hourly P&L
                now = datetime.now()
                date_key = now.strftime('%Y-%m-%d')
                hour_key = now.strftime('%Y-%m-%d %H:00')
                self.daily_pnl[date_key] += pnl_usdt
                self.hourly_pnl[hour_key] += pnl_usdt
                
                # Save trade to database
                duration_minutes = (time.time() - position['timestamp']) / 60
                trade_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usdt': pnl_usdt,
                    'exit_reason': exit_reason,
                    'duration_minutes': int(duration_minutes)
                }
                self.db.save_trade(trade_data)
                
                self.logger.info(f"Position closed: {symbol} - {exit_reason} - P&L: {pnl_pct:.2f}% (${pnl_usdt:.2f})")
                del self.positions[symbol]

    def create_dashboard(self) -> Layout:
        """Create rich dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="performance", ratio=2),
            Layout(name="positions", ratio=1)
        )
        
        layout["right"].split_column(
            Layout(name="trades", ratio=2),
            Layout(name="market", ratio=1)
        )
        
        return layout

    def update_dashboard(self, layout: Layout):
        """Update dashboard with current data including crypto holdings"""
        try:
            # Get fresh balance data
            balance = self.get_balance()
            
            if not balance:
                layout["header"].update(Panel("❌ Error fetching balance data", style="bold red"))
                return
            
            # Header with portfolio overview
            if self.initial_balance == 0:
                self.initial_balance = self.portfolio_value_usdt
            
            total_return_pct = ((self.portfolio_value_usdt - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
            
            runtime = time.time() - self.start_time
            runtime_str = str(timedelta(seconds=int(runtime)))
            
            # Trading mode indicator
            mode_indicator = "🔴 LIVE TRADING" if not self.exchange.sandbox else "🟡 TESTNET"
            
            header_text = f"🤖 Binance Trading Bot | {mode_indicator} | Runtime: {runtime_str} | Portfolio: ${self.portfolio_value_usdt:.2f} | Return: {total_return_pct:+.2f}%"
            header_color = "bold green" if total_return_pct >= 0 else "bold red"
            layout["header"].update(Panel(header_text, style=header_color))
            
            # Performance metrics
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            usdt_balance = self.crypto_holdings.get('USDT', {})
            free_usdt = usdt_balance.get('free', 0)
            used_usdt = usdt_balance.get('used', 0)
            
            performance_table = Table(title="Performance & Balance", box=box.ROUNDED)
            performance_table.add_column("Metric", style="cyan")
            performance_table.add_column("Value", style="magenta")
            
            performance_table.add_row("Portfolio Value", f"${self.portfolio_value_usdt:.2f}")
            performance_table.add_row("Free USDT", f"${free_usdt:.2f}")
            performance_table.add_row("Used USDT", f"${used_usdt:.2f}")
            performance_table.add_row("Initial Value", f"${self.initial_balance:.2f}")
            performance_table.add_row("Total P&L", f"${self.total_pnl:+.2f}")
            performance_table.add_row("Return %", f"{total_return_pct:+.2f}%")
            performance_table.add_row("Total Trades", str(self.total_trades))
            performance_table.add_row("Win Rate", f"{win_rate:.1f}%")
            performance_table.add_row("Max Drawdown", f"{self.max_drawdown:.2f}%")
            
            layout["performance"].update(performance_table)
            
            # Crypto Holdings
            holdings_table = Table(title="Crypto Holdings", box=box.ROUNDED)
            holdings_table.add_column("Asset", style="cyan")
            holdings_table.add_column("Total", style="yellow")
            holdings_table.add_column("Free", style="green")
            holdings_table.add_column("Used", style="red")
            holdings_table.add_column("Price (USDT)", style="blue")
            holdings_table.add_column("Value (USDT)", style="magenta")
            holdings_table.add_column("% of Portfolio", style="white")
            
            # Sort by value descending
            sorted_holdings = sorted(
                self.crypto_holdings.items(), 
                key=lambda x: x[1].get('value_usdt', x[1]['total'] if x[0] == 'USDT' else 0), 
                reverse=True
            )
            
            for currency, amounts in sorted_holdings:
                total = amounts['total']
                free = amounts['free']
                used = amounts['used']
                
                if currency == 'USDT':
                    price = 1.0
                    value_usdt = total
                else:
                    price = amounts.get('price_usdt', 0)
                    value_usdt = amounts.get('value_usdt', 0)
                
                portfolio_pct = (value_usdt / self.portfolio_value_usdt * 100) if self.portfolio_value_usdt > 0 else 0
                
                # Color coding based on usage
                if used > 0:
                    asset_style = "[yellow]" + currency + "[/yellow]"  # Yellow if partially used
                else:
                    asset_style = currency
                
                holdings_table.add_row(
                    asset_style,
                    f"{total:.6f}",
                    f"{free:.6f}",
                    f"{used:.6f}" if used > 0 else "0",
                    f"${price:.4f}" if price > 0 else "N/A",
                    f"${value_usdt:.2f}",
                    f"{portfolio_pct:.1f}%"
                )
            
            if not self.crypto_holdings:
                holdings_table.add_row("No holdings found", "", "", "", "", "", "")
            
            layout["positions"].update(holdings_table)
            
            # Active Trading Positions
            positions_table = Table(title="Active Trading Positions", box=box.ROUNDED)
            positions_table.add_column("Symbol", style="cyan")
            positions_table.add_column("Side", style="green")
            positions_table.add_column("Amount", style="yellow")
            positions_table.add_column("Entry Price", style="blue")
            positions_table.add_column("Current Price", style="blue")
            positions_table.add_column("P&L %", style="magenta")
            positions_table.add_column("P&L $", style="magenta")
            positions_table.add_column("Duration", style="white")
            
            for symbol, position in self.positions.items():
                pnl_pct = position.get('current_pnl_pct', 0)
                pnl_usdt = position.get('current_pnl_usdt', 0)
                current_price = position.get('current_price', position['entry_price'])
                
                # Calculate duration
                duration_seconds = time.time() - position['timestamp']
                duration_minutes = int(duration_seconds / 60)
                duration_str = f"{duration_minutes}m"
                
                pnl_color = "green" if pnl_pct >= 0 else "red"
                
                positions_table.add_row(
                    symbol,
                    position['side'].upper(),
                    f"{position['amount']:.6f}",
                    f"{position['entry_price']:.4f}",
                    f"{current_price:.4f}",
                    f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]",
                    f"[{pnl_color}]${pnl_usdt:+.2f}[/{pnl_color}]",
                    duration_str
                )
            
            if not self.positions:
                positions_table.add_row("No active positions", "", "", "", "", "", "", "")
            
            # Add positions table as a separate section or combine with holdings
            combined_table = Table(title="Holdings & Positions", box=box.ROUNDED)
            combined_table.add_column("Type", style="white")
            combined_table.add_column("Asset/Symbol", style="cyan")
            combined_table.add_column("Amount/Side", style="yellow")
            combined_table.add_column("Value/P&L", style="magenta")
            combined_table.add_column("Status", style="green")
            
            # Add holdings
            for currency, amounts in sorted_holdings[:5]:  # Top 5 holdings
                value_usdt = amounts.get('value_usdt', amounts['total'] if currency == 'USDT' else 0)
                portfolio_pct = (value_usdt / self.portfolio_value_usdt * 100) if self.portfolio_value_usdt > 0 else 0
                
                status = "Available" if amounts['used'] == 0 else f"{amounts['used']:.6f} in use"
                
                combined_table.add_row(
                    "💰 Holding",
                    currency,
                    f"{amounts['total']:.6f}",
                    f"${value_usdt:.2f} ({portfolio_pct:.1f}%)",
                    status
                )
            
            # Add active positions
            for symbol, position in self.positions.items():
                pnl_usdt = position.get('current_pnl_usdt', 0)
                pnl_pct = position.get('current_pnl_pct', 0)
                pnl_color = "green" if pnl_pct >= 0 else "red"
                
                combined_table.add_row(
                    "📈 Position",
                    symbol,
                    f"{position['side'].upper()}",
                    f"[{pnl_color}]${pnl_usdt:+.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]",
                    "Active"
                )
            
            layout["positions"].update(combined_table)
            
            # Recent trades (same as before)
            trades_df = self.db.get_trades(days=1)
            trades_table = Table(title="Recent Trades (Today)", box=box.ROUNDED)
            trades_table.add_column("Time", style="cyan")
            trades_table.add_column("Symbol", style="yellow")
            trades_table.add_column("Side", style="green")
            trades_table.add_column("Amount", style="blue")
            trades_table.add_column("P&L %", style="magenta")
            trades_table.add_column("P&L $", style="magenta")
            trades_table.add_column("Reason", style="white")
            
            for _, trade in trades_df.head(8).iterrows():
                pnl_color = "green" if trade['pnl_pct'] >= 0 else "red"
                timestamp = pd.to_datetime(trade['timestamp']).strftime('%H:%M:%S')
                
                trades_table.add_row(
                    timestamp,
                    trade['symbol'],
                    trade['side'].upper(),
                    f"{trade['amount']:.6f}",
                    f"[{pnl_color}]{trade['pnl_pct']:+.2f}%[/{pnl_color}]",
                    f"[{pnl_color}]${trade['pnl_usdt']:+.2f}[/{pnl_color}]",
                    trade['exit_reason']
                )
            
            if trades_df.empty:
                trades_table.add_row("No trades today", "", "", "", "", "", "")
            
            layout["trades"].update(trades_table)
            
            # Market overview with more detailed info
            market_table = Table(title="Market Overview & Signals", box=box.ROUNDED)
            market_table.add_column("Symbol", style="cyan")
            market_table.add_column("Price", style="yellow")
            market_table.add_column("24h Change", style="magenta")
            market_table.add_column("Volume", style="blue")
            market_table.add_column("Signal", style="green")
            market_table.add_column("Confidence", style="white")
            market_table.add_column("Action", style="red")
            
            for symbol in self.symbols:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    df = self.get_market_data(symbol, limit=50)
                    
                    if not df.empty:
                        signal, confidence = self.identify_opportunity(df)
                        latest = df.iloc[-1]
                        
                        change_24h = ticker.get('percentage', 0)
                        change_color = "green" if change_24h >= 0 else "red"
                        
                        signal_color = "green" if signal == 'buy' else "red" if signal == 'sell' else "yellow"
                        
                        # Determine action
                        if symbol in self.positions:
                            action = "🔄 Managing"
                        elif len(self.positions) >= self.max_positions:
                            action = "⏸️ Max positions"
                        elif signal in ['buy', 'sell'] and confidence > 0.28:
                            action = f"🎯 Ready to {signal.upper()}"
                        else:
                            action = "👀 Watching"
                        
                        # Volume indicator
                        volume_status = "🔥" if latest['volume_ratio'] > 2 else "📈" if latest['volume_ratio'] > 1.5 else "📊"
                        
                        market_table.add_row(
                            symbol,
                            f"${ticker['last']:.4f}",
                            f"[{change_color}]{change_24h:+.2f}%[/{change_color}]",
                            f"{volume_status} {latest['volume_ratio']:.1f}x",
                            f"[{signal_color}]{signal.upper()}[/{signal_color}]",
                            f"{confidence:.2f}",
                            action
                        )
                except Exception as e:
                    market_table.add_row(symbol, "Error", "Error", "Error", "Error", "Error", str(e)[:20])
            
            layout["market"].update(market_table)
            
            # Footer with trading status
            active_positions = len(self.positions)
            trading_status = "🟢 ACTIVE" if self.running else "🔴 STOPPED"
            mode_text = "LIVE TRADING" if not self.exchange.sandbox else "TESTNET MODE"
            
            footer_text = f"{trading_status} | {mode_text} | Positions: {active_positions}/{self.max_positions} | Free USDT: ${free_usdt:.2f} | Last Update: {datetime.now().strftime('%H:%M:%S')} | Press 'q' to quit"
            footer_color = "bold green" if self.running else "bold red"
            layout["footer"].update(Panel(footer_text, style=footer_color))
            
        except Exception as e:
            error_msg = f"❌ Dashboard Error: {str(e)}"
            layout["header"].update(Panel(error_msg, style="bold red"))
            self.logger.error(f"Dashboard update error: {e}")

    def run_cli_dashboard(self):
        """Run the CLI dashboard"""
        layout = self.create_dashboard()
        
        with Live(layout, refresh_per_second=1, screen=True):
            while self.cli_mode:
                self.update_dashboard(layout)
                time.sleep(1)

    def run_strategy(self):
        """Main trading loop with enhanced debugging"""
        self.running = True
        self.logger.info("Starting trading bot...")
        
        # Initialize balance
        self.get_balance()
        
        while self.running:
            try:
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    # Get market data
                    df = self.get_market_data(symbol)
                    if df.empty:
                        continue
                    
                    # Manage existing positions
                    if symbol in self.positions:
                        self.manage_position(symbol, df)
                        continue
                    
                    # Skip if maximum positions reached
                    if len(self.positions) >= self.max_positions:
                        self.logger.info(f"Skipping {symbol}: Max positions reached ({len(self.positions)}/{self.max_positions})")
                        continue
                    
                    # Look for new opportunities
                    signal, confidence = self.identify_opportunity(df)
                    
                    # Enhanced logging for debugging
                    if signal in ['buy', 'sell']:
                        current_price = df.iloc[-1]['close']
                        amount = self.calculate_position_size(symbol, signal, current_price)
                        balance = self.get_balance()
                        usdt_balance = balance.get('USDT', {}).get('free', 0)
                        
                        self.logger.info(f"SIGNAL DETECTED: {symbol}")
                        self.logger.info(f"  Signal: {signal}")
                        self.logger.info(f"  Confidence: {confidence:.4f} (Required: >0.3)")
                        self.logger.info(f"  Current Price: ${current_price:.4f}")
                        self.logger.info(f"  USDT Balance: ${usdt_balance:.2f}")
                        self.logger.info(f"  Calculated Amount: {amount:.6f}")
                        self.logger.info(f"  Active Positions: {len(self.positions)}/{self.max_positions}")
                        
                        # Check each condition
                        if confidence <= 0.28:
                            self.logger.warning(f"  ❌ TRADE BLOCKED: Confidence too low ({confidence:.4f} <= 0.28)")
                            continue
                        
                        if usdt_balance < 10:
                            self.logger.warning(f"  ❌ TRADE BLOCKED: Insufficient USDT balance (${usdt_balance:.2f} < $10)")
                            continue
                        
                        if amount <= 0:
                            self.logger.warning(f"  ❌ TRADE BLOCKED: Invalid position size ({amount:.6f})")
                            continue
                        
                        # Check minimum order size for the symbol
                        try:
                            markets = self.exchange.load_markets()
                            market_info = markets.get(symbol, {})
                            min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0)
                            min_cost = market_info.get('limits', {}).get('cost', {}).get('min', 0)
                            
                            order_value = amount * current_price
                            
                            self.logger.info(f"  Order Value: ${order_value:.2f}")
                            self.logger.info(f"  Min Amount: {min_amount}")
                            self.logger.info(f"  Min Cost: ${min_cost}")
                            
                            if amount < min_amount:
                                self.logger.warning(f"  ❌ TRADE BLOCKED: Amount too small ({amount:.6f} < {min_amount})")
                                continue
                                
                            if order_value < min_cost:
                                self.logger.warning(f"  ❌ TRADE BLOCKED: Order value too small (${order_value:.2f} < ${min_cost})")
                                continue
                                
                        except Exception as e:
                            self.logger.error(f"  Error checking market limits: {e}")
                        
                        # If we get here, all conditions are met
                        self.logger.info(f"  ✅ ALL CONDITIONS MET - PLACING ORDER")
                        
                        # Place order
                        order = self.place_order(symbol, signal, amount)
                        
                        if order:
                            # Track position
                            self.positions[symbol] = {
                                'side': signal,
                                'amount': amount,
                                'entry_price': current_price,
                                'timestamp': time.time(),
                                'order_id': order.get('id'),
                                'current_pnl_pct': 0,
                                'current_pnl_usdt': 0,
                                'current_price': current_price
                            }
                            
                            self.logger.info(f"✅ NEW POSITION: {signal} {amount:.6f} {symbol} at {current_price:.4f} (Confidence: {confidence:.2f})")
                        else:
                            self.logger.error(f"❌ ORDER FAILED for {symbol}")
                    
                    # Wait before next iteration
                    if self.running:
                        time.sleep(30)
                    
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                if self.running:
                    time.sleep(60)

    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.cli_mode = False

def main():
    parser = argparse.ArgumentParser(description='Binance Trading Bot with CLI Dashboard')
    parser.add_argument('--api-key', required=True, help='Binance API Key')
    parser.add_argument('--api-secret', required=True, help='Binance API Secret')
    parser.add_argument('--testnet', action='store_true', help='Use Binance testnet')
    parser.add_argument('--dashboard-only', action='store_true', help='Run dashboard only (no trading)')
    parser.add_argument('--show-stats', action='store_true', help='Show trading statistics and exit')
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = BinanceTradingBot(args.api_key, args.api_secret, args.testnet)
    
    if args.show_stats:
        # Show statistics and exit
        trades_df = bot.db.get_trades(days=30)
        
        if not trades_df.empty:
            console = Console()
            
            # Summary statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['pnl_usdt'].sum()
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            stats_table = Table(title="Trading Statistics (Last 30 Days)", box=box.ROUNDED)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="magenta")
            
            stats_table.add_row("Total Trades", str(total_trades))
            stats_table.add_row("Winning Trades", str(winning_trades))
            stats_table.add_row("Losing Trades", str(losing_trades))
            stats_table.add_row("Win Rate", f"{win_rate:.2f}%")
            stats_table.add_row("Total P&L", f"${total_pnl:.2f}")
            stats_table.add_row("Avg P&L per Trade", f"${avg_pnl_per_trade:.2f}")
            stats_table.add_row("Best Trade", f"${trades_df['pnl_usdt'].max():.2f}")
            stats_table.add_row("Worst Trade", f"${trades_df['pnl_usdt'].min():.2f}")
            
            console.print(stats_table)
            
            # Daily breakdown
            daily_stats = trades_df.groupby(trades_df['timestamp'].str[:10]).agg({
                'pnl_usdt': ['sum', 'count'],
                'pnl_pct': 'mean'
            }).round(2)
            
            daily_table = Table(title="Daily Performance", box=box.ROUNDED)
            daily_table.add_column("Date", style="cyan")
            daily_table.add_column("Trades", style="yellow")
            daily_table.add_column("P&L ($)", style="magenta")
            daily_table.add_column("Avg P&L (%)", style="green")
            
            for date, row in daily_stats.iterrows():
                pnl_color = "green" if row[('pnl_usdt', 'sum')] >= 0 else "red"
                daily_table.add_row(
                    date,
                    str(int(row[('pnl_usdt', 'count')])),
                    f"[{pnl_color}]${row[('pnl_usdt', 'sum')]:+.2f}[/{pnl_color}]",
                    f"{row[('pnl_pct', 'mean')]:+.2f}%"
                )
            
            console.print(daily_table)
        else:
            print("No trading data found.")
        
        return
    
    try:
        if args.dashboard_only:
            # Run dashboard only
            bot.cli_mode = True
            bot.run_cli_dashboard()
        else:
            # Start trading bot in background thread
            trading_thread = threading.Thread(target=bot.run_strategy)
            trading_thread.daemon = True
            trading_thread.start()
            
            # Run CLI dashboard
            bot.cli_mode = True
            
            print("Starting Binance Trading Bot with Live Dashboard...")
            print("Press Ctrl+C to stop")
            time.sleep(2)
            try:
                bot.run_cli_dashboard()
            except KeyboardInterrupt:
                print("\nShutting down...")
                bot.stop()
                trading_thread.join(timeout=5)
                
    except KeyboardInterrupt:
        print("\nShutting down...")
        bot.stop()
    except Exception as e:
        print(f"Error: {e}")
        bot.stop()

class TradingBotCLI:
    """Extended CLI commands for the trading bot"""
    
    def __init__(self, bot: BinanceTradingBot):
        self.bot = bot
        self.console = Console()
    
    def show_portfolio_analysis(self):
        """Show detailed portfolio analysis"""
        trades_df = self.bot.db.get_trades(days=30)
        
        if trades_df.empty:
            self.console.print("[red]No trading data available[/red]")
            return
        
        # Portfolio metrics
        total_pnl = trades_df['pnl_usdt'].sum()
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        max_consecutive_losses = self._calculate_max_consecutive_losses(trades_df)
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl_usdt'] > 0]['pnl_usdt'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_usdt'] < 0]['pnl_usdt'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Create analysis table
        analysis_table = Table(title="Portfolio Analysis", box=box.ROUNDED)
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="magenta")
        analysis_table.add_column("Benchmark", style="yellow")
        analysis_table.add_column("Status", style="green")
        
        # Add metrics with benchmarks
        metrics = [
            ("Win Rate", f"{winning_trades/total_trades*100:.1f}%", ">60%", "✅" if winning_trades/total_trades > 0.6 else "⚠️"),
            ("Profit Factor", f"{profit_factor:.2f}", ">1.5", "✅" if profit_factor > 1.5 else "⚠️"),
            ("Sharpe Ratio", f"{sharpe_ratio:.2f}", ">1.0", "✅" if sharpe_ratio > 1.0 else "⚠️"),
            ("Max Consecutive Losses", str(max_consecutive_losses), "<5", "✅" if max_consecutive_losses < 5 else "⚠️"),
            ("Avg Trade Duration", f"{trades_df['duration_minutes'].mean():.1f} min", "<60 min", "✅" if trades_df['duration_minutes'].mean() < 60 else "⚠️"),
            ("Total P&L", f"${total_pnl:.2f}", ">0", "✅" if total_pnl > 0 else "❌"),
        ]
        
        for metric, value, benchmark, status in metrics:
            analysis_table.add_row(metric, value, benchmark, status)
        
        self.console.print(analysis_table)
        
        # Symbol performance breakdown
        symbol_stats = trades_df.groupby('symbol').agg({
            'pnl_usdt': ['sum', 'count', 'mean'],
            'pnl_pct': 'mean'
        }).round(2)
        
        symbol_table = Table(title="Performance by Symbol", box=box.ROUNDED)
        symbol_table.add_column("Symbol", style="cyan")
        symbol_table.add_column("Trades", style="yellow")
        symbol_table.add_column("Total P&L", style="magenta")
        symbol_table.add_column("Avg P&L", style="green")
        symbol_table.add_column("Avg %", style="blue")
        
        for symbol, row in symbol_stats.iterrows():
            pnl_color = "green" if row[('pnl_usdt', 'sum')] >= 0 else "red"
            symbol_table.add_row(
                symbol,
                str(int(row[('pnl_usdt', 'count')])),
                f"[{pnl_color}]${row[('pnl_usdt', 'sum')]:+.2f}[/{pnl_color}]",
                f"[{pnl_color}]${row[('pnl_usdt', 'mean')]:+.2f}[/{pnl_color}]",
                f"{row[('pnl_pct', 'mean')]:+.2f}%"
            )
        
        self.console.print(symbol_table)
    
    def _calculate_max_consecutive_losses(self, trades_df):
        """Calculate maximum consecutive losses"""
        consecutive_losses = 0
        max_consecutive = 0
        
        for _, trade in trades_df.iterrows():
            if trade['pnl_usdt'] < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def show_risk_report(self):
        """Show risk management report"""
        trades_df = self.bot.db.get_trades(days=30)
        
        if trades_df.empty:
            self.console.print("[red]No trading data available[/red]")
            return
        
        # Risk calculations
        daily_returns = trades_df.groupby(trades_df['timestamp'].str[:10])['pnl_usdt'].sum()
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(daily_returns, 5)
        
        # Maximum drawdown
        cumulative_pnl = daily_returns.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max)
        max_drawdown = drawdown.min()
        
        # Risk metrics table
        risk_table = Table(title="Risk Management Report", box=box.ROUNDED)
        risk_table.add_column("Risk Metric", style="cyan")
        risk_table.add_column("Value", style="magenta")
        risk_table.add_column("Assessment", style="yellow")
        
        risk_metrics = [
            ("Daily VaR (95%)", f"${var_95:.2f}", "Max expected daily loss"),
            ("Max Drawdown", f"${max_drawdown:.2f}", "Largest peak-to-trough loss"),
            ("Volatility", f"${daily_returns.std():.2f}", "Daily P&L standard deviation"),
            ("Risk/Reward Ratio", f"{abs(trades_df[trades_df['pnl_usdt'] > 0]['pnl_usdt'].mean() / trades_df[trades_df['pnl_usdt'] < 0]['pnl_usdt'].mean()):.2f}", "Avg win / Avg loss"),
        ]
        
        for metric, value, assessment in risk_metrics:
            risk_table.add_row(metric, value, assessment)
        
        self.console.print(risk_table)
        
        # Position sizing analysis
        balance = self.bot.get_balance()
        current_balance = balance.get('USDT', {}).get('total', 0)
        
        position_table = Table(title="Position Sizing Analysis", box=box.ROUNDED)
        position_table.add_column("Parameter", style="cyan")
        position_table.add_column("Current", style="magenta")
        position_table.add_column("Recommended", style="green")
        
        recommended_risk = min(0.02, 1000 / current_balance) if current_balance > 0 else 0.02
        
        position_table.add_row("Risk per Trade", f"{self.bot.risk_per_trade*100:.1f}%", f"{recommended_risk*100:.1f}%")
        position_table.add_row("Max Positions", str(self.bot.max_positions), "3-5")
        position_table.add_row("Current Balance", f"${current_balance:.2f}", "Monitor closely")
        
        self.console.print(position_table)
    
    def show_market_analysis(self):
        """Show current market analysis for all symbols"""
        market_table = Table(title="Detailed Market Analysis", box=box.ROUNDED)
        market_table.add_column("Symbol", style="cyan")
        market_table.add_column("Price", style="yellow")
        market_table.add_column("RSI", style="magenta")
        market_table.add_column("BB Position", style="green")
        market_table.add_column("Volume", style="blue")
        market_table.add_column("Signal", style="red")
        market_table.add_column("Confidence", style="white")
        
        for symbol in self.bot.symbols:
            try:
                df = self.bot.get_market_data(symbol, limit=50)
                if df.empty:
                    continue
                
                latest = df.iloc[-1]
                signal, confidence = self.bot.identify_opportunity(df)
                
                # RSI status
                rsi_status = "Oversold" if latest['rsi'] < 30 else "Overbought" if latest['rsi'] > 70 else "Neutral"
                
                # Bollinger Band position
                bb_position = "Below" if latest['close'] < latest['bb_lower'] else "Above" if latest['close'] > latest['bb_upper'] else "Middle"
                
                # Volume status
                volume_status = "High" if latest['volume_ratio'] > 1.5 else "Normal" if latest['volume_ratio'] > 0.8 else "Low"
                
                signal_color = "green" if signal == 'buy' else "red" if signal == 'sell' else "yellow"
                
                market_table.add_row(
                    symbol,
                    f"${latest['close']:.4f}",
                    f"{latest['rsi']:.1f} ({rsi_status})",
                    bb_position,
                    volume_status,
                    f"[{signal_color}]{signal.upper()}[/{signal_color}]",
                    f"{confidence:.2f}"
                )
            except Exception as e:
                market_table.add_row(symbol, "Error", "Error", "Error", "Error", "Error", "Error")
        
        self.console.print(market_table)
    
    def export_trades(self, filename: str = None):
        """Export trades to CSV file"""
        if filename is None:
            filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        trades_df = self.bot.db.get_trades(days=365)  # Export last year
        
        if trades_df.empty:
            self.console.print("[red]No trades to export[/red]")
            return
        
        trades_df.to_csv(filename, index=False)
        self.console.print(f"[green]Exported {len(trades_df)} trades to {filename}[/green]")
    
    def interactive_commands(self):
        """Interactive command interface"""
        commands = {
            'portfolio': self.show_portfolio_analysis,
            'risk': self.show_risk_report,
            'market': self.show_market_analysis,
            'export': self.export_trades,
            'help': self.show_help,
            'quit': lambda: None,
            'exit': lambda: None,
        }
        
        self.console.print("\n[bold cyan]Trading Bot Interactive CLI[/bold cyan]")
        self.console.print("Type 'help' for available commands")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['quit', 'exit']:
                    break
                elif command in commands:
                    commands[command]()
                elif command.startswith('export '):
                    filename = command.split(' ', 1)[1]
                    self.export_trades(filename)
                else:
                    self.console.print(f"[red]Unknown command: {command}[/red]")
                    self.show_help()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def show_help(self):
        """Show available commands"""
        help_table = Table(title="Available Commands", box=box.ROUNDED)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="yellow")
        
        commands = [
            ("portfolio", "Show detailed portfolio analysis"),
            ("risk", "Show risk management report"),
            ("market", "Show current market analysis"),
            ("export [filename]", "Export trades to CSV file"),
            ("help", "Show this help message"),
            ("quit/exit", "Exit the CLI"),
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        self.console.print(help_table)

    # Enhanced main function with additional CLI options
def enhanced_main():
    parser = argparse.ArgumentParser(description='Advanced Binance Trading Bot with Comprehensive CLI')
    parser.add_argument('--api-key', required=True, help='Binance API Key')
    parser.add_argument('--api-secret', required=True, help='Binance API Secret')
    parser.add_argument('--testnet', action='store_true', help='Use Binance testnet')
    parser.add_argument('--mode', choices=['trade', 'dashboard', 'stats', 'interactive'], 
                        default='trade', help='Bot operation mode')
    parser.add_argument('--export-trades', help='Export trades to specified CSV file')
    parser.add_argument('--days', type=int, default=30, help='Number of days for statistics')
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = BinanceTradingBot(args.api_key, args.api_secret, args.testnet)
    cli = TradingBotCLI(bot)
    
    try:
        if args.mode == 'stats':
            # Show statistics
            cli.show_portfolio_analysis()
            cli.show_risk_report()
            
        elif args.mode == 'interactive':
            # Interactive CLI mode
            cli.interactive_commands()
            
        elif args.mode == 'dashboard':
            # Dashboard only mode
            bot.cli_mode = True
            print("Starting dashboard... Press Ctrl+C to stop")
            bot.run_cli_dashboard()
            
        elif args.mode == 'trade':
            # Full trading mode with dashboard
            trading_thread = threading.Thread(target=bot.run_strategy)
            trading_thread.daemon = True
            trading_thread.start()
            
            bot.cli_mode = True
            print("Starting Binance Trading Bot with Live Dashboard...")
            print("Press Ctrl+C to stop")
            time.sleep(2)
            
            bot.run_cli_dashboard()
            
        # Handle export if requested
        if args.export_trades:
            cli.export_trades(args.export_trades)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        bot.stop()
    except Exception as e:
        print(f"Error: {e}")
        bot.stop()

if __name__ == "__main__":
    # Check if rich is installed
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.layout import Layout
        from rich.live import Live
        from rich.text import Text
        from rich import box
    except ImportError:
        print("Please install required packages:")
        print("pip install rich ccxt pandas numpy ta python-binance")
        sys.exit(1)
    
    # Use enhanced main function
    enhanced_main()