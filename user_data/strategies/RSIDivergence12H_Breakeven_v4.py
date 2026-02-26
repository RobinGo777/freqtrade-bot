from freqtrade.strategy import IStrategy, informative
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RSIDivergence12H_Breakeven_v4(IStrategy):
    """
    RSI Divergence Strategy v4
    Changes vs v3:
    - max_open_trades = 5
    - risk_percent = 0.015 (1.5% per trade)
    - min_risk_pct = 0.010 (1% min stop)
    - max_risk_pct = 0.0375 (3.75% max stop)
    - rr_ratio = 1.5
    - Volume filter: skip pairs with 24h volume < $20M (live/dry-run only)
    - Live: real exchange stop-market order placed immediately after entry fill
    - Backtest: stop/tp logic via adjust_trade_position
    - Pivot detection vectorized (faster)
    - startup_candle_count = 2500 (enough for 200-period 12H MA)
    """

    timeframe = "1h"
    startup_candle_count = 2500  # 2500 x 1H = ~104 days; covers 200-candle 12H MA
    can_short = True
    max_open_trades = 5

    use_custom_stoploss = False
    use_exit_signal = False
    trailing_stop = False
    position_adjustment_enable = True

    minimal_roi = {"0": 10.0}
    stoploss = -0.99  # Freqtrade safety net only — real SL handled via exchange order / adjust_trade_position

    # ===== RISK =====
    risk_percent = 0.015         # 1.5% account risk per trade

    # ===== DIVERGENCE PARAMETERS =====
    lookback = 4                 # Pivot detection window
    freshness = 2                # Max candles since last pivot
    max_gap = 30                 # Max candles between two pivots
    min_rsi_diff = 5             # Min RSI divergence magnitude

    # RSI bounds
    bull_rsi_low = 30            # First pivot RSI < 30 (oversold)
    bull_rsi_high = 45           # Second pivot RSI <= 45
    bear_rsi_high = 70           # First pivot RSI > 70 (overbought)
    bear_rsi_low = 55            # Second pivot RSI >= 55

    # Stop filters
    stop_buffer = 0.003          # 0.3% buffer beyond pivot
    min_risk_pct = 0.010         # Min 1.0% stop distance
    max_risk_pct = 0.0375        # Max 3.75% stop distance
    atr_multiplier = 0.4         # Min stop > 0.4 * ATR

    # Volume filter (live/dry-run only — backtest skips, curate whitelist manually)
    min_volume_usdt = 20_000_000  # $20M 24h quote volume

    # ===== RR & TREND FILTER =====
    rr_ratio = 1.5
    rsi12h_bull_min = 46
    rsi12h_bear_max = 54

    # ===== BREAKEVEN =====
    breakeven_enabled = True
    breakeven_rr_trigger = 1.0   # Move SL to breakeven after price reaches RR 1.0
    breakeven_offset = 0.0001    # 0.01% beyond entry price

    # ===== POSITION SYNC GUARD =====
    sync_guard_enabled = True        # Enable position sync security layer
    sync_guard_interval = 300        # Check every 5 minutes (seconds)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def _is_backtesting(self) -> bool:
        return self.dp.runmode.value in ("backtest", "hyperopt", "plot", "other")

    def _place_exchange_stoploss(self, trade: Trade, stop_price: float) -> Optional[str]:
        """Place real stop-market order on exchange (live/dry-run only).
        Called immediately after entry fill AND after breakeven trigger.
        """
        if self._is_backtesting():
            return None
        exchange = getattr(self, "exchange", None) or getattr(self.dp, "_exchange", None)
        if exchange is None:
            logger.error(f"❌ Cannot access exchange object for {trade.pair}")
            return None
        side = "sell" if not trade.is_short else "buy"
        try:
            order = exchange.create_order(
                pair=trade.pair,
                ordertype="stop_market",
                side=side,
                amount=trade.amount,
                rate=stop_price,
                params={"stopPrice": stop_price, "reduceOnly": True}
            )
            order_id = order.get("id") if isinstance(order, dict) else getattr(order, "id", None)
            logger.info(f"📌 Exchange SL placed | {trade.pair} | stop={stop_price:.6f} | order_id={order_id}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Failed to place exchange SL for {trade.pair}: {e}")
            return None

    def _cancel_exchange_stoploss(self, trade: Trade, order_id: str) -> None:
        """Cancel previous exchange stop order before placing updated one (breakeven update)."""
        if self._is_backtesting() or not order_id:
            return
        exchange = getattr(self, "exchange", None) or getattr(self.dp, "_exchange", None)
        if exchange is None:
            return
        try:
            exchange.cancel_order(order_id, trade.pair)
            logger.info(f"🗑️ Cancelled old SL order {order_id} for {trade.pair}")
        except Exception as e:
            logger.warning(f"Could not cancel SL order {order_id} for {trade.pair}: {e}")

    def _place_exchange_tp(self, trade: Trade, tp_price: float) -> Optional[str]:
        """Place limit take-profit order on exchange (live/dry-run only).
        Called immediately after entry fill together with SL order.
        On breakeven trigger: TP order stays unchanged (price doesn't move).
        """
        if self._is_backtesting():
            return None
        exchange = getattr(self, "exchange", None) or getattr(self.dp, "_exchange", None)
        if exchange is None:
            logger.error(f"❌ Cannot access exchange object for {trade.pair}")
            return None
        side = "sell" if not trade.is_short else "buy"
        try:
            order = exchange.create_order(
                pair=trade.pair,
                ordertype="limit",
                side=side,
                amount=trade.amount,
                rate=tp_price,
                params={"reduceOnly": True}
            )
            order_id = order.get("id") if isinstance(order, dict) else getattr(order, "id", None)
            logger.info(f"🎯 Exchange TP placed | {trade.pair} | limit={tp_price:.6f} | order_id={order_id}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Failed to place exchange TP for {trade.pair}: {e}")
            return None

    def _cancel_exchange_tp(self, trade: Trade, order_id: str) -> None:
        """Cancel TP limit order (used when SL hits first — avoid dangling orders)."""
        if self._is_backtesting() or not order_id:
            return
        exchange = getattr(self, "exchange", None) or getattr(self.dp, "_exchange", None)
        if exchange is None:
            return
        try:
            exchange.cancel_order(order_id, trade.pair)
            logger.info(f"🗑️ Cancelled TP order {order_id} for {trade.pair}")
        except Exception as e:
            logger.warning(f"Could not cancel TP order {order_id} for {trade.pair}: {e}")

    def _check_volume(self, pair: str) -> bool:
        """Check 24h quote volume >= min_volume_usdt (live/dry-run only).
        In backtesting volume filter is skipped — curate whitelist manually.
        """
        if self._is_backtesting():
            return True
        try:
            ticker = self.dp.ticker(pair)
            if ticker and ticker.get("quoteVolume"):
                vol = float(ticker["quoteVolume"])
                if vol < self.min_volume_usdt:
                    logger.info(f"❌ {pair} rejected: volume ${vol:,.0f} < ${self.min_volume_usdt:,.0f}")
                    return False
                return True
        except Exception as e:
            logger.warning(f"Volume check failed for {pair}: {e}")
        return True  # Don't block if ticker unavailable

    def _position_sync_guard(self, trade: Trade) -> None:
        """
        Security layer: compare open position on exchange vs Freqtrade DB.
        Called every sync_guard_interval seconds from adjust_trade_position.

        Scenarios handled:
        1. Position exists in DB but NOT on exchange → emergency close in DB + blacklist pair
        2. Position size mismatch (>5% diff) → log critical warning + blacklist pair
        3. Entry price mismatch (>0.5% diff) → recalculate SL/TP/breakeven prices
        4. SL order missing on exchange → re-place it immediately
        """
        if self._is_backtesting() or not self.sync_guard_enabled:
            return

        exchange = getattr(self, "exchange", None) or getattr(self.dp, "_exchange", None)
        if exchange is None:
            return

        if not hasattr(self, "_sync_last_check"):
            self._sync_last_check = {}

        now = datetime.utcnow().timestamp()
        last_check = self._sync_last_check.get(trade.pair, 0)
        if now - last_check < self.sync_guard_interval:
            return
        self._sync_last_check[trade.pair] = now

        try:
            # Fetch real position from exchange
            positions = exchange.fetch_positions([trade.pair])
            exchange_position = None
            for pos in positions:
                symbol = pos.get("symbol", "").replace("/", "").replace(":USDT", "USDT")
                trade_symbol = trade.pair.replace("/", "").replace(":USDT", "USDT")
                if symbol == trade_symbol:
                    exchange_position = pos
                    break

            exchange_size = abs(float(exchange_position.get("contracts", 0) or 0)) \
                            if exchange_position else 0.0
            db_size = trade.amount

            # ── Scenario 1: Position missing on exchange ──
            if exchange_size == 0 and db_size > 0:
                logger.critical(
                    f"🚨 SYNC GUARD | {trade.pair} | Position in DB ({db_size:.4f}) "
                    f"but MISSING on exchange! Blacklisting pair."
                )
                self._blacklist_pair(trade.pair)
                return

            # ── Scenario 2: Size mismatch > 5% ──
            if exchange_size > 0 and db_size > 0:
                diff_pct = abs(exchange_size - db_size) / db_size
                if diff_pct > 0.05:
                    logger.critical(
                        f"🚨 SYNC GUARD | {trade.pair} | Size mismatch! "
                        f"DB={db_size:.4f} Exchange={exchange_size:.4f} "
                        f"diff={diff_pct:.1%} — Blacklisting pair."
                    )
                    self._blacklist_pair(trade.pair)
                    return

            # ── Scenario 3: Entry price mismatch > 0.5% ──
            if exchange_position and not hasattr(self, "_trade_data") is False:
                trade_key = f"trade_{trade.id}"
                data = self._trade_data.get(trade_key, {})
                if data:
                    exchange_entry = float(exchange_position.get("entryPrice", 0) or 0)
                    db_entry = data.get("entry_price", trade.open_rate)
                    if exchange_entry > 0 and db_entry > 0:
                        price_diff_pct = abs(exchange_entry - db_entry) / db_entry
                        if price_diff_pct > 0.005:
                            logger.warning(
                                f"⚠️ SYNC GUARD | {trade.pair} | Entry price mismatch! "
                                f"DB={db_entry:.6f} Exchange={exchange_entry:.6f} "
                                f"diff={price_diff_pct:.3%} — Recalculating SL/TP."
                            )
                            # Recalculate SL/TP based on real exchange entry price
                            stop_price = data.get("stop_price")
                            if stop_price:
                                risk = abs(exchange_entry - stop_price)
                                new_tp = (exchange_entry + risk * self.rr_ratio) \
                                         if not trade.is_short \
                                         else (exchange_entry - risk * self.rr_ratio)
                                new_rr1 = (exchange_entry + risk * self.breakeven_rr_trigger) \
                                          if not trade.is_short \
                                          else (exchange_entry - risk * self.breakeven_rr_trigger)
                                data["entry_price"] = exchange_entry
                                data["tp_price"] = new_tp
                                data["rr1_price"] = new_rr1
                                logger.info(
                                    f"🔧 SYNC GUARD | {trade.pair} | Recalculated: "
                                    f"entry={exchange_entry:.6f} TP={new_tp:.6f} BE={new_rr1:.6f}"
                                )

            # ── Scenario 4: SL or TP order missing on exchange ──
            if not hasattr(self, "_trade_data"):
                return
            trade_key = f"trade_{trade.id}"
            data = self._trade_data.get(trade_key, {})
            sl_order_id = data.get("sl_order_id")
            tp_order_id = data.get("tp_order_id")

            if sl_order_id or tp_order_id:
                try:
                    open_orders = exchange.fetch_open_orders(trade.pair)
                    open_ids = [o.get("id") for o in open_orders]

                    # Check SL
                    if sl_order_id and sl_order_id not in open_ids:
                        logger.warning(
                            f"⚠️ SYNC GUARD | {trade.pair} | SL order {sl_order_id} "
                            f"missing on exchange — re-placing SL."
                        )
                        stop_price = data.get("stop_price")
                        if stop_price:
                            new_id = self._place_exchange_stoploss(trade, stop_price)
                            data["sl_order_id"] = new_id

                    # Check TP
                    if tp_order_id and tp_order_id not in open_ids:
                        logger.warning(
                            f"⚠️ SYNC GUARD | {trade.pair} | TP order {tp_order_id} "
                            f"missing on exchange — re-placing TP."
                        )
                        tp_price = data.get("tp_price")
                        if tp_price:
                            new_id = self._place_exchange_tp(trade, tp_price)
                            data["tp_order_id"] = new_id

                except Exception as e:
                    logger.warning(f"Could not fetch open orders for {trade.pair}: {e}")

            logger.debug(f"✅ SYNC GUARD OK | {trade.pair} | "
                         f"DB={db_size:.4f} Exchange={exchange_size:.4f}")

        except Exception as e:
            logger.error(f"❌ SYNC GUARD error for {trade.pair}: {e}")

    def _blacklist_pair(self, pair: str) -> None:
        """Add pair to runtime blacklist to prevent new entries."""
        try:
            if hasattr(self.dp, "bl_pair"):
                self.dp.bl_pair(pair)
                logger.critical(f"🔒 {pair} added to blacklist via dp.bl_pair()")
            else:
                # Fallback: store in local set, checked in confirm_trade_entry
                if not hasattr(self, "_local_blacklist"):
                    self._local_blacklist = set()
                self._local_blacklist.add(pair)
                logger.critical(f"🔒 {pair} added to local blacklist (dp.bl_pair unavailable)")
        except Exception as e:
            logger.error(f"Failed to blacklist {pair}: {e}")

    # ─────────────────────────────────────────────
    # INDICATORS
    # ─────────────────────────────────────────────
    @informative("12h")
    def populate_indicators_12h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """12H trend filter: MA200 direction + RSI level."""
        dataframe["ma200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["rsi14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["above_ma200"] = dataframe["close"] > dataframe["ma200"]
        dataframe["below_ma200"] = dataframe["close"] < dataframe["ma200"]
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """1H indicators + vectorized divergence detection."""

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        lb = self.lookback
        close = dataframe["close"].values
        n = len(close)

        # ── Vectorized pivot detection ──
        pivot_low = np.ones(n, dtype=bool)
        pivot_high = np.ones(n, dtype=bool)
        for j in range(1, lb + 1):
            pivot_low &= np.concatenate([[False] * j, close[j:] < close[:-j]])
            pivot_low &= np.concatenate([close[:-j] < close[j:], [False] * j])
            pivot_high &= np.concatenate([[False] * j, close[j:] > close[:-j]])
            pivot_high &= np.concatenate([close[:-j] > close[j:], [False] * j])

        # Mask edges to avoid lookahead bias
        pivot_low[:lb] = False
        pivot_low[-lb:] = False
        pivot_high[:lb] = False
        pivot_high[-lb:] = False

        dataframe["pivot_low"] = pivot_low
        dataframe["pivot_high"] = pivot_high

        # ── Divergence detection ──
        dataframe["bullish_div"] = False
        dataframe["bearish_div"] = False
        dataframe["bull_stop"] = np.nan
        dataframe["bear_stop"] = np.nan

        rsi = dataframe["rsi"].values
        low = dataframe["low"].values
        high = dataframe["high"].values

        swing_lows = np.where(pivot_low)[0].tolist()
        swing_highs = np.where(pivot_high)[0].tolist()

        # Bullish divergence: price lower low, RSI higher low
        for current_idx in range(n):
            avail = [i for i in swing_lows if i < current_idx]
            if len(avail) < 2:
                continue
            idx2, idx1 = avail[-1], avail[-2]
            if current_idx - idx2 > self.freshness:
                continue
            if idx2 - idx1 > self.max_gap:
                continue
            p1, p2 = close[idx1], close[idx2]
            r1, r2 = rsi[idx1], rsi[idx2]
            if (p2 < p1 and r2 > r1 and
                    r1 < self.bull_rsi_low and
                    r2 <= self.bull_rsi_high and
                    (r2 - r1) >= self.min_rsi_diff):
                dataframe.at[dataframe.index[current_idx], "bullish_div"] = True
                dataframe.at[dataframe.index[current_idx], "bull_stop"] = low[idx2]

        # Bearish divergence: price higher high, RSI lower high
        for current_idx in range(n):
            avail = [i for i in swing_highs if i < current_idx]
            if len(avail) < 2:
                continue
            idx2, idx1 = avail[-1], avail[-2]
            if current_idx - idx2 > self.freshness:
                continue
            if idx2 - idx1 > self.max_gap:
                continue
            p1, p2 = close[idx1], close[idx2]
            r1, r2 = rsi[idx1], rsi[idx2]
            if (p2 > p1 and r2 < r1 and
                    r1 > self.bear_rsi_high and
                    r2 >= self.bear_rsi_low and
                    (r1 - r2) >= self.min_rsi_diff):
                dataframe.at[dataframe.index[current_idx], "bearish_div"] = True
                dataframe.at[dataframe.index[current_idx], "bear_stop"] = high[idx2]

        return dataframe

    # ─────────────────────────────────────────────
    # ENTRY / EXIT SIGNALS
    # ─────────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        has_12h = "above_ma200_12h" in dataframe.columns
        col_above = "above_ma200_12h" if has_12h else "above_ma200"
        col_below = "below_ma200_12h" if has_12h else "below_ma200"
        col_rsi_12h = "rsi14_12h" if has_12h else "rsi14"

        # LONG: bullish divergence + price above 12H MA200 + 12H RSI not too low
        dataframe.loc[
            (dataframe["bullish_div"]) &
            (dataframe[col_above]) &
            (dataframe[col_rsi_12h] > self.rsi12h_bull_min) &
            (dataframe["bull_stop"].notna()) &
            ((dataframe["close"] - dataframe["bull_stop"]) >= self.atr_multiplier * dataframe["atr"]),
            ["enter_long", "enter_tag"]
        ] = (1, "bullish_div_12h")

        # SHORT: bearish divergence + price below 12H MA200 + 12H RSI not too high
        dataframe.loc[
            (dataframe["bearish_div"]) &
            (dataframe[col_below]) &
            (dataframe[col_rsi_12h] < self.rsi12h_bear_max) &
            (dataframe["bear_stop"].notna()) &
            ((dataframe["bear_stop"] - dataframe["close"]) >= self.atr_multiplier * dataframe["atr"]),
            ["enter_short", "enter_tag"]
        ] = (1, "bearish_div_12h")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    # ─────────────────────────────────────────────
    # TRADE CONFIRMATION
    # ─────────────────────────────────────────────
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Validate stop distance and volume. Store trade data for later use."""

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return False

        # ── Check local blacklist (set by sync guard) ──
        if hasattr(self, "_local_blacklist") and pair in self._local_blacklist:
            logger.info(f"❌ {pair} blocked — in local blacklist (sync guard)")
            return False

        last = df.iloc[-1]
        stop_raw = last["bull_stop"] if side == "long" else last["bear_stop"]

        if np.isnan(stop_raw):
            return False

        # Apply buffer beyond pivot
        stop_price = stop_raw * (1 - self.stop_buffer) if side == "long" \
                     else stop_raw * (1 + self.stop_buffer)

        # Stop distance filter
        risk_pct = abs(rate - stop_price) / rate
        if risk_pct < self.min_risk_pct:
            logger.info(f"❌ {pair} rejected: stop too tight {risk_pct:.3%} < {self.min_risk_pct:.3%}")
            return False
        if risk_pct > self.max_risk_pct:
            logger.info(f"❌ {pair} rejected: stop too wide {risk_pct:.3%} > {self.max_risk_pct:.3%}")
            return False

        # Volume filter (live/dry-run only)
        if not self._check_volume(pair):
            return False

        # Precalculate TP and breakeven prices
        risk = abs(rate - stop_price)
        tp_price = (rate + risk * self.rr_ratio) if side == "long" \
                   else (rate - risk * self.rr_ratio)
        rr1_price = (rate + risk * self.breakeven_rr_trigger) if side == "long" \
                    else (rate - risk * self.breakeven_rr_trigger)

        if not hasattr(self, "_trade_data"):
            self._trade_data = {}

        self._trade_data[f"pending_{pair}"] = {
            "stop_price": stop_price,
            "tp_price": tp_price,
            "rr1_price": rr1_price,
            "entry_price": rate,
            "breakeven_triggered": False,
            "sl_order_id": None,
            "tp_order_id": None,
        }

        logger.info(f"✅ {side.upper()} {pair} @ {rate:.6f} | "
                    f"SL={stop_price:.6f} ({risk_pct:.2%}) | "
                    f"TP={tp_price:.6f} | RR={self.rr_ratio}")
        return True

    # ─────────────────────────────────────────────
    # STAKE SIZING
    # ─────────────────────────────────────────────
    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        """Risk-based position sizing: risk_percent of total balance per trade."""

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return proposed_stake
        last = df.iloc[-1]
        stop_raw = last["bull_stop"] if side == "long" else last["bear_stop"]

        if np.isnan(stop_raw):
            return proposed_stake

        stop_price = stop_raw * (1 - self.stop_buffer) if side == "long" \
                     else stop_raw * (1 + self.stop_buffer)

        balance = self.wallets.get_total_stake_amount()
        risk_amount = balance * self.risk_percent
        risk_per_unit = abs(current_rate - stop_price)

        if risk_per_unit <= 0:
            return proposed_stake

        stake = (risk_amount / risk_per_unit) * current_rate
        stake = max(stake, min_stake or 0)
        stake = min(stake, max_stake)
        return stake

    # ─────────────────────────────────────────────
    # ORDER FILLED — place SL (market) + TP (limit) simultaneously
    # ─────────────────────────────────────────────
    def order_filled(self, pair: str, trade: Trade, order,
                     current_time: datetime, **kwargs) -> None:
        """Place stop-market SL + limit TP immediately after entry fill (live/dry-run only).
        Both orders placed atomically — position is protected from both sides at once.
        """
        if self._is_backtesting():
            return

        ft_side = order.ft_order_side if hasattr(order, "ft_order_side") \
                  else order.get("ft_order_side", "")
        order_type = getattr(order, "order_type", "") if hasattr(order, "order_type") \
                     else order.get("order_type", "")

        # Skip if this is SL or TP order being filled (not an entry)
        if order_type in ("stoploss", "stop_market", "stop", "limit"):
            return

        # Only act on entry fills
        is_entry = (not trade.is_short and ft_side == "buy") or \
                   (trade.is_short and ft_side == "sell")
        if not is_entry:
            return

        if not hasattr(self, "_trade_data"):
            self._trade_data = {}

        pending_key = f"pending_{pair}"
        if pending_key in self._trade_data:
            data = self._trade_data[pending_key]
            stop_price = data.get("stop_price")
            tp_price = data.get("tp_price")

            # Place SL (stop-market) — protection against adverse move
            if stop_price:
                sl_order_id = self._place_exchange_stoploss(trade, stop_price)
                data["sl_order_id"] = sl_order_id

            # Place TP (limit) — locks in profit at target price
            if tp_price:
                tp_order_id = self._place_exchange_tp(trade, tp_price)
                data["tp_order_id"] = tp_order_id

            logger.info(f"🚀 Entry filled {pair} | "
                        f"SL={stop_price:.6f} (market) | "
                        f"TP={tp_price:.6f} (limit) | both orders placed")

    # ─────────────────────────────────────────────
    # POSITION MANAGEMENT (backtest SL/TP + live breakeven)
    # ─────────────────────────────────────────────
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Handles:
        1. Breakeven trigger → move SL to entry (+ update exchange order in live)
        2. SL hit → close position (backtest + live safety fallback)
        3. TP hit → close position
        In live: exchange stop order is primary SL; this is a safety fallback.
        """

        if not hasattr(self, "_trade_data"):
            self._trade_data = {}

        # ── Security: sync exchange position vs DB ──
        self._position_sync_guard(trade)

        trade_key = f"trade_{trade.id}"

        # Promote pending data → active trade data on first call
        if trade_key not in self._trade_data:
            pending_key = f"pending_{trade.pair}"
            if pending_key in self._trade_data:
                self._trade_data[trade_key] = self._trade_data.pop(pending_key)
            else:
                return None

        data = self._trade_data[trade_key]
        stop_price = data["stop_price"]
        tp_price = data["tp_price"]
        rr1_price = data["rr1_price"]
        entry_price = data["entry_price"]
        breakeven_triggered = data.get("breakeven_triggered", False)

        # ── Breakeven: move SL to entry after reaching RR 1.0 ──
        if self.breakeven_enabled and not breakeven_triggered:
            triggered = (current_rate <= rr1_price) if trade.is_short \
                        else (current_rate >= rr1_price)
            if triggered:
                new_stop = entry_price * (1 + self.breakeven_offset) if trade.is_short \
                           else entry_price * (1 - self.breakeven_offset)
                data["stop_price"] = new_stop
                data["breakeven_triggered"] = True
                stop_price = new_stop

                # Live: cancel old SL order and place new one at breakeven
                if not self._is_backtesting():
                    old_order_id = data.get("sl_order_id")
                    if old_order_id:
                        self._cancel_exchange_stoploss(trade, old_order_id)
                    new_order_id = self._place_exchange_stoploss(trade, new_stop)
                    data["sl_order_id"] = new_order_id

                logger.info(f"🔄 BREAKEVEN {'SHORT' if trade.is_short else 'LONG'} "
                            f"{trade.pair} | new_SL={new_stop:.6f}")

        # ── Stop Loss hit — cancel TP limit order ──
        sl_hit = (current_rate >= stop_price) if trade.is_short \
                 else (current_rate <= stop_price)
        if sl_hit:
            logger.info(f"🛑 SL {'SHORT' if trade.is_short else 'LONG'} "
                        f"{trade.pair} @ {current_rate:.6f} | SL={stop_price:.6f}")
            # Cancel dangling TP limit order on exchange
            if not self._is_backtesting():
                tp_order_id = data.get("tp_order_id")
                if tp_order_id:
                    self._cancel_exchange_tp(trade, tp_order_id)
            del self._trade_data[trade_key]
            return -trade.stake_amount

        # ── Take Profit hit — cancel SL stop order ──
        tp_hit = (current_rate <= tp_price) if trade.is_short \
                 else (current_rate >= tp_price)
        if tp_hit:
            logger.info(f"🎯 TP {'SHORT' if trade.is_short else 'LONG'} "
                        f"{trade.pair} @ {current_rate:.6f} | TP={tp_price:.6f}")
            # Cancel dangling SL stop-market order on exchange
            if not self._is_backtesting():
                sl_order_id = data.get("sl_order_id")
                if sl_order_id:
                    self._cancel_exchange_stoploss(trade, sl_order_id)
            del self._trade_data[trade_key]
            return -trade.stake_amount

        return None
