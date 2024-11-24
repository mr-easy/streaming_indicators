# Streaming Indicators 

A python library for computing technical analysis indicators on streaming data.

## Installation
```
pip install streaming-indicators
```
## Why another TA library?
There are many other technical analysis python packages, most notably ta-lib, then why another library?  
All other libraries work on static data, you can not add values to any indicator. But in real-time trading system, price values (ticks/candles) keeps streaming, and indicators should update on real-time. This library is for that purpose.

## Usage
Each indicator is a class, and is statefull. It will have 3 main functions:
1. Constructor: initialise all parameters such as period.
2. update: To add new data point in the indicator computation. Returns the new value of the indicator.
3. compute: Compute indicator value with a new data point, but don't update it's state. This is useful in some cases, for example, compute indictor on ltp, but don't update it.

## List of indicators (and usage)
- Simple Moving Average (SMA)  
```
import streaming_indicators as si

period = 14
SMA = si.SMA(period)
for idx, candle in candles.iterrows():
    sma = SMA.update(candle['close'])
    print(sma)
```
- Exponential Moving Average (EMA)  
```
period = 14
EMA = si.EMA(period)
for idx, candle in candles.iterrows():
    ema = EMA.update(candle['close'])
    print(ema)
```
- Weighted Moving Average (WMA)  
- Smoothed Moving Average (SMMA)  
- Volume Weighted Average Price (VWAP)
Computes VWAP using hlc3 and volume, anchors at first candle.
```
VWAP = si.VWAP()
for idx, candle in candles.iterrows():
    vwap = VWAP.update(candle)
    print(vwap)
```
- Relative Strength Index (RSI)  
```
period = 14
RSI = si.RSI(period)
for idx, candle in candles.iterrows():
    rsi = RSI.update(candle['close'])
    print(rsi)
```
- Central Pivot Range (CPR)  
- True Range (TRANGE)  
- Average True Range (ATR)  
```
atr_period = 20
ATR = si.ATR(atr_period)
for idx, candle in candles.iterrows():
    atr = ATR.update(candle)  # Assumes candle to have 'open',high','low','close' - TODO: give multiple inputs to update.
    print(atr)
```
- Bollinger Bands (BBands)  
- SuperTrend (SuperTrend)   
```
st_atr_length = 10
st_factor = 3
ST = si.SuperTrend(st_atr_length, st_factor)
for idx, candle in candles.iterrows():
    st = ST.update(candle)
    print(st) # (st_direction:1/-1, band_value)
```
To use some historical candles to initiate, use: `ST = si.SuperTrend(st_atr_length, st_factor, candles=initial_candles)` where `initial_candles` is pandas dataframe with `open,high,low,close` columns, and requires talib package.
- Heikin Ashi Candlesticks (HeikinAshi)  
```
HA = si.HeikinAshi()
for idx, candle in candles.iterrows():
    ha_candle = HA.update(candle)
    print(ha_candle) # {'close': float, 'open': float, 'high': float, 'low': float}
```
- Renko Bricks (Renko)  
```
# For fixed brick size
brick_size = 20
Renko = si.Renko()
for idx, candle in candles.iterrows():
    bricks = Renko.update(candle['close'], brick_size)
    print(bricks) # [{'direction': 1/-1, 'brick_num': int, 'wick_size': float, 'brick_size': float, 'brick_end_price': float, 'price': float}, {}]: list of bricks formed after this candle
```
```
# For brick size using ATR
atr_period = 20
ATR = si.ATR(atr_period)
Renko = si.Renko()
for idx, candle in candles.iterrows():
    atr = ATR.update(candle)
    print(atr)
    bricks = Renko.update(candle['close'], atr)
    print(bricks)
```
- Order Checking (IsOrder)  
Checks if the running sequence is in a given order, eg increasing, decreasing, exponential, etc. Useful when checking if consecutive n candles/ltps were increasing.
```
period = 10
all_increasing = si.IsOrder('>', period)
for idx, candle in candles.iterrows():
    is_increasing = all_increasing.update(candle['close'])
    print(is_increasing) # True/False
```
- HalfTrend (`HalfTrend`)  
HalfTrend indicator by Alex Orekhov (everget) in tradingview. Refered it's pine script. `trend = 0` for uptrend and `1` for downtrend.
```
HT = si.HalfTrend(amplitude=2, channel_deviation=2, atr_period=100)
for idx, candle in candles.iterrows():
    trend, half_trend, up, down, atr_high, atr_low = HT.update(candle)
```
- CWA 2-Sigma (`CWA2Sigma`)  
As discussed by Mr Rakesh Pujara in his [interview](https://www.youtube.com/watch?v=tSlfPgaWIu4).
```
CWA2Sigma = si.CWA2Sigma(bb_period=50, bb_width=2, ema_period=100, atr_period=14, atr_factor=1.8, sl_perc=20)
for idx, candle in candles.iterrows():
    cwa_signal,cwa_entry_price = CWA2Sigma.update(candle)
```

## Changelogs and TODOs
- [Version History](version_history.md)
- [TODOs](TODO.md)

If you find this repo useful, do consider giving a star. Contributions are most welcome.