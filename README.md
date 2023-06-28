# Streaming Indicators 

A python library for computing technical analysis indicators on streaming data.

## Installation
```
pip install streaming-indicators
```
## Why another TA library?
There are many other technical analysis python packages, most notably ta-lib, then why another library?  
All other libraries work on static data, but you can not add values to any indicator. But in real system, data values keeps coming, and indicators should keep updating. This library is for that purpose.

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
- Relative Strength Index (RSI)
```
period = 14
RSI = si.RSI(period)
for idx, candle in candles.iterrows():
    rsi = RSI.update(candle['close'])
    print(rsi)
```
- True Range (TR)
- Average True Range (ATR)
```
atr_period = 20
ATR = si.ATR(atr_period)
for idx, candle in candles.iterrows():
    atr = ATR.update(candle)  # Assumes candle to have 'open',high','low','close' - TODO: give multiple inputs to update.
    print(atr)
```
- Heikin Ashi Candlesticks (HeikinAshi)
- Renko Bricks (Renko)
```
# For fixed brick size
brick_size = 20
Renko = si.Renko()
for idx, candle in candles.iterrows():
    bricks = Renko.update(candle['close'], brick_size)
    print(bricks)
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

## TODO
- Not all indicators current support compute method.
- Add documentation.
- HeikinAshi,ATR depends on key names, eg ('open','close'). Should be independent, i.e. given in input.
- Implement more indicators.