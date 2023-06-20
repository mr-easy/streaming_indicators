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

## Example
```
from streaming_indicators import SMA

sma = SMA(10)
for i in range(20):
    print(i, sma.update(i))
```

## List of indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- True Range (TR)
- Average True Range (ATR)
- Heikin Ashi Candlesticks (HeikinAshi)
- Renko Bricks (Renko)

## TODO
- Not all indicators current support compute method.
- Add documentation.
- HeikinAshi depends on key names, eg ('open','close'). Should be independent, i.e. given in input.
- Implement more indicators.