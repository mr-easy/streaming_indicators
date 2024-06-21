# Version History

### 0.1.1
#### changes
- added `RollingStat` abstract class for indicators which require computing stat on a rolling window.
- added `Max` and `Min` using `RollingStat`
- changed `SMA` and `SD` to use `RollingStat`
#### todo
- can we have better implementations of `SMA` and `SD`, currently computing on whole data.
### 0.1.0
#### changes
- added `SD` and `BBands`.
- using dequeue instead of list in `SMA`, `EMA`.
- added `compute` in `WMA`, `SMMA`.
- added type annotations.
#### todo
- convert candle inputs to open, high, low and close.
