# Version History

### 0.1.6
- added `VWAP`.
### 0.1.5
- BUG FIX: All `RollingStat` subclasses were passing None in points, instead of points.
### 0.1.4
- added `CPR`.
### 0.1.3
- added `CWA2Sigma`.
### 0.1.2
- added `HalfTrend`.
### 0.1.1
- added `RollingStat` abstract class for indicators which require computing stat on a rolling window.
- added `Max` and `Min` using `RollingStat`
- changed `SMA` and `SD` to use `RollingStat`
### 0.1.0
- added `SD` and `BBands`.
- using dequeue instead of list in `SMA`, `EMA`.
- added `compute` in `WMA`, `SMMA`.
- added type annotations.