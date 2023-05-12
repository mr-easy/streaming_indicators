import numpy as np


class SMA:
    def __init__(self, period):
        self.period = period
        self.points = []
        self.value = None
    def compute(self, point):
        return np.mean(self.points + [point])
    def update(self, point):
        self.points.append(float(point))
        self.points = self.points[-self.period:]
        if(len(self.points) == self.period):
            self.value = np.mean(self.points)
        return self.value
    
class EMA:
    def __init__(self, period, smoothing_factor=2):
        self.period = period
        self.smoothing_factor = smoothing_factor
        self.mult = self.smoothing_factor / (1+self.period)
        self.points = []
        self.value = None
    def update(self, point):
        self.points.append(point)
        self.points = self.points[-(self.period+1):]
        if(len(self.points) == self.period):
            self.value = np.mean(self.points) # Simple SMA
        elif(len(self.points) > self.period):
            self.value = (point * self.mult) + (self.value * (1-self.mult))
        return self.value
    
class RSI:
    def __init__(self, period):
        self.period = period
        self.points = []
        self.losses = []
        self.gains = []
        self.avg_gain = None
        self.avg_loss = None
        self.rsi = None
        self.value = None
    def update(self, point):
        self.points.append(float(point))
        if(len(self.points) > 1):
            diff = self.points[-1] - self.points[-2]
            if(diff >= 0):
                self.gains.append(diff)
                self.losses.append(0)
            else:
                self.gains.append(0)
                self.losses.append(-diff)
        self.points = self.points[-(self.period+1):]
        self.gains = self.gains[-(self.period):]
        self.losses = self.losses[-(self.period):]

        if(len(self.points) == self.period+1):
            if(self.avg_gain is None):
                self.avg_gain = np.mean(self.gains)
                self.avg_loss = np.mean(self.losses)
            else:
                self.avg_gain = ((self.avg_gain*(self.period-1)) + self.gains[-1])/self.period
                self.avg_loss = ((self.avg_loss*(self.period-1)) + self.losses[-1])/self.period
            rs = self.avg_gain / self.avg_loss
            self.rsi = 100 - (100/(1+rs))
            self.value = self.rsi
        return self.value

class HeikinAshi:
    def __init__(self):
        self.value = None
    def update(self, candle):
        ha = {}
        ha['close'] = round((candle.open+candle.high+candle.low+candle.close)/4,4)
        if(self.value is None):
            # no previous candle
            ha['open'] = candle.open
        else:
            ha['open'] = round((self.value['open']+self.value['close'])/2,4)
        ha['high'] = max(candle.high, ha['open'], ha['close'])
        ha['low'] = min(candle.low, ha['open'], ha['close'])
        self.value = ha
        return self.value