import numpy as np


class SMA:
    def __init__(self, period):
        self.period = period
        self.points = []
        self.value = None
    def compute(self, point):
        return np.mean(self.points + [float(point)])
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
    # def compute(self, point):
    #     points = self.points + [float(point)]
        
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

class TRANGE:
    '''True Range'''
    def __init__(self):
        self.prev_close = None
        self.value = None
    def compute(self, candle):
        if(self.prev_close is None):
            return candle['high'] - candle['low']
        else:
            return max(
                candle['high'] - candle['low'],
                abs(candle['high'] - self.prev_close),
                abs(candle['low'] - self.prev_close)
            )
    def update(self, candle):
        self.value = self.compute(candle)
        self.prev_close = candle['close']
        return self.value

class ATR:
    '''Average True Range'''
    def __init__(self, period):
        self.period = period
        self.period_1 = period-1
        self.TR = TRANGE()
        self.atr = 0
        self.value = None
        self.count = 0
    def compute(self, candle):
        tr = self.TR.compute(candle)
        if(self.count < self.period):
            return None
        elif(self.count == self.period):
            return (self.atr + tr)/self.period
        else:
            return (self.atr*self.period_1 + tr)/self.period

    def update(self, candle):
        self.count += 1
        tr = self.TR.update(candle)
        if(self.count < self.period):
            self.atr += tr
            return None
        if(self.count == self.period):
            self.atr += tr
            self.atr /= self.period
        else:
            self.atr = (self.atr*self.period_1 + tr)/self.period
        self.value = self.atr
        return self.value

class HeikinAshi:
    def __init__(self):
        self.value = None

    def compute(self, candle):
        ha = {}
        ha['close'] = round((candle.open+candle.high+candle.low+candle.close)/4,4)
        if(self.value is None):
            # no previous candle
            ha['open'] = candle.open
        else:
            ha['open'] = round((self.value['open']+self.value['close'])/2,4)
        ha['high'] = max(candle.high, ha['open'], ha['close'])
        ha['low'] = min(candle.low, ha['open'], ha['close'])
        return ha

    def update(self, candle):
        self.value = self.compute(candle)
        return self.value

class Renko:
    def __init__(self, start_price=None):
        self.bricks = []
        self.current_direction = 0
        self.brick_end_price = start_price
        self.pwick = 0   # positive wick
        self.nwick = 0   # negative wick
        self.brick_num = 0
        self.value = None
        
    def _create_brick(self, direction, brick_size):
        brick = {
            'direction': direction,
            'brick_num': self.brick_num,
            'wick_size': self.nwick if direction==1 else self.pwick,
            'brick_size': brick_size
        }
        self.bricks.append(brick)
        self.brick_num += 1
        self.current_direction = direction
        self.pwick = 0
        self.nwick = 0
        return brick        
        
    def update(self, price, brick_size):
        if(self.brick_end_price is None):
            self.brick_end_price = price
            return None
        bricks = None
        change = round(price - self.brick_end_price, 2)
        self.pwick = max(change, self.pwick)
        self.nwick = min(-change, self.nwick)
        if(self.current_direction == 0):
            direction = 0
            if(change >= brick_size): direction = 1
            elif(-change >= brick_size): direction = -1
            if(direction != 0):
                #print("firect brick direction:", str(direction))
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(direction, brick_size) for i in range(num_bricks)]
                self.brick_end_price = self.brick_end_price + direction*num_bricks*brick_size
                
        elif(self.current_direction == 1):
            if(change >= brick_size):
                # more bricks in +1 direction
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(1, brick_size) for i in range(num_bricks)]
                self.brick_end_price = self.brick_end_price + num_bricks*brick_size
                
            elif(-change >= 2*brick_size):
                # reverse direction to -1
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(-1, brick_size) for i in range(num_bricks-1)]
                self.brick_end_price = self.brick_end_price - num_bricks*brick_size
                
        elif(self.current_direction == -1):
            if(-change >= brick_size):
                # more bricks in -1 direction
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(-1, brick_size) for i in range(num_bricks)]
                self.brick_end_price = self.brick_end_price - num_bricks*brick_size
                
            elif(change >= 2*brick_size):
                # reverse direction to +1
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(1, brick_size) for i in range(num_bricks-1)]
                self.brick_end_price = self.brick_end_price + num_bricks*brick_size

        self.value = bricks
        return bricks