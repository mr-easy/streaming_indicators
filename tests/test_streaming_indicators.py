import pytest
import numpy as np

import streaming_indicators as si


def test_rollingstat_assertion():
    with pytest.raises(AssertionError):
        si.RollingStat(period=1, func=max)


def test_max_min_sma_sd_compute_update():
    data = [1, 3, 2, 5, 4]
    period = 3
    max_rs = si.Max(period)
    min_rs = si.Min(period)
    sma_rs = si.SMA(period)
    sd_rs = si.SD(period)

    results = []
    for x in data:
        m = max_rs.update(x)
        n = min_rs.update(x)
        s = sma_rs.update(x)
        d = sd_rs.update(x)
        results.append((m, n, s, d))

    # Only last 3 points should produce values
    # Last window: [2,5,4]
    assert results[-1][0] == 5  # max
    assert results[-1][1] == 2  # min
    assert pytest.approx(results[-1][2]) == np.mean([2,5,4])  # sma
    assert pytest.approx(results[-1][3]) == np.std([2,5,4])  # sd


def test_ema_simple():
    period = 3
    ema = si.EMA(period)
    # feed first 3 points -> simple SMA
    ema.update(1)
    ema.update(2)
    v = ema.update(3)
    assert pytest.approx(v) == np.mean([1, 2, 3])
    # next update -> exponential
    e2 = ema.update(4)
    mult = 2 / (1 + period)
    expected = 4 * mult + v * (1 - mult)
    assert pytest.approx(e2) == expected


def test_wma_compute_update():
    period = 2
    wma = si.WMA(period)
    # first update returns None until full
    assert wma.update(1) is None
    # second
    v = wma.update(2)
    # weights [1,2], sum=3, (1*1+2*2)/3 = 5/3
    assert pytest.approx(v) == (1*1 + 2*2) / 3


def test_smma_delegates_to_ema():
    period = 3
    smma = si.SMMA(period)
    # period*2-1 = 5 for underlying EMA
    # feed 5 points
    vals = [1,2,3,4,5]
    for x in vals:
        out = smma.update(x)
    # after 5, simple avg = mean(vals)
    assert pytest.approx(out) == np.mean(vals)


def test_rma_behavior():
    period = 3
    rma = si.RMA(period)
    # first
    v1 = rma.update(1)
    assert pytest.approx(v1) == 1.0000
    # next
    v2 = rma.update(2)
    expected2 = round((1/period) * 2 + (1 - 1/period) * v1, 4)
    assert pytest.approx(v2) == expected2


def test_vwap_compute_update():
    vwap = si.VWAP()
    # first candle
    c1 = {'high':1, 'low':1, 'close':1, 'volume':1}
    assert vwap.update(c1) == 1.0
    # second
    c2 = {'high':2, 'low':2, 'close':2, 'volume':2}
    v = vwap.update(c2)
    # tpv_sum = 1*1 + (2+2+2)/3*2 = 1 + 4 = 5; vol_sum = 3; vwap = 5/3
    assert pytest.approx(v) == 5/3


def test_trange_compute_update():
    tr = si.TRANGE()
    c1 = {'high':10, 'low':7, 'close':8}
    # compute initial
    assert tr.compute(c1) == 3
    # update sets prev_close
    v1 = tr.update(c1)
    assert v1 == 3
    c2 = {'high':12, 'low':9, 'close':10}
    # compute true range
    comp = tr.compute(c2)
    expected = max(12-9, abs(12-8), abs(9-8))
    assert comp == expected


def test_cpr_compute_update():
    cpr = si.CPR()
    c = {'high': 10, 'low': 8, 'close': 9}
    comp = cpr.compute(c)
    # cpr=(10+8+9)/3=9; bc=(10+8)/2=9; tc=9+(9-9)=9
    assert comp == (9.0, 9.0, 9.0)
    upd = cpr.update(c)
    assert upd == comp


def test_heikin_ashi_initial_and_update():
    ha = si.HeikinAshi()
    c1 = {'open':1, 'high':4, 'low':0, 'close':2}
    h1 = ha.compute(c1)
    # close=(1+4+0+2)/4=1.75, open=c1['open']=1
    assert h1['close'] == round((1+4+0+2)/4,4)
    assert h1['open'] == 1
    # update stores value
    h2 = ha.update(c1)
    assert ha.value == h2


def test_renko_initial_and_bricks():
    rk = si.Renko(start_price=10)
    # initial call
    assert rk.update(10, brick_size=1) is None
    # price moves up by 2
    bricks = rk.update(12, brick_size=1)
    assert isinstance(bricks, list)
    assert len(bricks) == 2
    # each brick direction=1
    assert all(b['direction'] == 1 for b in bricks)


def test_isorder_increasing_and_reset():
    iso = si.IsOrder('>', length=3)
    seq = [1,2,3]
    res = [iso.update(x) for x in seq]
    # third update should return True
    assert res[2] is True
    # breaking the order
    assert iso.update(1) is False
    # then 2,3 again True
    iso2 = si.IsOrder('>', length=2)
    assert iso2.update(5) is False
    assert iso2.update(6) is True

# Advanced Indicators Tests

def test_atr_compute_update():
    period = 3
    atr = si.ATR(period)
    c1 = {'high':5, 'low':1, 'close':4}
    c2 = {'high':6, 'low':2, 'close':5}
    c3 = {'high':7, 'low':3, 'close':6}
    # first two updates should return None
    assert atr.update(c1) is None
    assert atr.update(c2) is None
    # third update: initial ATR = (4+4+4)/3 = 4
    v3 = atr.update(c3)
    assert pytest.approx(v3) == 4.0
    # fourth update uses smoothing: (4*(3-1)+4)/3 = 4
    c4 = {'high':8, 'low':4, 'close':7}
    v4 = atr.update(c4)
    assert pytest.approx(v4) == 4.0


def test_bbands_compute_update():
    period, mult = 3, 2
    bb = si.BBands(period, mult)
    data = [1, 2, 3, 4]
    res = [bb.update(x) for x in data]
    # Only last values valid
    ub, mb, lb = res[-1]
    window = [2,3,4]
    mean = np.mean(window)
    sd = np.std(window)
    assert pytest.approx(mb) == mean
    assert pytest.approx(ub) == mean + mult * sd
    assert pytest.approx(lb) == mean - mult * sd


def test_rsi_simple():
    period = 2
    rsi = si.RSI(period)
    # updates: first no output, second no output, third gives initial RSI
    assert rsi.update(1) is None
    assert rsi.update(2) is None
    v3 = rsi.update(3)
    # gains=[1,1], losses=[0,0] -> avg_gain=1, avg_loss=0 -> RSI=100
    assert pytest.approx(v3) == 100.0
    # next downward movement: diff = 2-3 = -1
    v4 = rsi.update(2)
    # expected RSI = 100 - (100/(1 + ( ( (1*(1)) + 0 )/2 ) / ( ( (0*(1)) + 1 )/2 ) )) = 50
    assert pytest.approx(v4) == 50.0


def test_plus_minus_di():
    period = 2
    plus = si.PLUS_DI(period)
    minus = si.MINUS_DI(period)
    c1 = {'high':1, 'low':1, 'close':1}
    c2 = {'high':3, 'low':0, 'close':2}
    assert plus.update(c1) is None
    assert minus.update(c1) is None
    plus_di = plus.update(c2)
    minus_di = minus.update(c2)
    # up=2, down=1 -> plus_di = 100*2/3, minus_di = 0
    assert pytest.approx(plus_di, rel=1e-3) == 100 * 2/3
    assert pytest.approx(minus_di) == 0.0


def test_supertrend_initial():
    st = si.SuperTrend(atr_length=3, factor=1)
    c = {'high':5, 'low':1, 'close':4, 'open':4, 'volume':1}
    # before ATR has period data, compute and update return None tuple
    assert st.compute(c) == (None, None)
    assert st.update(c) == (None, None)


def test_cwa2sigma_initial():
    cw = si.CWA2Sigma(bb_period=3, bb_width=1, ema_period=2, atr_period=2, atr_factor=1, sl_perc=20)
    # no data -> compute should not change signal
    c = {'high':1, 'low':1, 'close':1, 'open':1, 'volume':1}
    sig, entry = cw.compute(c)
    assert sig == 0
    assert entry is None
