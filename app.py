"""
Personal F&O Scanner (analysis only, NO order placement)
Run:  pip install streamlit yfinance pandas numpy plotly
      streamlit run app.py
Data: yfinance (free, thoda delayed). Live broker data ke liye fetch() ko
Angel One / Upstox adapter se replace karna (Step 2).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from broker import Angel, telegram
except Exception:  # SmartApi/dotenv install nahi to yfinance chalega
    Angel, telegram = None, (lambda m: None)


@st.cache_resource
def get_broker():
    try:
        return Angel()
    except Exception as e:
        return str(e)


B = get_broker() if Angel else None
LIVE = B is not None and not isinstance(B, str)

st.set_page_config(page_title="F&O Scanner", layout="wide")
TFS = {"15m": ("15m", "30d"), "30m": ("30m", "30d"), "1h": ("60m", "120d"), "1D": ("1d", "1y")}
INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "SENSEX": "^BSESN"}
DEFAULT_WL = "RELIANCE,HDFCBANK,ICICIBANK,SBIN,TCS,INFY,TATAMOTORS,AXISBANK,ITC,LT,MARUTI,BAJFINANCE"

# ---------------- sidebar settings ----------------
sb = st.sidebar
sb.header("Key Levels")
PL = sb.number_input("Pivot Length", 2, 20, 6)
PM = sb.number_input("Pivot Memory", 4, 40, 10)
NL = sb.number_input("Number of Levels", 1, 8, 4)
sb.caption("Data: " + ("Angel One LIVE" if LIVE else "yfinance (delayed)" + (f" | Angel: {B}" if isinstance(B, str) else "")))
USE_OI = sb.checkbox("OI + Option chain score", LIVE, disabled=not LIVE)
ALERT = sb.checkbox("Telegram alerts", False)
sb.header("Risk")
CAP = sb.number_input("Capital (₹)", 10000, 10**8, 200000, 10000)
RISK = sb.number_input("Risk % per trade", 0.1, 5.0, 1.0, 0.1)
LOT = sb.number_input("Lot size (symbol ka, NSE se check karo)", 1, 10000, 500)
MIN_SCORE = sb.slider("Min score for signal", 40, 100, 70)
if LIVE and sb.checkbox("Saare F&O stocks (slow)", False):
    DEFAULT_WL = ",".join(B.fno_stocks())
WL = [s.strip().upper() for s in sb.text_area("Watchlist", DEFAULT_WL).split(",") if s.strip()]


def ticker(s):
    return INDEX.get(s, s + ".NS")


@st.cache_data(ttl=60, show_spinner=False)
def fetch(sym, tf):
    iv, per = TFS[tf]
    if LIVE:
        try:
            df = B.candles(sym, tf)
            if len(df):
                return df
        except Exception:
            pass
    try:
        df = yf.Ticker(ticker(sym)).history(period=per, interval=iv)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.columns = ["o", "h", "l", "c", "v"]
        return df
    except Exception:
        return pd.DataFrame()


# ---------------- indicators ----------------
def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def supertrend_dir(df, n=10, m=3.0):
    a = atr(df, n).values
    hl2 = ((df.h + df.l) / 2).values
    c = df.c.values
    ub, lb = hl2 + m * a, hl2 - m * a
    d = np.ones(len(df))
    for i in range(1, len(df)):
        if c[i] > ub[i - 1]:
            d[i] = 1
        elif c[i] < lb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
            if d[i] > 0:
                lb[i] = max(lb[i], lb[i - 1])
            else:
                ub[i] = min(ub[i], ub[i - 1])
    return d[-1]


def vwap(df, intraday):
    tp = (df.h + df.l + df.c) / 3
    if intraday:
        day = df.index.date
        return (tp * df.v).groupby(day).cumsum() / df.v.groupby(day).cumsum().replace(0, np.nan)
    return (tp * df.v).rolling(20).sum() / df.v.rolling(20).sum().replace(0, np.nan)


# ---------------- Key Levels ----------------
def key_levels(df, length, memory, n_levels):
    """Pivot high/low -> last `memory` pivots -> cluster (ATR*0.5) -> top zones."""
    w = 2 * length + 1
    ph = df.h[df.h == df.h.rolling(w, center=True).max()]
    pl = df.l[df.l == df.l.rolling(w, center=True).min()]
    piv = pd.concat([ph, pl]).sort_index().tail(memory)
    if piv.empty:
        return []
    width = atr(df).iloc[-1] * 0.5
    prices = sorted(piv.values)
    clusters, cur = [], [prices[0]]
    for p in prices[1:]:
        if p - np.mean(cur) <= width:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    zones = []
    for cl in clusters:
        mid = float(np.mean(cl))
        zones.append(dict(lo=mid - width / 2, hi=mid + width / 2, mid=mid,
                          n=len(cl), strength=round(100 * len(cl) / len(piv))))
    zones.sort(key=lambda z: -z["n"])
    return sorted(zones[:n_levels], key=lambda z: z["mid"])


# ---------------- scoring (0-100) ----------------
def analyse(sym, tf, oi=None):
    df = fetch(sym, tf)
    if len(df) < 60:
        return None
    c, o, h, l = df.c.iloc[-1], df.o.iloc[-1], df.h.iloc[-1], df.l.iloc[-1]
    a = atr(df).iloc[-1]
    e20, e50 = df.c.ewm(span=20).mean().iloc[-1], df.c.ewm(span=50).mean().iloc[-1]
    vw = vwap(df, tf != "1D").iloc[-1]
    r = rsi(df.c).iloc[-1]
    st_dir = supertrend_dir(df)
    zones = key_levels(df, PL, PM, NL)
    bull = bear = 0.0
    why = []
    # 1 trend (25)
    if c > e20 > e50:
        bull += 20; why.append("EMA up")
    elif c < e20 < e50:
        bear += 20; why.append("EMA down")
    if not np.isnan(vw):
        bull, bear = (bull + 5, bear) if c > vw else (bull, bear + 5)
    # 2 level reaction (25)
    body = abs(c - o)
    for z in zones:
        if l <= z["hi"] + 0.25 * a and c > z["mid"] and c > o and (min(o, c) - l) >= body:
            bull += 25; why.append(f"Support rejection {z['mid']:.1f}"); break
        if h >= z["lo"] - 0.25 * a and c < z["mid"] and c < o and (h - max(o, c)) >= body:
            bear += 25; why.append(f"Resistance rejection {z['mid']:.1f}"); break
    # 3 volume (15)
    vol_spike = bool(df.v.iloc[-1] > 1.5 * df.v.iloc[-21:-1].mean())
    if vol_spike:
        if c > o: bull += 15
        else: bear += 15
        why.append("Vol spike")
    # 4 momentum (15)
    if r > 55: bull += 7.5
    elif r < 45: bear += 7.5
    if st_dir > 0: bull += 7.5
    else: bear += 7.5
    # 5 futures OI buildup (10) + 6 options PCR/walls (10) - sirf broker data ke saath
    mx = 80
    if oi:
        mx = 100
        b = oi.get("buildup", "")
        if b in ("Long buildup", "Short covering"): bull += 10; why.append(b)
        elif b in ("Short buildup", "Long unwinding"): bear += 10; why.append(b)
        pcr = oi.get("pcr")
        if pcr:
            if pcr >= 1.1 and c > oi["put_wall"]: bull += 10; why.append(f"PCR {pcr}")
            elif pcr <= 0.8 and c < oi["call_wall"]: bear += 10; why.append(f"PCR {pcr}")
    bull, bear = round(bull / mx * 100), round(bear / mx * 100)
    side = "BULL" if bull >= bear else "BEAR"
    nearest = min(zones, key=lambda z: abs(z["mid"] - c)) if zones else None
    return dict(df=df, c=c, a=a, zones=zones, bull=bull, bear=bear, side=side,
                score=max(bull, bear), vol=vol_spike, why=why, nearest=nearest)


def mtf(sym):
    oi = None
    if LIVE and USE_OI:
        try:
            oi = B.derivs(sym)
        except Exception:
            oi = None
    res = {tf: analyse(sym, tf, oi) for tf in TFS}
    ok = {k: v for k, v in res.items() if v}
    if len(ok) < 3:
        return None
    sides = [v["side"] for v in ok.values()]
    n_bull = sides.count("BULL")
    n_bear = sides.count("BEAR")
    d1, h1 = ok.get("1D"), ok.get("1h")
    bias = "BULL" if n_bull >= 3 else "BEAR" if n_bear >= 3 else "MIXED"
    trig = ok.get("15m") or ok["30m"]
    sig = "NO TRADE"
    if bias != "MIXED" and trig["score"] >= MIN_SCORE and trig["side"] == bias and \
            (not d1 or d1["side"] == bias) and (not h1 or h1["side"] == bias):
        sig = "BUY CE / FUT LONG" if bias == "BULL" else "BUY PE / FUT SHORT"
    return dict(oi=oi, res=res, bias=bias, sig=sig, align=f"{max(n_bull, n_bear)}/{len(ok)}")


def plan(r, side):
    c, a = r["c"], r["a"]
    risk = 1.5 * a
    zs = r["zones"]
    if side == "BULL":
        sl = c - risk
        tg = [z["mid"] for z in zs if z["mid"] > c + 0.3 * a][:3]
        tg += [c + risk * k for k in (2, 3, 4)][len(tg):]
    else:
        sl = c + risk
        tg = [z["mid"] for z in sorted(zs, key=lambda z: -z["mid"]) if z["mid"] < c - 0.3 * a][:3]
        tg += [c - risk * k for k in (2, 3, 4)][len(tg):]
    rr = abs(tg[0] - c) / risk
    lots = int((CAP * RISK / 100) // (risk * LOT))
    return dict(entry=c, sl=sl, targets=tg, rr=rr, lots=lots, risk_amt=lots * risk * LOT)


# ---------------- UI ----------------
t1, t2, t3 = st.tabs(["Scanner", "Chart", "Trade Plan"])

with t1:
    st.caption("Analysis only. Koi auto order nahi. Data yfinance (delayed) - paper trade pehle.")
    if st.button("Scan now", type="primary"):
        rows, bar = [], st.progress(0)
        for i, s in enumerate(list(INDEX) + WL):
            m = mtf(s)
            bar.progress((i + 1) / (len(WL) + len(INDEX)))
            if not m:
                continue
            base = m["res"].get("15m") or m["res"]["30m"]
            nz = base["nearest"]
            row = {"Symbol": s}
            for tf in TFS:
                v = m["res"].get(tf)
                row[tf] = f"{'▲' if v['side']=='BULL' else '▼'} {v['score']}" if v else "-"
            oi = m["oi"] or {}
            if ALERT and m["sig"] != "NO TRADE":
                telegram(f"{s}: {m['sig']} | score {base['score']} | align {m['align']}")
            row.update({"OI": oi.get("buildup", ""), "PCR": oi.get("pcr"), "Align": m["align"], "Bias": m["bias"], "Signal": m["sig"],
                        "Vol": "🔥" if base["vol"] else "",
                        "Level": round(nz["mid"], 1) if nz else None,
                        "Dist %": round(abs(base["c"] - nz["mid"]) / base["c"] * 100, 2) if nz else None})
            rows.append(row)
        bar.empty()
        st.session_state["scan"] = pd.DataFrame(rows)
    if "scan" in st.session_state:
        d = st.session_state["scan"]
        f = st.radio("Filter", ["All", "Signals only", "Bullish", "Bearish"], horizontal=True)
        if f == "Signals only": d = d[d.Signal != "NO TRADE"]
        elif f == "Bullish": d = d[d.Bias == "BULL"]
        elif f == "Bearish": d = d[d.Bias == "BEAR"]
        st.dataframe(d, use_container_width=True, hide_index=True)

with t2:
    sym = st.selectbox("Symbol", list(INDEX) + WL, key="sym")
    tf = st.radio("Timeframe", list(TFS), horizontal=True, index=3)
    r = analyse(sym, tf)
    if r:
        df = r["df"].tail(150)
        fig = go.Figure(go.Candlestick(x=df.index, open=df.o, high=df.h, low=df.l, close=df.c))
        for z in r["zones"]:
            fig.add_hrect(y0=z["lo"], y1=z["hi"], fillcolor="orange", opacity=0.25, line_width=0,
                          annotation_text=f"{z['strength']}%", annotation_position="right")
        fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.update_xaxes(type="category", showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Bull {r['bull']} | Bear {r['bear']} | " + ", ".join(r["why"]))
    else:
        st.warning("Data nahi mila.")

with t3:
    sym = st.session_state.get("sym", WL[0])
    m = mtf(sym)
    if m:
        st.subheader(f"{sym}: {m['sig']} (bias {m['bias']}, align {m['align']})")
        base = m["res"].get("15m") or m["res"]["30m"]
        side = m["bias"] if m["bias"] != "MIXED" else base["side"]
        p = plan(base, side)
        c1, c2, c3 = st.columns(3)
        c1.metric("Entry", f"{p['entry']:.2f}")
        c2.metric("Stop-loss", f"{p['sl']:.2f}")
        c3.metric("R:R (T1)", f"{p['rr']:.1f}")
        st.write("Targets:", ", ".join(f"{t:.2f}" for t in p["targets"]))
        st.write(f"Lots: **{p['lots']}** (risk ≈ ₹{p['risk_amt']:,.0f}). "
                 "Strike: ATM / 1-step ITM, nearest expiry (broker option chain se confirm karo).")
        o = m["oi"]
        if o and "pcr" in o:
            st.write(f"**Options ({o['expiry']})**: PCR {o['pcr']} | Call wall (resistance) {o['call_wall']:.0f} | "
                     f"Put wall (support) {o['put_wall']:.0f} | Max pain {o['max_pain']:.0f} | ATM {o['atm']:.0f}")
            if "chain" in o:
                st.dataframe(o["chain"].style.background_gradient(cmap="Oranges"), hide_index=True)
        if LIVE:
            lt = B.lot(sym)
            if lt:
                st.caption(f"Lot size (instrument master): {lt}")
        if p["rr"] < 2:
            st.warning("R:R 1:2 se kam hai - trade skip karo.")
        if m["sig"] == "NO TRADE":
            st.info("Score/alignment kam hai - abhi trade nahi.")
