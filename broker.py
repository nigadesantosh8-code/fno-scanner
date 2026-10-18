"""Angel One SmartAPI adapter (analysis only, no orders).
.env keys: ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PASSWORD (MPIN), ANGEL_TOTP_SECRET,
           TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional)"""
import datetime as dt
import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
SCRIP = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
# symbol: (spot exchange, spot token, derivatives exchange)
IDX = {"NIFTY": ("NSE", "99926000", "NFO"), "BANKNIFTY": ("NSE", "99926009", "NFO"),
       "FINNIFTY": ("NSE", "99926037", "NFO"), "SENSEX": ("BSE", "99919000", "BFO")}
IV = {"15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR", "1D": "ONE_DAY"}
DAYS = {"15m": 30, "30m": 30, "1h": 120, "1D": 365}
SNAP = "oi_snap.json"


class Angel:
    def __init__(self):
        import pyotp
        from SmartApi import SmartConnect
        self.s = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
        d = self.s.generateSession(os.environ["ANGEL_CLIENT_CODE"], os.environ["ANGEL_PASSWORD"],
                                   pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now())
        if not d.get("status"):
            raise RuntimeError(d.get("message", "login failed"))
        self.m = self._master()
        self.t0 = 0.0

    def _master(self):
        """Instrument master, downloaded once per day. Lot sizes/expiries come from here."""
        f = f"scrip_{dt.date.today()}.json"
        if not os.path.exists(f):
            with open(f, "wb") as fh:
                fh.write(requests.get(SCRIP, timeout=120).content)
        m = pd.DataFrame(json.load(open(f)))
        m["strike"] = m.strike.astype(float) / 100
        m["lotsize"] = m.lotsize.astype(int)
        m["exp"] = pd.to_datetime(m.expiry, format="%d%b%Y", errors="coerce")
        return m

    def _wait(self):  # Angel limit ~3 req/sec
        time.sleep(max(0, 0.4 - (time.time() - self.t0)))
        self.t0 = time.time()

    def fno_stocks(self):
        return sorted(self.m[self.m.instrumenttype == "FUTSTK"].name.unique())

    def _fut(self, sym):
        fx = IDX[sym][2] if sym in IDX else "NFO"
        r = self.m[(self.m.name == sym) & (self.m.exch_seg == fx) &
                   (self.m.instrumenttype.isin(["FUTIDX", "FUTSTK"])) &
                   (self.m.exp >= pd.Timestamp.today().normalize())].sort_values("exp")
        return r.iloc[0] if len(r) else None

    def lot(self, sym):
        f = self._fut(sym)
        return int(f.lotsize) if f is not None else None

    def candles(self, sym, tf):
        if sym in IDX:
            ex, tok = IDX[sym][:2]
        else:
            r = self.m[(self.m.exch_seg == "NSE") & (self.m.symbol == sym + "-EQ")]
            if r.empty:
                return pd.DataFrame()
            ex, tok = "NSE", r.token.iloc[0]
        now = dt.datetime.now()
        self._wait()
        d = self.s.getCandleData({"exchange": ex, "symboltoken": tok, "interval": IV[tf],
                                  "fromdate": (now - dt.timedelta(days=DAYS[tf])).strftime("%Y-%m-%d %H:%M"),
                                  "todate": now.strftime("%Y-%m-%d %H:%M")})
        if not d or not d.get("data"):
            return pd.DataFrame()
        df = pd.DataFrame(d["data"], columns=["t", "o", "h", "l", "c", "v"])
        df.index = pd.to_datetime(df.pop("t"))
        return df.astype(float)

    def _quotes(self, fx, tokens):
        self._wait()
        d = self.s.getMarketData("FULL", {fx: list(tokens)})
        return (d or {}).get("data", {}).get("fetched", []) or []

    def derivs(self, sym):
        """Futures OI buildup + option chain stats (PCR, walls, max pain)."""
        out = {}
        f = self._fut(sym)
        fx = IDX[sym][2] if sym in IDX else "NFO"
        if f is not None:
            q = self._quotes(fx, [f.token])
            if q:
                q = q[0]
                oi, chg = float(q.get("opnInterest", 0)), float(q.get("netChange", 0))
                out["fut_ltp"] = float(q.get("ltp", 0))
                snap = json.load(open(SNAP)) if os.path.exists(SNAP) else {}
                prev = snap.get(sym)
                today = str(dt.date.today())
                if prev and prev["d"] == today and oi != prev["oi"]:
                    up = oi > prev["oi"]
                    out["buildup"] = ("Long buildup" if chg > 0 else "Short buildup") if up else \
                                     ("Short covering" if chg > 0 else "Long unwinding")
                    out["oi_chg"] = oi - prev["oi"]
                else:
                    out["buildup"] = "n/a (next scan)"
                if not prev or prev["d"] != today or oi != prev["oi"]:
                    snap[sym] = {"d": today, "oi": oi}
                    json.dump(snap, open(SNAP, "w"))
        o = self.m[(self.m.name == sym) & (self.m.exch_seg == fx) &
                   (self.m.instrumenttype.isin(["OPTIDX", "OPTSTK"])) &
                   (self.m.exp >= pd.Timestamp.today().normalize())]
        if o.empty:
            return out
        o = o[o.exp == o.exp.min()]
        out["expiry"] = o.expiry.iloc[0]
        strikes = sorted(o.strike.unique())
        ref = out.get("fut_ltp") or strikes[len(strikes) // 2]  # futures LTP ~ spot
        atm = min(strikes, key=lambda k: abs(k - ref))
        i = strikes.index(atm)
        keep = strikes[max(0, i - 12): i + 13]
        o = o[o.strike.isin(keep)].copy()
        o["type"] = o.symbol.str[-2:]
        q = pd.DataFrame(self._quotes(fx, o.token.tolist()))
        if q.empty:
            return out
        q = q.rename(columns={"symbolToken": "token"})[["token", "opnInterest", "ltp"]]
        o = o.merge(q, on="token")
        ce, pe = o[o.type == "CE"], o[o.type == "PE"]
        out["atm"] = atm
        out["pcr"] = round(pe.opnInterest.sum() / max(ce.opnInterest.sum(), 1), 2)
        out["call_wall"] = ce.loc[ce.opnInterest.idxmax(), "strike"]
        out["put_wall"] = pe.loc[pe.opnInterest.idxmax(), "strike"]
        pain = {k: (ce.opnInterest * (k - ce.strike).clip(lower=0)).sum() +
                   (pe.opnInterest * (pe.strike - k).clip(lower=0)).sum() for k in keep}
        out["max_pain"] = min(pain, key=pain.get)
        out["chain"] = o.pivot_table(index="strike", columns="type", values="opnInterest").reset_index()
        return out


def telegram(msg):
    t, c = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if t and c:
        try:
            requests.post(f"https://api.telegram.org/bot{t}/sendMessage", data={"chat_id": c, "text": msg}, timeout=10)
        except Exception:
            pass
