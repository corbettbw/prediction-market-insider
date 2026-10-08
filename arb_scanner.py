#!/usr/bin/env python3
"""
Scan Kalshi vs Polymarket for cross-venue arbitrage on binary markets.

Arb condition: buy YES on one venue + NO on the other for less than $1.00
(after fees). Exactly one leg pays $1 at resolution, so profit per contract
pair = 1 - cost.

    pip install requests rapidfuzz
    python arb_scanner.py                       # one-shot scan
    python arb_scanner.py --watch 15            # poll every 15s, print on open/close

Read-only. No API keys needed. Market matching is fuzzy on titles, so every
hit needs a manual check of the resolution rules on both sides.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from rapidfuzz import fuzz, process, utils

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))


def get(url, **params):
    for attempt in range(5):
        r = S.get(url, params=params, timeout=30)
        if r.status_code in (429, 502, 503, 504) and attempt < 4:
            time.sleep(2 ** attempt)  # back off on rate limits / transient errors
            continue
        break
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} for {r.url}\nResponse body: {r.text[:500]}", response=r)
    return r.json()


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------- data ----------

def kalshi_row(m):
    ya, na = num(m.get("yes_ask_dollars")), num(m.get("no_ask_dollars"))
    if not ya or not na or m["ticker"].startswith("KXMVE"):  # empty book / combo market
        return None
    return {
        "ticker": m["ticker"],
        "title": f'{m.get("title", "")} {m.get("yes_sub_title", "")}'.strip(),
        "yes_ask": ya,
        "no_ask": na,
        # NO ask size == YES bid size on a binary book
        "yes_ask_sz": num(m.get("yes_ask_size_fp")) or 0,
        "no_ask_sz": num(m.get("yes_bid_size_fp")) or 0,
        "close": m.get("close_time"),
    }


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _split(edges):
    """Consecutive (lo, hi) windows over sorted edges, open-ended at both extremes."""
    bounds = [None] + list(edges) + [None]
    return list(zip(bounds[:-1], bounds[1:]))


def _kalshi_chain(lo, hi):
    """One sequential cursor chain over markets created in [lo, hi)."""
    out, cursor, raw = {}, None, 0
    while True:
        params = {"status": "open", "limit": 1000, "mve_filter": "exclude"}
        if lo:
            params["min_created_ts"] = lo
        if hi:
            params["max_created_ts"] = hi
        if cursor:
            params["cursor"] = cursor
        d = get(f"{KALSHI}/markets", **params)
        raw += len(d["markets"])
        for r in map(kalshi_row, d["markets"]):
            if r:
                out[r["ticker"]] = r
        cursor = d.get("cursor")
        if not cursor:
            return out, raw


def fetch_kalshi(workers=6):
    """A cursor chain is sequential, but disjoint created-time windows are independent chains,
    so run one chain per window concurrently. (min/max_created_ts is documented as compatible
    with status=open; mve_filter=exclude drops the auto-generated combo markets.)"""
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    edges, t = [], start
    while t < now:
        edges.append(int(t.timestamp()))
        t += timedelta(days=90)
    out, raw = {}, 0
    wins = _split(edges)
    with ThreadPoolExecutor(workers) as ex:
        for i, (rows, n) in enumerate(ex.map(lambda w: _kalshi_chain(*w), wins), 1):
            out.update(rows)
            raw += n
    print(f"Kalshi: scanned {raw} raw markets, kept {len(out)} with a live YES and NO ask", flush=True)
    return list(out.values())


def _poly_row(m):
    if m.get("active") is False or m.get("acceptingOrders") is False:
        return None
    try:
        outcomes = json.loads(m["outcomes"])
        tokens = json.loads(m["clobTokenIds"])
    except (KeyError, TypeError, ValueError):
        return None
    bid, ask = num(m.get("bestBid")), num(m.get("bestAsk"))
    if outcomes != ["Yes", "No"] or not bid or not ask:
        return None
    return {
        "slug": m.get("slug"),
        "title": m["question"],
        "yes_ask": ask,
        "no_ask": 1 - bid,  # No book mirrors the Yes book
        "yes_sz": None,
        "no_sz": None,
        "yes_tok": tokens[0],
        "no_tok": tokens[1],
        "end": m.get("endDate"),
    }


def _poly_chain(lo, hi, min_liquidity):
    """One sequential keyset chain over markets with endDate in [lo, hi)."""
    out, cursor = {}, None
    while True:
        params = {"limit": 100, "closed": "false"}
        if min_liquidity:
            params["liquidity_num_min"] = min_liquidity
        if lo:
            params["end_date_min"] = lo
        if hi:
            params["end_date_max"] = hi
        if cursor:
            params["after_cursor"] = cursor
        resp = get(f"{GAMMA}/markets/keyset", **params)
        for r in map(_poly_row, resp.get("markets", [])):
            if r:
                out[r["slug"]] = r
        cursor = resp.get("next_cursor")
        if not cursor or not resp.get("markets"):
            return out


def fetch_poly(min_liquidity=0, workers=6):
    """Gamma /markets/keyset (offset is rejected, max 100/page). One chain per end-date window,
    run concurrently; windows are narrow where markets are dense (near-term) and wide further out.
    Markets with no endDate are not returned by a date-filtered query."""
    now = datetime.now(timezone.utc)
    days = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730, 1095]
    wins = _split([_iso(now + timedelta(days=d)) for d in days])
    out, done = {}, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_poly_chain, lo, hi, min_liquidity) for lo, hi in wins]
        for f in futs:
            out.update(f.result())
            done += 1
            print(f"  Polymarket: {done}/{len(wins)} windows done, {len(out)} markets", flush=True)
    return list(out.values())


def poly_best_ask(token):
    """(price, size) of the best ask from the live CLOB book, or None."""
    book = get(f"{CLOB}/book", token_id=token)
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
    return min(asks) if asks else None


# ---------- matching / math ----------

def parse_dt(s):
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (AttributeError, ValueError):
        return None


def date_gap_days(a, b):
    a, b = parse_dt(a), parse_dt(b)
    return abs((a - b).days) if a and b else 0


def kalshi_fee(p):
    """Approx. Kalshi taker fee per contract: 0.07 * p * (1 - p). Check current fee schedule."""
    return 0.07 * p * (1 - p)


def edge(kp, pp, poly_fee_rate=0.0):
    return 1 - kp - pp - kalshi_fee(kp) - poly_fee_rate * pp * (1 - pp)


def legs(k, p):
    """Both directions: (label, kalshi_price, kalshi_size, poly_price, poly_size, poly_token)."""
    return [
        ("Kalshi YES + Poly NO", k["yes_ask"], k["yes_ask_sz"], p["no_ask"], p["no_sz"], p["no_tok"]),
        ("Poly YES + Kalshi NO", k["no_ask"], k["no_ask_sz"], p["yes_ask"], p["yes_sz"], p["yes_tok"]),
    ]


def match_pairs(kal, poly, cutoff, gap_days, chunk=500):
    """Best Kalshi title for each Polymarket title. Vectorized (rapidfuzz cdist, all cores),
    with titles normalized once instead of on every comparison."""
    kt = [utils.default_process(k["title"]) for k in kal]
    pairs = []
    t0 = time.time()
    for i in range(0, len(poly), chunk):
        block = poly[i:i + chunk]
        pt = [utils.default_process(p["title"]) for p in block]
        sc = process.cdist(pt, kt, scorer=fuzz.token_sort_ratio, dtype=np.uint8,
                           score_cutoff=cutoff, workers=-1)
        best = sc.argmax(axis=1)
        for row, j in enumerate(best):
            score = sc[row, j]
            if score >= cutoff and date_gap_days(kal[j]["close"], block[row]["end"]) <= gap_days:
                pairs.append((kal[j], block[row], float(score)))
        print(f"  matched {min(i + chunk, len(poly))}/{len(poly)} ({time.time() - t0:.0f}s)", flush=True)
    return pairs


def find_hits(pairs, min_edge, poly_fee_rate):
    hits = []
    for k, p, score in pairs:
        for label, kp, ksz, pp, psz, tok in legs(k, p):
            e = edge(kp, pp, poly_fee_rate)
            if e >= min_edge:
                size = min(ksz, psz) if psz is not None else ksz
                hits.append({"edge": e, "cost": kp + pp, "label": label, "score": score,
                             "k": k, "p": p, "kp": kp, "pp": pp, "ksz": ksz, "size": size, "tok": tok})
    return sorted(hits, key=lambda h: -h["edge"])


def verify(h, poly_fee_rate):
    """Re-price the Polymarket leg from the live book and attach executable size."""
    best = poly_best_ask(h["tok"])
    if not best:
        return None
    pp, psz = best
    h["pp"], h["cost"] = pp, h["kp"] + pp
    h["edge"] = edge(h["kp"], pp, poly_fee_rate)
    h["size"] = min(h["ksz"], psz)
    return h


# ---------- output ----------

def show(h, tag=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f'\n[{ts}] {tag}{h["label"]}  edge={h["edge"]:.3f}  cost={h["cost"]:.3f}  '
          f'match={h["score"]:.0f}  size={h["size"]:.0f}')
    print(f'  Kalshi: {h["k"]["ticker"]} | {h["k"]["title"]} @ {h["kp"]:.3f}')
    print(f'  Poly:   {h["p"]["slug"]} | {h["p"]["title"]} @ {h["pp"]:.3f}')


# ---------- modes ----------

def discover(a):
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:  # the two venues are independent, so overlap the fetches
        fk, fp = ex.submit(fetch_kalshi, a.workers), ex.submit(fetch_poly, a.min_liquidity, a.workers)
        kal, poly = fk.result(), fp.result()
    print(f"Kalshi: {len(kal)} markets | Polymarket: {len(poly)} markets | fetched in {time.time() - t0:.0f}s")
    return match_pairs(kal, poly, a.match, a.date_gap)


def best_edge(pr, poly_fee):
    k, p, _ = pr
    return max(edge(kp, pp, poly_fee) for _, kp, _, pp, _, _ in legs(k, p))


def show_near(pr, poly_fee):
    k, p, score = pr
    print(f"  best edge {best_edge(pr, poly_fee):+.3f}  match={score:.0f}")
    print(f"    Kalshi: {k['ticker']} | {k['title']}")
    print(f"    Poly:   {p['slug']} | {p['title']}")


def one_shot(a):
    pairs = discover(a)
    hits = find_hits(pairs, a.min_edge, a.poly_fee)[: a.top * 3]
    if not a.no_verify:
        hits = [h for h in (verify(h, a.poly_fee) for h in hits) if h and h["edge"] >= a.min_edge]
    hits = sorted(hits, key=lambda h: -h["edge"])[: a.top]
    for h in hits:
        show(h)
    if not hits:
        print(f"No opportunities above threshold ({len(pairs)} matched pairs). Closest to an arb:")
        for pr in sorted(pairs, key=lambda pr: -best_edge(pr, a.poly_fee))[:5]:
            show_near(pr, a.poly_fee)


def refresh(pairs):
    """Re-price tracked pairs: Kalshi in batches by ticker, Polymarket from live books."""
    tickers = sorted({k["ticker"] for k, _, _ in pairs})
    fresh = {}
    for i in range(0, len(tickers), 100):
        for m in get(f"{KALSHI}/markets", tickers=",".join(tickers[i:i + 100]))["markets"]:
            r = kalshi_row(m)
            if r:
                fresh[r["ticker"]] = r

    def book(tok):
        try:
            return poly_best_ask(tok)
        except requests.RequestException:
            return None

    toks = sorted({t for _, p, _ in pairs for t in (p["yes_tok"], p["no_tok"])})
    with ThreadPoolExecutor(8) as ex:
        books = dict(zip(toks, ex.map(book, toks)))

    out = []
    for k, p, score in pairs:
        k2, y, n = fresh.get(k["ticker"]), books.get(p["yes_tok"]), books.get(p["no_tok"])
        if k2 and y and n:
            out.append((k2, {**p, "yes_ask": y[0], "yes_sz": y[1], "no_ask": n[0], "no_sz": n[1]}, score))
    return out


def load_pairs(a):
    """Matched pairs from the on-disk cache if fresh, otherwise a full (slow) discovery."""
    if a.cache and os.path.exists(a.cache) and time.time() - os.path.getmtime(a.cache) < a.rediscover:
        with open(a.cache) as f:
            pairs = [tuple(x) for x in json.load(f)]
        age = (time.time() - os.path.getmtime(a.cache)) / 60
        print(f"Loaded {len(pairs)} matched pairs from {a.cache} ({age:.0f} min old)")
        return pairs
    pairs = discover(a)
    if a.cache:
        with open(a.cache, "w") as f:
            json.dump(pairs, f)
    return pairs


def pick_tracked(pairs, a):
    """Rank matched pairs by how close they are to an arb, keep the top `--track` within `--band`."""
    ranked = sorted(pairs, key=lambda pr: -best_edge(pr, a.poly_fee))
    ranked = [pr for pr in ranked if best_edge(pr, a.poly_fee) >= a.min_edge - a.band]
    tracked = ranked[: a.track]
    print(f"Matched {len(pairs)} pairs; tracking the {len(tracked)} closest to an arb. Top 3:")
    for pr in tracked[:3]:
        show_near(pr, a.poly_fee)
    return tracked


def watch(a):
    bg = ThreadPoolExecutor(1)  # rediscovery runs in the background so polling never pauses for it
    tracked = pick_tracked(load_pairs(a), a)
    active, pending, last_disc, cycle = {}, None, time.time(), 0
    while True:
        try:
            if pending is None and time.time() - last_disc > a.rediscover:
                print("Refreshing market universe in the background...")
                pending = bg.submit(discover, a)
            if pending is not None and pending.done():
                try:
                    pairs = pending.result()
                    if a.cache:
                        with open(a.cache, "w") as f:
                            json.dump(pairs, f)
                    tracked = pick_tracked(pairs, a)
                except Exception as e:  # keep polling the old set if a refresh fails
                    print(f"rediscovery failed: {e}")
                pending, last_disc = None, time.time()

            hits = {(h["k"]["ticker"], h["p"]["slug"], h["label"]): h
                    for h in find_hits(refresh(tracked), a.min_edge, a.poly_fee)}
            for key, h in hits.items():
                if key not in active:
                    show(h, "OPEN   ")
            for key, h in active.items():
                if key not in hits:
                    print(f'\n[{datetime.now():%H:%M:%S}] CLOSED {h["label"]} | {h["k"]["ticker"]}')
            active = hits
        except requests.RequestException as e:
            print(f"request error: {e}")

        cycle += 1
        if a.cycles and cycle >= a.cycles:
            return
        time.sleep(a.watch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.01, help="min profit per $1 payout, after fees")
    ap.add_argument("--match", type=float, default=88, help="title similarity cutoff (0-100)")
    ap.add_argument("--date-gap", type=int, default=7, help="max days between close dates")
    ap.add_argument("--poly-fee", type=float, default=0.0, help="Polymarket fee rate, same p*(1-p) shape")
    ap.add_argument("--min-liquidity", type=float, default=1000,
                    help="Polymarket: skip markets below this liquidity (USD); 0 = no filter")
    ap.add_argument("--workers", type=int, default=6, help="concurrent requests per venue")
    ap.add_argument("--top", type=int, default=20, help="one-shot: max results")
    ap.add_argument("--no-verify", action="store_true", help="one-shot: skip live CLOB re-check")
    ap.add_argument("--watch", type=float, default=0, metavar="SECS", help="poll interval; 0 = one-shot")
    ap.add_argument("--band", type=float, default=0.05, help="watch: track pairs within this edge of threshold")
    ap.add_argument("--track", type=int, default=300, help="watch: max pairs to poll (closest to an arb first)")
    ap.add_argument("--cache", default="pairs_cache.json",
                    help="watch: cache of matched pairs so restarts skip the slow fetch; '' disables")
    ap.add_argument("--rediscover", type=float, default=1800, help="watch: seconds between full re-matches")
    ap.add_argument("--cycles", type=int, default=0, help="watch: stop after N polls (0 = forever)")
    a = ap.parse_args()

    if a.watch:
        try:
            watch(a)
        except KeyboardInterrupt:
            pass
    else:
        one_shot(a)


if __name__ == "__main__":
    main()