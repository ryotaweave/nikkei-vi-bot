#!/usr/bin/env python3
"""
日経平均VI (Nikkei 225 Volatility Index) -> Microsoft Teams notifier.

Posts the latest VI close with quantified context and a chart.

Data sources (all free, no API keys):
  * VI daily OHLC     : indexes.nikkei.co.jp  ...nikkei_stock_average_vi_daily_jp.csv
  * Nikkei 225 daily  : indexes.nikkei.co.jp  ...nikkei_stock_average_daily_jp.csv
  * Macro comparators : query1.finance.yahoo.com/v8/finance/chart/<symbol>

Design notes:
  * We report *co-movements*, never causes. Attributing a VI move to a macro
    driver is inherently speculative, so the card presents same-day changes in
    related markets and lets the reader judge.
  * Regime labels are percentile-based (self-calibrating against the last year)
    rather than hard-coded thresholds that drift as the vol regime changes.
  * De-duplication is by the rate's own date (see tibor-teams-bot for why):
    GitHub's cron is best-effort, so "is it today?" checks silently drop data.

Modes:
  --prepare    fetch + compute + write chart PNG and out/card.json (no posting)
  --post       POST out/card.json to the webhook, then record state
  --dry-run    prepare, then print the card instead of posting
Run with no flags to do prepare+post in one go (local use).
"""

import csv
import datetime as dt
import io
import json
import os
import re
import statistics
import sys
from zoneinfo import ZoneInfo

import requests

import matplotlib
matplotlib.use("Agg")            # headless: no display on CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

JST = ZoneInfo("Asia/Tokyo")
UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept-Language": "ja,en;q=0.9",
}

VI_CSV = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_vi_daily_jp.csv"
N225_CSV = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_jp.csv"
VI_PAGE = "https://indexes.nikkei.co.jp/nkave/index/profile?idx=nk225vi"

# label, yahoo symbol, kind ("pct" -> show % change, "bp" -> yield in bp)
MACRO = [
    ("ドル円", "USDJPY=X", "pct"),
    ("米VIX", "^VIX", "pct"),
    ("S&P500", "^GSPC", "pct"),
    ("米10年金利", "^TNX", "bp"),
]

CARD_JSON = "out/card.json"


def log(msg):
    print(f"[vi-bot] {msg}", flush=True)


# ---------------------------------------------------------------- data loading

def fetch_nikkei_csv(url):
    """Return [(date, close, open, high, low), ...] oldest-first."""
    r = requests.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    text = r.content.decode("shift_jis", errors="replace")
    rows = []
    for parts in csv.reader(io.StringIO(text)):
        if len(parts) < 5:
            continue
        d = parts[0].strip()
        if not re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", d):
            continue        # header row and the trailing copyright notice
        try:
            vals = [float(p.replace(",", "")) for p in parts[1:5]]
        except ValueError:
            continue
        y, m, day = (int(x) for x in d.split("/"))
        rows.append((dt.date(y, m, day), *vals))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise RuntimeError(f"no data rows parsed from {url}")
    return rows


def fetch_yahoo(symbol, rng="1mo"):
    """Return [(date, close), ...] oldest-first, or [] on failure."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        r = requests.get(url, headers=UA, timeout=30,
                         params={"range": rng, "interval": "1d"})
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        ts = res["timestamp"]
        closes = res["indicators"]["quote"][0]["close"]
        out = []
        for t, c in zip(ts, closes):
            if c is None:
                continue
            d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date()
            out.append((d, float(c)))
        return out
    except Exception as e:                      # a missing comparator must not
        log(f"WARNING: macro fetch failed for {symbol}: {e}")
        return []                              # kill the whole notification


# ------------------------------------------------------------------- analytics

def pct_change(new, old):
    return (new / old - 1.0) * 100.0 if old else 0.0


def percentile_rank(values, x):
    """% of observations at or below x."""
    if not values:
        return None
    return 100.0 * sum(1 for v in values if v <= x) / len(values)


def realized_vol(closes, window=20):
    """Annualised close-to-close volatility in % (needs window+1 closes)."""
    if len(closes) < window + 1:
        return None
    import math
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    rets = rets[-window:]
    if len(rets) < 2:
        return None
    return statistics.stdev(rets) * math.sqrt(252) * 100.0


def correlation(xs, ys):
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    try:
        return statistics.correlation(xs, ys)
    except Exception:
        return None


def regime_label(pct_1y):
    """Percentile-based regime, so it self-calibrates to the current era."""
    if pct_1y is None:
        return "—"
    if pct_1y >= 98:
        return "極度の緊張（過去1年で最高水準）"
    if pct_1y >= 90:
        return "高い（警戒領域）"
    if pct_1y >= 75:
        return "やや高い"
    if pct_1y >= 25:
        return "平常レンジ"
    if pct_1y >= 10:
        return "低い（落ち着き）"
    return "極めて低い"


def fmt_signed(x, digits=2, unit=""):
    return f"{'+' if x >= 0 else '-'}{abs(x):.{digits}f}{unit}"


def arrow(x):
    return "↑" if x > 0 else ("↓" if x < 0 else "→")


def analyse():
    vi = fetch_nikkei_csv(VI_CSV)
    n225 = fetch_nikkei_csv(N225_CSV)
    log(f"VI rows={len(vi)} (through {vi[-1][0]}), N225 rows={len(n225)}")

    d, close, open_, high, low = vi[-1]
    prev_close = vi[-2][1]
    chg = close - prev_close
    chg_pct = pct_change(close, prev_close)

    vi_closes = [r[1] for r in vi]
    closes_1y = vi_closes[-250:]
    p1y = percentile_rank(closes_1y, close)
    p3y = percentile_rank(vi_closes, close)
    med1y = statistics.median(closes_1y)

    def back(n):
        return vi_closes[-1 - n] if len(vi_closes) > n else None

    wk, mo = back(5), back(21)

    # implied (VI) vs realised vol of the underlying -> variance risk premium
    n225_map = {r[0]: r[1] for r in n225}
    n225_closes = [r[1] for r in n225 if r[0] <= d]
    rv20 = realized_vol(n225_closes, 20)

    # 20d correlation of daily changes: VI vs Nikkei 225
    common = [x for x in (r[0] for r in vi) if x in n225_map][-21:]
    vi_map = {r[0]: r[1] for r in vi}
    vi_ret, nk_ret = [], []
    for i in range(1, len(common)):
        a, b = common[i - 1], common[i]
        vi_ret.append(pct_change(vi_map[b], vi_map[a]))
        nk_ret.append(pct_change(n225_map[b], n225_map[a]))
    corr = correlation(vi_ret, nk_ret)

    # Nikkei 225 same-day move
    nk_rows = [r for r in n225 if r[0] <= d]
    nk_today = nk_rows[-1] if nk_rows else None
    nk_chg_pct = (pct_change(nk_rows[-1][1], nk_rows[-2][1])
                  if len(nk_rows) >= 2 else None)

    # macro comparators
    macro = []
    for label, sym, kind in MACRO:
        series = fetch_yahoo(sym)
        if len(series) < 2:
            continue
        m_d, m_c = series[-1]
        _, m_prev = series[-2]
        if kind == "bp":
            delta = (m_c - m_prev) * 100.0        # ^TNX is in %, so 1.0 = 100bp
            value = f"{m_c:.2f}%  {arrow(delta)} {fmt_signed(delta, 0, 'bp')}"
        else:
            p = pct_change(m_c, m_prev)
            digits = 2 if m_c < 1000 else 0
            value = f"{m_c:,.{digits}f}  {arrow(p)} {fmt_signed(p, 2, '%')}"
        # Always stamp the observation date: these markets keep different hours
        # from Tokyo, so the newest bar is usually the *previous* session and
        # presenting it undated would imply a same-day close that doesn't exist.
        macro.append((f"{label}（{m_d:%m/%d}）", value))

    return {
        "date": d,
        "close": close, "open": open_, "high": high, "low": low,
        "prev_close": prev_close, "chg": chg, "chg_pct": chg_pct,
        "p1y": p1y, "p3y": p3y, "med1y": med1y,
        "wk_pct": pct_change(close, wk) if wk else None,
        "mo_pct": pct_change(close, mo) if mo else None,
        "rv20": rv20,
        "corr": corr,
        "n225": nk_today[1] if nk_today else None,
        "n225_date": nk_today[0] if nk_today else None,
        "n225_chg_pct": nk_chg_pct,
        "macro": macro,
        "vi_series": [(r[0], r[1]) for r in vi],
        "vi_ohlc": vi,
        "n225_series": [(r[0], r[1]) for r in n225],
    }


# ----------------------------------------------------------------------- chart

def make_chart(a, path, days=120):
    """VI line with the Nikkei 225 overlaid; English labels avoid CJK font deps."""
    vi = a["vi_series"][-days:]
    start = vi[0][0]
    nk = [(d, c) for d, c in a["n225_series"] if d >= start]
    ohlc = {d: (h, l) for d, _c, _o, h, l in a["vi_ohlc"] if d >= start}

    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    fig.patch.set_facecolor("white")

    vd = [d for d, _ in vi]
    vc = [c for _, c in vi]

    # intraday high-low band: a vol index can swing hugely within a session, so
    # the close alone understates the day's actual range.
    if all(d in ohlc for d in vd):
        ax.fill_between(vd, [ohlc[d][0] for d in vd], [ohlc[d][1] for d in vd],
                        color="#c2185b", alpha=0.13, linewidth=0, zorder=1,
                        label="intraday range")

    ax.plot(vd, vc, color="#c2185b", linewidth=1.9, zorder=3, label="VI close")

    med = statistics.median([c for _, c in a["vi_series"][-250:]])
    ax.axhline(med, color="#9e9e9e", linestyle="--", linewidth=1,
               zorder=1, label=f"1y median {med:.1f}")

    # highlight the latest point
    ax.scatter([vd[-1]], [vc[-1]], s=42, color="#c2185b", zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.annotate(f"{vc[-1]:.2f}", (vd[-1], vc[-1]), textcoords="offset points",
                xytext=(8, 6), fontsize=11, fontweight="bold", color="#c2185b")

    ax.set_ylabel("VI (implied volatility, %)", fontsize=9, color="#c2185b")
    ax.tick_params(axis="y", labelcolor="#c2185b", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot([d for d, _ in nk], [c for _, c in nk],
             color="#1565c0", linewidth=1.3, alpha=0.75, zorder=2,
             label="Nikkei 225 (right)")
    ax2.set_ylabel("Nikkei 225", fontsize=9, color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0", labelsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))

    ax.set_title(f"Nikkei 225 Volatility Index  —  {a['date']:%Y/%m/%d} close "
                 f"{a['close']:.2f} ({fmt_signed(a['chg_pct'], 2, '%')})",
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8,
              framealpha=0.9, ncol=3)

    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    log(f"chart written -> {path} ({os.path.getsize(path)} bytes, {len(vi)} points)")


# ------------------------------------------------------------------------ card

def build_card(a, chart_url):
    up = a["chg"] > 0
    colour = "Attention" if up else ("Good" if a["chg"] < 0 else "Default")

    facts = [
        {"title": "前日比", "value": f"{fmt_signed(a['chg'])}  ({fmt_signed(a['chg_pct'], 2, '%')})  "
                                    f"前日終値 {a['prev_close']:.2f}"},
        {"title": "日中レンジ", "value": f"始値 {a['open']:.2f} / 高値 {a['high']:.2f} / 安値 {a['low']:.2f}"},
    ]
    if a["wk_pct"] is not None:
        facts.append({"title": "1週間前比", "value": f"{fmt_signed(a['wk_pct'], 1, '%')}"})
    if a["mo_pct"] is not None:
        facts.append({"title": "1ヶ月前比", "value": f"{fmt_signed(a['mo_pct'], 1, '%')}"})
    if a["p1y"] is not None:
        facts.append({"title": "過去1年での位置", "value":
                      f"下位から {a['p1y']:.0f}%（{regime_label(a['p1y'])}）"
                      f"／中央値 {a['med1y']:.2f}"})
    if a["p3y"] is not None:
        facts.append({"title": "過去3年での位置", "value": f"下位から {a['p3y']:.0f}%"})
    if a["rv20"] is not None:
        spread = a["close"] - a["rv20"]
        facts.append({"title": "実現ボラとの差", "value":
                      f"VI {a['close']:.2f} − 日経225の実現ボラ(20日) {a['rv20']:.2f} "
                      f"= {fmt_signed(spread)}"})
    if a["corr"] is not None:
        facts.append({"title": "日経225との連動(20日)", "value":
                      f"日次変化の相関 {a['corr']:+.2f}（−1に近いほど逆行）"})

    macro_facts = []
    if a["n225"] is not None and a["n225_chg_pct"] is not None:
        lbl = "日経225"
        if a["n225_date"] != a["date"]:
            lbl = f"日経225（{a['n225_date']:%m/%d}時点）"
        macro_facts.append({"title": lbl, "value":
                            f"{a['n225']:,.2f}  {arrow(a['n225_chg_pct'])} "
                            f"{fmt_signed(a['n225_chg_pct'], 2, '%')}"})
    macro_facts += [{"title": t, "value": v} for t, v in a["macro"]]

    body = [
        {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "wrap": True,
         "text": "日経平均VI（日経平均ボラティリティー・インデックス）"},
        {"type": "TextBlock", "spacing": "None", "isSubtle": True, "wrap": True,
         "text": f"基準日: {a['date']:%Y/%m/%d}（終値）"},
        {"type": "ColumnSet", "spacing": "Small", "columns": [
            {"type": "Column", "width": "auto", "items": [
                {"type": "TextBlock", "text": f"{a['close']:.2f}",
                 "size": "ExtraLarge", "weight": "Bolder", "color": colour,
                 "spacing": "None"}]},
            {"type": "Column", "width": "stretch", "verticalContentAlignment": "Center",
             "items": [
                {"type": "TextBlock", "spacing": "None", "wrap": True,
                 "weight": "Bolder", "color": colour,
                 "text": f"{arrow(a['chg'])} {fmt_signed(a['chg'])} "
                         f"({fmt_signed(a['chg_pct'], 2, '%')})"},
                {"type": "TextBlock", "spacing": "None", "wrap": True,
                 "isSubtle": True, "text": regime_label(a["p1y"])}]},
        ]},
        {"type": "FactSet", "facts": facts},
        {"type": "TextBlock", "text": "同時に動いた主要指標", "weight": "Bolder",
         "separator": True, "spacing": "Medium", "wrap": True},
        {"type": "FactSet", "facts": macro_facts},
        {"type": "TextBlock", "size": "Small", "isSubtle": True, "wrap": True,
         "spacing": "Small",
         "text": "※ 同じ時期に動いた指標を並べたものです。VIの変動要因（因果）を"
                 "示すものではありません。判断はご自身で行ってください。"},
    ]

    if chart_url:
        body.append({"type": "Image", "url": chart_url, "size": "Stretch",
                     "altText": "日経平均VIと日経225の推移", "spacing": "Medium"})

    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": body,
                "actions": [{"type": "Action.OpenUrl", "title": "日経公式ページを開く",
                             "url": VI_PAGE}],
            },
        }],
    }


# ------------------------------------------------------------------ state / io

def read_last_posted(path):
    try:
        with open(path, encoding="utf-8") as f:
            s = f.read().strip()
        return dt.date.fromisoformat(s) if s else None
    except (FileNotFoundError, ValueError):
        return None


def write_last_posted(path, d):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(d.isoformat())


def post_to_teams(webhook, card):
    r = requests.post(webhook, json=card, timeout=40)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"Teams webhook returned {r.status_code}: {r.text[:400]}")
    log(f"posted to Teams (HTTP {r.status_code})")


# ------------------------------------------------------------------------ main

def main():
    args = set(sys.argv[1:])
    prepare = "--prepare" in args
    post = "--post" in args
    dry = "--dry-run" in args
    if not (prepare or post):
        prepare = post = True          # local one-shot

    state_file = os.environ.get("STATE_FILE", "state/last_posted.txt")
    chart_path = os.environ.get("CHART_PATH", "charts/latest.png")
    always = os.environ.get("ALWAYS_POST", "").strip() in ("1", "true", "True")
    webhook = os.environ.get("TEAMS_WEBHOOK_URL", "").strip()

    if prepare:
        a = analyse()
        log(f"VI {a['date']} close={a['close']:.2f} "
            f"({fmt_signed(a['chg'])}, {fmt_signed(a['chg_pct'], 2, '%')})  "
            f"1y pct={a['p1y']:.0f}%  rv20={a['rv20'] and round(a['rv20'], 2)}")

        last = read_last_posted(state_file)
        log(f"last posted: {last}")
        if last is not None and a["date"] <= last and not always:
            log(f"VI for {a['date']} already posted — nothing new, exiting.")
            return 0

        make_chart(a, chart_path)

        base = os.environ.get("CHART_URL_BASE", "").strip()
        chart_url = f"{base}?v={a['date']:%Y%m%d}" if base else ""
        if not chart_url:
            log("note: CHART_URL_BASE unset — card will have no image "
                "(set it in CI so Teams can fetch the PNG)")

        card = build_card(a, chart_url)
        os.makedirs(os.path.dirname(CARD_JSON) or ".", exist_ok=True)
        with open(CARD_JSON, "w", encoding="utf-8") as f:
            json.dump({"date": a["date"].isoformat(), "card": card}, f,
                      ensure_ascii=False, indent=1)
        log(f"card written -> {CARD_JSON}")

        if dry:
            print(json.dumps(card, ensure_ascii=False, indent=2))
            log("DRY RUN — not posting.")
            return 0

    if post:
        if not os.path.exists(CARD_JSON):
            log("nothing prepared (no new data) — skipping post.")
            return 0
        if not webhook:
            log("TEAMS_WEBHOOK_URL not set — skipping post "
                "(card is ready in out/card.json).")
            return 0
        with open(CARD_JSON, encoding="utf-8") as f:
            payload = json.load(f)
        post_to_teams(webhook, payload["card"])
        d = dt.date.fromisoformat(payload["date"])
        write_last_posted(state_file, d)
        log(f"recorded last-posted {d} -> {state_file}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
