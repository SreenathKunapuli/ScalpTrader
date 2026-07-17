"use client";
// Closed-trade blotter with "Today" summary + tape + P&L histogram above history.
import { useEffect, useState, useMemo } from "react";
import { api } from "@/lib/api";
import { fmtET } from "@/lib/time";

// ──────────────────────────── types ────────────────────────────

type TodayTrade = {
  symbol: string;
  side: string;
  qty: number;
  entry_ts: string;
  exit_ts: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  holding_seconds: number;
};

type TodaySummary = {
  n: number;
  total_pnl: number;
  hit_rate: number;
  avg_win: number;
  avg_loss: number;
};

type TodayResp = {
  trades: TodayTrade[];
  summary: TodaySummary;
};

type Trade = {
  symbol: string; side: string; qty: number; entry_ts: string; exit_ts: string;
  entry_price: number; exit_price: number; pnl: number;
  holding_seconds: number; signal_scores: Record<string, unknown>;
};

// ──────────────────────────── helpers ───────────────────────────

function formatHold(s: number): string {
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function pnlColor(v: number): string {
  return v >= 0 ? "text-emerald-400" : "text-red-400";
}

// ──────────────────────────── histogram ─────────────────────────

function PnlHistogram({ trades }: { trades: TodayTrade[] }) {
  const BINS = 10;

  const { bins, edges } = useMemo(() => {
    if (trades.length === 0) return { bins: [], edges: [] };
    const vals = trades.map((t) => t.pnl);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    // Protect against all trades having the same P&L
    const span = hi - lo === 0 ? 1 : hi - lo;
    const step = span / BINS;

    const edges: number[] = Array.from({ length: BINS + 1 }, (_, i) => lo + i * step);
    const counts: number[] = Array(BINS).fill(0);
    for (const v of vals) {
      const idx = Math.min(Math.floor((v - lo) / step), BINS - 1);
      counts[idx]++;
    }
    const maxCount = Math.max(...counts, 1);
    const bins = counts.map((c, i) => ({
      count: c,
      height: Math.round((c / maxCount) * 100),
      midpoint: edges[i] + step / 2,
      isProfit: edges[i] + step / 2 >= 0,
    }));
    return { bins, edges };
  }, [trades]);

  if (trades.length === 0) return null;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <h2 className="mb-3 text-sm font-semibold text-gray-400">Today P&L Distribution</h2>
      <div className="flex items-end gap-1" style={{ height: 80 }}>
        {bins.map((b, i) => (
          <div key={i} className="flex flex-1 flex-col items-center justify-end" style={{ height: "100%" }}>
            <div
              title={`${b.count} trade${b.count !== 1 ? "s" : ""} (midpoint $${b.midpoint.toFixed(2)})`}
              className={`w-full rounded-t ${b.isProfit ? "bg-emerald-600" : "bg-red-600"} ${b.count === 0 ? "opacity-20" : ""}`}
              style={{ height: `${b.height}%`, minHeight: b.count > 0 ? 4 : 0 }}
            />
          </div>
        ))}
      </div>
      {/* axis labels: just min and max */}
      <div className="mt-1 flex justify-between text-xs text-gray-600">
        <span>${edges[0]?.toFixed(2) ?? ""}</span>
        <span>${edges[edges.length - 1]?.toFixed(2) ?? ""}</span>
      </div>
    </div>
  );
}

// ──────────────────────────── summary card ──────────────────────

function SummaryCard({ label, value, tone }: { label: string; value: string; tone?: "pos" | "neg" | "neutral" }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${
        tone === "pos" ? "text-emerald-400" : tone === "neg" ? "text-red-400" : "text-white"
      }`}>
        {value}
      </div>
    </div>
  );
}

// ──────────────────────────── page ──────────────────────────────

export default function Trades() {
  // Today state
  const [today, setToday] = useState<TodayResp | null>(null);

  // History state
  const [trades, setTrades] = useState<Trade[]>([]);
  const [open, setOpen] = useState<number | null>(null);

  function loadToday() {
    api<TodayResp>("/trades/today").then(setToday).catch(() => {});
  }

  useEffect(() => {
    loadToday();
    const id = setInterval(loadToday, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    api<Trade[]>("/trades?limit=200").then(setTrades).catch(() => {});
  }, []);

  const summary = today?.summary;
  const todayTrades = today?.trades ?? [];

  return (
    <main className="mx-auto max-w-5xl space-y-6 p-6">
      <h1 className="text-xl font-bold text-white">Trades</h1>

      {/* ── Today section ── */}
      <section className="space-y-4">
        <h2 className="text-base font-semibold text-gray-300">Today</h2>

        {/* Summary strip */}
        {summary ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <SummaryCard label="Trades" value={String(summary.n)} />
            <SummaryCard
              label="Total P&L"
              value={`$${summary.total_pnl.toFixed(2)}`}
              tone={summary.total_pnl >= 0 ? "pos" : "neg"}
            />
            <SummaryCard
              label="Hit Rate"
              value={`${(summary.hit_rate * 100).toFixed(1)}%`}
            />
            <SummaryCard
              label="Avg Win"
              value={summary.avg_win != null ? `$${summary.avg_win.toFixed(2)}` : "—"}
              tone="pos"
            />
            <SummaryCard
              label="Avg Loss"
              value={summary.avg_loss != null ? `$${summary.avg_loss.toFixed(2)}` : "—"}
              tone="neg"
            />
          </div>
        ) : (
          <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 text-center text-sm text-gray-600">
            No today data yet.
          </div>
        )}

        {/* P&L Histogram */}
        <PnlHistogram trades={todayTrades} />

        {/* Compact tape */}
        {todayTrades.length > 0 && (
          <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-gray-500">
                <tr>
                  <th className="pb-2 pr-4">Exit (ET)</th>
                  <th className="pb-2 pr-4">Symbol</th>
                  <th className="pb-2 pr-4">Qty</th>
                  <th className="pb-2 pr-4">Entry → Exit px</th>
                  <th className="pb-2">P&L</th>
                </tr>
              </thead>
              <tbody className="text-gray-200">
                {todayTrades.map((t, i) => (
                  <tr key={i} className="border-t border-gray-800 hover:bg-gray-800/50">
                    <td className="py-1.5 pr-4 text-xs text-gray-400">{fmtET(t.exit_ts)}</td>
                    <td className="py-1.5 pr-4 font-medium">{t.symbol}</td>
                    <td className="py-1.5 pr-4">{t.qty}</td>
                    <td className="py-1.5 pr-4">
                      ${t.entry_price.toFixed(2)} → ${t.exit_price.toFixed(2)}
                    </td>
                    <td className={`py-1.5 font-semibold ${pnlColor(t.pnl)}`}>
                      ${t.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {todayTrades.length === 0 && summary?.n === 0 && (
          <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 text-center text-sm text-gray-600">
            No trades today.
          </div>
        )}
      </section>

      {/* ── History section (existing, unchanged logic) ── */}
      <section className="space-y-4">
        <h2 className="text-base font-semibold text-gray-300">History</h2>
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-gray-500">
              <tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry → Exit</th><th>Held</th><th>P&L</th></tr>
            </thead>
            <tbody className="text-gray-200">
              {trades.map((t, i) => (
                <>
                  <tr
                    key={i}
                    onClick={() => setOpen(open === i ? null : i)}
                    className="cursor-pointer border-t border-gray-800 hover:bg-gray-800/50"
                  >
                    <td className="py-2 font-medium">{t.symbol}</td>
                    <td className="capitalize">{t.side}</td>
                    <td>{t.qty}</td>
                    <td>${t.entry_price.toFixed(2)} → ${t.exit_price.toFixed(2)}</td>
                    <td>{formatHold(t.holding_seconds)}</td>
                    <td className={t.pnl >= 0 ? "text-emerald-400" : "text-red-400"}>${t.pnl.toFixed(2)}</td>
                  </tr>
                  {open === i && (
                    <tr className="border-t border-gray-800 bg-black/40">
                      <td colSpan={6} className="px-4 py-3 text-xs text-gray-400">
                        <b>Entry signals:</b>{" "}
                        {Object.entries(t.signal_scores ?? {}).map(([k, v]) => (
                          <span key={k} className="mr-4">
                            {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
                          </span>
                        ))}
                        {Object.keys(t.signal_scores ?? {}).length === 0 && "none recorded"}
                      </td>
                    </tr>
                  )}
                </>
              ))}
              {trades.length === 0 && (
                <tr><td colSpan={6} className="py-4 text-center text-gray-600">no closed trades yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
