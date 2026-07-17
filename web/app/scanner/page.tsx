"use client";
// Scanner watchlist — polls /scanner/watchlist every 5 s, shows live-slot dot.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { fmtET } from "@/lib/time";

type WatchRow = {
  ts: string;
  symbol: string;
  score: number;
  price: number;
  gain_pct: number;
  relvol: number;
  spread_bps: number;
  streamed: boolean;
};

type WatchlistResp = {
  ts: string | null;
  rows: WatchRow[];
};

export default function Scanner() {
  const [data, setData] = useState<WatchlistResp | null>(null);
  const [error, setError] = useState(false);

  function load() {
    api<WatchlistResp>("/scanner/watchlist")
      .then((d) => { setData(d); setError(false); })
      .catch(() => setError(true));
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  const rows = data?.rows ?? [];
  const asOf = data?.ts;

  return (
    <main className="mx-auto max-w-6xl space-y-4 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Scanner Watchlist</h1>
        {asOf && (
          <span className="text-xs text-gray-500">
            as of {fmtET(asOf)}
          </span>
        )}
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-gray-500">
            <tr>
              <th className="pb-2 pr-4">Symbol</th>
              <th className="pb-2 pr-4">Price</th>
              <th className="pb-2 pr-4">Gain %</th>
              <th className="pb-2 pr-4">Rel Vol</th>
              <th className="pb-2 pr-4">Spread bps</th>
              <th className="pb-2 pr-4">Score</th>
              <th className="pb-2">Live</th>
            </tr>
          </thead>
          <tbody className="text-gray-200">
            {rows.map((r) => (
              <tr key={r.symbol} className="border-t border-gray-800 hover:bg-gray-800/50">
                <td className="py-2 pr-4 font-medium">{r.symbol}</td>
                <td className="py-2 pr-4">${r.price.toFixed(2)}</td>
                <td className={`py-2 pr-4 font-medium ${r.gain_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {r.gain_pct >= 0 ? "+" : ""}{r.gain_pct.toFixed(2)}%
                </td>
                <td className="py-2 pr-4">{r.relvol.toFixed(2)}x</td>
                <td className="py-2 pr-4">{r.spread_bps.toFixed(1)}</td>
                <td className="py-2 pr-4">
                  <span className={`font-semibold ${r.score >= 0.7 ? "text-emerald-400" : r.score >= 0.4 ? "text-yellow-400" : "text-gray-400"}`}>
                    {r.score.toFixed(3)}
                  </span>
                </td>
                <td className="py-2">
                  {r.streamed ? (
                    <span
                      title="Holds a live WebSocket slot"
                      className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.5)]"
                    />
                  ) : (
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-gray-700" />
                  )}
                </td>
              </tr>
            ))}
            {rows.length === 0 && !error && (
              <tr>
                <td colSpan={7} className="py-8 text-center text-gray-600">
                  Scanner idle — engine not running.
                </td>
              </tr>
            )}
            {error && (
              <tr>
                <td colSpan={7} className="py-8 text-center text-red-700">
                  Failed to load watchlist.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
