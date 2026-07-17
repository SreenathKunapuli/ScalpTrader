"use client";
// Read-only view of engine status/tier/scalp profile/kill state and staleness.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const TIERS: Record<string, Record<string, string>> = {
  low: {
    Universe: "10 ETFs (SPY, QQQ, IWM, DIA, XLK, XLF, XLE, XLV, GLD, TLT)",
    Direction: "Long only", "Max position": "5% of equity", "Max gross": "40%",
    "Max open positions": "6", "Daily loss kill": "1.0%", "Max drawdown kill": "5%",
    Stop: "1.5 × ATR(14, 5m)", "Confidence threshold": "0.60",
    Rebalance: "daily at 15:45 ET", "Risk per trade": "0.25%",
  },
  medium: {
    Universe: "10 ETFs + 10 large caps", Direction: "Long only",
    "Max position": "10% of equity", "Max gross": "80%", "Max open positions": "10",
    "Daily loss kill": "2.0%", "Max drawdown kill": "10%", Stop: "2.0 × ATR",
    "Confidence threshold": "0.55", Rebalance: "every 15 min", "Risk per trade": "0.50%",
  },
  high: {
    Universe: "10 ETFs + 10 large caps", Direction: "Long + short",
    "Max position": "20% of equity", "Max gross": "150%", "Max open positions": "15",
    "Daily loss kill": "4.0%", "Max drawdown kill": "15%", Stop: "2.5 × ATR",
    "Confidence threshold": "0.52", Rebalance: "every 5 min", "Risk per trade": "1.00%",
  },
};

type EngineStatus = {
  status: string;
  tier: string;
  halted_reason: string;
  trading_mode: string;
  heartbeat_age_s: number | null;
  last_data_ts: string | null;
};

type StalenessSymbol = {
  age_s: number;
  gap_p50_s: number | null;
  gap_p95_s: number | null;
};

const statusColor: Record<string, string> = {
  RUNNING: "text-emerald-400",
  PAUSED: "text-amber-400",
  HALTED: "text-red-400",
  STOPPED: "text-gray-400",
};

function fmt(v: number | null, unit = "s"): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(2)}${unit}`;
}

export default function Settings() {
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [staleness, setStaleness] = useState<Record<string, StalenessSymbol>>({});
  const [stalenessErr, setStalenessErr] = useState(false);

  useEffect(() => {
    api<EngineStatus>("/engine/status")
      .then(setEngine)
      .catch(() => {});
    api<Record<string, StalenessSymbol>>("/staleness")
      .then(setStaleness)
      .catch(() => setStalenessErr(true));
  }, []);

  const tier = engine?.tier ?? "medium";
  const mode = engine?.trading_mode ?? "paper";
  const engineStatus = engine?.status ?? "…";
  const isHalted = engineStatus === "HALTED";
  const hbAge = engine?.heartbeat_age_s ?? null;

  const staleSorted = Object.entries(staleness).sort(
    ([, a], [, b]) => (b.age_s ?? 0) - (a.age_s ?? 0)
  );

  return (
    <main className="mx-auto max-w-3xl space-y-4 p-6">
      <h1 className="text-xl font-bold text-white">Settings</h1>

      {/* Engine status card */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 text-sm space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Engine</div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          <Row label="Status">
            <span className={`font-semibold ${statusColor[engineStatus] ?? "text-gray-300"}`}>
              {engineStatus}
            </span>
            {isHalted && engine?.halted_reason && (
              <span className="ml-2 text-red-400 text-xs">({engine.halted_reason})</span>
            )}
          </Row>
          <Row label="Trading mode">
            <span className="font-semibold uppercase text-sky-300">{mode}</span>
          </Row>
          <Row label="Active tier">
            <span className="font-semibold capitalize text-emerald-400">{tier}</span>
          </Row>
          <Row label="Heartbeat age">
            <span className={hbAge !== null && hbAge > 60 ? "text-amber-400" : "text-gray-200"}>
              {hbAge !== null ? fmt(hbAge) : "—"}
            </span>
          </Row>
          <Row label="Last data">
            <span className="text-gray-200">
              {engine?.last_data_ts
                ? new Date(engine.last_data_ts).toLocaleTimeString()
                : "—"}
            </span>
          </Row>
        </div>
      </div>

      {/* Tier parameters card */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 text-sm">
        <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
          Tier parameters — {tier}
        </div>
        <table className="w-full">
          <tbody>
            {Object.entries(TIERS[tier] ?? {}).map(([k, v]) => (
              <tr key={k} className="border-t border-gray-800">
                <td className="py-2 pr-4 text-gray-500">{k}</td>
                <td className="text-gray-200">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-4 text-xs text-gray-600">
          Tier parameters are code-reviewed constants (engine/scalpengine/config/tiers.py) —
          not editable from the UI by design. Secrets and endpoints live in .env.
        </p>
      </div>

      {/* Staleness card */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 text-sm">
        <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
          Quote staleness
        </div>
        {stalenessErr ? (
          <p className="text-gray-600 text-xs">Staleness data unavailable.</p>
        ) : staleSorted.length === 0 ? (
          <p className="text-gray-600 text-xs">No quotes recorded yet (engine not streaming).</p>
        ) : (
          <table className="w-full">
            <thead className="text-left text-xs text-gray-500">
              <tr>
                <th className="pb-2 pr-4">Symbol</th>
                <th className="pb-2 pr-4">Age</th>
                <th className="pb-2 pr-4">Gap p50</th>
                <th className="pb-2">Gap p95</th>
              </tr>
            </thead>
            <tbody>
              {staleSorted.map(([sym, d]) => (
                <tr key={sym} className="border-t border-gray-800">
                  <td className="py-2 pr-4 font-medium text-white">{sym}</td>
                  <td className={`pr-4 ${d.age_s > 60 ? "text-amber-400" : "text-gray-200"}`}>
                    {fmt(d.age_s)}
                  </td>
                  <td className="pr-4 text-gray-200">{fmt(d.gap_p50_s)}</td>
                  <td className={d.gap_p95_s !== null && d.gap_p95_s > 5 ? "text-amber-400" : "text-gray-200"}>
                    {fmt(d.gap_p95_s)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mt-3 text-xs text-gray-600">
          Updated every engine heartbeat interval. Age = seconds since the newest exchange
          quote stamp. Gap p95 &gt; 5 s highlights symbols likely to need a paid feed.
        </p>
      </div>
    </main>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-gray-500 min-w-[120px]">{label}</span>
      <span>{children}</span>
    </div>
  );
}
