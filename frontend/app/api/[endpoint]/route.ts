import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const filenameMap: Record<string, string> = {
  "class-balance": "class_balance",
  "comparison-table": "comparison_table",
  "predictions": "predictions",
  "feature-importance": "feature_importance",
  "backtest": "backtest",
  "walk-forward": "walk_forward",
  "calibration": "calibration",
  "next-day": "next_day_signal",
  "ohlcv": "ohlcv",
};

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ endpoint: string }> }
) {
  const { endpoint } = await context.params;
  const market = request.nextUrl.searchParams.get("market")?.toLowerCase() === "ind" ? "ind" : "us";
  const filename = filenameMap[endpoint] || endpoint.replace(/-/g, "_");

  const filePath = path.join(process.cwd(), "artifacts", market, `${filename}.json`);

  if (fs.existsSync(filePath)) {
    try {
      const content = fs.readFileSync(filePath, "utf-8");
      return NextResponse.json(JSON.parse(content));
    } catch {
      return NextResponse.json({ error: "Failed to read artifact" }, { status: 500 });
    }
  }

  return NextResponse.json({ error: `Artifact ${endpoint} for ${market} not found` }, { status: 404 });
}
