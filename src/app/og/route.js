import { ImageResponse } from "next/og";

export const runtime = "edge";

// OG image generator, 1200×630.
//   /og?title=...&subtitle=...&meta=...&path=...
// Shared by the homepage, provider pages (/[org]), and recipe detail pages
// (/[org]/[repo]). Mirrors the vLLM blog OG design (top brand-gradient bar,
// off-canvas accent circles, system font) and adds a recipes-specific
// "vLLM / Recipes" wordmark in the top-left.
export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const title = searchParams.get("title") || "vLLM Recipes";
  const subtitle = searchParams.get("subtitle") || "";
  const meta = searchParams.get("meta") || "";
  const path = searchParams.get("path") || "";

  // Auto-scale title font; recipe titles can be long (hf_org/hf_repo).
  const titleSize = title.length > 48 ? "44px" : title.length > 32 ? "56px" : "68px";

  const logoUrl = "https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-text-light.png";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          position: "relative",
          background: "#ffffff",
          padding: "56px 80px 52px",
          fontFamily: "system-ui, -apple-system, sans-serif",
        }}
      >
        {/* Top brand-gradient bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            background: "linear-gradient(90deg, #fdb517 0%, #f59e0b 35%, #30a2ff 100%)",
          }}
        />

        {/* Accent circles, bottom-right off-canvas */}
        <div
          style={{
            position: "absolute",
            right: -180,
            bottom: -180,
            width: 560,
            height: 560,
            borderRadius: "50%",
            background: "rgba(253,181,23,0.10)",
            display: "flex",
          }}
        />
        <div
          style={{
            position: "absolute",
            right: -80,
            bottom: -80,
            width: 360,
            height: 360,
            borderRadius: "50%",
            background: "rgba(48,162,255,0.05)",
            display: "flex",
          }}
        />
        <div
          style={{
            position: "absolute",
            right: -40,
            bottom: -40,
            width: 220,
            height: 220,
            borderRadius: "50%",
            background: "rgba(253,181,23,0.06)",
            display: "flex",
          }}
        />

        {/* Top-left: vLLM logo + "/ Recipes" */}
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={logoUrl} alt="" width={120} height={46} style={{ height: 46, width: "auto" }} />
          <span style={{ fontSize: 28, color: "#cbd5e1", fontWeight: 300, lineHeight: 1 }}>/</span>
          <span style={{ fontSize: 26, color: "#111827", fontWeight: 600, letterSpacing: "-0.01em" }}>
            Recipes
          </span>
        </div>

        {/* Middle: title + subtitle + meta */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            maxWidth: 1000,
          }}
        >
          <div
            style={{
              fontSize: titleSize,
              fontWeight: 700,
              color: "#111827",
              lineHeight: 1.1,
              letterSpacing: "-0.025em",
              display: "flex",
            }}
          >
            {title}
          </div>
          {subtitle && (
            <div
              style={{
                marginTop: 20,
                fontSize: 26,
                color: "#30a2ff",
                fontWeight: 600,
                letterSpacing: "-0.01em",
                display: "flex",
              }}
            >
              {subtitle}
            </div>
          )}
          {meta && (
            <div
              style={{
                marginTop: 14,
                fontSize: 20,
                color: "#4b5563",
                fontWeight: 500,
                display: "flex",
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, monospace",
              }}
            >
              {meta}
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
          }}
        >
          <div style={{ display: "flex", color: "#9ca3af", fontSize: 16, fontWeight: 500 }}>
            recipes.vllm.ai
          </div>
          {path && (
            <div
              style={{
                display: "flex",
                color: "#9ca3af",
                fontSize: 16,
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, monospace",
              }}
            >
              {path}
            </div>
          )}
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}
