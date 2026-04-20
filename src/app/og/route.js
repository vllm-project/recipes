import { ImageResponse } from "next/og";

export const runtime = "edge";

// OG image generator, 1200×630.
//   /og?title=...&subtitle=...&meta=...&path=...
// Shared by the homepage, provider pages (/[org]), and recipe detail pages
// (/[org]/[repo]). Matches the vLLM blog OG design (top brand-gradient bar,
// three bottom-right ring accents, system font) and adds a recipes-specific
// "/ Recipes" wordmark next to the logo.
export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const title = searchParams.get("title") || "vLLM Recipes";
  const subtitle = searchParams.get("subtitle") || "";
  const meta = searchParams.get("meta") || "";
  const path = searchParams.get("path") || "";

  const baseUrl = new URL(request.url).origin;
  const logoUrl = `${baseUrl}/vLLM-Full-Logo.png`;

  const titleSize = title.length > 48 ? "44px" : title.length > 32 ? "56px" : "68px";

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "#ffffff",
          fontFamily: "system-ui, -apple-system, sans-serif",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Top brand-gradient bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "6px",
            background: "linear-gradient(90deg, #fdb517 0%, #f59e0b 35%, #30a2ff 100%)",
            display: "flex",
          }}
        />

        {/* Bottom-right concentric ring accents (matches blog OG) */}
        <div
          style={{
            position: "absolute",
            bottom: "-180px",
            right: "-180px",
            width: "480px",
            height: "480px",
            borderRadius: "50%",
            border: "48px solid rgba(253,181,23,0.10)",
            display: "flex",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: "-260px",
            right: "-260px",
            width: "640px",
            height: "640px",
            borderRadius: "50%",
            border: "40px solid rgba(253,181,23,0.06)",
            display: "flex",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: "-330px",
            right: "-330px",
            width: "800px",
            height: "800px",
            borderRadius: "50%",
            border: "32px solid rgba(48,162,255,0.05)",
            display: "flex",
          }}
        />

        {/* Content */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            height: "100%",
            padding: "56px 80px 52px",
          }}
        >
          {/* Logo + Recipes wordmark */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={logoUrl}
              alt="vLLM"
              width={180}
              height={46}
              style={{ objectFit: "contain", objectPosition: "left" }}
            />
            <span style={{ fontSize: 30, color: "#cbd5e1", fontWeight: 300, lineHeight: 1 }}>/</span>
            <span
              style={{
                fontSize: 26,
                color: "#111827",
                fontWeight: 600,
                letterSpacing: "-0.01em",
                display: "flex",
              }}
            >
              Recipes
            </span>
          </div>

          {/* Title + subtitle + meta */}
          <div style={{ display: "flex", flexDirection: "column", maxWidth: "980px" }}>
            <div
              style={{
                display: "flex",
                color: "#111827",
                fontSize: titleSize,
                fontWeight: 700,
                lineHeight: 1.2,
                letterSpacing: "-0.025em",
              }}
            >
              {title}
            </div>
            {subtitle && (
              <div
                style={{
                  display: "flex",
                  marginTop: 18,
                  fontSize: 26,
                  color: "#30a2ff",
                  fontWeight: 600,
                  letterSpacing: "-0.01em",
                }}
              >
                {subtitle}
              </div>
            )}
            {meta && (
              <div
                style={{
                  display: "flex",
                  marginTop: 12,
                  fontSize: 20,
                  color: "#4b5563",
                  fontWeight: 500,
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
      </div>
    ),
    { width: 1200, height: 630 }
  );
}
