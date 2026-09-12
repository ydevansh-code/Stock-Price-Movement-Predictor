export default function Skeleton({ height = 200 }: { height?: number }) {
  return (
    <div
      style={{
        height,
        borderRadius: "0.75rem",
        background: "linear-gradient(90deg, var(--surface) 25%, var(--border) 50%, var(--surface) 75%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.5s infinite",
      }}
    />
  );
}

// Add shimmer keyframe via inline style (no CSS file needed)
if (typeof document !== "undefined") {
  const s = document.createElement("style");
  s.textContent = "@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }";
  document.head.appendChild(s);
}
