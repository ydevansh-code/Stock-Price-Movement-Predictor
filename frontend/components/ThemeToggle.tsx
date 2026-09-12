"use client";
import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [light, setLight] = useState(false);

  useEffect(() => {
    document.body.className = light ? "light" : "";
  }, [light]);

  return (
    <button
      onClick={() => setLight(!light)}
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "0.5rem",
        color: "var(--text)",
        cursor: "pointer",
        padding: "0.4rem 0.8rem",
        fontSize: 13,
      }}
    >
      {light ? "🌙 Dark" : "☀️ Light"}
    </button>
  );
}
