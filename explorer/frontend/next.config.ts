import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow environment variable to override API URL at build time
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
  // Better Docker compatibility
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
};

export default nextConfig;
