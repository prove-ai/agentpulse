import type { NextConfig } from "next";

// Static export for GitHub Pages, served from the /agentpulse subpath
// (https://prove-ai.github.io/agentpulse/). `npm run build` writes the
// deployable site to out/ — copy its contents into ../docs to publish.
// The basePath only applies to production builds; `npm run dev` still
// serves at http://localhost:4700/.
const nextConfig: NextConfig = {
  output: "export",
  basePath: process.env.NODE_ENV === "production" ? "/agentpulse" : "",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
