import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
    outputFileTracingRoot: path.join(__dirname, ".."),
    allowedDevOrigins: ["192.168.1.156"],
};

export default nextConfig;
