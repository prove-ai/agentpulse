import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import SiteFooter from "@/components/SiteFooter";
import SiteHeader from "@/components/SiteHeader";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AgentPulse: drift investigation for multi-agent systems",
  description:
    "Open-source drift investigation for multi-agent systems: which agent drifted, where it originated, and what to check next.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      data-theme="light"
    >
      <body className="min-h-full flex flex-col">
        <div className="ambient" aria-hidden>
          <div className="blob blob-1" />
          <div className="blob blob-2" />
          <div className="blob blob-3" />
        </div>
        <SiteHeader />
        {children}
        <SiteFooter />
        <script
          data-goatcounter="https://agentpulse.goatcounter.com/count"
          async
          src="//gc.zgo.at/count.js"
        />
      </body>
    </html>
  );
}
