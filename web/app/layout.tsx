import type { Metadata } from "next";
import "@fontsource-variable/geist";
import "./globals.css";

export const metadata: Metadata = {
  title: "GoXLM — Local workspace",
  description: "Experimental local text completion with independently trained GoXLM models.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
