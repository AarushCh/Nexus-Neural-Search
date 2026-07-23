import type { Metadata } from "next";
import "./globals.css";
import { StoreProvider } from "@/lib/store";
import NeuralBg from "@/components/NeuralBg";

export const metadata: Metadata = {
  title: "Nexus | Neural Search",
  description:
    "A hybrid AI search engine that understands vibes and exact matches — movies, TV, anime, and documentaries.",
  openGraph: {
    title: "Nexus Neural Search",
    description: "Hybrid AI discovery for movies, TV, anime & documentaries.",
  },
};

// Set theme class before paint to avoid a flash of the wrong theme.
const themeInit = `(function(){try{var t=localStorage.getItem('theme')||'dark';if(t==='light')document.documentElement.classList.add('light-mode');}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700;800&family=Inter:wght@400;600&display=swap"
          rel="stylesheet"
        />
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>
        <StoreProvider>
          <NeuralBg />
          {children}
        </StoreProvider>
      </body>
    </html>
  );
}
