import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin", "latin-ext"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin", "latin-ext"],
  display: "swap",
});

export const metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "https://recipes.vllm.ai"),
  title: {
    default: "vLLM Recipes",
    template: "%s | vLLM Recipes",
  },
  description: "Deploy any model on any hardware with vLLM. Interactive command builder for model serving.",
  icons: {
    icon: { url: "https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-only-light.ico" },
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased bg-background text-foreground min-h-screen flex flex-col">
        {/* Global header */}
        <header className="border-b border-border bg-background/95 backdrop-blur-sm sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-only-light.ico"
                alt="vLLM"
                width={28}
                height={28}
                className="dark:invert"
              />
              <span className="font-bold text-base">Recipes</span>
            </Link>
            <nav className="flex items-center gap-5 text-sm text-muted-foreground">
              <a href="https://docs.vllm.ai" className="hover:text-foreground transition-colors hidden sm:inline">Docs</a>
              <a href="https://github.com/vllm-project/recipes" className="hover:text-foreground transition-colors hidden sm:inline">GitHub</a>
              <ThemeToggle />
            </nav>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1">
          {children}
        </div>

        {/* Global footer */}
        <footer className="border-t border-border mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 text-xs text-muted-foreground flex flex-wrap gap-4">
            <a href="https://github.com/vllm-project/recipes" className="hover:text-foreground transition-colors">GitHub</a>
            <a href="https://github.com/vllm-project/recipes/issues" className="hover:text-foreground transition-colors">Request a recipe</a>
            <a href="https://docs.vllm.ai" className="hover:text-foreground transition-colors">vLLM Docs</a>
            <a href="/models.json" className="hover:text-foreground transition-colors">API</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
