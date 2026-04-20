import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import { Analytics } from "@vercel/analytics/next";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { SearchBox } from "@/components/recipes/SearchBox";
import { getAllRecipes } from "@/lib/recipes";
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

export default async function RootLayout({ children }) {
  // Recipes are small (~50 KB) — fine to load into every page for the
  // global search box. Server-rendered, cached across requests.
  const recipes = getAllRecipes();
  // Drop the heavy `guide` field before sending to the client
  const searchRecipes = recipes.map((r) => ({
    hf_org: r.hf_org,
    hf_repo: r.hf_repo,
    hf_id: r.hf_id,
    meta: {
      title: r.meta.title,
      provider: r.meta.provider,
      description: r.meta.description,
      tasks: r.meta.tasks,
    },
    model: {
      architecture: r.model.architecture,
      parameter_count: r.model.parameter_count,
    },
  }));

  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased bg-background text-foreground min-h-screen flex flex-col">
        {/* Global header */}
        <header className="border-b border-border bg-background/95 backdrop-blur-sm sticky top-0 z-30">
          <div className="max-w-[1480px] mx-auto px-4 sm:px-6 h-16 flex items-center gap-4">
            <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity group shrink-0">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-text-light.png"
                alt="vLLM"
                width={96}
                height={36}
                className="h-8 w-auto dark:hidden"
              />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="https://docs.vllm.ai/en/latest/assets/logos/vllm-logo-text-dark.png"
                alt="vLLM"
                width={96}
                height={36}
                className="h-8 w-auto hidden dark:block"
              />
              <span className="text-muted-foreground/50 font-light text-xl leading-none">/</span>
              <span className="font-semibold text-base group-hover:text-vllm-blue transition-colors">Recipes</span>
            </Link>

            {/* Search — flex-1 so it expands to fill available space */}
            <div className="flex-1 flex justify-center max-w-xl mx-auto">
              <SearchBox recipes={searchRecipes} />
            </div>

            <nav className="flex items-center gap-4 text-sm text-muted-foreground shrink-0">
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
          <div className="max-w-[1480px] mx-auto px-4 sm:px-6 py-5 text-xs text-muted-foreground flex flex-wrap gap-x-5 gap-y-2">
            <a href="https://github.com/vllm-project/recipes" className="hover:text-foreground transition-colors">GitHub</a>
            <a href="https://github.com/vllm-project/recipes/issues" className="hover:text-foreground transition-colors">Request a recipe</a>
            <a href="https://docs.vllm.ai" className="hover:text-foreground transition-colors">Documentation</a>
            <a href="https://vllm.ai/#compatibility" className="hover:text-foreground transition-colors">Supported Models & Hardware</a>
            <a href="https://vllm.ai/#quick-start" className="hover:text-foreground transition-colors">Install vLLM</a>
            <a href="/models.json" className="hover:text-foreground transition-colors">JSON API</a>
          </div>
        </footer>
        <Analytics />
      </body>
    </html>
  );
}
