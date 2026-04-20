import Link from "next/link";

export default function NotFound() {
  return (
    <main className="max-w-6xl mx-auto px-4 py-20 text-center">
      <h1 className="text-2xl font-bold mb-2">404</h1>
      <p className="text-muted-foreground mb-4">This page could not be found.</p>
      <Link href="/" className="text-sm text-vllm-blue hover:underline">
        Back to recipes
      </Link>
    </main>
  );
}
