"use client";

export default function Error({ error, reset }) {
  return (
    <main className="max-w-6xl mx-auto px-4 py-20 text-center">
      <h1 className="text-2xl font-bold mb-2">Something went wrong</h1>
      <p className="text-muted-foreground mb-4">{error?.message || "An unexpected error occurred."}</p>
      <button
        onClick={reset}
        className="text-sm text-vllm-blue hover:underline"
      >
        Try again
      </button>
    </main>
  );
}
