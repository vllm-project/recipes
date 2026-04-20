"use client";

import { useMemo } from "react";
import { useSearchParams, useRouter } from "next/navigation";

export function RecipeFinder({ recipes, taxonomy }) {
  const searchParams = useSearchParams();
  const router = useRouter();

  const taskFilter = searchParams.get("task") || "";
  const providerFilter = searchParams.get("provider") || "";
  const hasFilters = taskFilter || providerFilter;

  const setFilter = (key, value) => {
    const sp = new URLSearchParams(searchParams.toString());
    if (value) sp.set(key, value);
    else sp.delete(key);
    const qs = sp.toString();
    router.replace(qs ? `?${qs}` : window.location.pathname, { scroll: false });
  };

  const toggleTask = (t) => setFilter("task", taskFilter === t ? "" : t);

  const providers = useMemo(
    () => [...new Set(recipes.map((r) => r.meta.provider))].sort(),
    [recipes]
  );

  // Collect tasks with counts
  const taskCounts = useMemo(() => {
    const counts = {};
    for (const r of recipes) {
      for (const t of r.meta.tasks || []) {
        counts[t] = (counts[t] || 0) + 1;
      }
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [recipes]);

  // Count filtered
  const filteredCount = useMemo(() => {
    return recipes.filter((r) => {
      if (providerFilter && r.meta.provider !== providerFilter) return false;
      if (taskFilter && !(r.meta.tasks || []).includes(taskFilter)) return false;
      return true;
    }).length;
  }, [recipes, taskFilter, providerFilter]);

  return (
    <div className="space-y-2.5">
      {/* Task tags */}
      <div className="flex flex-wrap items-center gap-1.5">
        {taskCounts.map(([task, count]) => {
          const active = taskFilter === task;
          return (
            <button
              key={task}
              onClick={() => toggleTask(task)}
              className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                active
                  ? "bg-vllm-blue text-white"
                  : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
              }`}
            >
              {task.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              <span className={active ? "text-white/70" : "text-muted-foreground"}>{count}</span>
            </button>
          );
        })}

        {/* Provider dropdown — compact, at the end */}
        <select
          value={providerFilter}
          onChange={(e) => setFilter("provider", e.target.value)}
          className="rounded-full border border-border bg-background px-2.5 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-vllm-blue/40 ml-1"
        >
          <option value="">All Providers</option>
          {providers.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>

        {hasFilters && (
          <button
            onClick={() => router.replace(window.location.pathname, { scroll: false })}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors ml-1"
          >
            Clear
          </button>
        )}

        <span className="text-xs text-muted-foreground ml-auto">
          {hasFilters ? `${filteredCount} / ${recipes.length}` : recipes.length}
        </span>
      </div>
    </div>
  );
}
