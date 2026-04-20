import * as React from "react";
import { cn } from "@/lib/utils";

function Card({ className, ...props }) {
  return (
    <div
      className={cn("bg-card text-card-foreground flex flex-col gap-4 rounded-xl border p-5 shadow-sm", className)}
      {...props}
    />
  );
}

function CardHeader({ className, ...props }) {
  return <div className={cn("flex flex-col gap-1", className)} {...props} />;
}

function CardTitle({ className, ...props }) {
  return <div className={cn("leading-none font-semibold", className)} {...props} />;
}

function CardDescription({ className, ...props }) {
  return <div className={cn("text-muted-foreground text-sm", className)} {...props} />;
}

function CardContent({ className, ...props }) {
  return <div className={cn("", className)} {...props} />;
}

export { Card, CardHeader, CardTitle, CardDescription, CardContent };
