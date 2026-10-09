import type { ComponentProps } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export function NativeSelect({ className, children, ...props }: ComponentProps<"select">) {
  return <span className="relative inline-flex min-w-0 items-center">
    <select className={cn("h-10 w-full appearance-none rounded-md border border-input bg-card py-2 pl-3 pr-10 text-sm disabled:opacity-50", className)} {...props}>{children}</select>
    <ChevronDown aria-hidden="true" className="pointer-events-none absolute right-3 h-4 w-4 shrink-0 text-muted-foreground" />
  </span>;
}
