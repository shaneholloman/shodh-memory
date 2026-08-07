import { cn } from "@/lib/utils";

/**
 * The centred title+body message. Before this, `Placeholder.tsx` and
 * `RecallView.tsx`'s `Message` hand-rolled the same shape at two different
 * sizes (`text-[15px]`/`max-w-sm` for a whole destination, `text-[13px]`/
 * `max-w-xs` for the narrower 340px result column) — real drift, not
 * cosmetic duplication, since a size chosen for one context silently leaked
 * into the other whenever someone copied a block. `size` keeps both call
 * sites, `panel` is the narrower default because more callers are panels
 * than whole pages.
 */
export function EmptyState({
  title,
  body,
  size = "panel",
  className,
}: {
  title: string;
  body: string;
  size?: "panel" | "page";
  className?: string;
}) {
  return (
    <div className={cn("grid h-full place-items-center", size === "page" ? "px-8" : "px-6", className)}>
      <div className={cn("text-center", size === "page" ? "max-w-sm" : "max-w-xs")}>
        <p
          className={cn(
            "font-medium tracking-tight",
            size === "page" ? "text-[15px]" : "text-[13px]",
          )}
        >
          {title}
        </p>
        <p
          className={cn(
            "text-muted-foreground leading-relaxed",
            size === "page" ? "mt-2 text-[13px]" : "mt-1.5 text-[12px]",
          )}
        >
          {body}
        </p>
      </div>
    </div>
  );
}
