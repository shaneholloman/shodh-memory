/**
 * A titled section in the right rail.
 *
 * Section headers are 12px medium in muted-foreground — sentence case, not
 * uppercase monospace. That single choice is most of what separates product
 * chrome from an instrument console.
 */
export function SidePanel({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-border border-b px-4 py-3.5">
      <h2 className="text-[12px] font-medium tracking-tight">{title}</h2>
      <p className="text-muted-foreground mt-0.5 text-[11px] leading-relaxed">
        {hint}
      </p>
      <div className="mt-3 text-[12px]">{children}</div>
    </section>
  );
}

/**
 * What a panel shows when its data is not available.
 *
 * Deliberately not the word "unavailable" four times down the rail: each panel
 * says what is missing in its own terms, so the rail reads as information
 * rather than as four identical failures.
 */
export function PanelEmpty({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground/70 text-[12px]">{children}</p>;
}
