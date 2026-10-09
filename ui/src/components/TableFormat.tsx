import { Button } from "./ui/button";

export default function TableFormat({ value, onChange, disabled = false }: {
  value: number; onChange: (count: 2 | 3) => void; disabled?: boolean;
}) {
  return <div role="group" aria-label="Table format" className="flex shrink-0 items-center gap-2">
    {([3, 2] as const).map(count => <Button key={count} type="button" disabled={disabled}
      aria-pressed={value === count} variant={value === count ? "default" : "secondary"}
      className="h-10 px-4 text-sm" onClick={() => onChange(count)}>{count === 3 ? "3-Max" : "HU"}</Button>)}
  </div>;
}
