"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** Portal keeps the interactive inspector outside the clipped poker table. */
export default function SeatInspection({ children, content, seat }: { children: ReactNode; content: ReactNode; seat: number }) {
  const [position, setPosition] = useState<{left:number; top:number} | null>(null);
  const anchor = useRef<HTMLButtonElement>(null), panel = useRef<HTMLDivElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const keep = () => { if (timer.current) clearTimeout(timer.current); };
  const closeSoon = () => { keep(); timer.current = setTimeout(()=>setPosition(null),200); };
  function open() {
    keep(); const rect = anchor.current!.getBoundingClientRect();
    setPosition({left:Math.max(12,Math.min(rect.left,window.innerWidth-Math.min(480,window.innerWidth-24)-12)), top:Math.max(12,Math.min(rect.bottom+8,window.innerHeight*0.2))});
  }
  useEffect(()=>{
    const close = () => setPosition(null);
    const key = (event: KeyboardEvent) => { if(event.key === "Escape") close(); };
    window.addEventListener("keydown",key); window.addEventListener("resize",close);
    return ()=>{ if(timer.current) clearTimeout(timer.current); window.removeEventListener("keydown",key); window.removeEventListener("resize",close); };
  },[]);
  return <>
    <button ref={anchor} type="button" aria-label={`Inspect player ${seat} range`} aria-expanded={position!==null} aria-controls={`seat-range-${seat}`}
      onMouseEnter={open} onMouseLeave={closeSoon} onFocus={open}
      onBlur={event=>{if(!panel.current?.contains(event.relatedTarget)) closeSoon();}}
      onClick={open} className="rounded-2xl focus-visible:outline-2 focus-visible:outline-primary">{children}</button>
    {position && createPortal(<div id={`seat-range-${seat}`} ref={panel} role="dialog" aria-label={`Player ${seat} range inspector`}
      onMouseEnter={keep} onMouseLeave={closeSoon} onFocus={keep}
      onBlur={event=>{if(!event.currentTarget.contains(event.relatedTarget) && event.relatedTarget!==anchor.current) closeSoon();}}
      className="fixed z-50 max-h-[78vh] overflow-auto rounded-lg border border-border bg-card p-3 text-foreground shadow-xl"
      style={{...position,width:"min(480px, calc(100vw - 24px))"}}>
      <button className="float-right ml-2 text-xs underline" aria-label="Close range inspector" onClick={()=>setPosition(null)}>Close</button>
      {content}
    </div>,document.body)}
  </>;
}
