"use client";
import { useState } from "react";
import TestTable from "@/components/TestTable";
import PolicyAnalysis from "@/components/PolicyAnalysis";
import { usePublishedPolicies } from "@/lib/use-published-policies";
import { Button } from "@/components/ui/button";
import { Sidebar, SidebarContent, SidebarHeader, SidebarProvider, SidebarTrigger, SidebarInset, useSidebar } from "@/components/ui/sidebar";

function CollapsedNavigationTrigger() {
  const { isMobile, state } = useSidebar();
  if (!isMobile && state === "expanded") return null;
  return <div className="px-4 pt-3"><SidebarTrigger /></div>;
}

export default function Page() {
  const [mainTab, setMainTab] = useState<"overview" | "case" | "test">("test");
  const { models, error, loading } = usePublishedPolicies();
  return <SidebarProvider>
    <Sidebar variant="floating" className="p-3">
      <SidebarHeader className="border-b border-border p-4"><div className="flex items-center gap-2"><SidebarTrigger /><h1 className="text-lg font-semibold">GTO Viewer</h1></div></SidebarHeader>
      <SidebarContent className="space-y-3 p-4">
        {([["overview", "Overview"], ["case", "Cas précis"], ["test", "Test Live"]] as const).map(([tab, label]) =>
          <Button key={tab} variant={mainTab === tab ? "default" : "secondary"} onClick={() => setMainTab(tab)}>{label}</Button>)}
      </SidebarContent>
    </Sidebar>
    <SidebarInset className="min-w-0">
      <CollapsedNavigationTrigger />
      <main className="space-y-4 px-4 py-3">
        {loading && <div role="status" className="text-sm text-muted-foreground">Loading published policies…</div>}
        {error && <div role="alert" className="text-sm text-destructive">{error}</div>}
        {mainTab !== "test" && <PolicyAnalysis mode={mainTab} policies={models} />}
        <div hidden={mainTab !== "test"}><TestTable policies={models} /></div>
      </main>
    </SidebarInset>
  </SidebarProvider>;
}
