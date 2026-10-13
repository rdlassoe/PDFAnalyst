"use client";

import { ToastProvider } from "@/components/Toaster";

export default function Providers({ children }: { children: React.ReactNode }) {
  return <ToastProvider>{children}</ToastProvider>;
}
