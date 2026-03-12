"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  PackageSearch,
  SearchCode,
  Bot,
  Activity,
  Warehouse,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { href: "/",           label: "Dashboard",     icon: LayoutDashboard },
  { href: "/sitemap",    label: "Site Map",       icon: Warehouse       },
  { href: "/bags",       label: "Bag Analyzer",  icon: PackageSearch   },
  { href: "/investigate",label: "Investigate",   icon: SearchCode      },
  { href: "/assistant",  label: "AI Assistant",  icon: Bot             },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <aside className="flex flex-col w-52 min-h-screen bg-[#111827] border-r border-[#1f2937] shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 py-4 border-b border-[#1f2937]">
        <Activity className="text-blue-400 shrink-0" size={20} />
        <span className="font-semibold text-sm text-slate-100 leading-tight truncate">
          AMR Intelligence<br />
          <span className="text-blue-400 font-normal text-xs">Platform</span>
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-0.5 flex-1 p-2.5 pt-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = path === href || (href !== "/" && path.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                active
                  ? "bg-blue-600/20 text-blue-400 border border-blue-600/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
              )}
            >
              <Icon size={15} className={clsx("shrink-0", active ? "text-blue-400" : "text-slate-500")} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-[#1f2937] text-xs text-slate-600 truncate">
        v0.1.0 · AMR Platform
      </div>
    </aside>
  );
}
