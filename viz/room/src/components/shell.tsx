/** Sidebar slot (IBM-2 pattern): each view renders its own controls and filters into the shell's left sidebar. */
import { createContext, useContext, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type Slot = { element: HTMLElement | null; closeMenu: () => void };
export const SidebarContext = createContext<Slot>({ element: null, closeMenu: () => {} });

/** Renders children into the sidebar; under SSR (render check) they render inline so they are still exercised. */
export function SidebarControls({ children }: { children: ReactNode }) {
  const { element } = useContext(SidebarContext);
  if (typeof document === 'undefined') return <div data-sidebar-ssr>{children}</div>;
  return element ? createPortal(children, element) : null;
}
export function SideGroup({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="side-group">
      <h2>{title}{right && <span className="side-right">{right}</span>}</h2>
      {children}
    </section>
  );
}
