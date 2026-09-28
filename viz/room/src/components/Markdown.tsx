/**
 * Small, safe markdown renderer (no innerHTML): headings, paragraphs, lists, code fences, tables, quotes, rules,
 * inline code/bold/italic/links. D-/P- ids become chips; relative .md links open in the docs reader.
 */
import { Fragment, type ReactNode } from 'react';
import { href } from '../lib/url';

function inline(src: string, key = 'i'): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*|_[^_\s][^_]*_)|(\[[^\]]+\]\([^)\s]+\))|(\b[DP]-\d{2,4}\b)/g;
  let last = 0, m: RegExpExecArray | null, i = 0;
  while ((m = re.exec(src))) {
    if (m.index > last) out.push(src.slice(last, m.index));
    const t = m[0];
    const k = `${key}-${i++}`;
    if (m[1]) out.push(<code key={k}>{t.slice(1, -1)}</code>);
    else if (m[2]) out.push(<strong key={k}>{inline(t.slice(2, -2), k)}</strong>);
    else if (m[3]) out.push(<em key={k}>{inline(t.slice(1, -1), k)}</em>);
    else if (m[4]) {
      const mm = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(t)!;
      const target = mm[2];
      const safe = /^(https?:|#|mailto:)/.test(target);
      const md = !safe && /\.md(#.*)?$/.test(target);
      if (md) out.push(<a key={k} href={href('knowledge', { tab: 'docs', doc: target.replace(/^\.\//, '').replace(/#.*$/, '') })}>{inline(mm[1], k)}</a>);
      else if (safe) out.push(<a key={k} href={target} target={target.startsWith('http') ? '_blank' : undefined} rel="noreferrer">{inline(mm[1], k)}</a>);
      else out.push(<span key={k} title={target}>{inline(mm[1], k)}</span>);
    } else if (m[5]) out.push(<a key={k} className="did" href={href('knowledge', { tab: 'decisions', d: t })}>{t}</a>);
    last = m.index + t.length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

function cells(line: string) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
}

export default function Markdown({ source, maxBlocks = 4000 }: { source: string; maxBlocks?: number }) {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let i = 0, b = 0;
  while (i < lines.length && blocks.length < maxBlocks) {
    const line = lines[i];
    const key = `b${b++}`;
    if (/^```/.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push(<pre key={key}><code>{body.join('\n')}</code></pre>);
      continue;
    }
    const h = /^(#{1,5})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      const Tag = (`h${level}` as 'h1');
      blocks.push(<Tag key={key}>{inline(h[2], key)}</Tag>);
      i++;
      continue;
    }
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { blocks.push(<hr key={key} className="soft" />); i++; continue; }
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const body: string[][] = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) body.push(cells(lines[i++]));
      blocks.push(
        <table key={key}>
          <thead><tr>{head.map((c, j) => <th key={j}>{inline(c, `${key}h${j}`)}</th>)}</tr></thead>
          <tbody>{body.map((r, ri) => <tr key={ri}>{r.map((c, j) => <td key={j}>{inline(c, `${key}r${ri}c${j}`)}</td>)}</tr>)}</tbody>
        </table>,
      );
      continue;
    }
    if (/^\s*>/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ''));
      blocks.push(<blockquote key={key}><Markdown source={body.join('\n')} /></blockquote>);
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d/.test(line);
      const items: string[] = [];
      while (i < lines.length && (/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) && !/^\s{4,}/.test(lines[i])) items.push(lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ''));
        else items[items.length - 1] += ` ${lines[i].trim()}`;
        i++;
      }
      const L = ordered ? 'ol' : 'ul';
      blocks.push(<L key={key}>{items.map((it, j) => <li key={j}>{inline(it, `${key}-${j}`)}</li>)}</L>);
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,5}\s|```|\s*[-*+]\s|\s*\d+[.)]\s|\s*>|\s*\|)/.test(lines[i])) para.push(lines[i++]);
    if (!para.length) { para.push(lines[i++]); }
    blocks.push(<p key={key}>{inline(para.join(' '), key)}</p>);
  }
  return <div className="md">{blocks.map((x, j) => <Fragment key={j}>{x}</Fragment>)}</div>;
}
