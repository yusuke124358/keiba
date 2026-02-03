#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import urlopen
from xml.etree import ElementTree as ET


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "paper"


def fetch_arxiv(query: str, max_results: int) -> list[dict]:
    url = (
        "https://export.arxiv.org/api/query?search_query="
        + quote_plus(query)
        + f"&start=0&max_results={max_results}"
    )
    with urlopen(url, timeout=20) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        link = ""
        for l in entry.findall("a:link", ns):
            if l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", "")
                break
        authors = [
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        entries.append(
            {
                "title": title,
                "summary": summary,
                "authors": authors,
                "link": link,
            }
        )
    return entries


def write_paper_md(out_dir: Path, entry: dict) -> Path:
    slug = slugify(entry.get("title", "paper"))
    path = out_dir / f"{slug}.md"
    if path.exists():
        return path
    text = [
        f"# {entry.get('title','').strip()}",
        "",
        "## Authors",
        ", ".join(entry.get("authors", [])) or "(unknown)",
        "",
        "## Link",
        entry.get("link", ""),
        "",
        "## Summary",
        entry.get("summary", "").strip(),
        "",
        "## Notes",
        "- leakage review: pending",
        "- reproducibility: pending",
    ]
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Search arXiv and store knowledge artifacts.")
    p.add_argument("query", help="Search query")
    p.add_argument("--max-results", type=int, default=5)
    args = p.parse_args()

    root = repo_root()
    papers_dir = root / "docs" / "knowledge" / "papers"
    logs_dir = root / "docs" / "knowledge" / "search_logs"
    papers_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    entries = fetch_arxiv(args.query, args.max_results)
    paper_paths = [str(write_paper_md(papers_dir, e)) for e in entries]

    log = {
        "query": args.query,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": entries,
        "paper_paths": paper_paths,
    }
    log_path = logs_dir / f"arxiv_{int(time.time())}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
