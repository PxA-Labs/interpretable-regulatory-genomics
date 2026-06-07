#!/usr/bin/env python3
"""
Convert all project-definition markdown documents to professionally styled PDFs.

Uses `markdown` for Markdown->HTML and `xhtml2pdf` for HTML->PDF.
Pure Python - no system dependencies (no GTK, no wkhtmltopdf).
"""

import os
import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import markdown
from xhtml2pdf import pisa

# ---------------------------------------------------------------------------
# CSS Stylesheet - professional, print-optimised
# ---------------------------------------------------------------------------

CSS = """
@page {
    size: A4;
    margin: 22mm 18mm 22mm 18mm;

    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 0mm;
        margin-left: 18mm;
        margin-right: 18mm;
        height: 10mm;
    }
}

/* -- Base -- */

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10px;
    line-height: 1.55;
    color: #1a1a2e;
}

/* -- Headings -- */

h1 {
    font-size: 22px;
    font-weight: bold;
    color: #0a1628;
    margin-top: 0px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 3px solid #1a56db;
}

h2 {
    font-size: 16px;
    font-weight: bold;
    color: #1e3a5f;
    margin-top: 20px;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1.5px solid #93c5fd;
}

h3 {
    font-size: 13px;
    font-weight: bold;
    color: #1e40af;
    margin-top: 14px;
    margin-bottom: 4px;
}

h4 {
    font-size: 11px;
    font-weight: bold;
    color: #374151;
    margin-top: 10px;
    margin-bottom: 4px;
}

/* -- Paragraphs -- */

p {
    margin-top: 3px;
    margin-bottom: 7px;
}

/* -- Lists -- */

ul, ol {
    margin-top: 3px;
    margin-bottom: 7px;
    padding-left: 18px;
}

li {
    margin-bottom: 2px;
}

/* -- Tables -- */

table {
    width: 98%;
    border-collapse: collapse;
    margin-top: 6px;
    margin-bottom: 10px;
    font-size: 9px;
    -pdf-keep-in-frame-mode: shrink;
}

th {
    background-color: #1e3a5f;
    color: #ffffff;
    font-weight: bold;
    text-align: left;
    padding: 4px 5px;
    border: 0.5px solid #1e3a5f;
    font-size: 8.5px;
}

td {
    padding: 3px 5px;
    border: 0.5px solid #d1d5db;
    vertical-align: top;
}

tr {
    background-color: #ffffff;
}

/* -- Code -- */

code {
    font-family: Courier, monospace;
    font-size: 8.5px;
    background-color: #f1f5f9;
    color: #be185d;
    padding: 1px 3px;
}

pre {
    background-color: #1e293b;
    color: #e2e8f0;
    padding: 10px 12px;
    font-size: 8px;
    line-height: 1.45;
    margin-top: 6px;
    margin-bottom: 10px;
    border-left: 4px solid #2563eb;
    -pdf-keep-in-frame-mode: shrink;
}

pre code {
    background: none;
    color: #e2e8f0;
    padding: 0;
    font-size: 8px;
}

/* -- Blockquotes -- */

blockquote {
    border-left: 4px solid #2563eb;
    background-color: #eff6ff;
    margin: 8px 0;
    padding: 8px 12px;
    color: #1e3a5f;
}

/* -- Horizontal rule -- */

hr {
    border: none;
    border-top: 1.5px solid #d1d5db;
    margin: 14px 0;
}

/* -- Links -- */

a {
    color: #2563eb;
    text-decoration: none;
}

/* -- Bold / italic -- */

strong {
    font-weight: bold;
    color: #111827;
}

em {
    font-style: italic;
    color: #374151;
}

/* -- Page break utility -- */

.page-break {
    page-break-before: always;
}

/* -- Cover style -- */

.doc-header {
    text-align: center;
    padding: 30px 0 20px 0;
    border-bottom: 3px solid #1a56db;
    margin-bottom: 20px;
}

.doc-header h1 {
    border-bottom: none;
    text-align: center;
}

.doc-subtitle {
    font-size: 11px;
    color: #6b7280;
    text-align: center;
    margin-top: 4px;
}
"""


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <style>
    {css}
    </style>
</head>
<body>
    {body}

    <div id="footerContent" style="text-align: center; font-size: 7px; color: #9ca3af; border-top: 0.5px solid #d1d5db; padding-top: 3px;">
        Interpretable ML for Regulatory Switches in Non-Coding DNA &mdash; Project-Definition Pack
    </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def md_to_html(md_text):
    """Convert Markdown to HTML."""
    extensions = [
        "tables",
        "fenced_code",
        "toc",
        "sane_lists",
    ]
    return markdown.markdown(
        md_text,
        extensions=extensions,
        output_format="html5",
    )


def add_alternating_row_colors(html_body):
    """
    Add alternating row background colors to tables since xhtml2pdf
    doesn't support nth-child CSS selectors.
    """
    import re

    def color_rows(match):
        table_html = match.group(0)
        # Find all <tr> in tbody (after </thead>)
        parts = table_html.split('</thead>')
        if len(parts) < 2:
            # No thead, color all rows after first
            rows = table_html.split('<tr>')
            result = rows[0]
            for i, row in enumerate(rows[1:], 1):
                if i % 2 == 0:
                    result += '<tr style="background-color: #f0f4ff;">' + row
                else:
                    result += '<tr>' + row
            return result
        else:
            head = parts[0] + '</thead>'
            body = parts[1]
            rows = body.split('<tr>')
            result = head + rows[0]
            for i, row in enumerate(rows[1:], 1):
                if i % 2 == 0:
                    result += '<tr style="background-color: #f0f4ff;">' + row
                else:
                    result += '<tr>' + row
            return result

    return re.sub(r'<table>.*?</table>', color_rows, html_body, flags=re.DOTALL)


def convert_file(md_path, pdf_path):
    """Convert a single Markdown file to a styled PDF."""
    print(f"  Converting: {os.path.basename(md_path)}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Extract title from first H1
    title = os.path.splitext(os.path.basename(md_path))[0]
    for line in md_text.splitlines():
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break

    body_html = md_to_html(md_text)
    body_html = add_alternating_row_colors(body_html)
    full_html = HTML_TEMPLATE.format(title=title, css=CSS, body=body_html)

    with open(pdf_path, "w+b") as pdf_file:
        status = pisa.CreatePDF(
            src=full_html,
            dest=pdf_file,
            encoding='utf-8'
        )

    if status.err:
        raise RuntimeError(f"xhtml2pdf reported {status.err} error(s)")

    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"    [OK] {os.path.basename(pdf_path)}  ({size_kb:.0f} KB)")


def main():
    docs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "docs"
    )
    pdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "docs", "pdf"
    )
    os.makedirs(pdf_dir, exist_ok=True)

    # Ordered list of documents to convert
    doc_files = [
        "README.md",
        "01-project-charter.md",
        "02-product-requirements-document.md",
        "03-technical-design-document.md",
        "04-research-design-document.md",
        "05-dataset-strategy.md",
        "06-modeling-roadmap.md",
        "07-compute-feasibility-memo.md",
        "08-experiment-tracking-mlops.md",
        "09-risk-register.md",
        "10-roadmap-milestones.md",
        "11-glossary-project-memory.md",
        "12-contributor-onboarding.md",
    ]

    print("=" * 65)
    print("  PROJECT-DEFINITION PACK -- Markdown -> PDF Conversion")
    print("=" * 65)
    print(f"  Source : {docs_dir}")
    print(f"  Output : {pdf_dir}")
    print("-" * 65)

    converted = 0
    failed = []

    for filename in doc_files:
        md_path = os.path.join(docs_dir, filename)
        if not os.path.exists(md_path):
            print(f"  [SKIP] Not found: {filename}")
            continue

        pdf_name = os.path.splitext(filename)[0] + ".pdf"
        pdf_path = os.path.join(pdf_dir, pdf_name)

        try:
            convert_file(md_path, pdf_path)
            converted += 1
        except Exception as e:
            print(f"    [FAIL] {e}")
            failed.append((filename, str(e)))

    print("-" * 65)
    print(f"  Done. {converted}/{len(doc_files)} documents converted.")
    if failed:
        print(f"  Failed: {len(failed)}")
        for name, err in failed:
            print(f"    - {name}: {err}")
    print(f"  PDFs saved to: {pdf_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
