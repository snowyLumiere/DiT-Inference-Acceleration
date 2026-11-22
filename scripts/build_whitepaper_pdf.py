"""Build the local whitepaper; publish only a newly rendered, validated PDF.

Dependencies for a real build: Python Markdown, PyMuPDF, and Chrome/Chromium.
--check validates the source list without importing render dependencies, starting
Chrome, or writing HTML/PDF. Formula/diagram assets are vendored with notices.
PDF structural validation does not replace visual proofreading.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape, unescape
from html.parser import HTMLParser
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from urllib.parse import quote, unquote, urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_NAME = "DiT_Inference_Acceleration_Whitepaper_v1.0.pdf"
HTML_NAME = "build_whitepaper.html"
OMNI_SOURCE = "docs/omni/inference-frameworks.md"
OMNI_PDF_NAME = "Omni_Inference_Frameworks.pdf"
OMNI_HTML_NAME = "Omni_Inference_Frameworks.html"
OMNI_TITLE = "Omni 与推理框架：阶段调度、传输和流式交付"
PUBLIC_AUTHOR = "SnowyLumiere"
PUBLIC_AUTHOR_URL = "https://github.com/SnowyLumiere"
EDITION_LABEL = "v1.0（r9 · 2026-11-20）"
WATERMARK_TEXT = "© 2026 SnowyLumiere · github.com/SnowyLumiere"

# Reader edition: two parts; extension and appendices are unnumbered back matter.
CHAPTER_FILES = [
    ("第一篇：计算、硬件与测量基础", [
        ("第1章 生成范式与模型地图", "docs/manuscript/01-paradigms.md"),
        ("第2章 从输入到输出的执行图与工作量", "docs/manuscript/02-execution-graph.md"),
        ("第3章 GPU 执行与性能约束", "docs/manuscript/03-gpu-execution.md"),
        ("第4章 状态、显存与数据生命周期", "docs/manuscript/04-memory-and-state.md"),
        ("第5章 性能分析与可复现实验", "docs/manuscript/05-measurement.md"),
    ]),
    ("第二篇：加速方法与模型实践", [
        ("第6章 减少网络求值：求解器、少步、CFG 与蒸馏", "docs/manuscript/06-evaluations-and-distillation.md"),
        ("第7章 跨步特征复用与自适应缓存", "docs/manuscript/07-feature-cache.md"),
        ("第8章 Dense Attention Kernel 与低精度 Attention", "docs/manuscript/08-attention-kernels.md"),
        ("第9章 稀疏、Token 压缩与替代 Attention", "docs/manuscript/09-sparsity-and-tokens.md"),
        ("第10章 Linear/GEMM 与低精度计算", "docs/manuscript/10-gemm-and-quantization.md"),
        ("第11章 算子融合、预计算、编译与 CUDA Graph", "docs/manuscript/11-fusion-and-graphs.md"),
        ("第12章 AR 历史 KV 与状态管理", "docs/manuscript/12-ar-state.md"),
        ("第13章 驻留、Offload 与数据搬运", "docs/manuscript/13-residency-and-transfer.md"),
        ("第14章 多 GPU 并行与通信", "docs/manuscript/14-parallelism.md"),
        ("第15章 VAE、流式执行与端到端交付", "docs/manuscript/15-decoding-and-streaming.md"),
        ("第16章 组合优化与回归验收", "docs/manuscript/16-composition-and-validation.md"),
    ]),
]

BACK_MATTER = [
    ("延伸阅读指引", "docs/manuscript/extension-omni.md"),
    ("附录 A 模型与实验配置", "docs/manuscript/appendix-a-model-contracts.md"),
    ("附录 B 复现入口与操作步骤", "docs/manuscript/appendix-b-reproduction.md"),
    ("附录 C 来源与实验数据", "docs/manuscript/appendix-c-evidence.md"),
]

# These are repository entry links, not additional embedded chapters or code.
LINKED_SOURCES = [
    ("发布说明与配套文件", "RELEASE_DIT_INFRA.md"),
    ("代码案例入口与复现说明", "examples/README.md"),
]


class BuildError(RuntimeError):
    pass


DIAGRAM_MARKER_BASE = 'https://dit-diagram.invalid/'
DIAGRAM_DPI = 300
WATERMARK_DPI = 150
WATERMARK_OPACITY = 0.09


def render_watermark_png(text, width=595.0, height=842.0, *, dpi=WATERMARK_DPI,
                         opacity=WATERMARK_OPACITY, angle_deg=26.0):
    """Render a transparent non-text diagonal watermark PNG using only PyMuPDF.

    Placing the watermark as a raster Image XObject behind page content
    (overlay=False) avoids selectable text spans, preserves copy/paste, and
    prevents Zotero Reading Mode from mixing watermark words into paragraphs.
    """
    if not text or not text.strip():
        raise BuildError('Watermark text must be nonempty')
    if type(dpi) is not int or not 72 <= dpi <= 300:
        raise BuildError('Watermark DPI must be an integer from 72 to 300')
    if not (0.01 <= opacity <= 0.30):
        raise BuildError('Watermark opacity must be between 0.01 and 0.30')
    import fitz
    with fitz.open() as scratch:
        page = scratch.new_page(width=width, height=height)
        rot = fitz.Matrix(angle_deg)
        fontsize = 11.5
        # Balanced staggered placements wholly within A4 bounds (no clipped edge text).
        x_scale = width / 595.0
        y_scale = height / 842.0
        placements = [
            (48 * x_scale, 235 * y_scale),
            (305 * x_scale, 295 * y_scale),
            (48 * x_scale, 485 * y_scale),
            (305 * x_scale, 545 * y_scale),
            (48 * x_scale, 735 * y_scale),
            (305 * x_scale, 795 * y_scale),
        ]
        for x, y in placements:
            pt = fitz.Point(x, y)
            page.insert_text(
                pt, text, fontname="helv", fontsize=fontsize,
                color=(0.15, 0.22, 0.33), morph=(pt, rot), fill_opacity=opacity,
            )
        pixmap = page.get_pixmap(dpi=dpi, alpha=True)
        return pixmap.tobytes("png")


def apply_pdf_watermark(document, text=WATERMARK_TEXT, *, skip_first_page=True,
                        dpi=WATERMARK_DPI, opacity=WATERMARK_OPACITY):
    """Stamp a shared non-text background watermark onto body pages."""
    cache = {}
    pages_watermarked = 0
    for index, page in enumerate(document):
        if skip_first_page and index == 0:
            continue
        key = (round(page.rect.width, 2), round(page.rect.height, 2))
        if key not in cache:
            png = render_watermark_png(text, width=page.rect.width, height=page.rect.height,
                                       dpi=dpi, opacity=opacity)
            xref = page.insert_image(page.rect, stream=png, keep_proportion=False, overlay=False)
            cache[key] = {'xref': xref, 'bytes': len(png),
                          'sha256': hashlib.sha256(png).hexdigest()}
        else:
            page.insert_image(page.rect, xref=cache[key]['xref'],
                              keep_proportion=False, overlay=False)
        pages_watermarked += 1
    primary = next(iter(cache.values()), {'bytes': 0, 'sha256': None})
    return {'method': 'non_text_background_png_xobject', 'text': text, 'dpi': dpi,
            'opacity': opacity, 'overlay': False, 'skip_first_page': skip_first_page,
            'pages_watermarked': pages_watermarked, 'png_bytes': primary['bytes'],
            'png_sha256': primary['sha256']}


def mark_mermaid_for_pdf(html):
    """Add temporary whole-diagram links so Chrome supplies exact PDF rectangles.

    The published offline HTML keeps its original Mermaid SVG. These links are
    only present in Chrome's staged input and are deleted before publication.
    """
    if DIAGRAM_MARKER_BASE in html:
        raise BuildError('Reserved diagram marker URL already present in HTML')
    markers = []

    def wrap(match):
        identifier = f'{len(markers) + 1:04d}'
        markers.append({'id': identifier, 'uri': DIAGRAM_MARKER_BASE + identifier,
                        'source_sha256': hashlib.sha256(match[0].encode('utf-8')).hexdigest()})
        return ('<a href="' + markers[-1]['uri']
                + '" style="display:block;break-inside:avoid;text-decoration:none;color:inherit">'
                + match[0] + '</a>')

    staged = re.sub(r'<div class="mermaid(?: [^"]*)?">.*?</div>', wrap, html, flags=re.S)
    return staged, markers


def rasterize_mermaid_pdf(path, markers, *, dpi=DIAGRAM_DPI):
    """Replace each complete diagram with one opaque image, preserving body text.

    Reading-mode engines may split loose SVG paths and labels into unrelated
    text and image blocks. A single image XObject keeps each diagram together.
    All geometry and pixel data are validated before changing the staged PDF.
    """
    report = {'method': 'whole_diagram_image_xobject', 'dpi': dpi,
              'expected_diagrams': len(markers), 'rasterized_diagrams': 0,
              'temporary_marker_links_removed': 0, 'diagrams': [],
              'offline_html': 'original Mermaid SVG',
              'body_text_and_math': 'retained outside diagram rectangles'}
    if not markers:
        return report
    if type(dpi) is not int or not 150 <= dpi <= 600:
        raise BuildError('Diagram raster DPI must be an integer from 150 to 600')
    import fitz
    expected = {row['uri']: row for row in markers}
    if len(expected) != len(markers):
        raise BuildError('Duplicate expected diagram marker')
    try:
        with fitz.open(path) as document:
            found = {}
            for page in document:
                links = page.get_links()
                for link in links:
                    uri = link.get('uri', '')
                    if not uri.startswith(DIAGRAM_MARKER_BASE):
                        continue
                    if uri not in expected:
                        raise BuildError('Unexpected diagram marker: ' + uri)
                    if uri in found:
                        raise BuildError('Diagram marker is duplicated or split across pages: ' + uri)
                    rect = fitz.Rect(link['from'])
                    values = tuple(rect)
                    if (not all(math.isfinite(value) for value in values)
                            or rect.is_empty or rect.is_infinite or rect.width < 10 or rect.height < 10
                            or not (page.rect + (-0.1, -0.1, 0.1, 0.1)).contains(rect)):
                        raise BuildError('Invalid or out-of-page diagram rectangle: ' + uri)
                    for other in links:
                        if other['xref'] != link['xref'] and rect.intersects(other['from']):
                            raise BuildError('Diagram overlaps another link or body navigation: ' + uri)
                    # A body line crossing the rectangle would be damaged by
                    # redaction. Diagram labels must fit wholly inside it.
                    for block in page.get_text('dict')['blocks']:
                        for line in block.get('lines', []):
                            for span in line['spans']:
                                bounds = fitz.Rect(span['bbox'])
                                if (span['text'].strip() and rect.intersects(bounds)
                                        and not (rect + (-0.5, -0.5, 0.5, 0.5)).contains(bounds)):
                                    raise BuildError('Text crosses a diagram boundary: ' + uri)
                    for prior in found.values():
                        if prior['page'] == page.number and rect.intersects(prior['rect']):
                            raise BuildError('Diagram rectangles overlap: ' + uri)
                    found[uri] = {'page': page.number, 'rect': rect, 'link': link}
            if set(found) != set(expected):
                raise BuildError('Missing diagram markers; Mermaid must render as complete single-page figures')
            # Render all figures before the first mutation, including figures
            # that share a page. Images have an opaque white background.
            for marker in markers:
                item = found[marker['uri']]
                page = document[item['page']]
                pixmap = page.get_pixmap(dpi=dpi, clip=item['rect'], alpha=False)
                item['png'] = pixmap.tobytes('png')
                item['pixel_size'] = [pixmap.width, pixmap.height]
            for page_number in sorted({item['page'] for item in found.values()}):
                page = document[page_number]
                items = [found[row['uri']] for row in markers if found[row['uri']]['page'] == page_number]
                for item in items:
                    page.delete_link(item['link'])
                    page.add_redact_annot(item['rect'], fill=(1, 1, 1), cross_out=False)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                      graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                                      text=fitz.PDF_REDACT_TEXT_REMOVE)
                for item in items:
                    if page.get_text('text', clip=item['rect']).strip():
                        raise BuildError('Original diagram text remains after redaction')
                    item['image_xref'] = page.insert_image(item['rect'], stream=item['png'],
                                                          keep_proportion=False, overlay=True)
            for marker in markers:
                item = found[marker['uri']]
                report['diagrams'].append({
                    'id': marker['id'], 'source_sha256': marker['source_sha256'],
                    'page': item['page'] + 1, 'rect_points': list(item['rect']),
                    'pixel_size': item['pixel_size'], 'png_bytes': len(item['png']),
                    'png_sha256': hashlib.sha256(item['png']).hexdigest(),
                    'image_xobject_created': bool(item['image_xref']),
                })
            for page in document:
                if any(link.get('uri', '').startswith(DIAGRAM_MARKER_BASE) for link in page.get_links()):
                    raise BuildError('Temporary diagram link remains in PDF')
            report['rasterized_diagrams'] = len(found)
            report['temporary_marker_links_removed'] = len(found)
            with tempfile.TemporaryDirectory(prefix='.diagram-images-', dir=Path(path).parent) as stage:
                clean = Path(stage) / 'rasterized.pdf'
                document.save(clean, garbage=4, deflate=True)
                os.replace(clean, path)
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError('Cannot export complete diagram images: ' + str(exc)) from exc
    return report


def validate_sources(repo_root):
    paths = [path for _, chapters in CHAPTER_FILES for _, path in chapters]
    paths += [path for _, path in BACK_MATTER]
    if len({chapter_id(path) for path in paths}) != len(paths):
        raise BuildError("Chapter namespace collision; choose distinct source paths")
    guides = [f'docs/manuscript/part-{index}.md' for index in range(1, len(CHAPTER_FILES) + 1)]
    missing = [path for path in paths + guides + [p for _, p in LINKED_SOURCES]
               if not (repo_root / path).is_file()]
    if missing:
        raise BuildError("Missing required source files: " + ", ".join(missing))
    return paths


def chapter_id(path):
    return "chapter-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", path)


def heading_slug(value, separator="-"):
    """Match the repository's GitHub-style Markdown heading fragments."""
    value = value.replace("`", "").lower()
    value = "".join(c for c in value
                    if unicodedata.category(c)[0] not in ("P", "S") or c in "-_")
    return re.sub(r"\s", separator, value.strip())


def heading_slugger():
    used = set()
    def slug(value, separator="-"):
        base = heading_slug(value, separator)
        candidate, number = base, 0
        while candidate in used:
            number += 1
            candidate = f'{base}-{number}'
        used.add(candidate)
        return candidate
    return slug


def scope_heading_ids(body, source_path):
    prefix = chapter_id(source_path) + "--"
    return re.sub(r'''(?<![\w:-])id=(["'])(.*?)\1''',
                  lambda m: 'id=' + m[1] + prefix + m[2] + m[1], body)


TABLE_SPLIT_MARKER = "reader:table-may-split"


def mark_long_tables(body):
    """Let an explicitly marked tall table continue on the next page.

    Tables keep page-break-inside: avoid by default, so short tables never leave
    an orphan row. A source comment <!-- reader:table-may-split --> directly
    before a table opts it into splitting between rows with a repeated header.
    """
    return re.sub(r'<!--\s*' + re.escape(TABLE_SPLIT_MARKER) + r'\s*-->\s*<table>',
                  '<table class="long-table">', body)


def rewrite_file_links(body, source_path, repo_root, included):
    """Resolve chapter-relative files even when Chrome loads a temporary HTML.

    Embedded chapters and their headings become internal anchors. Other files
    use absolute URIs only while Chrome renders; publication makes them relative.
    """
    def replace(match):
        value = unescape(match.group("url"))
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            return match.group(0)
        if not parsed.path:
            if parsed.fragment:
                href = "#" + chapter_id(source_path) + "--" + unquote(parsed.fragment)
                return match.group("prefix") + match.group("quote") + escape(href, quote=True) + match.group("quote")
            return match.group(0)
        target = (repo_root / source_path).parent.joinpath(unquote(parsed.path)).resolve()
        try:
            relative = target.relative_to(repo_root).as_posix()
        except ValueError:
            raise BuildError("Local link escapes repository: " + value)
        if relative in included and not parsed.query:
            href = "#" + chapter_id(relative)
            if parsed.fragment:
                href += "--" + unquote(parsed.fragment)
        else:
            href = urlunsplit((*urlsplit(target.as_uri())[:3], parsed.query, parsed.fragment))
        return match.group("prefix") + match.group("quote") + escape(href, quote=True) + match.group("quote")

    return re.sub(r'''(?P<prefix>(?<![\w:-])(?:href|src)\s*=\s*)(?P<quote>["'])(?P<url>.*?)(?P=quote)''',
                  replace, body, flags=re.IGNORECASE)


def portable_file_uri(value, repo_root, artifact_parent):
    """Return a URL relative to the delivered artifact; reject escaped files."""
    parsed = urlsplit(value)
    if parsed.scheme != "file":
        return value
    if parsed.netloc not in ("", "localhost"):
        raise BuildError("Remote file URI is not portable: " + value)
    target = Path(unquote(parsed.path)).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError as exc:
        raise BuildError("Local URI escapes repository: " + value) from exc
    path = quote(Path(os.path.relpath(target, artifact_parent.resolve())).as_posix(), safe="/-._~")
    return urlunsplit(("", "", path, parsed.query, parsed.fragment))


def portable_html(html, repo_root, artifact_parent):
    def replace(match):
        value = portable_file_uri(unescape(match["url"]), repo_root, artifact_parent)
        return match["prefix"] + match["quote"] + escape(value, quote=True) + match["quote"]
    return re.sub(r'''(?P<prefix>(?<![\w:-])(?:href|src)\s*=\s*)(?P<quote>["'])(?P<url>.*?)(?P=quote)''',
                  replace, html, flags=re.IGNORECASE)


def portable_pdf_links(path, repo_root, artifact_parent):
    """Rewrite Chrome's file actions to relative URI actions before hashing."""
    import fitz
    changed = 0
    with fitz.open(path) as document:
        for page in document:
            for link in page.get_links():
                action_type = document.xref_get_key(link['xref'], 'A/S')
                if action_type == ('name', '/Launch'):
                    raise BuildError('Unsupported PDF Launch action; use a relative URI companion link')
                kind, value = document.xref_get_key(link["xref"], "A/URI")
                if kind != "string":
                    value = link.get("uri") or ""
                if not value.startswith("file:"):
                    continue
                uri = portable_file_uri(value, repo_root, artifact_parent)
                # Updating through the URI API avoids platform-specific Launch
                # file actions. Some readers still disable relative file links.
                page.delete_link(link)
                page.insert_link({"kind": fitz.LINK_URI, "from": link["from"], "uri": uri})
                changed += 1
        if changed:
            updated = path.with_name("portable.pdf")
            document.save(updated, garbage=3, deflate=True)
    if changed:
        os.replace(updated, path)
    return {"rewritten_local_links": changed, "local_link_format": "relative_URI",
            "base": "artifact_directory", "requires_companion_files": True}


def render_asset_records(html, repo_root):
    """Fingerprint local render resources, including fonts referenced by CSS."""
    pending = []
    for value in re.findall(r'''\b(?:href|src)=["']([^"']+)["']''', html):
        url = urlsplit(unescape(value))
        if url.scheme == 'file':
            path = Path(unquote(url.path)).resolve()
            if path.suffix.lower() in {'.css', '.js', '.svg', '.png', '.jpg', '.jpeg', '.woff', '.woff2', '.ttf'}:
                pending.append(path)
    records = {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(repo_root).as_posix()
        if relative in records: continue
        if not path.is_file(): raise BuildError('Missing render asset: ' + relative)
        content = path.read_bytes()
        records[relative] = {'path': relative, 'bytes': len(content),
                             'sha256': hashlib.sha256(content).hexdigest()}
        if path.suffix == '.css':
            for value in re.findall(r'''url\(\s*["']?([^\s)'";]+)''', content.decode()):
                url = urlsplit(value)
                if url.scheme or url.netloc:
                    raise BuildError('Nonlocal CSS dependency: ' + value)
                pending.append((path.parent / unquote(url.path)).resolve())
    return [records[k] for k in sorted(records)]

def _escaped(text, position):
    backslashes = 0
    while position > 0 and text[position - 1] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _inline_code_segments(text):
    """Keep matched backtick runs opaque, including multiline code spans."""
    position = plain_start = 0
    while position < len(text):
        if text[position] != chr(96) or _escaped(text, position):
            position += 1
            continue
        end = position + 1
        while end < len(text) and text[end] == chr(96):
            end += 1
        width = end - position
        closing = None
        for match in re.finditer(r"\x60+", text[end:]):
            if len(match.group()) == width:
                closing = (end + match.start(), end + match.end())
                break
        if closing is None:
            position = end
            continue
        if position > plain_start:
            yield False, text[plain_start:position]
        yield True, text[position:closing[1]]
        position = plain_start = closing[1]
    if plain_start < len(text):
        yield False, text[plain_start:]


def _markdown_segments(text):
    """Separate fenced/inline code without searching math across code boundaries."""
    plain, code, fence = [], [], None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^[ \t]*(\x60{3,}|~{3,})([^\r\n]*)", line)
        if fence is not None:
            code.append(line)
            if (match and match[1][0] == fence[0] and len(match[1]) >= fence[1]
                    and not match[2].strip()):
                yield True, "".join(code)
                code, fence = [], None
        elif match and not (match[1][0] == chr(96) and chr(96) in match[2]):
            yield from _inline_code_segments("".join(plain))
            plain = []
            code, fence = [line], (match[1][0], len(match[1]))
        else:
            plain.append(line)
    if fence is not None:
        yield True, "".join(code)
    yield from _inline_code_segments("".join(plain))


def protect_math(text):
    """Stash complete dollar-delimited math before Markdown touches TeX.

    Code remains byte-for-byte intact. Tokens contain no Markdown punctuation;
    HTML-escaped restoration after Markdown lets KaTeX see the original TeX.
    """
    prefix = "WHITEPAPERMATH" + hashlib.sha256(text.encode()).hexdigest()[:16].upper()
    while prefix in text:
        prefix += "X"
    formulas, output = {}, []
    for is_code, segment in _markdown_segments(text):
        if is_code:
            output.append(segment)
            continue
        position = plain_start = 0
        while position < len(segment):
            if segment[position] != "$" or _escaped(segment, position):
                position += 1
                continue
            delimiter = "$$" if segment.startswith("$$", position) else "$"
            end = position + len(delimiter)
            closing = None
            while end < len(segment):
                if delimiter == "$" and segment.startswith("\n\n", end):
                    break
                if segment.startswith(delimiter, end) and not _escaped(segment, end):
                    if delimiter == "$" and segment.startswith("$$", end):
                        end += 2
                        continue
                    closing = end + len(delimiter)
                    break
                end += 1
            if closing is None:
                position += len(delimiter)
                continue
            token = f"{prefix}TOKEN{len(formulas)}END"
            formulas[token] = segment[position:closing]
            output.extend((segment[plain_start:position], token))
            position = plain_start = closing
        output.append(segment[plain_start:])
    return "".join(output), formulas


def restore_math(body, formulas):
    # KaTeX's $$ delimiter supplies display layout; no raw HTML is introduced
    # into code blocks, tables, or inline paragraphs by this restoration.
    for token, formula in formulas.items():
        body = body.replace(token, escape(formula, quote=False))
    return body


def _prepare_lists(text):
    """Allow paragraph-adjacent lists and the repo's 2/4-space child bullets."""
    mask = bytearray(len(text))
    offset = 0
    for is_code, segment in _markdown_segments(text):
        if is_code:
            mask[offset:offset + len(segment)] = b"\x01" * len(segment)
        offset += len(segment)
    output, levels = [], []
    previous_blank, previous_marker, offset = True, False, 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^( *)(?:[*+-]|\d+\.)[ \t]+\S", line)
        indent = len(match[1]) if match else 0
        marker = bool(match and not mask[offset + indent] and (levels or indent < 4))
        if re.fullmatch(r"[ \t]*([*_-])(?:[ \t]*\1){2,}[ \t\r\n]*", line):
            marker = False
        if marker:
            if not previous_blank and not previous_marker:
                output.append("\n")
            while levels and indent < levels[-1][0]:
                levels.pop()
            if not levels:
                levels.append((indent, indent))
            elif indent > levels[-1][0]:
                levels.append((indent, levels[-1][1] + 4))
            output.append(" " * levels[-1][1] + line[indent:])
        else:
            output.append(line)
            first = len(line) - len(line.lstrip(" \t"))
            if line.strip() and first == 0 and not mask[offset]:
                levels = []
        previous_blank, previous_marker = not line.strip(), marker
        offset += len(line)
    return "".join(output)


def preprocess_markdown(text):
    callout_types = {
        "NOTE": "ℹ️ NOTE", "TIP": "💡 TIP", "IMPORTANT": "📌 IMPORTANT",
        "WARNING": "⚠️ WARNING", "CAUTION": "🛑 CAUTION", "CRITICAL": "🚨 CRITICAL",
    }
    output = []
    for is_code, segment in _markdown_segments(_prepare_lists(text)):
        if is_code:
            lines = segment.splitlines(keepends=True)
            opening = re.match(r"^[ \t]*(\x60{3,}|~{3,})mermaid[ \t]*$", lines[0].rstrip("\r\n"))
            closing = (re.fullmatch(r"[ \t]*(\x60+|~+)[ \t]*", lines[-1].rstrip("\r\n"))
                       if len(lines) > 1 else None)
            if (opening and closing and opening[1][0] == closing[1][0]
                    and len(closing[1]) >= len(opening[1])):
                diagram = "".join(lines[1:-1])
                diagram_class = (' mermaid-chart' if re.match(r'\s*xychart', diagram)
                                 else ' mermaid-sequence' if re.match(r'\s*sequenceDiagram', diagram)
                                 else '')
                output.append('\n<div class="mermaid' + diagram_class + '">\n' + diagram + "\n</div>\n")
            else:
                output.append(segment)
        else:
            output.append(re.sub(
                r"(?m)^>[ \t]*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION|CRITICAL)\][ \t]*([^\n]*)",
                lambda match: f'> **{callout_types[match[1]]}** {match[2]}', segment,
            ))
    return "".join(output)


def html_header():
    """Shared offline formula, diagram and print styling for both documents."""
    svg_wm = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='360' height='220'>"
        "<text x='180' y='110' text-anchor='middle' font-family='sans-serif' font-size='13' "
        "fill='#334155' fill-opacity='0.065' transform='rotate(-26 180 110)'>"
        "© 2026 SnowyLumiere · github.com/SnowyLumiere</text></svg>"
    )
    svg_wm_uri = "data:image/svg+xml;utf8," + quote(svg_wm, safe="")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="author" content="{PUBLIC_AUTHOR}">
  <title>DiT 推理加速：原理与工程实践</title>
  
  <!-- KaTeX -->
  <link rel="stylesheet" href="VENDOR_ROOT/katex-0.16.8/katex.min.css">
  <script defer src="VENDOR_ROOT/katex-0.16.8/katex.min.js"></script>
  <script defer src="VENDOR_ROOT/katex-0.16.8/auto-render.min.js"
          onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
  
  <!-- Mermaid -->
  <script src="VENDOR_ROOT/mermaid-10.9.3/mermaid.min.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      mermaid.initialize({{ startOnLoad: true, theme: 'neutral',
        themeVariables: {{ xyChart: {{ plotColorPalette: '#2563eb' }} }} }});
    }});
  </script>

  <style>
    @page {{
      size: A4;
      margin: 18.5mm 16mm 18.5mm 16mm;
      @bottom-right {{
        content: counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 9pt;
        color: #718096;
      }}
      @bottom-left {{
        content: "DiT 推理加速：原理与工程实践 · © 2026 SnowyLumiere (github.com/SnowyLumiere)";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 8pt;
        color: #94a3b8;
      }}
    }}
    @page :first {{
      margin: 0;
      @bottom-right {{ content: none; }}
      @bottom-left {{ content: none; }}
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      font-size: 10.5pt;
      line-height: 1.56;
      color: #2d3748;
      background-color: #ffffff;
      margin: 0;
      padding: 0;
    }}

    /* Cover Page */
    .cover-page {{
      height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 60px 50px;
      box-sizing: border-box;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
      color: #ffffff;
      page-break-after: always;
    }}
    .cover-badge {{
      display: inline-block;
      align-self: flex-start;
      background: rgba(56, 189, 248, 0.2);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #38bdf8;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 11pt;
      font-weight: 600;
      letter-spacing: 1px;
      margin-bottom: 24px;
    }}
    .cover-title {{
      font-size: 32pt;
      font-weight: 800;
      line-height: 1.25;
      margin: 0 0 16px 0;
      color: #f8fafc;
    }}
    .cover-subtitle {{
      font-size: 15pt;
      color: #94a3b8;
      font-weight: 400;
      line-height: 1.5;
      margin-bottom: 22px;
    }}
    .cover-author {{
      font-size: 11pt;
      color: #e2e8f0;
      margin-bottom: 36px;
      letter-spacing: 0.3px;
    }}
    .cover-author a {{
      color: #38bdf8;
      text-decoration: none;
    }}
    .cover-meta-box {{
      background: rgba(255, 255, 255, 0.05);
      border-left: 4px solid #38bdf8;
      padding: 20px 24px;
      border-radius: 4px;
      margin-top: auto;
    }}
    .cover-meta-item {{
      font-size: 10.5pt;
      color: #cbd5e1;
      margin: 6px 0;
    }}
    .cover-meta-item strong {{
      color: #ffffff;
    }}
    .edition-colophon {{
      margin-top: 8px;
      padding: 8px 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #2563eb;
      border-radius: 4px;
      font-size: 8.8pt;
      line-height: 1.48;
      color: #334155;
      break-inside: avoid;
    }}
    .edition-colophon p {{
      margin: 2px 0;
      text-align: left;
    }}
    #reading-guide {{
      margin-bottom: 14px;
    }}
    #reading-guide h1 {{
      margin-top: 4px;
      margin-bottom: 10px;
    }}
    #reading-guide > p {{
      font-size: 9.7pt;
      line-height: 1.5;
      margin-bottom: 6px;
    }}
    #global-contents h1 {{
      margin-top: 12px;
      margin-bottom: 8px;
    }}
    .companion-meta {{
      font-size: 9pt;
      color: #475569;
      margin: -6px 0 12px 0;
      text-align: left;
    }}
    @media screen {{
      body {{
        max-width: 920px;
        margin: 0 auto;
        padding: 24px 28px 64px;
        position: relative;
      }}
      body::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("{svg_wm_uri}");
        background-repeat: repeat;
        pointer-events: none;
        user-select: none;
        z-index: 0;
      }}
      body > * {{
        position: relative;
        z-index: 1;
      }}
      .cover-page {{
        height: auto;
        min-height: 440px;
        border-radius: 10px;
        margin-bottom: 32px;
      }}
      .katex-display {{
        overflow-x: auto;
        overflow-y: hidden;
        padding: 4px 0;
      }}
      table {{
        display: block;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
      }}
      @media (max-width: 640px) {{
        body {{ padding: 14px 14px 48px; font-size: 10pt; }}
        .cover-page {{ padding: 32px 22px; }}
        .cover-title {{ font-size: 24pt; }}
        .contents-grid {{ grid-template-columns: 1fr; }}
        .contents-backmatter ul {{ columns: 1; }}
      }}
    }}""" + """

    /* Part Header */
    .part-banner {
      background: #f1f5f9;
      border-left: 6px solid #2563eb;
      padding: 16px 20px;
      margin-top: 20px;
      margin-bottom: 25px;
      border-radius: 0 4px 4px 0;
      page-break-before: always;
      break-inside: avoid;
      break-after: avoid-page;
    }
    .part-title {
      font-size: 20pt;
      font-weight: 800;
      color: #1e3a8a;
      margin: 0;
    }
    .contents {
      break-before: auto;
      font-size: 9.5pt;
    }
    .contents-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .contents-backmatter ul { columns: 2; column-gap: 18px; }
    .contents li { break-inside: avoid; }
    .contents ul, .part-banner ul {
      list-style: none;
      padding-left: 0;
      margin: 6px 0;
    }
    .contents li, .part-banner li { margin: 3.5px 0; }
    .contents a, .part-banner a { color: #1e3a8a; text-decoration: none; }
    .contents h2 { font-size: 11.5pt; margin-top: 10px; margin-bottom: 6px; }
    .part-banner p { margin-top: 12px; }

    /* Chapter Header */
    .chapter-container {
      margin-bottom: 30px;
    }
    .last-chapter, .last-chapter > :last-child {
      margin-bottom: 0;
    }
    .chapter-divider {
      break-before: auto;
    }

    h1 {
      font-size: 18pt;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 8px;
      margin-top: 20px;
      margin-bottom: 16px;
    }
    h2 {
      font-size: 14pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 18px;
      margin-bottom: 12px;
      border-left: 4px solid #3b82f6;
      padding-left: 10px;
    }
    h3 {
      font-size: 12pt;
      font-weight: 600;
      color: #334155;
      margin-top: 18px;
      margin-bottom: 8px;
    }
    p {
      margin: 0 0 8.5px 0;
      text-align: justify;
      orphans: 3;
      widows: 3;
    }
    a {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    img {
      max-width: 100%;
      height: auto;
    }

    /* Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 9.5pt;
      page-break-inside: avoid;
      font-variant-numeric: tabular-nums lining-nums;
    }
    table.long-table {
      page-break-inside: auto;
      break-inside: auto;
    }
    table.long-table thead { display: table-header-group; }
    table.long-table tr {
      page-break-inside: avoid;
      break-inside: avoid;
    }
    th, td {
      border: 1px solid #cbd5e1;
      padding: 5px 8px;
      text-align: left;
    }
    th {
      background-color: #f8fafc;
      font-weight: 600;
      color: #1e293b;
    }
    th[style*="text-align: right"], td[style*="text-align: right"] {
      white-space: nowrap;
    }
    th { line-break: strict; }
    tr:nth-child(even) {
      background-color: #f1f5f9;
    }

    /* Code blocks */
    pre {
      background-color: #0f172a;
      color: #f8fafc;
      padding: 12px 16px;
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 8.5pt;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 14px 0;
      page-break-inside: avoid;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 9pt;
      background-color: #f1f5f9;
      color: #0f172a;
      padding: 2px 5px;
      border-radius: 4px;
      overflow-wrap: anywhere;
    }
    pre code {
      background-color: transparent;
      color: inherit;
      padding: 0;
    }

    /* Blockquotes */
    blockquote {
      margin: 16px 0;
      padding: 12px 18px;
      background-color: #f8fafc;
      border-left: 4px solid #64748b;
      color: #334155;
      border-radius: 0 4px 4px 0;
      page-break-inside: avoid;
    }
    blockquote p {
      margin: 0;
    }

    /* Mermaid container */
    .mermaid {
      margin: 20px 0;
      text-align: center;
      page-break-inside: avoid;
    }
    .diagram-figure { margin: 16px 0; break-inside: avoid; }
    .diagram-figure figcaption { font-size: 9.5pt; color: #475569; }
    .diagram-figure .mermaid { margin-top: 8px; }
    @media print {
      html, body { background: transparent !important; background-color: transparent !important; }
      h1, h2, h3, h4 { break-after: avoid-page; page-break-after: avoid; }
      .mermaid svg {
        width: auto !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 125mm !important;
      }
      .mermaid-chart svg { max-height: 84mm !important; }
      .mermaid-sequence svg { max-height: 100mm !important; }
    }
    
    hr {
      border: 0;
      border-top: 1px solid #e2e8f0;
      margin: 24px 0;
    }
  </style>
</head>
<body>
"""


def render_html(repo_root, *, source_records=None):
    included = set(validate_sources(repo_root))
    try:
        import markdown
    except ImportError as exc:
        raise BuildError("Python Markdown is required; use an interpreter with markdown and PyMuPDF installed") from exc
    html_parts = []
    if source_records is None:
        source_records = []
    # Record the bytes actually consumed, rather than re-reading sources after
    # Chrome exits (other editors may have changed a chapter in the meantime).
    def read_source(path, role):
        content = (repo_root / path).read_bytes()
        source_records.append({"path": path, "role": role, "bytes": len(content),
                               "sha256": hashlib.sha256(content).hexdigest()})
        return content.decode("utf-8")

    for _, path in LINKED_SOURCES:
        read_source(path, "repository_entry")

    header = html_header()
    html_parts.append(header)
    
    # 1. Cover Page
    cover = f"""
  <div class="cover-page">
    <div class="cover-badge">Diffusion Transformer · AI Infrastructure</div>
    <div class="cover-title">DiT 推理加速<br/>原理与工程实践</div>
    <div class="cover-subtitle">计算图、GPU 执行、状态与通信：视频 DiT 与 AR-chunk 的优化方法（源码解析与合成实验）</div>
    <div class="cover-author">作者：{PUBLIC_AUTHOR} · <a href="{PUBLIC_AUTHOR_URL}">github.com/{PUBLIC_AUTHOR}</a> · 读者版 {EDITION_LABEL}</div>
    
    <div class="cover-meta-box">
      <div class="cover-meta-item"><strong>计算与算子</strong>：采样、Attention、低精度与 Kernel 执行</div>
      <div class="cover-meta-item"><strong>状态与系统</strong>：KV 生命周期、显存、通信与流式调度</div>
      <div class="cover-meta-item"><strong>方法实践</strong>：公开源码分析；RTX 5090 / SM90 单卡合成案例</div>
      <div class="cover-meta-item"><strong>版权与归属</strong>：© 2026 {PUBLIC_AUTHOR} · 仅供阅读与学习参考，详见 COPYRIGHT.md</div>
    </div>
  </div>
"""
    html_parts.append(cover)
    entries = "；".join(
        f'<a href="{escape((repo_root / path).as_uri(), quote=True)}">{escape(title)}</a>'
        for title, path in LINKED_SOURCES
    )
    html_parts.append(f"""
    <div id="reading-guide" class="chapter-container">
      <h1>阅读说明</h1>
      <p>第一篇用五章建立生成范式、执行图、GPU、显存和测量的共同基础；第二篇按加速方法组织十一章，
      将原理、实现、模型差异及验证放在同一处。Wan2.2 与 SolarWM 构成双向扩散和连续 AR-chunk 的两条主线，
      H3 与 ABot-World 用于架构和训练对照。Omni 框架已拆为独立配套文档，末尾附阅读指引。</p>
      <p>实测覆盖 RTX 5090 与数据中心 SM90 GPU（80 GB）的单卡算子、搬运和 CPU 状态小例，以及 50 层 AR 规模 KV 容量与 Gather 汇总。
      整模性能、多卡扩展与视频质量对照需另建基线。各实验的配置与适用边界见附录 A、C，测量原则见第 5 章。</p>
      <p>阅读入口：<a href="#global-contents">总目录</a>。配套入口：{entries}。PDF 和 HTML 与配套目录放在同一根目录，代码和证据摘录使用相对路径。
      模型权重、完整 trace 和视频不随包分发。部分 PDF 阅读器禁用本地文件跳转时，可打开同目录 HTML 或按附录路径查阅。</p>
      <p>PDF 中流程图按 300 DPI 整图嵌入，避免 Zotero“阅读模式”把节点、文字和箭头拆开；正文各页底层带有不干扰文本选择的轻量归属水印。
      重排模式仍可能遗漏行内公式；核对公式时请使用普通 PDF 页面视图或同目录 HTML。</p>
      <div class="edition-colophon">
        <p><strong>版本与署名</strong>：作者 {PUBLIC_AUTHOR}（<a href="{PUBLIC_AUTHOR_URL}">{PUBLIC_AUTHOR_URL}</a>）· 内容版本 {EDITION_LABEL} · 公开资料核对截止 2026-11。</p>
        <p><strong>推荐引用</strong>：{PUBLIC_AUTHOR}. 《DiT 推理加速：原理与工程实践（源码解析与合成实验）》, v1.0 (Reader Edition r9), 2026.</p>
        <p><strong>权限与勘误</strong>：原创文稿、图与示例保留全部权利，仅供阅读参考（见 COPYRIGHT.md）；勘误与反馈请通过 GitHub 主页提交。</p>
      </div>
    </div>
    """)

    def chapter_links(chapters):
        return '<ul>' + ''.join(
            f'<li><a href="#{chapter_id(path)}">{escape(title)}</a></li>'
            for title, path in chapters) + '</ul>'

    html_parts.append('<nav id="global-contents" class="contents"><h1>总目录</h1>'
                      '<a href="#reading-guide">阅读说明</a><div class="contents-grid">')
    for part_idx, (part_name, chapters) in enumerate(CHAPTER_FILES, 1):
        html_parts.append(f'<section><h2><a href="#part-{part_idx}">{escape(part_name)}</a></h2>')
        html_parts.append(chapter_links(chapters) + '</section>')
    html_parts.append('</div><section class="contents-backmatter"><h2>延伸与附录</h2>'
                      + chapter_links(BACK_MATTER) + '</section></nav>')

    def add_chapter(rel_path, *, last=False):
        raw_md = read_source(rel_path, "embedded_chapter")
        protected_md, formulas = protect_math(raw_md)
        body_html = markdown.markdown(
            preprocess_markdown(protected_md),
            extensions=['tables', 'fenced_code', 'codehilite', 'toc'],
            extension_configs={'toc': {'slugify': heading_slugger()}}
        )
        body_html = scope_heading_ids(body_html, rel_path)
        body_html = rewrite_file_links(body_html, rel_path, repo_root, included)
        body_html = restore_math(body_html, formulas)
        body_html = mark_long_tables(body_html)
        body_html = re.sub(r'<p>(图示：.*?)</p>\s*(<div class="mermaid[^"]*">.*?</div>)',
                           r'<figure class="diagram-figure"><figcaption>\1</figcaption>\2</figure>',
                           body_html, flags=re.S)
        last_class = 'last-chapter' if last else ''
        html_parts.append(f'<div id="{chapter_id(rel_path)}" '
                          f'class="chapter-container chapter-divider {last_class}">'
                          f'<!-- source: {escape(rel_path)} -->{body_html}</div>')

    for part_idx, (part_name, chapters) in enumerate(CHAPTER_FILES, 1):
        guide_path = f'docs/manuscript/part-{part_idx}.md'
        guide = read_source(guide_path, 'part_guide')
        guide = re.sub(r'^# [^\n]+\n', '', guide, count=1)
        guide = re.sub(r'^\[返回总目录\]\([^\n]+\)\s*$', '', guide, flags=re.M)
        guide_html = markdown.markdown(preprocess_markdown(guide), extensions=['tables', 'fenced_code'])
        guide_html = rewrite_file_links(guide_html, guide_path, repo_root, included)
        html_parts.append(f"""
        <nav id="part-{part_idx}" class="part-banner">
          <div class="part-title">{part_name}</div>
          <p>本篇目录</p>
          {guide_html}
          <a href="#global-contents">返回总目录</a>
        </nav>
        """)
        for _, rel_path in chapters:
            add_chapter(rel_path)
    for index, (_, rel_path) in enumerate(BACK_MATTER):
        add_chapter(rel_path, last=index == len(BACK_MATTER) - 1)

    html_parts.append("""
</body>
</html>
""")
    
    rendered = "\n".join(html_parts).replace("VENDOR_ROOT", (repo_root / "assets/vendor").as_uri())
    return "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"


def validate_omni_sources(repo_root):
    if not (repo_root / OMNI_SOURCE).is_file():
        raise BuildError("Missing required source files: " + OMNI_SOURCE)
    return [OMNI_SOURCE]


class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag in ('h1', 'h2', 'h3'):
            self.current = {'level': int(tag[1]), 'id': dict(attrs).get('id'), 'text': ''}

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'] += data

    def handle_endtag(self, tag):
        if self.current is not None and tag == 'h' + str(self.current['level']):
            self.rows.append(self.current)
            self.current = None


def render_omni_html(repo_root, *, source_records=None):
    validate_omni_sources(repo_root)
    try:
        import markdown
    except ImportError as exc:
        raise BuildError("Python Markdown is required to render the Omni companion") from exc
    content = (repo_root / OMNI_SOURCE).read_bytes()
    if source_records is not None:
        source_records.append({'path': OMNI_SOURCE, 'role': 'embedded_chapter',
                               'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    # Chapter references remain readable without a second document's local files.
    def local_reference(match):
        label, target = match.groups()
        return match.group(0) if urlsplit(target).scheme or target.startswith('#') else label
    source = re.sub(r'(?<!!)\[([^\]]+)\]\(([^\s)]+)\)', local_reference, content.decode('utf-8'))
    protected, formulas = protect_math(source)
    body = markdown.markdown(preprocess_markdown(protected),
                             extensions=['tables', 'fenced_code', 'codehilite', 'toc'],
                             extension_configs={'toc': {'slugify': heading_slugger()}})
    body = mark_long_tables(restore_math(body, formulas))
    headings = HeadingParser(); headings.feed(body)
    meta = (f'<p class="companion-meta">作者：{PUBLIC_AUTHOR}（<a href="{PUBLIC_AUTHOR_URL}">'
            f'github.com/{PUBLIC_AUTHOR}</a>）· 《DiT 推理加速》独立配套 · {EDITION_LABEL}</p>')
    nav = '<nav class="companion-toc"><p><strong>目录</strong></p><ul>' + ''.join(
        f'<li><a href="#{escape(row["id"], quote=True)}">{escape(row["text"])}</a></li>'
        for row in headings.rows if row['level'] == 2) + '</ul></nav>'
    body = body.replace('</h1>', '</h1>' + meta + nav, 1)
    header = html_header().replace('<title>DiT 推理加速：原理与工程实践</title>', f'<title>{OMNI_TITLE}</title>')
    header = header.replace(
        'content: "DiT 推理加速：原理与工程实践 · © 2026 SnowyLumiere (github.com/SnowyLumiere)";',
        'content: "Omni 与推理框架 · © 2026 SnowyLumiere (github.com/SnowyLumiere)";',
    )
    css = '''<style>
@page :first { margin: 18.5mm 16mm; @bottom-right { content: counter(page); }
  @bottom-left { content: "Omni 与推理框架 · © 2026 SnowyLumiere (github.com/SnowyLumiere)"; } }
h1 { font-size: 21pt; line-height: 1.35; margin-top: 0; }
.companion-toc { background: #f8fafc; border-left: 3px solid #3b82f6;
  padding: 8px 14px; margin: 10px 0 18px; break-inside: avoid; }
.companion-toc ul { margin: 0; padding-left: 20px; }
.companion-toc a { color: #1e3a8a; text-decoration: none; }
</style>'''
    rendered = header.replace('</head>', css + '</head>') + body + '</body></html>\n'
    return rendered.replace('VENDOR_ROOT', (repo_root / 'assets/vendor').as_uri())


def save_public_pdf(document, path, metadata_values):
    """Rewrite completely: incremental saves leave old metadata recoverable."""
    cleaned = dict(author='', creator='', producer='', creationDate='', modDate='')
    cleaned.update(metadata_values)
    cleaned.update(creator='', producer='', creationDate='', modDate='')
    document.set_metadata(cleaned)
    document.del_xml_metadata()
    with tempfile.TemporaryDirectory(prefix='.pdf-clean-', dir=Path(path).parent) as stage:
        clean = Path(stage) / 'clean.pdf'
        document.save(clean, garbage=4, deflate=True)
        os.replace(clean, path)


def add_omni_navigation(path, html):
    """Locate actual headings, excluding duplicate text in the contents list."""
    import fitz
    headings = HeadingParser(); headings.feed(html)
    with fitz.open(path) as document:
        toc, targets = [], {}
        for row in headings.rows:
            matches = [(i, rect) for i, page in enumerate(document)
                       for rect in page.search_for(row['text'])]
            if not matches:
                raise BuildError('Missing rendered Omni heading: ' + row['text'])
            # The contents precede the body. Use the last exact heading occurrence.
            page, rect = matches[-1]
            target = {'kind': fitz.LINK_GOTO, 'page': page,
                      'to': fitz.Point(0, max(0, rect.y0 - 14)), 'zoom': 0}
            targets[row['id']] = target
            toc.append([row['level'], row['text'], page + 1, target])
        for page in document:
            for link in page.get_links():
                if link['kind'] == fitz.LINK_NAMED:
                    anchor = unquote(link['nameddest'])
                    if anchor not in targets:
                        raise BuildError('Unresolved Omni destination: ' + anchor)
                    page.delete_link(link)
                    page.insert_link(dict(targets[anchor], **{'from': link['from']}))
        explicit_internal_links(document)
        document.set_toc(toc, collapse=1)
        watermark = apply_pdf_watermark(document, WATERMARK_TEXT, skip_first_page=False)
        save_public_pdf(document, path, dict(title=OMNI_TITLE, author=PUBLIC_AUTHOR,
                        subject='Omni 与 DiT、vLLM-Omni、SGLang-Omni 的源码分析',
                        keywords='Omni, DiT, vLLM-Omni, SGLang-Omni, streaming'))
    return {'bookmarks': len(toc), 'destinations_verified': True,
            'actual_heading_coordinates': True, 'watermark': watermark}


def find_chrome(explicit=None):
    candidates = ([explicit] if explicit else [
        os.environ.get("CHROME_BIN"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "google-chrome", "chromium", "chromium-browser",
    ])
    for candidate in filter(None, candidates):
        executable = shutil.which(str(candidate))
        if executable:
            return executable
    raise BuildError("Chrome/Chromium not found; pass --chrome or set CHROME_BIN")


def validate_pdf(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise BuildError("Chrome did not create a new nonempty PDF")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise BuildError("Chrome output has no PDF header")
        stream.seek(max(0, path.stat().st_size - 1024))
        if b"%%EOF" not in stream.read():
            raise BuildError("Chrome output has no PDF EOF marker")
    try:
        import fitz
    except ImportError as exc:
        raise BuildError("PyMuPDF (fitz) is required to validate the new PDF") from exc
    try:
        with fitz.open(path) as document:
            if (not document.is_pdf or document.needs_pass or document.is_repaired
                    or document.page_count < 1):
                raise BuildError("New PDF is empty, encrypted, or structurally damaged")
            for page in range(document.page_count):
                document.load_page(page)
            return document.page_count
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError("Cannot parse the new PDF: " + str(exc)) from exc


def explicit_internal_links(document):
    """Replace Chrome named destinations with explicit page targets.

    Chrome builds differ: some write long percent-encoded /Dests names that are
    cut at the 127-byte PDF name limit. Explicit targets behave the same in every
    reader, and the then-unused name dictionary is dropped. Bookmarks open by
    default, matching earlier releases.
    """
    import fitz
    converted = 0
    for page in document:
        for link in page.get_links():
            if link['kind'] != fitz.LINK_NAMED:
                continue
            if link.get('page', -1) < 0:
                raise BuildError('Unresolved PDF destination: ' + str(link.get('nameddest')))
            page.delete_link(link)
            page.insert_link({'kind': fitz.LINK_GOTO, 'from': link['from'], 'page': link['page'],
                              'to': link['to'], 'zoom': 0})
            converted += 1
    catalog = document.pdf_catalog()
    if document.xref_get_key(catalog, 'Dests')[0] != 'null':
        document.xref_set_key(catalog, 'Dests', 'null')
    document.set_pagemode('UseOutlines')
    return converted


def add_pdf_navigation(path):
    """Resolve rendered destinations rather than guessing pages from TOC text."""
    import fitz
    with fitz.open(path) as document:
        names = document.resolve_names()
        def page_for(anchor):
            if anchor not in names or names[anchor].get('page', -1) < 0:
                raise BuildError("Missing rendered navigation destination: " + anchor)
            return names[anchor]['page'] + 1
        toc = [[1, '封面', 1], [1, '阅读说明', page_for('reading-guide')],
               [1, '总目录', page_for('global-contents')]]
        for index, (part, chapters) in enumerate(CHAPTER_FILES, 1):
            toc.append([1, part, page_for(f'part-{index}')])
            for title, source in chapters:
                toc.append([2, title, page_for(chapter_id(source))])
        for title, source in BACK_MATTER:
            toc.append([1, title, page_for(chapter_id(source))])
        explicit_internal_links(document)
        document.set_toc(toc, collapse=1)
        watermark = apply_pdf_watermark(document, WATERMARK_TEXT, skip_first_page=True)
        metadata_values = dict(
            title='DiT 推理加速：原理与工程实践', author=PUBLIC_AUTHOR,
            subject='计算与测量基础、加速方法、公开源码分析与合成实验',
            keywords='DiT, GPU, inference, AR-chunk')
        save_public_pdf(document, path, metadata_values)
    return {'bookmarks': len(toc), 'destinations_verified': True, 'watermark': watermark}


def build(repo_root, output_pdf, output_html, chrome, *, timeout=180, receipt_path=None, omni=False):
    repo_root = Path(repo_root).resolve(strict=True)
    output_pdf, output_html = Path(output_pdf).absolute(), Path(output_html).absolute()
    receipt_path = Path(receipt_path).absolute() if receipt_path else output_pdf.with_suffix(".build-receipt.json")
    outputs = (output_pdf, output_html, receipt_path)
    if (len({path.resolve() for path in outputs}) != len(outputs)
            or any(path.is_symlink() or (path.exists() and not path.is_file()) for path in outputs)):
        raise BuildError("PDF, HTML and receipt must be distinct regular output paths, not symlinks")
    if timeout <= 0:
        raise BuildError("--timeout must be positive")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    builder_bytes = Path(__file__).read_bytes()
    sources = []
    html = (render_omni_html if omni else render_html)(repo_root, source_records=sources)
    print_html, diagram_markers = mark_mermaid_for_pdf(html)
    render_assets = render_asset_records(html, repo_root)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Each staged file lives on its publication filesystem. Never give Chrome
    # the existing output filename: a failed render cannot reuse an old PDF.
    with tempfile.TemporaryDirectory(prefix=".whitepaper-pdf-", dir=output_pdf.parent) as pdf_stage, \
            tempfile.TemporaryDirectory(prefix=".whitepaper-html-", dir=output_html.parent) as html_stage, \
            tempfile.TemporaryDirectory(prefix=".whitepaper-receipt-", dir=receipt_path.parent) as receipt_stage:
        staged_pdf = Path(pdf_stage) / "new.pdf"
        staged_html = Path(html_stage) / "document.html"
        staged_html.write_text(print_html, encoding="utf-8")
        cmd = [
            str(chrome),
            "--headless",
            "--disable-gpu",
            "--disable-background-networking",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=12000",
            "--no-pdf-header-footer",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={Path(pdf_stage) / 'chrome-profile'}",
            f"--print-to-pdf={staged_pdf}",
            staged_html.as_uri(),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise BuildError(f"Chrome exceeded {timeout}s; previous PDF retained") from exc
        if result.returncode != 0:
            raise BuildError(f"Chrome exited {result.returncode}; previous PDF retained.\n"
                             + (result.stderr or "")[-4000:])
        page_count = validate_pdf(staged_pdf)
        diagram_export = rasterize_mermaid_pdf(staged_pdf, diagram_markers)
        link_report = portable_pdf_links(staged_pdf, repo_root, output_pdf.parent)
        navigation = (add_omni_navigation(staged_pdf, html) if omni
                      else add_pdf_navigation(staged_pdf))
        validate_pdf(staged_pdf)
        staged_html.write_text(portable_html(html, repo_root, output_html.parent), encoding="utf-8")
        size = staged_pdf.stat().st_size
        pdf_hash = hashlib.sha256(staged_pdf.read_bytes()).hexdigest()
        html_bytes = staged_html.read_bytes()
        dependencies = {}
        for package in ("Markdown", "PyMuPDF"):
            try:
                dependencies[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                dependencies[package] = None
        source_json = json.dumps(sources, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
        receipt = {
            "schema_version": 2,
            "status": "new_render_validated",
            "document": "omni" if omni else "main",
            "started_at": started_at,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds_through_validation": time.monotonic() - started,
            "repo_root": str(repo_root),
            "builder": {"path": str(Path(__file__).resolve()), "bytes": len(builder_bytes),
                        "sha256": hashlib.sha256(builder_bytes).hexdigest()},
            "runtime": {"python_executable": sys.executable, "python_version": sys.version,
                        "dependencies": dependencies},
            "sources": sources,
            "render_assets": render_assets,
            "sources_sha256": hashlib.sha256(source_json).hexdigest(),
            "embedded_sources": sum(item["role"] == "embedded_chapter" for item in sources),
            "repository_entry_links": sum(item["role"] == "repository_entry" for item in sources),
            "render": {"command": cmd, "returncode": result.returncode,
                       "timeout_seconds": timeout, "isolated_temporary_profile": True,
                       "remote_dependencies": [],
                       "vendored_dependencies": ["KaTeX 0.16.8", "Mermaid 10.9.3"],
                       "diagram_export": diagram_export,
                       "diagrams_and_math_visually_verified": False},
            "links": link_report,
            "navigation": navigation,
            "artifact_path_base": "repo_root",
            "pdf": {"path": os.path.relpath(output_pdf.resolve(), repo_root), "bytes": size, "pages": page_count,
                    "sha256": pdf_hash},
            "html": {"path": os.path.relpath(output_html.resolve(), repo_root), "bytes": len(html_bytes),
                     "sha256": hashlib.sha256(html_bytes).hexdigest()},
            # The receipt is prepared before the final PDF commit. A consumer
            # must compare this SHA with the artifact, including after failures.
            "artifact_sha_match_required": True,
        }
        staged_receipt = Path(receipt_stage) / "receipt.json"
        receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        staged_receipt.write_bytes(receipt_bytes)
        # Publish only after all render/validation checks. PDF replacement is
        # the final commit step; HTML/receipt errors retain the previous PDF.
        # Each replacement is atomic; consumers verify PDF/HTML against receipt
        # hashes because the three files are not a single filesystem transaction.
        os.replace(staged_html, output_html)
        os.replace(staged_receipt, receipt_path)
        os.replace(staged_pdf, output_pdf)
    return {"pdf": str(output_pdf), "html": str(output_html),
            "pages": page_count, "bytes": size, "sha256": pdf_hash,
            "receipt": str(receipt_path), "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "embedded_sources": receipt["embedded_sources"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--receipt", type=Path, help="build receipt; default: PDF path with .build-receipt.json suffix")
    parser.add_argument("--chrome", help="Chrome executable; otherwise CHROME_BIN or autodetect")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--omni", action="store_true", help="build the independent Omni companion")
    parser.add_argument("--check", action="store_true", help="check required sources only; no writes or Chrome")
    args = parser.parse_args(argv)
    try:
        root = args.repo_root.resolve(strict=True)
        if args.timeout <= 0:
            raise BuildError("--timeout must be positive")
        if args.check:
            paths = validate_omni_sources(root) if args.omni else validate_sources(root)
            entries = 0 if args.omni else len(LINKED_SOURCES)
            print(f"CHECK OK: {len(paths)} embedded sources; {entries} repository entry links; root={root}")
            return 0
        result = build(root, args.output_pdf or root / (OMNI_PDF_NAME if args.omni else PDF_NAME),
                       args.output_html or root / (OMNI_HTML_NAME if args.omni else HTML_NAME),
                       find_chrome(args.chrome), timeout=args.timeout, receipt_path=args.receipt, omni=args.omni)
    except (BuildError, OSError, ValueError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    print(f"SUCCESS: newly rendered and validated '{result['pdf']}'")
    print(f"Page Count: {result['pages']} | Bytes: {result['bytes']} | Embedded sources: {result['embedded_sources']}")
    print(f"PDF SHA256: {result['sha256']}")
    print(f"Build receipt: {result['receipt']} | SHA256: {result['receipt_sha256']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
