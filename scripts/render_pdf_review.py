#!/usr/bin/env python3
"""Render every PDF page and two-page sheets for manual layout review.

Requires PyMuPDF. Geometry flags are inspection hints, not visual acceptance.
The output directory must be new; source PDF is never changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--dpi', type=int, default=144)
    args = parser.parse_args()
    if not 96 <= args.dpi <= 300:
        parser.error('--dpi must be between 96 and 300')
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error('--output-dir must be new')
    import fitz
    args.output_dir.mkdir(parents=True)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'pdf_sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
              'renderer': 'PyMuPDF '+fitz.VersionBind, 'dpi': args.dpi,
              'manual_visual_acceptance': False, 'pages': [], 'contact_sheets': []}
    with fitz.open(args.pdf) as document:
        for index, page in enumerate(document):
            spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                     for line in block['lines'] for span in line['spans'] if span['text'].strip()]
            outside = [s['text'] for s in spans if fitz.Rect(s['bbox']).x0 < -0.5
                       or fitz.Rect(s['bbox']).x1 > page.rect.width+0.5
                       or fitz.Rect(s['bbox']).y0 < -0.5
                       or fitz.Rect(s['bbox']).y1 > page.rect.height+0.5]
            name = f'page-{index+1:03}.png'
            page.get_pixmap(dpi=args.dpi, alpha=False).save(args.output_dir/name)
            report['pages'].append({'page': index+1, 'png': name,
                                    'png_sha256': hashlib.sha256((args.output_dir/name).read_bytes()).hexdigest(),
                                    'text_spans': len(spans), 'outside_page_text': outside,
                                    'smallest_text_pt': min((s['size'] for s in spans), default=None)})
        for index in range(0, len(document), 2):
            with fitz.open() as sheet:
                rect = document[index].rect
                page = sheet.new_page(width=rect.width*2, height=rect.height)
                page.show_pdf_page(fitz.Rect(0,0,rect.width,rect.height), document,index)
                if index+1 < len(document):
                    page.show_pdf_page(fitz.Rect(rect.width,0,rect.width*2,rect.height),document,index+1)
                name = f'contact-{index+1:03}-{min(index+2,len(document)):03}.png'
                page.get_pixmap(dpi=args.dpi, alpha=False).save(args.output_dir/name)
                report['contact_sheets'].append(name)
    (args.output_dir/'render-receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pages':len(report['pages']),'sheets':len(report['contact_sheets']),
                      'geometry_flags':[p['page'] for p in report['pages'] if p['outside_page_text']],
                      'pdf_sha256':report['pdf_sha256']},ensure_ascii=False))


if __name__ == '__main__':
    main()
