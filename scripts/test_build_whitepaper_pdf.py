"""CPU-only builder checks. Chrome is always mocked; all PDFs are tiny fixtures."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "build_whitepaper_pdf", Path(__file__).with_name("build_whitepaper_pdf.py")
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class SourceChecks(unittest.TestCase):
    def test_omni_check_is_independent_of_main_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / builder.OMNI_SOURCE
            source.parent.mkdir(parents=True); source.write_text('# Omni\n')
            output = io.StringIO()
            with mock.patch.object(builder, 'find_chrome', side_effect=AssertionError('Chrome called')), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(builder.main(['--repo-root', directory, '--omni', '--check']), 0)
            self.assertIn('1 embedded sources; 0 repository entry links', output.getvalue())

    def test_omni_default_artifacts_do_not_replace_main_book(self):
        fake = {'pdf': 'omni.pdf', 'pages': 1, 'bytes': 1, 'embedded_sources': 1,
                'sha256': 'a', 'receipt': 'r', 'receipt_sha256': 'b'}
        with mock.patch.object(builder, 'build', return_value=fake) as build, \
                mock.patch.object(builder, 'find_chrome', return_value='chrome'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(builder.main(['--omni']), 0)
        self.assertEqual(build.call_args.args[1].name, builder.OMNI_PDF_NAME)
        self.assertEqual(build.call_args.args[2].name, builder.OMNI_HTML_NAME)
        self.assertTrue(build.call_args.kwargs['omni'])

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF required')
    def test_full_rewrite_removes_old_metadata_and_xml_bytes(self):
        import fitz
        marker = 'renderer-private-marker-12345'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.pdf'
            with fitz.open() as document:
                document.new_page().insert_text((40, 40), 'Public text')
                document.set_metadata({'creator': marker, 'producer': marker, 'author': marker,
                                       'creationDate': 'D:20261031010000Z'})
                document.set_xml_metadata('<metadata>' + marker + '</metadata>')
                document.save(path)
            with fitz.open(path) as document:
                builder.save_public_pdf(document, path, {'title': 'Public document'})
            self.assertNotIn(marker.encode(), path.read_bytes())
            self.assertNotIn(b'D:20261031010000Z', path.read_bytes())
            with fitz.open(path) as document:
                self.assertFalse(document.get_xml_metadata())
                for key in ('author', 'creator', 'producer', 'creationDate', 'modDate'):
                    self.assertFalse(document.metadata[key])
                self.assertIn('Public text', document[0].get_text())

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF required')
    def test_omni_bookmark_targets_heading_instead_of_contents_copy(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'omni.pdf'
            with fitz.open() as document:
                document.new_page().insert_text((40, 40), 'Stage scheduling')
                document.new_page().insert_text((40, 140), 'Stage scheduling')
                document.save(path)
            result = builder.add_omni_navigation(path, '<h1 id="stage">Stage scheduling</h1>')
            self.assertTrue(result['actual_heading_coordinates'])
            with fitz.open(path) as document:
                self.assertEqual(document.get_toc(), [[1, 'Stage scheduling', 2]])
                self.assertGreater(document.get_toc(simple=False)[0][3]['to'].y, 100)

    @unittest.skipUnless(importlib.util.find_spec('markdown'), 'Markdown required')
    def test_omni_render_tracks_one_source_and_reuses_offline_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); path = root / builder.OMNI_SOURCE
            path.parent.mkdir(parents=True)
            path.write_text('# Omni\n\n## Stage\n\n[Chapter](../manuscript/06.md)\n\n$x^2$\n')
            records = []
            value = builder.render_omni_html(root, source_records=records)
            self.assertIn(builder.OMNI_TITLE, value)
            self.assertIn('companion-toc', value)
            self.assertNotIn('href="../manuscript', value)
            self.assertIn('katex-0.16.8', value)
            self.assertIn('$x^2$', value)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_two_parts_have_sixteen_chapters_and_separate_back_matter(self):
        self.assertEqual([len(chapters) for _, chapters in builder.CHAPTER_FILES], [5, 11])
        paths = [path for _, chapters in builder.CHAPTER_FILES for _, path in chapters]
        numbered = [Path(path).name[:2] for path in paths if Path(path).name[:2].isdigit()]
        self.assertEqual(numbered, [f"{i:02}" for i in range(1, 17)])
        self.assertEqual(len(builder.BACK_MATTER), 4)
        self.assertEqual(Path(builder.BACK_MATTER[0][1]).name, 'extension-omni.md')
        self.assertTrue(all(path.startswith('docs/manuscript/') for path in paths))
        self.assertTrue(all(path.startswith('docs/manuscript/') for path in builder.validate_sources(builder.REPO_ROOT)))

    def test_required_missing_source_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(builder.BuildError, "Missing required source"):
                builder.validate_sources(Path(directory))

    def test_source_check_does_not_render_or_find_chrome(self):
        with mock.patch.object(builder, "find_chrome", side_effect=AssertionError("Chrome called")), \
                mock.patch.object(builder, "build", side_effect=AssertionError("build called")), \
                mock.patch.object(builder, "validate_sources", return_value=["a.md"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(builder.main(["--check"]), 0)

    @unittest.skipUnless(importlib.util.find_spec('markdown'), 'Markdown required')
    def test_only_marked_tables_may_split_across_pages(self):
        import markdown
        table = "| a | b |\n|---|---|\n" + "| x | y |\n" * 8
        source = table + "\n<!-- reader:table-may-split -->\n\n" + table
        html = builder.mark_long_tables(markdown.markdown(source, extensions=["tables"]))
        self.assertEqual(html.count('<table class="long-table">'), 1)
        self.assertEqual(html.count('<table>'), 1)
        self.assertLess(html.index('<table>'), html.index('<table class="long-table">'))

    def test_header_avoids_pdfkit_cover_clip_bug_and_has_screen_layout(self):
        header = builder.html_header()
        self.assertNotIn('-webkit-background-clip: text', header)
        self.assertNotIn('-webkit-text-fill-color: transparent', header)
        self.assertIn('<meta name="viewport"', header)
        self.assertIn('@media screen', header)
        self.assertIn(builder.PUBLIC_AUTHOR, header)

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF required')
    def test_watermark_is_non_text_background_image_and_preserves_explicit_author(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wm.pdf'
            with fitz.open() as document:
                document.new_page().insert_text((40, 40), 'Cover page text')
                document.new_page().insert_text((40, 40), 'Body page text')
                document.new_page().insert_text((40, 40), 'Second body page')
                report = builder.apply_pdf_watermark(document, skip_first_page=True)
                builder.save_public_pdf(document, path, {'title': 'Doc', 'author': builder.PUBLIC_AUTHOR})
            self.assertEqual(report['pages_watermarked'], 2)
            self.assertFalse(report['overlay'])
            self.assertGreater(report['png_bytes'], 100)
            with fitz.open(path) as document:
                self.assertEqual(document.metadata['author'], builder.PUBLIC_AUTHOR)
                for key in ('creator', 'producer', 'creationDate', 'modDate'):
                    self.assertFalse(document.metadata[key])
                self.assertEqual(len(document[0].get_images()), 0)
                self.assertEqual(len(document[1].get_images()), 1)
                self.assertEqual(len(document[2].get_images()), 1)
                # Shared single Image XObject xref across pages of equal dimensions.
                self.assertEqual(document[1].get_images()[0][0], document[2].get_images()[0][0])
                # Watermark must never introduce selectable text.
                self.assertEqual(document[1].get_text().strip(), 'Body page text')
                self.assertNotIn(builder.PUBLIC_AUTHOR, document[1].get_text())

    def test_math_and_mermaid_preprocessing_is_retained(self):
        value = builder.preprocess_markdown("$x^2$\n```mermaid\ngraph TD\n A --> B\n```\n")
        self.assertIn("$x^2$", value)
        self.assertIn('<div class="mermaid">', value)
        self.assertIn("A --> B", value)

    def test_render_records_exact_consumed_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "chapter.md").write_bytes(b"# chapter\n")
            (root / "entry.md").write_bytes(b"# linked entry\n")
            (root / 'docs/manuscript').mkdir(parents=True)
            (root / 'docs/manuscript/part-1.md').write_text('# Part\n\nAR reading route\n\n- [Chapter](../../chapter.md)\n')
            records = []
            fake_markdown = SimpleNamespace(markdown=lambda text, **kwargs: "<p>" + text + "</p>")
            with mock.patch.object(builder, "CHAPTER_FILES", [("Part", [("Chapter", "chapter.md")])]), \
                    mock.patch.object(builder, "BACK_MATTER", []), \
                    mock.patch.object(builder, "LINKED_SOURCES", [("Entry", "entry.md")]), \
                    mock.patch.dict(sys.modules, {"markdown": fake_markdown}):
                rendered = builder.render_html(root, source_records=records)
            self.assertIn('id="chapter-chapter-md"', rendered)
            self.assertIn("katex-0.16.8", rendered)
            self.assertIn("mermaid-10.9.3", rendered)
            self.assertIn("<title>DiT 推理加速：原理与工程实践</title>", rendered)
            self.assertIn('id="global-contents"', rendered)
            self.assertIn('id="part-1"', rendered)
            self.assertIn('本篇目录', rendered)
            self.assertIn('AR reading route', rendered)
            self.assertEqual({item["path"] for item in records}, {"chapter.md", "entry.md", 'docs/manuscript/part-1.md'})
            for item in records:
                content = (root / item["path"]).read_bytes()
                self.assertEqual(item["bytes"], len(content))
                self.assertEqual(item["sha256"], hashlib.sha256(content).hexdigest())

    def test_chapter_images_and_source_links_resolve_from_the_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            body = '<img src="figures/a%20b.svg"><a href="../cases/demo.md">case</a>'
            body += '<a href="../../examples/README.md#run">code</a><a href="https://example.org/x">web</a>'
            value = builder.rewrite_file_links(body, "docs/chapters/README.md", root,
                                                {"docs/cases/demo.md"})
            self.assertIn((root / "docs/chapters/figures/a b.svg").as_uri(), value)
            self.assertIn('href="#chapter-docs-cases-demo-md"', value)
            self.assertIn((root / "examples/README.md").as_uri() + "#run", value)
            self.assertIn('href="https://example.org/x"', value)

    def test_heading_fragments_are_scoped_to_the_correct_chapter(self):
        body = '<a href="#安装">same</a><a href="b.md#安装">other</a>'
        value = builder.rewrite_file_links(body, 'a.md', Path('/book'), {'a.md', 'b.md'})
        self.assertIn('href="#chapter-a-md--安装"', value)
        self.assertIn('href="#chapter-b-md--安装"', value)

    def test_saved_html_uses_relative_companion_paths(self):
        root = Path('/book')
        value = builder.portable_html('<a href="file:///book/docs/a%20b.md#run">x</a>', root, root)
        self.assertEqual(value, '<a href="docs/a%20b.md#run">x</a>')
        with self.assertRaisesRegex(builder.BuildError, 'escapes repository'):
            builder.portable_html('<img src="file:///private/a.png">', root, root)

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF required")
    def test_real_pdf_file_actions_become_relative_uri_actions(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); path = root / 'x.pdf'
            with fitz.open() as document:
                page = document.new_page()
                page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(10, 10, 90, 30),
                                  'uri': (root / 'docs/a b.md').as_uri() + '#run'})
                document.save(path)
            report = builder.portable_pdf_links(path, root, root)
            self.assertEqual(report['rewritten_local_links'], 1)
            with fitz.open(path) as document:
                link = document[0].get_links()[0]
                self.assertEqual(document.xref_get_key(link['xref'], 'A/URI'),
                                 ('string', 'docs/a%20b.md#run'))
                self.assertEqual(document.xref_get_key(link['xref'], 'A/S'), ('name', '/URI'))

    def test_data_attributes_are_not_rewritten_as_links_or_ids(self):
        body = '<div data-id="keep" data-href="a.md" data-src="b.png" id="actual"></div>'
        scoped = builder.scope_heading_ids(body, 'a.md')
        self.assertIn('data-id="keep"', scoped)
        value = builder.rewrite_file_links(scoped, 'a.md', Path('/book'), {'a.md'})
        self.assertIn('data-href="a.md" data-src="b.png"', value)

    @unittest.skipUnless(importlib.util.find_spec("markdown"), "Markdown required")
    def test_duplicate_heading_fragments_follow_github_numbering(self):
        import markdown
        body = markdown.markdown('# 安装\n\n# 安装\n', extensions=['toc'],
                                 extension_configs={'toc': {'slugify': builder.heading_slugger()}})
        self.assertIn('id="安装-1"', body)
        self.assertNotIn('id="安装_1"', body)

    def test_chapter_namespace_collisions_fail_before_rendering(self):
        with mock.patch.object(builder, 'CHAPTER_FILES', [('Part', [('A', 'a/b.md'), ('B', 'a-b.md')])]):
            with self.assertRaisesRegex(builder.BuildError, 'namespace collision'):
                builder.validate_sources(Path('/book'))

    def test_missing_css_font_is_a_build_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'a.css').write_text('@font-face{src:url(missing.woff2)}')
            with self.assertRaisesRegex(builder.BuildError, 'Missing render asset'):
                builder.render_asset_records('<link href="'+(root/'a.css').as_uri()+'">', root)

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF required")
    def test_launch_action_is_rejected_before_pdf_publication(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); path = root/'launch.pdf'
            with fitz.open() as document:
                page = document.new_page()
                page.insert_link({'kind': fitz.LINK_LAUNCH, 'from': fitz.Rect(10,10,90,30), 'file': '/private/example'})
                document.save(path)
            with self.assertRaisesRegex(builder.BuildError, 'Launch action'):
                builder.portable_pdf_links(path, root, root)

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF required")
    def test_named_destinations_become_explicit_targets(self):
        import fitz
        # Fixture: one 127-byte named destination, as written by some Chrome builds.
        with fitz.open() as source:
            source.new_page(); source.new_page()
            first = source[0]
            first.insert_link({'kind': fitz.LINK_GOTO, 'from': fitz.Rect(10, 10, 60, 30),
                               'page': 1, 'to': fitz.Point(0, 0), 'zoom': 0})
            first = source.reload_page(first)
            link_xref, second_xref = first.get_links()[0]['xref'], source[1].xref
            del first
            source.xref_set_key(link_xref, 'Dest', '/' + 'x' * 127)
            source.xref_set_key(link_xref, 'A', 'null')
            source.xref_set_key(source.pdf_catalog(), 'Dests',
                                f'<< /{"x" * 127} [{second_xref} 0 R /XYZ 0 700 0] >>')
            fixture = source.tobytes()
        with fitz.open('pdf', fixture) as document:
            self.assertEqual(document[0].get_links()[0]['kind'], fitz.LINK_NAMED)
            self.assertEqual(builder.explicit_internal_links(document), 1)
            link = document[0].get_links()[0]
            self.assertEqual((link['kind'], link['page']), (fitz.LINK_GOTO, 1))
            self.assertEqual(document.xref_get_key(document.pdf_catalog(), 'Dests')[0], 'null')
            self.assertEqual(document.pagemode, 'UseOutlines')

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF required")
    def test_missing_rendered_navigation_is_a_build_failure(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'no-destinations.pdf'
            with fitz.open() as document:
                document.new_page().insert_text((40, 40), 'No navigation targets')
                document.save(path)
            with self.assertRaisesRegex(builder.BuildError, 'Missing rendered navigation'):
                builder.add_pdf_navigation(path)


class MarkdownRegressionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import markdown
        except ImportError:
            raise unittest.SkipTest("Markdown required")
        cls.markdown = markdown

    def render(self, source):
        protected, formulas = builder.protect_math(source)
        body = self.markdown.markdown(
            builder.preprocess_markdown(protected), extensions=["tables", "fenced_code", "codehilite"]
        )
        return builder.restore_math(body, formulas)

    def test_paragraph_adjacent_star_list_is_an_actual_list(self):
        body = self.render("其中：\n* Batch size\n* Channels\n")
        self.assertIn("<p>其中：</p>", body)
        self.assertIn("<ul>", body)
        self.assertEqual(body.count("<li>"), 2)
        self.assertNotIn("* Batch size", body)

    def test_callout_header_does_not_swallow_the_next_quote_marker(self):
        body = self.render('> [!NOTE]\n> 后续正文\n')
        self.assertIn('后续正文', body)
        self.assertNotIn('&gt;', body)

    def test_two_and_four_space_nested_lists_keep_their_hierarchy(self):
        from xml.etree import ElementTree
        for indent in (2, 4):
            with self.subTest(indent=indent):
                source = "说明：\n* 父项\n" + " " * indent + "* 子一\n"
                source += " " * indent + "* 子二\n* 末项\n"
                root = ElementTree.fromstring("<root>" + self.render(source) + "</root>")
                self.assertEqual(len(root.findall("./ul/li")), 2)
                self.assertEqual(len(root.findall("./ul/li/ul/li")), 2)

    def test_teacache_display_formula_survives_without_emphasis_damage(self):
        from html import unescape
        formula = r"$$\Delta_{\text{rel}}(t) = \frac{\| c(t) - c(t_{\text{prev}}) \|_1}{\| c(t_{\text{prev}}) \|_1}$$"
        for separator in ("", "\n"):
            with self.subTest(separator=repr(separator)):
                body = self.render("TeaCache 相对变化：" + separator + formula + "\n其中 $c(t)$ 是条件。")
                self.assertIn(formula, unescape(body))
                self.assertNotIn("<em>", body)
                self.assertNotIn("<strong>", body)

    def test_inline_multisubscripts_and_html_characters_reach_katex_unchanged(self):
        from html import unescape
        formula = r"$X_{\text{layer46\_out}} - X_{\text{layer3\_in}} < a_i_j \land b_k_l & c_m_n$"
        body = self.render("内联 " + formula + " 结束。")
        self.assertIn(formula, unescape(body))
        self.assertIn("&lt;", body)
        self.assertIn("&amp;", body)
        self.assertNotIn("<em>", body)
        self.assertNotIn("<strong>", body)

    def test_math_operator_lines_are_not_treated_as_list_items(self):
        from html import unescape
        formula = "$$\nX_a_b = 1\n* X_c_d + 2\n$$"
        body = self.render("公式：\n" + formula + "\n* 唯一列表项\n")
        self.assertIn(formula, unescape(body))
        self.assertEqual(body.count("<li>"), 1)

    def test_code_fences_and_multiline_inline_code_are_not_rewritten(self):
        tick = chr(96)
        source = tick * 4 + "python\nvalue = '$x_i_j$'\n* literal\n" + tick * 3 + "\n" + tick * 4 + "\n\n"
        source += "~~~text\n$$y_a_b$$\n> [!NOTE] literal\n* literal\n~~~\n\n"
        source += tick * 2 + " inline $z_a_b$\n* still code " + tick * 2 + "\n"
        protected, formulas = builder.protect_math(source)
        self.assertEqual(formulas, {})
        self.assertEqual(protected, source)
        self.assertEqual(builder.preprocess_markdown(protected), source)
        body = self.render(source)
        self.assertNotIn("<li>", body)
        self.assertIn("$x_i_j$", body)
        self.assertIn("$$y_a_b$$", body)
        self.assertIn("$z_a_b$", body)

    def test_formula_scanning_cannot_cross_inline_code_or_an_unclosed_fence(self):
        tick = chr(96)
        source = "$unclosed " + tick + "$literal_a_b$" + tick + " outside $valid_i_j$"
        protected, formulas = builder.protect_math(source)
        self.assertEqual(list(formulas.values()), ["$valid_i_j$"])
        self.assertIn(tick + "$literal_a_b$" + tick, protected)
        source = "~~~python\n* literal\nvalue = '$not_math_i_j$'\n"
        protected, formulas = builder.protect_math(source)
        self.assertEqual(formulas, {})
        self.assertEqual(builder.preprocess_markdown(protected), source)

    def test_indented_code_and_escaped_or_unmatched_dollars_remain_literal(self):
        body = self.render("代码：\n\n    value = '$x_i_j$'\n    * literal\n")
        self.assertIn("$x_i_j$", body)
        self.assertNotIn("<li>", body)
        source = r"literal \$5 and unmatched $"
        protected, formulas = builder.protect_math(source)
        self.assertEqual(protected, source)
        self.assertEqual(formulas, {})


class DiagramImageChecks(unittest.TestCase):
    def setUp(self):
        if not importlib.util.find_spec('fitz'):
            self.skipTest('PyMuPDF required')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'figures.pdf'
        self.markers = [{'id': '0001', 'uri': builder.DIAGRAM_MARKER_BASE + '0001',
                         'source_sha256': hashlib.sha256(b'diagram source').hexdigest()}]

    def fixture(self, *, missing=False, duplicate=False, extra=False, crossing=False,
                rectangle=None, external_inside=False):
        import fitz
        rect = fitz.Rect(rectangle or (40, 90, 360, 200))
        with fitz.open() as document:
            page = document.new_page(width=400, height=400)
            page.insert_text((40, 35), 'Body heading remains selectable')
            page.insert_text((60, 140), 'NODE LEFT -> NODE RIGHT')
            page.draw_rect((50, 110, 350, 180), color=(0, 0, 0), fill=(.9, .9, .9), overlay=False)
            page.insert_text((40, 240), 'Body formula E = m*c^2 remains')
            page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(40, 270, 250, 290),
                              'uri': 'https://example.com/body'})
            if crossing:
                page.insert_text((20, 120), 'Body text crosses the diagram boundary')
            if external_inside:
                page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(60, 120, 180, 145),
                                  'uri': 'https://example.com/inside'})
            if not missing:
                page.insert_link({'kind': fitz.LINK_URI, 'from': rect, 'uri': self.markers[0]['uri']})
            if duplicate:
                other = document.new_page(width=400, height=400)
                other.insert_link({'kind': fitz.LINK_URI, 'from': rect, 'uri': self.markers[0]['uri']})
            if extra:
                page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(40, 310, 360, 350),
                                  'uri': builder.DIAGRAM_MARKER_BASE + '9999'})
            document.save(self.path)

    def test_stage_links_wrap_only_complete_mermaid_blocks(self):
        html = ('<p>Body before</p><div class="mermaid">graph TD; A-->B</div>'
                '<figure><figcaption>Caption stays outside</figcaption>'
                '<div class="mermaid mermaid-chart">xychart-beta</div></figure><p>Body after</p>')
        staged, markers = builder.mark_mermaid_for_pdf(html)
        self.assertEqual([row['id'] for row in markers], ['0001', '0002'])
        self.assertEqual(staged.count(builder.DIAGRAM_MARKER_BASE), 2)
        self.assertIn('<p>Body before</p><a ', staged)
        self.assertIn('Caption stays outside</figcaption><a ', staged)
        self.assertIn('</div></a></figure><p>Body after</p>', staged)
        self.assertNotIn(builder.DIAGRAM_MARKER_BASE, html)

    def test_existing_reserved_marker_is_rejected(self):
        with self.assertRaisesRegex(builder.BuildError, 'Reserved diagram marker'):
            builder.mark_mermaid_for_pdf('<a href="' + self.markers[0]['uri'] + '">old</a>')

    def test_complete_figure_becomes_one_image_without_hidden_label_text(self):
        import fitz
        self.fixture()
        result = builder.rasterize_mermaid_pdf(self.path, self.markers)
        self.assertEqual(result['expected_diagrams'], 1)
        self.assertEqual(result['rasterized_diagrams'], 1)
        self.assertEqual(result['temporary_marker_links_removed'], 1)
        self.assertEqual(result['dpi'], 300)
        record = result['diagrams'][0]
        self.assertEqual(record['page'], 1)
        self.assertEqual(record['source_sha256'], self.markers[0]['source_sha256'])
        self.assertGreater(record['pixel_size'][0], 1300)
        self.assertGreater(record['pixel_size'][1], 450)
        self.assertEqual(len(record['png_sha256']), 64)
        with fitz.open(self.path) as document:
            page = document[0]
            self.assertEqual(len(page.get_images()), 1)
            self.assertIn('Body heading remains selectable', page.get_text())
            self.assertIn('Body formula E = m*c^2 remains', page.get_text())
            self.assertNotIn('NODE LEFT', page.get_text())
            self.assertFalse(page.get_text('text', clip=fitz.Rect(record['rect_points'])).strip())
            self.assertEqual([link.get('uri') for link in page.get_links()], ['https://example.com/body'])
            image = page.get_image_info(xrefs=True)[0]
            self.assertEqual(list(image['bbox']), record['rect_points'])

    def test_multiple_figures_on_a_page_are_snapshotted_before_redaction(self):
        import fitz
        self.fixture()
        second = {'id': '0002', 'uri': builder.DIAGRAM_MARKER_BASE + '0002',
                  'source_sha256': hashlib.sha256(b'second').hexdigest()}
        with fitz.open(self.path) as document:
            page = document[0]
            page.insert_text((50, 325), 'SECOND DIAGRAM')
            page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(40, 300, 360, 350), 'uri': second['uri']})
            data = document.tobytes()
        self.path.write_bytes(data)
        result = builder.rasterize_mermaid_pdf(self.path, self.markers + [second])
        self.assertEqual(result['rasterized_diagrams'], 2)
        with fitz.open(self.path) as document:
            self.assertEqual(len(document[0].get_images()), 2)
            self.assertNotIn('NODE LEFT', document[0].get_text())
            self.assertNotIn('SECOND DIAGRAM', document[0].get_text())

    def test_bad_marker_geometry_or_coverage_never_changes_input_pdf(self):
        cases = [({'missing': True}, 'Missing diagram markers'),
                 ({'duplicate': True}, 'duplicated or split'),
                 ({'extra': True}, 'Unexpected diagram marker'),
                 ({'rectangle': (380, 390, 450, 430)}, 'out-of-page'),
                 ({'crossing': True}, 'Text crosses'),
                 ({'external_inside': True}, 'overlaps another link')]
        for options, message in cases:
            with self.subTest(options=options):
                self.path.unlink(missing_ok=True)
                self.fixture(**options)
                original = self.path.read_bytes()
                with self.assertRaisesRegex(builder.BuildError, message):
                    builder.rasterize_mermaid_pdf(self.path, self.markers)
                self.assertEqual(self.path.read_bytes(), original)

    def test_no_figures_keeps_pdf_bytes_unchanged(self):
        self.fixture(missing=True)
        original = self.path.read_bytes()
        report = builder.rasterize_mermaid_pdf(self.path, [])
        self.assertEqual(report['rasterized_diagrams'], 0)
        self.assertEqual(self.path.read_bytes(), original)


class BuildChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pdf = self.root / "existing.pdf"
        self.html = self.root / "existing.html"
        self.receipt = self.pdf.with_suffix(".build-receipt.json")
        self.pdf.write_bytes(b"OLD PDF MUST REMAIN")
        self.html.write_text("OLD HTML MUST REMAIN")
        self.receipt.write_bytes(b"OLD RECEIPT MUST REMAIN")
        def render(root, *, source_records):
            source_records.append({"path": "fixture.md", "role": "embedded_chapter", "bytes": 3,
                                   "sha256": hashlib.sha256(b"new").hexdigest()})
            return "<html>new document</html>"
        self.render = mock.patch.object(builder, "render_html", side_effect=render)
        self.render.start()
        self.addCleanup(self.render.stop)
        self.navigation = mock.patch.object(builder, "add_pdf_navigation", return_value={'bookmarks': 0})
        self.navigation.start()
        self.addCleanup(self.navigation.stop)

    def run_build(self, fake):
        with mock.patch.object(builder.subprocess, "run", side_effect=fake):
            return builder.build(self.root, self.pdf, self.html, "/fake/chrome", timeout=9)

    def assert_old(self):
        self.assertEqual(self.pdf.read_bytes(), b"OLD PDF MUST REMAIN")
        self.assertEqual(self.html.read_text(), "OLD HTML MUST REMAIN")
        self.assertEqual(self.receipt.read_bytes(), b"OLD RECEIPT MUST REMAIN")

    @staticmethod
    def fake_render(payload=None, code=0):
        def run(command, **kwargs):
            target = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--print-to-pdf=")))
            if payload is not None:
                target.write_bytes(payload)
            return subprocess.CompletedProcess(command, code, stdout="", stderr="fixture stderr")
        return run

    def test_nonzero_exit_never_reuses_existing_pdf(self):
        with self.assertRaisesRegex(builder.BuildError, "Chrome exited 7"):
            self.run_build(self.fake_render(b"new output even on failed exit", 7))
        self.assert_old()

    def test_zero_exit_without_new_pdf_is_failure(self):
        with self.assertRaisesRegex(builder.BuildError, "new nonempty PDF"):
            self.run_build(self.fake_render())
        self.assert_old()

    def test_empty_new_pdf_is_failure(self):
        with self.assertRaisesRegex(builder.BuildError, "new nonempty PDF"):
            self.run_build(self.fake_render(b""))
        self.assert_old()

    def test_non_pdf_output_is_failure(self):
        with self.assertRaisesRegex(builder.BuildError, "PDF header"):
            self.run_build(self.fake_render(b"not a pdf"))
        self.assert_old()

    def test_truncated_pdf_is_failure(self):
        with self.assertRaisesRegex(builder.BuildError, "EOF marker"):
            self.run_build(self.fake_render(b"%PDF-1.7\ntruncated"))
        self.assert_old()

    def test_pdf_markers_without_valid_document_are_failure(self):
        with self.assertRaises(builder.BuildError):
            self.run_build(self.fake_render(b"%PDF-1.7\ninvalid objects\n%%EOF\n"))
        self.assert_old()

    def test_chrome_timeout_preserves_previous_outputs(self):
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaisesRegex(builder.BuildError, "exceeded 9s"):
            self.run_build(timeout)
        self.assert_old()

    def test_chrome_launch_failure_preserves_previous_outputs(self):
        with self.assertRaises(OSError):
            self.run_build(mock.Mock(side_effect=FileNotFoundError("fixture executable missing")))
        self.assert_old()

    def test_valid_new_pdf_is_published_and_temp_files_are_removed(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF required for the real PDF-parser fixture")
        with fitz.open() as document:
            document.new_page().insert_text((40, 40), "fixture PDF")
            payload = document.tobytes()
        commands = []
        def render(command, **kwargs):
            commands.append(command)
            return self.fake_render(payload)(command, **kwargs)
        result = self.run_build(render)
        self.assertEqual(self.pdf.read_bytes(), payload)
        self.assertEqual(self.html.read_text(), "<html>new document</html>")
        self.assertEqual(result["pages"], 1)
        target = next(x.split("=", 1)[1] for x in commands[0] if x.startswith("--print-to-pdf="))
        self.assertNotEqual(Path(target), self.pdf)
        self.assertFalse(Path(target).parent.exists())
        self.assertTrue(any(x.startswith("--user-data-dir=") for x in commands[0]))
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["pdf"]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt["pdf"]["pages"], 1)
        self.assertEqual(receipt["embedded_sources"], 1)
        self.assertEqual(receipt["sources"][0]["sha256"], hashlib.sha256(b"new").hexdigest())
        self.assertEqual(receipt["render"]["returncode"], 0)
        self.assertTrue(receipt["render"]["isolated_temporary_profile"])
        self.assertFalse(receipt["render"]["diagrams_and_math_visually_verified"])
        self.assertEqual(receipt['pdf']['path'], 'existing.pdf')
        self.assertTrue(receipt["artifact_sha_match_required"])
        self.assertEqual(result["receipt_sha256"], hashlib.sha256(self.receipt.read_bytes()).hexdigest())

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF required')
    def test_diagram_export_failure_preserves_all_previous_outputs(self):
        import fitz
        with fitz.open() as document:
            document.new_page().insert_text((40, 40), 'No diagram marker in this Chrome fixture')
            payload = document.tobytes()
        html = '<html><div class="mermaid">graph TD; A-->B</div></html>'
        with mock.patch.object(builder, 'render_html', return_value=html):
            with self.assertRaisesRegex(builder.BuildError, 'Missing diagram markers'):
                self.run_build(self.fake_render(payload))
        self.assert_old()

    def test_receipt_publication_failure_preserves_previous_pdf(self):
        real_replace = builder.os.replace
        def replace(source, target):
            if target == self.receipt:
                raise OSError("fixture receipt publication failed")
            return real_replace(source, target)
        with mock.patch.object(builder, "validate_pdf", return_value=1), \
                mock.patch.object(builder, "portable_pdf_links", return_value={}), \
                mock.patch.object(builder.os, "replace", side_effect=replace):
            with self.assertRaisesRegex(OSError, "receipt publication failed"):
                self.run_build(self.fake_render(b"validated fixture bytes"))
        self.assertEqual(self.pdf.read_bytes(), b"OLD PDF MUST REMAIN")
        self.assertEqual(self.receipt.read_bytes(), b"OLD RECEIPT MUST REMAIN")

    def test_pdf_commit_failure_cannot_make_old_pdf_match_new_receipt(self):
        real_replace = builder.os.replace
        def replace(source, target):
            if target == self.pdf:
                raise OSError("fixture PDF publication failed")
            return real_replace(source, target)
        with mock.patch.object(builder, "validate_pdf", return_value=1), \
                mock.patch.object(builder, "portable_pdf_links", return_value={}), \
                mock.patch.object(builder.os, "replace", side_effect=replace):
            with self.assertRaisesRegex(OSError, "PDF publication failed"):
                self.run_build(self.fake_render(b"validated fixture bytes"))
        receipt = json.loads(self.receipt.read_text())
        self.assertTrue(receipt["artifact_sha_match_required"])
        self.assertNotEqual(receipt["pdf"]["sha256"], hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        self.assertEqual(self.pdf.read_bytes(), b"OLD PDF MUST REMAIN")

    def test_output_paths_must_be_distinct(self):
        with self.assertRaisesRegex(builder.BuildError, "distinct regular output paths"):
            builder.build(self.root, self.pdf, self.html, "/fake/chrome", receipt_path=self.pdf)
        self.assert_old()

    def test_cli_failure_returns_nonzero(self):
        with mock.patch.object(builder, "find_chrome", return_value="/fake/chrome"), \
                mock.patch.object(builder, "build", side_effect=builder.BuildError("fixture failure")), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(builder.main([]), 1)
        self.assertIn("fixture failure", stderr.getvalue())
        self.assert_old()


if __name__ == "__main__":
    unittest.main()
