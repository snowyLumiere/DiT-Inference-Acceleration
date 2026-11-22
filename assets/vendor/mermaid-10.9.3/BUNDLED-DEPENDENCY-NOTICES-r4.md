# Mermaid 10.9.3 bundled dependency attribution

The original Mermaid browser bundle is unchanged. Its own LICENSE is MIT. Embedded components retain their own terms; that project LICENSE does not replace them.

For copied code identified in the bundle's own licence comments, see the [r6 embedded notices supplement](BUNDLED-EMBEDDED-NOTICES-r6.md). It adds the complete js-yaml 4.1.0 notice and the MIT material for the Thenable, jQuery-event, Bezier and Framer snippets; those items are outside this package-lock dependency-closure table.

This conservative notice collection follows the declared production dependency closure of the official v10.9.3 pnpm-lock.yaml. It includes optional/type/CLI or unused dependencies and is not a claim that every listed component executes or was bundled. Full notices are copied without alteration from fixed official npm packages; where the published package omitted a licence text, the same publisher's original text at the package's pinned gitHead is retained and identified when available. The dagre-d3-es publisher gap described below is an explicit exception. No package source tree, npm archive or node_modules is included.

Official source lock: https://raw.githubusercontent.com/mermaid-js/mermaid/v10.9.3/pnpm-lock.yaml
Lock SHA256: a3d9e621ed8ad4e89dea942fc163fd72a17ecfa1030590cb67b740e784413163

Observed in the bundle: DOMPurify 3.1.6 (Apache-2.0 OR MPL-2.0), ELK layout implementation (locked elkjs 0.9.2, EPL-2.0), plus MIT/ISC/BSD components. The lock records Mermaid's embedded KaTeX0.16.11 separately from the top-level KaTeX0.16.8 resources used for the manual's formula rendering.

ELK corresponding publisher source at its package gitHead: https://github.com/kieler/elkjs/tree/72b471e65f5c5d7b0e94450b14c14bdf12384f82
ELK Java source project: https://github.com/eclipse-elk/elk . The publisher describes the generated JavaScript build in that source repository. Refer to the included EPL-2.0 and its source-availability provisions. This notice collection does not make a legal conclusion or grant a new licence.

Mermaid upstream patched cytoscape3.28.1. Source patch: https://github.com/mermaid-js/mermaid/blob/v10.9.3/patches/cytoscape%403.28.1.patch .

| Component | Declared licence | Original source | Retained notice files |
|---|---|---|---|
| @braintree/sanitize-url@6.0.4 | MIT | git+https://github.com/braintree/sanitize-url.git | `licenses/braintree--sanitize-url-6.0.4/LICENSE` |
| @types/d3-scale@4.0.8 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--d3-scale-4.0.8/LICENSE` |
| @types/d3-scale-chromatic@3.0.3 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--d3-scale-chromatic-3.0.3/LICENSE` |
| @types/d3-time@3.0.3 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--d3-time-3.0.3/LICENSE` |
| @types/debug@4.1.12 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--debug-4.1.12/LICENSE` |
| @types/mdast@3.0.15 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--mdast-3.0.15/LICENSE` |
| @types/ms@0.7.34 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--ms-0.7.34/LICENSE` |
| @types/unist@2.0.10 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped.git | `licenses/types--unist-2.0.10/LICENSE` |
| character-entities@2.0.2 | MIT | git+https://github.com/wooorm/character-entities.git | `licenses/character-entities-2.0.2/license` |
| commander@7.2.0 | MIT | git+https://github.com/tj/commander.js.git | `licenses/commander-7.2.0/LICENSE` |
| commander@8.3.0 | MIT | git+https://github.com/tj/commander.js.git | `licenses/commander-8.3.0/LICENSE` |
| cose-base@1.0.3 | MIT | git+https://github.com/iVis-at-Bilkent/cose-base.git | `licenses/cose-base-1.0.3/LICENSE` |
| cytoscape@3.28.1 | MIT | git+https://github.com/cytoscape/cytoscape.js.git | `licenses/cytoscape-3.28.1/LICENSE` |
| cytoscape-cose-bilkent@4.1.0 | MIT | git+https://github.com/cytoscape/cytoscape.js-cose-bilkent.git | `licenses/cytoscape-cose-bilkent-4.1.0/LICENSE` |
| d3@7.8.5 | ISC | git+https://github.com/d3/d3.git | `licenses/d3-7.8.5/LICENSE` |
| d3-array@2.12.1 | BSD-3-Clause | git+https://github.com/d3/d3-array.git | `licenses/d3-array-2.12.1/LICENSE` |
| d3-array@3.2.4 | ISC | git+https://github.com/d3/d3-array.git | `licenses/d3-array-3.2.4/LICENSE` |
| d3-axis@3.0.0 | ISC | git+https://github.com/d3/d3-axis.git | `licenses/d3-axis-3.0.0/LICENSE` |
| d3-brush@3.0.0 | ISC | git+https://github.com/d3/d3-brush.git | `licenses/d3-brush-3.0.0/LICENSE` |
| d3-chord@3.0.1 | ISC | git+https://github.com/d3/d3-chord.git | `licenses/d3-chord-3.0.1/LICENSE` |
| d3-color@3.1.0 | ISC | git+https://github.com/d3/d3-color.git | `licenses/d3-color-3.1.0/LICENSE` |
| d3-contour@4.0.2 | ISC | git+https://github.com/d3/d3-contour.git | `licenses/d3-contour-4.0.2/LICENSE` |
| d3-delaunay@6.0.4 | ISC | git+https://github.com/d3/d3-delaunay.git | `licenses/d3-delaunay-6.0.4/LICENSE` |
| d3-dispatch@3.0.1 | ISC | git+https://github.com/d3/d3-dispatch.git | `licenses/d3-dispatch-3.0.1/LICENSE` |
| d3-drag@3.0.0 | ISC | git+https://github.com/d3/d3-drag.git | `licenses/d3-drag-3.0.0/LICENSE` |
| d3-dsv@3.0.1 | ISC | git+https://github.com/d3/d3-dsv.git | `licenses/d3-dsv-3.0.1/LICENSE` |
| d3-ease@3.0.1 | BSD-3-Clause | git+https://github.com/d3/d3-ease.git | `licenses/d3-ease-3.0.1/LICENSE` |
| d3-fetch@3.0.1 | ISC | git+https://github.com/d3/d3-fetch.git | `licenses/d3-fetch-3.0.1/LICENSE` |
| d3-force@3.0.0 | ISC | git+https://github.com/d3/d3-force.git | `licenses/d3-force-3.0.0/LICENSE` |
| d3-format@3.1.0 | ISC | git+https://github.com/d3/d3-format.git | `licenses/d3-format-3.1.0/LICENSE` |
| d3-geo@3.1.0 | ISC | git+https://github.com/d3/d3-geo.git | `licenses/d3-geo-3.1.0/LICENSE` |
| d3-hierarchy@3.1.2 | ISC | git+https://github.com/d3/d3-hierarchy.git | `licenses/d3-hierarchy-3.1.2/LICENSE` |
| d3-interpolate@3.0.1 | ISC | git+https://github.com/d3/d3-interpolate.git | `licenses/d3-interpolate-3.0.1/LICENSE` |
| d3-path@1.0.9 | BSD-3-Clause | git+https://github.com/d3/d3-path.git | `licenses/d3-path-1.0.9/LICENSE` |
| d3-path@3.1.0 | ISC | git+https://github.com/d3/d3-path.git | `licenses/d3-path-3.1.0/LICENSE` |
| d3-polygon@3.0.1 | ISC | git+https://github.com/d3/d3-polygon.git | `licenses/d3-polygon-3.0.1/LICENSE` |
| d3-quadtree@3.0.1 | ISC | git+https://github.com/d3/d3-quadtree.git | `licenses/d3-quadtree-3.0.1/LICENSE` |
| d3-random@3.0.1 | ISC | git+https://github.com/d3/d3-random.git | `licenses/d3-random-3.0.1/LICENSE` |
| d3-sankey@0.12.3 | BSD-3-Clause | git+https://github.com/d3/d3-sankey.git | `licenses/d3-sankey-0.12.3/LICENSE` |
| d3-scale@4.0.2 | ISC | git+https://github.com/d3/d3-scale.git | `licenses/d3-scale-4.0.2/LICENSE` |
| d3-scale-chromatic@3.0.0 | ISC | git+https://github.com/d3/d3-scale-chromatic.git | `licenses/d3-scale-chromatic-3.0.0/LICENSE` |
| d3-selection@3.0.0 | ISC | git+https://github.com/d3/d3-selection.git | `licenses/d3-selection-3.0.0/LICENSE` |
| d3-shape@1.3.7 | BSD-3-Clause | git+https://github.com/d3/d3-shape.git | `licenses/d3-shape-1.3.7/LICENSE` |
| d3-shape@3.2.0 | ISC | git+https://github.com/d3/d3-shape.git | `licenses/d3-shape-3.2.0/LICENSE` |
| d3-time@3.1.0 | ISC | git+https://github.com/d3/d3-time.git | `licenses/d3-time-3.1.0/LICENSE` |
| d3-time-format@4.1.0 | ISC | git+https://github.com/d3/d3-time-format.git | `licenses/d3-time-format-4.1.0/LICENSE` |
| d3-timer@3.0.1 | ISC | git+https://github.com/d3/d3-timer.git | `licenses/d3-timer-3.0.1/LICENSE` |
| d3-transition@3.0.1 | ISC | git+https://github.com/d3/d3-transition.git | `licenses/d3-transition-3.0.1/LICENSE` |
| d3-zoom@3.0.0 | ISC | git+https://github.com/d3/d3-zoom.git | `licenses/d3-zoom-3.0.0/LICENSE` |
| dagre-d3-es@7.0.10 | MIT declared in publisher manifest; see disclosed gap | git+https://github.com/tbo47/dagre-es.git | `licenses/dagre-d3-es-7.0.10/UPSTREAM-dagre-d3-0.6.4-LICENSE.txt` (original upstream project) |
| dayjs@1.11.10 | MIT | git+https://github.com/iamkun/dayjs.git | `licenses/dayjs-1.11.10/LICENSE` |
| debug@4.3.4 | MIT | git://github.com/debug-js/debug.git | `licenses/debug-4.3.4/LICENSE` |
| decode-named-character-reference@1.0.2 | MIT | git+https://github.com/wooorm/decode-named-character-reference.git | `licenses/decode-named-character-reference-1.0.2/license` |
| delaunator@5.0.1 | ISC | git+https://github.com/mapbox/delaunator.git | `licenses/delaunator-5.0.1/LICENSE` |
| dequal@2.0.3 | MIT | git+https://github.com/lukeed/dequal.git | `licenses/dequal-2.0.3/license` |
| diff@5.1.0 | BSD-3-Clause | git://github.com/kpdecker/jsdiff.git | `licenses/diff-5.1.0/LICENSE` |
| dompurify@3.1.6 | (MPL-2.0 OR Apache-2.0) | git://github.com/cure53/DOMPurify.git | `licenses/dompurify-3.1.6/LICENSE` |
| elkjs@0.9.2 | EPL-2.0 | git+https://github.com/kieler/elkjs.git | `licenses/elkjs-0.9.2/LICENSE.md` |
| has-flag@4.0.0 | MIT | git+https://github.com/sindresorhus/has-flag.git | `licenses/has-flag-4.0.0/license` |
| heap@0.2.7 | MIT | git://github.com/qiao/heap.js.git | `licenses/heap-0.2.7/LICENSE` |
| iconv-lite@0.6.3 | MIT | git://github.com/ashtuchkin/iconv-lite.git | `licenses/iconv-lite-0.6.3/LICENSE` |
| internmap@1.0.1 | ISC | git+https://github.com/mbostock/internmap.git | `licenses/internmap-1.0.1/LICENSE` |
| internmap@2.0.3 | ISC | git+https://github.com/mbostock/internmap.git | `licenses/internmap-2.0.3/LICENSE` |
| katex@0.16.11 | MIT | git+https://github.com/KaTeX/KaTeX.git | `licenses/katex-0.16.11/LICENSE` |
| khroma@2.1.0 | null | git+https://github.com/fabiospampinato/khroma.git | `licenses/khroma-2.1.0/license` |
| kleur@4.1.5 | MIT | git+https://github.com/lukeed/kleur.git | `licenses/kleur-4.1.5/license` |
| layout-base@1.0.2 | MIT | git+https://github.com/iVis-at-Bilkent/layout-base.git | `licenses/layout-base-1.0.2/LICENSE` |
| lodash@4.17.21 | MIT | git+https://github.com/lodash/lodash.git | `licenses/lodash-4.17.21/LICENSE` |
| lodash-es@4.17.21 | MIT | git+https://github.com/lodash/lodash.git | `licenses/lodash-es-4.17.21/LICENSE` |
| mdast-util-from-markdown@1.3.1 | MIT | git+https://github.com/syntax-tree/mdast-util-from-markdown.git | `licenses/mdast-util-from-markdown-1.3.1/license` |
| mdast-util-to-string@3.2.0 | MIT | git+https://github.com/syntax-tree/mdast-util-to-string.git | `licenses/mdast-util-to-string-3.2.0/license` |
| micromark@3.2.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-3.2.0/LICENSE-from-upstream.txt` |
| micromark-core-commonmark@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-core-commonmark-1.1.0/LICENSE-from-upstream.txt` |
| micromark-factory-destination@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-factory-destination-1.1.0/LICENSE-from-upstream.txt` |
| micromark-factory-label@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-factory-label-1.1.0/LICENSE-from-upstream.txt` |
| micromark-factory-space@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-factory-space-1.1.0/LICENSE-from-upstream.txt` |
| micromark-factory-title@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-factory-title-1.1.0/LICENSE-from-upstream.txt` |
| micromark-factory-whitespace@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-factory-whitespace-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-character@1.2.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-character-1.2.0/LICENSE-from-upstream.txt` |
| micromark-util-chunked@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-chunked-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-classify-character@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-classify-character-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-combine-extensions@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-combine-extensions-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-decode-numeric-character-reference@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-decode-numeric-character-reference-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-decode-string@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-decode-string-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-encode@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-encode-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-html-tag-name@1.2.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-html-tag-name-1.2.0/LICENSE-from-upstream.txt` |
| micromark-util-normalize-identifier@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-normalize-identifier-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-resolve-all@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-resolve-all-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-sanitize-uri@1.2.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-sanitize-uri-1.2.0/LICENSE-from-upstream.txt` |
| micromark-util-subtokenize@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-subtokenize-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-symbol@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-symbol-1.1.0/LICENSE-from-upstream.txt` |
| micromark-util-types@1.1.0 | MIT | git+https://github.com/micromark/micromark.git#main | `licenses/micromark-util-types-1.1.0/LICENSE-from-upstream.txt` |
| mri@1.2.0 | MIT | git+https://github.com/lukeed/mri.git | `licenses/mri-1.2.0/license.md` |
| ms@2.1.2 | MIT | git+https://github.com/zeit/ms.git | `licenses/ms-2.1.2/license.md` |
| non-layered-tidy-tree-layout@2.0.2 | MIT | https://github.com/stetrevor/non-layered-tidy-tree-layout.git | `licenses/non-layered-tidy-tree-layout-2.0.2/LICENSE` |
| robust-predicates@3.0.2 | Unlicense | git+https://github.com/mourner/robust-predicates.git | `licenses/robust-predicates-3.0.2/LICENSE` |
| rw@1.3.3 | BSD-3-Clause | git+ssh://git@github.com/mbostock/rw.git | `licenses/rw-1.3.3/LICENSE` |
| sade@1.8.1 | MIT | git+https://github.com/lukeed/sade.git | `licenses/sade-1.8.1/license` |
| safer-buffer@2.1.2 | MIT | git+https://github.com/ChALkeR/safer-buffer.git | `licenses/safer-buffer-2.1.2/LICENSE` |
| stylis@4.3.1 | MIT | git+https://github.com/thysultan/stylis.js.git | `licenses/stylis-4.3.1/LICENSE` |
| supports-color@8.1.1 | MIT | git+https://github.com/chalk/supports-color.git | `licenses/supports-color-8.1.1/license` |
| ts-dedent@2.2.0 | MIT | git+https://github.com/tamino-martinius/node-ts-dedent.git | `licenses/ts-dedent-2.2.0/LICENSE` |
| unist-util-stringify-position@3.0.3 | MIT | git+https://github.com/syntax-tree/unist-util-stringify-position.git | `licenses/unist-util-stringify-position-3.0.3/license` |
| uuid@9.0.1 | MIT | git+https://github.com/uuidjs/uuid.git | `licenses/uuid-9.0.1/LICENSE.md` |
| uvu@0.5.6 | MIT | git+https://github.com/lukeed/uvu.git | `licenses/uvu-0.5.6/license` |
| web-worker@1.3.0 | Apache-2.0 | git+https://github.com/developit/web-worker.git | `licenses/web-worker-1.3.0/LICENSE` |

## Disclosed publisher notice gap: dagre-d3-es7.0.10

The fixed publisher npm manifest declares MIT, while its published tarball and the recursive Git tree at gitHead f14d9a1a7dd13d2448b86e4aa8e496e1e585596a contain no named LICENSE/NOTICE file. Its fixed README explicitly identifies this as a fork of dagre-d3. The original upstream dagre-d3 v0.6.4 Chris Pettitt MIT text is retained above as an attributed upstream reference. That reference does not establish the exact fork base or confirm the copyright chain of any additional fork work. No missing author statement has been invented and no legal conclusion is made.
