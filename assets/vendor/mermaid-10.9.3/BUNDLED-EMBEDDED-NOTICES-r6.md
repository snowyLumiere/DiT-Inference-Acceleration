# Mermaid 10.9.3 embedded notices supplement (r6)

This supplements [the fixed dependency notice collection](BUNDLED-DEPENDENCY-NOTICES-r4.md). A package dependency closure does not enumerate code copied into a dependency's own source. The unchanged browser bundle contains all six notice blocks preserved in [BUNDLE-LICENSE-COMMENTS.txt](BUNDLE-LICENSE-COMMENTS.txt); this supplement accounts for each block and supplies the missing MIT material.

The browser bundle and its embedded comments are unchanged. The original copyright notices and licence designations are preserved, not replaced by the reader edition's copyright. No additional author's permission or legal clearance is asserted here.

## Coverage of the embedded notice blocks

| Embedded component | Preserved material and full terms |
|---|---|
| DOMPurify 3.1.6, Cure53 and other contributors | Its original Apache-2.0 OR MPL-2.0 notice remains in the bundle; the complete publisher [licence file](licenses/dompurify-3.1.6/LICENSE) was already included. |
| js-yaml 4.1.0 | The bundle identifies the component but omits its author. The complete, unchanged publisher [MIT licence](licenses/js-yaml-4.1.0/LICENSE) restores `Copyright (C) 2011-2015 by Vitaly Puzrin` and the permission/disclaimer text. |
| Thenable 1.0.7, Ralf S. Engelschall | [Original notice and full referenced MIT terms](licenses/cytoscape-embedded-mit/THENABLE-1.0.7-NOTICE-AND-MIT.txt). The notice is copied verbatim from the original 1.0.7 source. That source names MIT by URL rather than containing a separate full licence; the linked file identifies the separately reproduced MIT text. |
| Event object based on jQuery events | The bundle's original jQuery MIT/source links remain intact. The complete [upstream jQuery MIT notice](licenses/cytoscape-embedded-mit/UPSTREAM-jquery-3.7.1-LICENSE.txt) is additionally retained as a fixed upstream reference. |
| Bezier curve function generator, Gaetan Renaudeau | The bundle's copyright and MIT designation remain intact. The complete [original author's MIT licence](licenses/cytoscape-embedded-mit/UPSTREAM-bezier-easing-20140717-LICENSE.txt), including the publisher's spelling `Gaëtan`, is additionally retained without alteration. |
| Runge-Kutta spring generator adapted from Framer.js, Koen Bok | The bundle's copyright and MIT designation remain intact. The complete [original Framer MIT licence](licenses/cytoscape-embedded-mit/UPSTREAM-Framer-20140624-LICENSE.txt) is additionally retained without alteration. |

The four Cytoscape snippets are identified by their matching notices in the fixed `cytoscape.js v3.28.1` source files below. For the jQuery, Bezier and Framer snippets, the immediate source does not pin the exact revision originally imported from those projects. Their full upstream licence files below are retained as attributed upstream references, not a claim that the entire named reference version was bundled or that its exact import base was established. The explicit MIT designations in the unchanged bundled snippets remain the licence evidence for those snippets.

## Fixed source fingerprints

The SHA256 values below describe the original UTF-8 source/notice bytes retrieved from the publisher, before any extraction or transcription. Source implementations and archives used to verify provenance are not distributed in the reader package. Only the licence files and original notice material identified above are included.

| Publisher source | Bytes | SHA256 |
|---|---:|---|
| [js-yaml 4.1.0 LICENSE](https://raw.githubusercontent.com/nodeca/js-yaml/4.1.0/LICENSE) | 1084 | `a07bc24468b9654ce76a547d47a2db282d07733b715db4c73a98bd63961f9550` |
| [Thenable 1.0.7 thenable.js](https://raw.githubusercontent.com/rse/thenable/1.0.7/thenable.js) | 8614 | `15d5267aeaa15d3dfeac1ca6a97378189d693d75ee26130afa342b6fcdfb97ed` |
| [Cytoscape v3.28.1 src/promise.js](https://raw.githubusercontent.com/cytoscape/cytoscape.js/v3.28.1/src/promise.js) | 7591 | `f12e7bfaa4eeadff9545f48668593c9e8dc1e03403b980dc8e26720d0bbdf868` |
| [Cytoscape v3.28.1 src/event.js](https://raw.githubusercontent.com/cytoscape/cytoscape.js/v3.28.1/src/event.js) | 2921 | `5cb8280758bf91767096f0b711ae2eb3e42cbce72107196be30d6d1a28ded015` |
| [Cytoscape v3.28.1 src/core/animation/cubic-bezier.js](https://raw.githubusercontent.com/cytoscape/cytoscape.js/v3.28.1/src/core/animation/cubic-bezier.js) | 3843 | `ac0c873e3efc9209efdfca1276bce21ad9d42bff4d81fa0834c698c60b2ed939` |
| [Cytoscape v3.28.1 src/core/animation/spring.js](https://raw.githubusercontent.com/cytoscape/cytoscape.js/v3.28.1/src/core/animation/spring.js) | 3190 | `a44025d0fbdc257abd02beb46a8e3dd427dc6ba3a81804a4966f115bfb7fe103` |
| [jQuery 3.7.1 LICENSE.txt](https://raw.githubusercontent.com/jquery/jquery/3.7.1/LICENSE.txt) | 1097 | `d4db9ebe6f29f5168eac45ad713f055623ac5d0dcd5ba92da23d650ae012020d` |
| [bezier-easing LICENSE, commit abe28d9ccebafd6cd68e0d379ca8f86da6f38ec9](https://raw.githubusercontent.com/gre/bezier-easing/abe28d9ccebafd6cd68e0d379ca8f86da6f38ec9/LICENSE) | 1061 | `32bfd1097d45b41ca6792bf07b389b26ee4045e7f7db28349f4199eef889c516` |
| [Framer LICENSE, commit a41d843dc1371dd0cdc87d1fbbb3f833a5d24d58](https://raw.githubusercontent.com/koenbok/Framer/a41d843dc1371dd0cdc87d1fbbb3f833a5d24d58/LICENSE) | 1075 | `298012b8b6c965d489221b5b79be5f77ca2027ec65d47a80b16b73812cb60a46` |

The complete MIT permission/disclaimer text accompanying the Thenable notice is reproduced from the [Open Source Initiative MIT text](https://opensource.org/license/mit), as referenced by the original author's `http://opensource.org/licenses/MIT` notice. The original copyright line is retained; no template copyright holder or date has been filled in.

The previously disclosed `dagre-d3-es@7.0.10` publisher notice gap remains documented in the [dependency notices](BUNDLED-DEPENDENCY-NOTICES-r4.md#disclosed-publisher-notice-gap-dagre-d3-es7010); this supplement does not invent a missing fork-author statement.
