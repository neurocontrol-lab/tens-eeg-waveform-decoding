# Embedded Markdown renderer

`markdown-it` 15.0.3, MIT licensed. Source: https://github.com/markdown-it/markdown-it.
The browser UMD bundle and LICENSE are copied unchanged from the official npm
package. `provenance.json` records its package URL, SHA-512 archive integrity and
SHA-256 file digests. The report builder verifies the vendored files before use.

`experiments.report_markdown` embeds the bundle and complete license into generated
HTML. There is no runtime CDN request. Rendering disables raw HTML and retains
markdown-it's default link validation. Relative documentation links resolve to the
repository's Docs directory. If JavaScript is disabled, the escaped Markdown source
remains readable. This changes report formatting, not scientific results.
