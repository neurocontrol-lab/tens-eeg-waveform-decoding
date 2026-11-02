"""Offline JavaScript Markdown rendering for standalone scientific reports."""
import hashlib
import html
import json
from pathlib import Path

ASSETS = Path(__file__).with_name('report_assets') / 'markdown_it'


def markdown_details(title, markdown):
    return ('<details><summary>' + html.escape(title) + '</summary>'
            '<div class="report-markdown"><pre class="markdown-source">' +
            html.escape(markdown) + '</pre></div></details>')


def markdown_support():
    metadata = json.loads((ASSETS / 'provenance.json').read_text())
    for name, digest in metadata['sha256'].items():
        if hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Vendored Markdown renderer changed: {name}')
    license_text = (ASSETS / 'LICENSE').read_text(encoding='utf-8')
    bundle = (ASSETS / 'markdown-it.umd.min.js').read_text(encoding='utf-8')
    bundle = bundle.replace('</script', '<\\/script')
    return '''<style>
.report-markdown{font:15px/1.65 system-ui,sans-serif;overflow-wrap:anywhere;overflow-x:auto;padding:8px 0}
.report-markdown h1{font-size:1.7em}.report-markdown h2{font-size:1.35em;margin-top:1.8em}
.report-markdown h3{font-size:1.15em;margin-top:1.5em}.report-markdown p{margin:1em 0}
.report-markdown ul,.report-markdown ol{padding-left:1.8em}.report-markdown li{margin:.4em 0}
.report-markdown table{font-size:14px;border-collapse:collapse;margin:1em 0}
.report-markdown th{background:#edf3f8}.report-markdown td,.report-markdown th{border:1px solid #ccd7e2;padding:8px}
.report-markdown code{font:0.9em ui-monospace,Consolas,monospace;background:#eef2f6;padding:2px 4px;border-radius:3px}
.report-markdown pre{font:14px/1.5 ui-monospace,Consolas,monospace;background:#eef2f6;padding:14px;white-space:pre;overflow:auto}
.report-markdown pre code{background:none;padding:0}.report-markdown blockquote{border-left:4px solid #879db1;margin-left:0;padding-left:16px;color:#425466}
</style><script data-vendor="markdown-it" data-version="''' + metadata['version'] + '''">
/* markdown-it — MIT license
''' + license_text + '*/\n' + bundle + '''
</script><script>
function renderReportMarkdown(){
if(typeof markdownit!=='function')return;
const md=markdownit({html:false,linkify:false,typographer:false});
const defaultLink=md.renderer.rules.link_open||function(tokens,index,options,env,self){return self.renderToken(tokens,index,options)};
md.renderer.rules.link_open=function(tokens,index,options,env,self){
const token=tokens[index],href=token.attrGet('href');
if(href&&!/^(?:[a-z][a-z0-9+.-]*:|#)/i.test(href)){
token.attrSet('href',new URL(href,'https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/').href);
}
return defaultLink(tokens,index,options,env,self);
};
for(const container of document.querySelectorAll('.report-markdown')){
const source=container.querySelector('.markdown-source');
if(source){container.innerHTML=md.render(source.textContent);container.dataset.rendered='true';}
}
}
renderReportMarkdown();
</script>'''
