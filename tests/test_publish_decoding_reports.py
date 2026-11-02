import base64
import pytest
from experiments.publish_decoding_reports import standalone


def test_publication_embeds_linked_summary_assets_without_copying_other_files(tmp_path):
    (tmp_path / 'index.html').write_text('<h1>Results</h1><img src="plot.png"><a href="summary.csv">Download</a>')
    (tmp_path / 'plot.png').write_bytes(b'figure-bytes')
    (tmp_path / 'summary.csv').write_text('class,recall\nSine,0.2\n')
    (tmp_path / 'epochs.npy').write_bytes(b'PRIVATE_RAW_SENTINEL')
    page, manifest = standalone(tmp_path, 'E015')
    assert base64.b64encode(b'figure-bytes').decode() in page
    assert 'download="summary.csv"' in page
    assert 'PRIVATE_RAW_SENTINEL' not in page and 'epochs.npy' not in manifest
    assert set(manifest) == {'plot.png', 'summary.csv'}


@pytest.mark.parametrize('asset', ['epochs.npy', '../outside.json'])
def test_publication_rejects_raw_array_or_out_of_report_asset(tmp_path, asset):
    (tmp_path / 'index.html').write_text(f'<h1>Results</h1><a href="{asset}">Asset</a>')
    with pytest.raises(ValueError, match='Unexpected report asset'):
        standalone(tmp_path, 'E015')
