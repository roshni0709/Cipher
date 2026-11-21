"""Small pure-helper regressions for the demo's product navigation."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace

SOURCE = Path(__file__).parents[1] / 'pricematch' / 'demo' / 'app.py'


def helpers(meta, state=None):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'_name_slugs', '_share_url', '_pin_product', '_seed', '_split_agent_answer'}
    funcs = ast.Module(body=[n for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name in names],
                       type_ignores=[])
    namespace = {'re': re, '_CLUSTER_META': meta,
                 'st': SimpleNamespace(session_state=state if state is not None else {})}
    exec(compile(funcs, str(SOURCE), 'exec'), namespace)
    return namespace


def test_display_names_preserve_citations_and_avoid_prefix_collisions():
    ns = helpers({'boat-141': ('boAt 141', 'audio'),
                  'boat-141-pro': ('boAt 141 Pro', 'audio')})
    assert ns['_name_slugs']('boat-141-pro [1], boat-141 [2]') == 'boAt 141 Pro [1], boAt 141 [2]'


def test_display_name_does_not_replace_unknown_longer_slug():
    ns = helpers({'boat-141': ('boAt 141', 'audio')})
    assert ns['_name_slugs']('boat-141-other [1]') == 'boat-141-other [1]'


def test_share_link_preserves_theme_without_fixed_host_or_port():
    ns = helpers({}, {'light': True})
    assert ns['_share_url']('maroon-kurti') == '?pin=maroon-kurti&light=1'
    ns['st'].session_state['light'] = False
    assert ns['_share_url']('maroon-kurti') == '?pin=maroon-kurti'


def test_product_pin_uses_exact_id_and_catalog_name():
    ns = helpers({'noise-colorfit-pro5': ('Noise ColorFit Pro 5 Smartwatch', 'electronics')})
    ns['_pin_product']('noise-colorfit-pro5')
    assert ns['st'].session_state == {'pin': 'noise-colorfit-pro5', 'q': 'Noise ColorFit Pro 5 Smartwatch'}


def test_fresh_search_clears_pin():
    ns = helpers({}, {'pin': 'old'})
    ns['_seed']('q', 'new search')
    assert ns['st'].session_state == {'q': 'new search'}


def test_review_summary_is_visible_not_buried_in_receipts():
    ns = helpers({})
    result = ns['_split_agent_answer']('* [tool: review_summary] watch: review trust low. Suspicious review patterns. [2]')
    assert result['main'] == ['Review check: watch: review trust low. Suspicious review patterns. [2]']
    assert result['receipts'] == []
