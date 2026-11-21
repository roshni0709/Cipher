"""Light-only widget regression coverage without importing the running app."""
import ast
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).parents[1] / 'pricematch' / 'demo' / 'app.py'


def light_css(enabled=True):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    palette = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_LIGHT' for t in n.targets))
    block = next(n for n in tree.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'st.session_state.light')
    rendered = []
    st = SimpleNamespace(session_state=SimpleNamespace(light=enabled),
                         markdown=lambda value, **kwargs: rendered.append(value))
    exec(compile(ast.Module(body=[palette, block], type_ignores=[]), str(SOURCE), 'exec'), {'st': st})
    return '\n'.join(rendered)


def test_widget_repairs_are_not_emitted_in_dark_mode():
    assert light_css(False) == ''


def test_light_link_repairs_cover_nested_labels_and_interaction_states():
    css = light_css()
    for selector in ('[data-testid="stLinkButton"] a *', 'a[data-testid^="stBaseLinkButton"] *',
                     'a:visited', 'a:hover', 'a:focus-visible'):
        assert selector in css
    assert '-webkit-text-fill-color: #18181b !important' in css


def test_light_alert_repairs_cover_inner_surface_not_only_wrapper():
    css = light_css()
    assert '[data-testid="stAlertContainer"] {' in css
    assert '[data-testid="stAlertContainer"] *' in css
    assert 'background: #ffffff !important' in css


def test_light_picker_repairs_cover_input_placeholder_and_arrow():
    css = light_css()
    for selector in ('[data-testid="stMultiSelect"] input::placeholder',
                     '[data-testid="stMultiSelect"] [role="group"]',
                     '[data-testid="stMultiSelect"] button svg',
                     '[data-testid="stSelectbox"] input::placeholder'):
        assert selector in css
    assert '-webkit-text-fill-color: #52525b !important' in css
