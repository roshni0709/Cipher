from pricematch.collector.extract import extract_product

JSONLD_HTML = """
<html><head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "Product", "name": "Test Kettle 1.5L",
 "image": ["https://img.example/k.jpg"],
 "offers": {"@type": "Offer", "price": "649.00", "priceCurrency": "INR"}}
</script></head><body></body></html>
"""

OG_HTML = """
<html><head>
<meta property="og:title" content="OG Kettle"/>
<meta property="og:price:amount" content="599"/>
</head><body></body></html>
"""


def test_jsonld():
    p = extract_product(JSONLD_HTML, "https://example.in/p/1")
    assert p["title"] == "Test Kettle 1.5L" and p["price"] == 649.0
    assert p["image_url"] == "https://img.example/k.jpg"


def test_opengraph_fallback():
    p = extract_product(OG_HTML, "https://example.in/p/2")
    assert p["price"] == 599.0


def test_no_data():
    assert extract_product("<html></html>", "https://example.in/p/3") is None


def test_jsonld_trailing_junk():
    # Seen in the wild (Croma product page, Nov 2026): a valid Product block
    # followed by an extra closing brace. Parse the first complete value.
    html = JSONLD_HTML.replace("</script>", "}\n</script>")
    p = extract_product(html, "https://example.in/p/4")
    assert p["title"] == "Test Kettle 1.5L" and p["price"] == 649.0


def test_jsonld_unparseable_still_skipped():
    html = '<script type="application/ld+json">{not json at all</script>'
    assert extract_product(html, "https://example.in/p/5") is None
