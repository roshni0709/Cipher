import pytest
from pricematch.rag.pipeline import RagPipeline
from pricematch.pipeline import seed


@pytest.fixture(scope='module')
def pipe(tmp_path_factory):
    path = str(tmp_path_factory.mktemp('reviewquery') / 'reviews.db')
    seed(path)
    return RagPipeline(path, retriever_backend='tfidf', generator_backend='template')


@pytest.mark.parametrize('name,cid,signal', [
    ('Noise ColorFit Pro 5', 'noise-colorfit-pro5', 'No manipulation signals detected'),
    ('Fire-Boltt Phoenix Pro', 'fire-boltt-phoenix-pro', 'within a 7-day window'),
    ('Boult Z40 Pro', 'boult-z40-pro', 'within a 7-day window'),
    ('Portronics Toad 23', 'portronics-toad-23', 'near-identical text'),
    ('H&M Relaxed Tee', 'hm-relaxed-tee', 'no critical balance'),
])
def test_review_question_answers_for_exact_named_product(pipe, name, cid, signal):
    result = pipe.answer(f'Are the reviews for {name} genuine?')
    docs = [doc for doc, _ in result['retrieved']]
    assert {d.meta.get('cluster_id') for d in docs} == {cid}
    reviews = [d for d in docs if d.meta.get('tool') == 'review_summary']
    assert len(reviews) == 1
    assert signal in reviews[0].text
    assert 'Heuristic signals, not proof' in reviews[0].text
    assert 'Demo review data, not live store reviews' in reviews[0].text
    assert reviews[0].text in result['answer']
    assert result['generator'] == 'template'


def test_review_question_bypasses_freeform_generator(pipe):
    pipe.generator_backend = 'auto'
    pipe._get_llm = lambda: pytest.fail('Review verdicts must not need a model download')
    result = pipe.answer('Are the reviews for Boult Z40 Pro genuine?')
    assert result['generator'] == 'template'
    assert 'Suspicious review patterns' in result['answer']
