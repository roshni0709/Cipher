from pricematch.data import sample
from pricematch.eval.run import evaluate_one
from sklearn.model_selection import train_test_split


def test_pairs_have_both_labels():
    df = sample.pairs()
    assert set(df["label"].unique()) == {0, 1}
    assert (df["label"] == 1).sum() >= 6  # one+ positive per multi-offer cluster


def test_tfidf_beats_chance_on_sample():
    df = sample.pairs()
    train, test = train_test_split(df, test_size=0.4, random_state=1, stratify=df["label"])
    report = evaluate_one("tfidf", train, test)
    assert report["f1"] > 0.5  # electronics model numbers make this easy
