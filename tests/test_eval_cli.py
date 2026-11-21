from pricematch.eval.run import main


def test_eval_sample_runs(capsys):
    main(["--dataset", "sample", "--matcher", "tfidf"])
    out = capsys.readouterr().out
    assert "[tfidf]" in out and "F1=" in out
