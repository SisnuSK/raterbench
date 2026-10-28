from raterbench.perturb.edits import gender_swap, typo_noise


def test_gender_swap_basic_and_case() -> None:
    out, n = gender_swap("My husband said he loves His job. She told Him.")
    assert out == "My wife said she loves Her job. He told Her."
    assert n == 5  # husband, he, His, She, Him


def test_gender_swap_her_object_vs_possessive() -> None:
    assert gender_swap("I told her that")[0] == "I told him that"
    assert gender_swap("I met her.")[0] == "I met him."
    assert gender_swap("her mom is nice")[0] == "his dad is nice"


def test_gender_swap_leaves_other_words_alone() -> None:
    text = "The theme of the shell is hence there."
    assert gender_swap(text) == (text, 0)


def test_typo_noise_is_seeded_and_spares_hashtags_mentions_urls() -> None:
    text = "Climate action matters #ClimateChange @someone https://example.com/page today"
    a, na = typo_noise(text, seed="x", rate=0.5)
    b, nb = typo_noise(text, seed="x", rate=0.5)
    assert (a, na) == (b, nb)
    assert na >= 1 and a != text
    for keep in ("#ClimateChange", "@someone", "https://example.com/page"):
        assert keep in a


def test_typo_noise_never_touches_urls() -> None:
    text = "see https://example.com/longpath and www.example.org/abcd for details please"
    for s in range(30):
        out, _ = typo_noise(text, seed=str(s), rate=1.0)
        assert "https://example.com/longpath" in out and "www.example.org/abcd" in out


def test_typo_noise_no_eligible_words() -> None:
    assert typo_noise("a b c #tag", seed="s") == ("a b c #tag", 0)
