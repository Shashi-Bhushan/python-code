def search_for_vowels(word: str) -> str:
    """Return any vowel found in supplied word"""
    vowels = set('aeiou')
    return vowels.intersection(set(word))


def search_for_letters(phrase: str, letters: str) -> set:
    """Returns a set of 'letters' found in 'phrase'"""
    return set(letters).intersection(set(phrase))
