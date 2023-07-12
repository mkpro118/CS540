from functools import wraps
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
)
import math
import random
import string

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import probability


class Indexifiers:
    ArgTypes = [str, bool]
    RetType = tuple[int] | tuple[tuple[int], dict[str, int]]
    FuncType = Callable[[str, bool], RetType]

    _lowercase = string.ascii_lowercase
    _uppercase = string.ascii_uppercase
    _digits = string.digits
    _whitespace = string.whitespace

    class Letters:
        @staticmethod
        def lowercase(
            text: str,
            return_mapping: bool = False
        ) -> tuple[int] | tuple[tuple[int], dict[str, int]]:
            mapping = {
                token: index for index, token in enumerate(
                    Indexifiers._lowercase,
                    start=1
                )
            }

            mapping.update({c: 0 for c in Indexifiers._whitespace})

            res = tuple(map(mapping.get, text))

            if None in res:
                raise ValueError('text contains non alpha-numeric characters')

            if return_mapping:
                return res, mapping  # type: ignore
            return res  # type: ignore

        @staticmethod
        def uppercase(
            text: str,
            return_mapping: bool = False
        ) -> tuple[int] | tuple[tuple[int], dict[str, int]]:
            mapping = {
                token: index for index, token in enumerate(
                    Indexifiers._uppercase,
                    start=1
                )
            }

            mapping.update({c: 0 for c in Indexifiers._whitespace})

            res = tuple(map(mapping.get, text))

            if None in res:
                raise ValueError('text contains non uppercase characters')

            if return_mapping:
                return res, mapping  # type: ignore
            return res  # type: ignore

        @staticmethod
        def alphabetic(
            text: str,
            return_mapping: bool = False
        ) -> tuple[int] | tuple[tuple[int], dict[str, int]]:
            mapping = {
                token: index for index, token in enumerate(
                    Indexifiers._uppercase + Indexifiers._lowercase,
                    start=1
                )
            }

            mapping.update({c: 0 for c in Indexifiers._whitespace})

            res = tuple(map(mapping.get, text))

            if None in res:
                raise ValueError('text contains non alpha-numeric characters')

            if return_mapping:
                return res, mapping  # type: ignore
            return res  # type: ignore

        @staticmethod
        def alpha_numeric(
            text: str,
            return_mapping: bool = False
        ) -> tuple[int] | tuple[tuple[int], dict[str, int]]:
            mapping = {
                token: index for index, token in enumerate(
                    Indexifiers._uppercase + Indexifiers._lowercase + Indexifiers._digits,
                    start=1
                )
            }

            mapping.update({c: 0 for c in Indexifiers._whitespace})

            res = tuple(map(mapping.get, text))

            if None in res:
                raise ValueError('text contains non alpha-numeric characters')

            if return_mapping:
                return res, mapping  # type: ignore
            return res  # type: ignore

    class Words:
        @staticmethod
        def case_sensitive(
            text: str,
            return_mapping: bool = False
        ) -> tuple[int] | tuple[tuple[int], dict[str, int]]:
            text = text.split()  # type: ignore
            mapping = {token: index for index,
                       token in enumerate(set(text), start=1)}

            text = tuple(map(mapping.get, text))  # type: ignore

            if return_mapping:
                return text, mapping  # type: ignore
            return text  # type: ignore

        @staticmethod
        def case_insensitive(
            text: str,
            return_mapping: bool = False
        ) -> tuple[int] | tuple[tuple[int], dict[str, int]]:
            text = text.lower().split()  # type: ignore
            mapping = {token: index for index,
                       token in enumerate(set(text), start=1)}

            text = tuple(map(mapping.get, text))  # type: ignore

            if return_mapping:
                return text, mapping  # type: ignore
            return text  # type: ignore


class Ngram:
    def __init__(self, token_indexifier: Optional[Indexifiers.FuncType] = None, **kwargs):
        self._token_indexifier = token_indexifier
        self._smoothing: bool = kwargs.get('smoothing', False)
        self._n: int = kwargs.get('n', 0)
        self._tpt: tuple = tuple()

    def _guess_token_indexifier(self, text: str):
        if any([x in text for x in string.whitespace]):
            self._token_indexifier = Indexifiers.Words.case_insensitive
        else:
            self._token_indexifier = Indexifiers.Letters.alpha_numeric

    def _build(self):
        raise NotImplementedError(
            'Arbitrary N-gram functionality is not yet implemented'
        )

    def fit(self, text: str, smoothing: bool = False) -> Self:
        if not self._token_indexifier:
            self._guess_token_indexifier(text)

        self._smoothing = smoothing

        assert self._token_indexifier
        indexified = self._token_indexifier(text, True)

        assert len(indexified) == 2, (
            'Expected the token_indexifier function to return 2 values, got '
            f'{len(indexified)}. If you are using a custom token_indexifier, '
            'it should take two positional parameters of types [str, bool] '
            'and, return type tuple[tuple[int], dict[str, int]]. '
            'The first parameter is the string to convert '
            '(the parameter given to fit), and the second is be an indicator '
            'to return the mapping used to indexify the string.'
        )

        self._idxs, self._mapping = indexified  # type: ignore[misc]
        self._inverse_mapping = {y: x for x, y in self._mapping.items()}
        self._inverse_mapping.update({0: ' '})

        self._build()

        return self

    def predict(self, token: str | Sequence[str]) -> tuple[str]:
        assert len(token) == self._n - 1, (
            f'length of `token` must be {self._n - 1}'
        )
        if isinstance(token, str):
            return self._predict(token),  # Evil comma hack, converts to tuple

        return tuple(map(self._predict, token))  # type: ignore

    def _predict(self, token: str) -> str:
        raise NotImplementedError(
            'Arbitrary N-gram prediction is not yet implemented')

    def _sequence_generator(self, length: int = 100, start: Optional[str] = None):
        min_letters = self._n - 1
        if start is None:
            start = ''

        # Use lower bound as 1 to avoid using whitespace as a
        # starting point
        while len(start) < min_letters:
            rand_idx = random.randrange(1, len(self._inverse_mapping))
            start += self._inverse_mapping[rand_idx]

        yield start
        for i in range(length - len(start)):
            next_ = self._predict(start)
            yield next_
            start = start + next_
            start = start[-min_letters:]

    def generate(self, length: int = 100, start: Optional[str] = None,
                 as_iter: bool = False) -> str:
        generator = self._sequence_generator(length, start)

        if as_iter:
            return generator

        return ''.join(generator)

    @property
    def transition_probability_table(self):
        return self._tpt

    @transition_probability_table.setter
    def transition_probability_table(self, value):
        raise ValueError(
            'Cannot set Transition Probability Table, it must be generated '
            'using the fit method'
        )


class Unigram(Ngram):
    def __init__(self, token_indexifier: Optional[Indexifiers.FuncType] = None):
        super().__init__(token_indexifier, n=1)

    def _build(self):
        self._tpt = probability.build_unigram(
            self._idxs,
            max(self._mapping.values()) + 1,
            smoothing=self._smoothing
        )

    def predict(self, token: str | Sequence[str]) -> tuple[str]:
        return self._predict(),  # Evil comma hack, converts to tuple

    def _predict(self) -> str:  # type: ignore[override]
        idx = max(range(len(self._tpt)), key=self._tpt.__getitem__)
        return self._inverse_mapping[idx]


class Bigram(Ngram):
    def __init__(self, token_indexifier: Optional[Indexifiers.FuncType] = None):
        super().__init__(token_indexifier, n=2)

    def _build(self):
        self._tpt = probability.build_bigram(
            self._idxs,
            max(self._mapping.values()) + 1,
            smoothing=self._smoothing
        )

    def _predict(self, token: str) -> str:
        idx = self._token_indexifier(token)[0]  # type: ignore
        assert isinstance(
            idx, int), f'required index of type {int}, found {idx = } {type(idx)} ({token = })'
        tpt = self._tpt[idx]

        print(f'{token = }', f'{tpt = }', sum(tpt))
        idx = random.choices(range(len(tpt)), weights=tpt)[0]
        print(f'{idx = }')

        print(f'{self._inverse_mapping[idx] = }')
        return self._inverse_mapping[idx]


class Trigram(Ngram):
    def __init__(self, token_indexifier: Optional[Indexifiers.FuncType] = None):
        super().__init__(token_indexifier, n=3)

    def _build(self):
        self._tpt = probability.build_trigram(
            self._idxs,
            max(self._mapping.values()) + 1,
            smoothing=self._smoothing
        )

    def _predict(self, token: str) -> str:
        print(self.__class__.__name__, token)
        idx1, idx2 = self._token_indexifier(token)  # type: ignore
        assert isinstance(idx1, int)
        assert isinstance(idx2, int)

        tpt = self._tpt[idx1][idx2]

        idx = random.choices(range(len(tpt)), weights=tpt)[0]
        return self._inverse_mapping[idx]


def isclose(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class NaiveBayesClassifier:
    def __init__(self, priors: Optional[dict[Any, float]] = None):
        self._priors = priors or {}
        assert isinstance(self._priors, dict), (
            '`priors` must be a dict of prior probabilities'
        )
        if self._priors:
            probs = sum(self._priors.values())
            assert isclose(1., probs), (
                'given priors do not add up to 1, '
                f'sum(priors) = {probs}'
            )

    def _compute_priors(self, labels: Sequence[Any]):
        self._priors = probability.compute_priors(labels)

    def _compute_posteriors(self, features: Sequence[Any], labels: Sequence[Any]):
        self._posteriors = probability.compute_posteriors(
            features, labels, self._priors)

    def fit(self, features: Sequence[Sequence[Any]], labels: Sequence[Any]) -> Self:
        if not self._priors:
            self._compute_priors(labels)

        self._compute_posteriors(features, labels)
        return self

    def predict(self, X: Sequence[Sequence[Any]]) -> tuple[Any]:
        predictions = []
        for feats in X:
            max_prob = float('-inf')
            predicted_label = None
            for label in self._priors.keys():
                prob = math.log(self._priors[label])
                _get = self._posteriors[label].get
                prob = sum(map(lambda x: math.log(_get(x, 1)), feats))
                if prob > max_prob:
                    max_prob = prob
                    predicted_label = label
            predictions.append(predicted_label)
        return tuple(predictions)  # type: ignore

    @property
    def priors(self):
        return self._priors

    @property
    def posteriors(self):
        return self._posteriors


def _():
    pass


FuncType = type(_)
del _


def save_result(func_or_filename):
    if s := isinstance(func_or_filename, str):
        file = func_or_filename
    elif isinstance(func_or_filename, FuncType):
        file = f'{func_or_filename.__name__}.txt'
    else:
        raise TypeError(f'save_result only accepts types {str} and {FuncType}')

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            with open(file, 'w') as f:
                f.write(result)
            return result
        return inner

    return decorator if s else decorator(func_or_filename)


@save_result
def q1():
    return 'Toy Story'


@save_result
def q2(script: str) -> str:
    model = Unigram(token_indexifier=Indexifiers.Letters.lowercase)
    model.fit(script)
    tpt = model.transition_probability_table

    return ', '.join(map(lambda x: f'{round(x, 4):.4f}', tpt))


@save_result
def q3(script: str) -> str:
    model = Bigram(token_indexifier=Indexifiers.Letters.lowercase)
    model.fit(script)
    tpt = model.transition_probability_table

    return '\n'.join(', '.join(map(lambda x: f'{round(x, 4):.4f}', row)) for row in tpt)


@save_result
def q4(script: str) -> str:
    model = Bigram(token_indexifier=Indexifiers.Letters.lowercase)
    model.fit(script, smoothing=True)
    tpt = model.transition_probability_table

    return '\n'.join(', '.join(map(lambda x: f'{x}', row)) for row in tpt)


@save_result
def q5(script: str) -> str:
    bigram_model = Bigram(token_indexifier=Indexifiers.Letters.lowercase)
    bigram_model.fit(script, smoothing=True)

    trigram_model = Trigram(token_indexifier=Indexifiers.Letters.lowercase)
    trigram_model.fit(script, smoothing=True)

    total = ''
    for letter in string.ascii_lowercase:
        start = bigram_model.predict(letter)[0]

        generator = trigram_model.generate(1000, start=(letter + start))

        res = f'{letter}{start}'

        for next_ in generator:
            res += next_
            if res[-2:] not in script:
                res += next_
                break

        generator = bigram_model.generate(1000 - len(res), start=res[-1])

        res += ''.join(generator)
        total += f'{res}\n'
    return total


@save_result
def q6(sentences: str) -> str:
    return random.choice(tuple(filter(len, sentences.splitlines())))


@save_result
def q7(script: str) -> str:
    model = Unigram(token_indexifier=Indexifiers.Letters.lowercase)
    model.fit(script)
    tpt = model.transition_probability_table

    return ', '.join(map(lambda x: f'{round(x, 4):.4f}', tpt))


@save_result
def q8(real_script: str, fake_script: str, clf: NaiveBayesClassifier) -> str:
    fake_data = [fake_script]
    real_data = [real_script]

    data = fake_data + real_data
    labels = (['Fake'] * len(fake_data)) + (['Real'] * len(real_data))

    clf.fit(data, labels)

    posteriors_fake = sorted(
        clf.posteriors['Fake'].items(), key=lambda x: x[0])

    return ', '.join(map(lambda x: f'{x[1]:.4f}', posteriors_fake))


@save_result
def q9(sentences: str, clf: NaiveBayesClassifier) -> str:
    data = list(map(lambda x: list(x), sentences.splitlines()))
    predictions = clf.predict(data)

    return ','.join(map(lambda x: '0' if x == 'Real' else '1', predictions))


def cleanup(script: str) -> str:
    return ' '.join([res for token in filter(len, script.split()) if len(res := ''.join(filter(str.isalpha, token.lower())))])


def download_script(title: str, filename: str):
    import requests

    from bs4 import BeautifulSoup

    SPACE = ' '
    HYPHEN = '-'

    title = title.replace(SPACE, HYPHEN)
    url = f'https://imsdb.com/scripts/{title}.html'

    with requests.get(url) as response:
        assert response.ok
        source = response.text

    soup = BeautifulSoup(source, 'lxml')

    container = soup.find('td', 'scrtext')

    assert container, 'Couldn\'t find <td class="scrtext">'
    script = cleanup(container.text)

    with open(filename, 'w') as f:
        f.write(script)


def get_script(script_file: str, txt_extension: str = '.txt') -> str:
    try:
        if txt_extension not in script_file:
            script_file += txt_extension
        with open(script_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        try:
            with open(script_file.lower().replace(' ', '_')) as f:
                return f.read()
        except FileNotFoundError:
            pass

        if txt_extension in script_file:
            title = script_file[:-len(txt_extension)]
        else:
            title = script_file
            script_file = f'{script_file}{txt_extension}'
        script_file = script_file.lower().replace(' ', '_')
        download_script(title, script_file)
        with open(script_file) as f:
            return f.read()


def main():
    real_script = get_script('toy_story')

    # q1()

    # q2(script)

    # q3(script)

    # q4(script)

    # sentences = q5(script)

    # q6(sentences)

    fake_script = get_script('fake_script')

    # q7(fake_script)

    clf = NaiveBayesClassifier(priors={'Fake': 0.44, 'Real': 0.56})

    q8(real_script, fake_script, clf)

    sentences = (f := open('q5.txt')).read().strip()

    f.close()

    q9(sentences, clf)

    print('done!')


if __name__ == '__main__':
    main()
