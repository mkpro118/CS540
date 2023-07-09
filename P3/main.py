from functools import wraps
from typing import Optional, Callable, Sequence
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import string
from transition_probability_table import (  # type: ignore[import]
    build_unigram,
    build_bigram,
    build_trigram,
)


def cleanup(script: str) -> str:
    return ' '.join([res for token in filter(len, script.split()) if len(res := ''.join(filter(str.isalpha, token.lower())))])


def download_script(title: str):
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

    with open('toy story.txt', 'w') as f:
        f.write(script)


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
    def __init__(self, token_indexifier: Optional[Indexifiers.FuncType] = None):
        self._token_indexifier = token_indexifier
        self._tpt: tuple = tuple()

    def _guess_token_indexifier(self, text: str):
        if any([x in text for x in string.whitespace]):
            self._token_indexifier = Indexifiers.Words.case_insensitive
        else:
            self._token_indexifier = Indexifiers.Letters.alpha_numeric

    def _build(self):
        raise NotImplementedError('Ngram functionality is not yet implemented')

    def fit(self, text: str) -> Self:
        if not self._token_indexifier:
            self._guess_token_indexifier(text)

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

        self._build()

        return self

    def predict(self, token: str | Sequence[str]) -> str | tuple[str]:
        if isinstance(token, str):
            return self._predict(token)

        return tuple(map(self._predict, token))  # type: ignore

    def _predict(self, token: str) -> str:
        if token not in self._mapping:
            raise ValueError(f'Model wasn\'t trained on {token = }')

        idx = max(range(len(self._tpt)), key=self._tpt.__getitem__)
        return self._inverse_mapping[idx]

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
    def _build(self):
        self._tpt = build_unigram(self._idxs, max(self._mapping.values()) + 1)


class Bigram(Ngram):
    def _build(self):
        self._tpt = build_bigram(self._idxs, max(self._mapping.values()) + 1)


class Trigram(Ngram):
    def _build(self):
        self._tpt = build_trigram(self._idxs, max(self._mapping.values()) + 1)


def save_result(func):
    file = f'{func.__name__}.txt'

    @wraps(func)
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        with open(file, 'w') as f:
            f.write(result)
        return result
    return inner


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


def main():
    with open('toy story.txt') as f:
        script = f.read()

    # q2(script)
    q3(script)


if __name__ == '__main__':
    main()
