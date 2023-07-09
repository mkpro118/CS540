from typing import Optional, Callable
import string
from transition_probability_table import build_bigram, build_trigram


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

    script = cleanup(container.text)

    with open('toy story.txt', 'w') as f:
        f.write(script)


class Indexifiers:
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


class Bigram:
    def __init__(self, token_indexifier: Optional[Callable[..., int | tuple[int]]] = None):
        self._token_indexifier = token_indexifier

    def fit(self, text: str) -> 'Bigram':
        if not self._token_indexifier:
            self._guess_token_indexifier(text)

        self._idxs = self._token_indexifier(text, True)  # type: ignore

        if isinstance(self._idxs, tuple):
            self._idxs = self._idxs[0]

        self._tpt = build_bigram()

        return self

    def _guess_token_indexifier(self, text):
        if any([x in text for x in string.whitespace]):
            self._token_indexifier = Indexifiers.Words.case_insensitive
        else:
            self._token_indexifier = Indexifiers.Letters.alpha_numeric


if __name__ == '__main__':
    # download_script('Toy Story')
    print(Indexifiers.Letters.lowercase('wow this must be amazing'))
