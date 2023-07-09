from typing import Optional, Callable


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
    import string

    _lowercase = string.ascii_lowercase
    _uppercase = string.ascii_uppercase
    _digits = string.digits

    del string

    @staticmethod
    def lowercase(token: str) -> tuple[int]:
        def convert(char: str) -> int:
            assert len(char) == 1, (
                f'{char = } is not a lowercase letter or space (\' \')!'
            )
            if char.isspace():
                return 0
            elif char in Indexifiers._lowercase:
                return Indexifiers._lowercase.index(char) + 1

            raise ValueError(
                f'{char = } is not a lowercase letter or space (\' \')!'
            )
        return tuple(map(convert, token))  # type: ignore

    @staticmethod
    def uppercase(token: str) -> tuple[int]:
        def convert(char: str) -> int:
            assert len(char) == 1, (
                f'{char = } is not a uppercase letter or space (\' \')!'
            )
            if char.isspace():
                return 0
            elif char in Indexifiers._uppercase:
                return Indexifiers._uppercase.index(char) + 1

            raise ValueError(
                f'{char = } is not a uppercase letter or space (\' \')!'
            )
        return tuple(map(convert, token))  # type: ignore

    @staticmethod
    def alphabetic(token: str) -> tuple[int]:
        def convert(char: str) -> int:
            lowercase_offset = 1 + len(Indexifiers._uppercase)
            assert len(char) == 1, (
                f'{char = } is not a alphabetic character or space (\' \')!'
            )
            if char.isspace():
                return 0
            elif char in Indexifiers._uppercase:
                return Indexifiers._uppercase.index(char) + 1
            elif char in Indexifiers._lowercase:
                return Indexifiers._lowercase.index(char) + lowercase_offset

            raise ValueError(
                f'{char = } is not a lowercase letter or space (\' \')!'
            )
        return tuple(map(convert, token))  # type: ignore

    @staticmethod
    def alpha_numeric(token: str) -> tuple[int]:
        def convert(char: str) -> int:
            lowercase_offset = 1 + len(Indexifiers._uppercase)
            digit_offset = lowercase_offset + len(Indexifiers._lowercase)
            assert len(char) == 1, (
                f'{char = } is not a alphabetic character or space (\' \')!'
            )
            if char.isspace():
                return 0
            elif char in Indexifiers._uppercase:
                return Indexifiers._uppercase.index(char) + 1
            elif char in Indexifiers._lowercase:
                return Indexifiers._lowercase.index(char) + lowercase_offset
            elif char in Indexifiers._lowercase:
                return Indexifiers._digits.index(char) + digit_offset

            raise ValueError(
                f'{char = } is not a lowercase letter or space (\' \')!'
            )
        return tuple(map(convert, token))  # type: ignore


class Bigram:
    import transition_probability_table

    def __init__(self, *, vocab_length: int=27,
                 vocab: Optional[set]=None,
                 token_indexifier: Callable[[str], int]):
        self.transition_matrix = []


if __name__ == '__main__':
    # download_script('Toy Story')
    print(Indexifiers.alpha_numeric('wow this must be amazing'))
