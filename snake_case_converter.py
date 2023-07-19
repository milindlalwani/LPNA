import re
import tokenize
import keyword
import token
from io import BytesIO
import argparse

def snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def modify_file(filename):
    with open(filename, 'r') as file:
        source = file.read()

    tokens = list(tokenize.tokenize(BytesIO(source.encode('utf-8')).readline))
    transformed = []

    for i, tok in enumerate(tokens):
        if tok.type == token.NAME and not keyword.iskeyword(tok.string) and tok.string[0].islower() and any(c.isupper() for c in tok.string):
            if i > 0 and tokens[i-1].type == token.OP and tokens[i-1].string == '.':
                # do not change attribute names
                transformed.append(tok)
            else:
                transformed.append(tokenize.TokenInfo(tok.type, snake_case(tok.string), tok.start, tok.end, tok.line))
        else:
            transformed.append(tok)

    transformed_source = tokenize.untokenize(transformed).decode('utf-8')

    with open(filename, 'w') as file:
        file.write(transformed_source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert camelCase variables in a Python file to snake_case.')
    parser.add_argument('filename', type=str, help='the Python file to convert')

    args = parser.parse_args()
    modify_file(args.filename)
