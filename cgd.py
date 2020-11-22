from typing import List, Optional, Tuple
import torch
import torch.utils.data as data
import string
from gensim.models import Word2Vec
import re

cpp_keywords = ('alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel', 'atomic_commit', 'atomic_noexcept', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 'const_cast', 'continue', 'co_await', 'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private', 'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq')
extra_keywords = ('main', 'NULL')

cwe_119_keywords = tuple(map(lambda v: v.strip(), open('./cwe-119.txt', 'r').read().split(',')))
cwe_399_keywords = tuple(map(lambda v: v.strip(), open('./cwe-399.txt', 'r').read().split(',')))

class Lexer(object):
    def __init__(self, code: str):
        self.code = code
    
    def _is_identifier_char(self, c: str) -> bool:
        if c in string.ascii_letters + string.digits + '_':
            return True
        return False

    def _is_digit(self, c: str) -> bool:
        if c in string.digits:
            return True
        return False

    def __iter__(self):
        self.i = 0
        return self

    def _peek(self) -> Optional[str]:
        if self.i < len(self.code):
            return self.code[self.i]
        raise StopIteration

    def _get(self) -> Optional[str]:
        if self.i < len(self.code):
            val = self.code[self.i] 
            self.i += 1
            return val
        raise StopIteration
    
    def _identifier(self) -> str:
        token = self._get()
        while self._is_identifier_char(self._peek()):
            token += self._get()
        return token

    def _number(self) -> str:
        token = self._get()
        while self._is_digit(self._peek()):
            token += self._get()
        return token
    
    def __next__(self) -> str:
        while True:
            c = self._peek()
            if c is None:
                break
            if not c.isspace():
                break
            self._get()
        
        c = self._peek()
        if c in string.ascii_letters:
            return self._identifier()
        elif c in string.digits:
            return self._number()
        elif c in '()[]{}<>=+-*/#.,:;\'"|':
            return self._get()
        else:
            return self._get()

def map_identifier(tokens: List[str], keywords: Tuple[str]):
    variable = {}
    variable_i = 1
    function = {}
    function_i = 1
    for i in range(len(tokens)):
        if re.match('^[a-zA-Z_][a-zA-Z0-9_]*$', tokens[i]) is not None and tokens[i] not in keywords:
            if i+1 < len(tokens) and tokens[i+1] == '(':
                # Functions
                if tokens[i] not in function:
                    function[tokens[i]] = f'FUN{function_i}'
                    function_i += 1
                tokens[i] = function[tokens[i]]
            else:
                # Variables
                if tokens[i] not in variable:
                    variable[tokens[i]] = f'VAR{variable_i}'
                    variable_i += 1
                tokens[i] = variable[tokens[i]]

def backward_slice(gadget: List[str]):
    pass

class CGDDataset(data.Dataset):
    def __init__(self, file: str, vector_length: int, keywords: Tuple[str]):
        super(CGDDataset, self).__init__()
        self.data = []
        self.token = []
        self.vector_length = vector_length
        with open(file, "r", encoding='utf-8') as f:
            info = ""
            code = ""
            value = 0
            while True:
                line = f.readline()
                if not line:
                    break

                if "---------------------------------" in line:
                    token = list(Lexer(code))
                    map_identifier(token, keywords)
                    self.data.append((info, code, value))
                    self.token.append(token)
                    info = ""
                    code = ""
                    value = 0
                    continue

                if line.strip().isdigit():
                    value = int(line)
                elif info == "":
                    info = line.strip()
                else:
                    code += line
            if info != "":
                token = list(Lexer(code))
                map_identifier(token, keywords)
                self.data.append((info, code, value))
                self.token.append(token)
        
        self.model = Word2Vec(self.token, min_count=1, size=self.vector_length, sg=1)
    
    def __getitem__(self, index):
        return self.token[index], self.data[index][2]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = CGDDataset('./VulDeePecker/CWE-119/CGD/cwe119_cgd.txt', 50, cpp_keywords + extra_keywords + cwe_119_keywords)

    for i in range(len(dataset.data)):
        print(dataset.token[i], dataset.data[i][1])
        input()
