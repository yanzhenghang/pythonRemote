#!/usr/bin/env python3
import base64
import gzip
from pathlib import Path


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def build_script(project_path,script_pyfile):
    # to_encode = list(Path(project_path).glob('*.py')) + [Path('setup.py')]
    to_encode = list(Path(project_path).glob('*.py'))
    to_encode1 = [Path('setup.py')]
    file_data = {str(path)[46:]: encode_file(path) for path in to_encode}
    file_data.update({str(to_encode1[0]): encode_file(to_encode1[0])})
    template = Path(script_pyfile).read_text('utf8')
    Path('build/script.py').write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')


if __name__ == '__main__':

    project_path = '/home/yanzhenghang/pythonRemote/kaggle_script/imet'
    #easy_gold
    script_pyfile = '/home/yanzhenghang/pythonRemote/kaggle_script/kaggle_scipt3.py'
    #'script_template.py'
    build_script(project_path,script_pyfile)
