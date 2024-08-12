import sys, os, re
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageMode
from PIL.Image import frombytes

# python casia_parser.py "..\data\casia\hwdb" test "C:\Users\maxos\Downloads\Gnt1.0TrainPart1" "C:\Users\maxos\Downloads\Gnt1.0TrainPart2" "C:\Users\maxos\Downloads\Gnt1.0TrainPart3" "C:\Users\maxos\Downloads\Gnt1.1TrainPart1" "C:\Users\maxos\Downloads\Gnt1.1TrainPart2"

def bytes_GB_char(label: bytes) -> str:
    decoded = ""
    method = None
    try:
        decoded = label.decode('gb2312')
        method = '2312'
    except:
        decoded = label.decode('gb18030')
        method = '18030'
    finally:
        return decoded, method

def gnt_parse_file(file: str, data_dict: dict[str: list[str]]) -> tuple[int, int]:
    file_length = Path(file).stat().st_size
    bytes_read = 0
    i = 0
    maxw, maxh = 0, 0
    with open(file, 'rb') as gnt_file:
        file_path = gnt_file.name
        file_name = next(s for s in gnt_file.name.split('\\') if '.gnt' in s).removesuffix('.gnt')
        while(bytes_read < file_length):
            b_length = gnt_file.read(4)
            length = np.frombuffer(b_length, dtype=np.uint32)[0]
            b_label = gnt_file.read(2)
            b_width = gnt_file.read(2)
            width = np.frombuffer(b_width, dtype=np.uint16)[0]
            b_height = gnt_file.read(2)
            height = np.frombuffer(b_height, dtype=np.uint16)[0]
            image_size = np.array([height]).astype(dtype=np.uint32)[0] * np.array([width]).astype(dtype=np.uint32)[0]
            b_img = gnt_file.read(image_size)
            img = frombytes("L", (width, height), b_img)
            try:
                label, method = bytes_GB_char(b_label)
            except:
                img.show()
                sys.exit(f'Error: Decoding failed! Byte value {b_label} is invalid')
            bytes_read += length
            maxw, maxh = max(maxw, width), max(maxh, height)
            data_dict['file_name'].append(file_path)
            data_dict['code'].append(file_name)
            data_dict['num'].append(i)
            data_dict['method'].append(method)
            data_dict['value'].append(b_label.hex())
            data_dict['character'].append(label)
            # print(f'fn: {data_dict['file_name'][i]}  c: {data_dict['code'][i]} n: {data_dict['num'][i]} m: {data_dict['method'][i]} v: {data_dict['value'][i]}  hz: {data_dict['character'][i]}')
            i += 1
    return (maxw, maxh)

def gnt_parse_directory(src_dir: str, dest_dir: str, data_dict: dict[str: list[str]]) -> None:
    files = Path(src_dir).glob('*.gnt')
    maxw, maxh = 0, 0
    for file in files:
        file_max_w, file_max_h = gnt_parse_file(file, data_dict) 
        maxw = max(file_max_w, maxw)
        maxh = max(file_max_h, maxh)
    print(f'{maxw=} {maxh=}')

def gnt_parse(src_dir: str, dest_dir: str, labels_name: str | None) -> None:
    data_dict = {
        'file_name': [],
        'code': [],
        'num': [],
        'method': [],
        'value': [],
        'character': [],
    }
    csv_path = f'{dest_dir}\\{labels_name}.csv'
    gnt_parse_directory(src_dir, dest_dir, data_dict)
    pd.DataFrame(data_dict).to_csv(csv_path, index=False)

def pot_parse():
    pass

def main():
    if len(sys.argv) < 3:
        sys.exit('ERROR: Too few arguments! (min. 4)\nPattern: casia_parser.py [new_directory_name] [labels_file_name] [old_directory_1 (...)]')
    new_directory = sys.argv[1]
    labels_file_name = sys.argv[2]
    for i in range(3, len(sys.argv)):
        mode = 'ol' if 'pot' in sys.argv[i].lower() else 'hw' if 'gnt' in sys.argv[i].lower() else None
        directory = sys.argv[i]
        match mode:
            case 'ol':
                pass
            case 'hw':
                gnt_parse(directory, new_directory, labels_file_name)
            case None:
                exit('ERROR: No valid directories found!\nDirectories must be labeled with POT or GNT (not case sensitive)')
    sys.argv[0]

if __name__ == '__main__':
    main()