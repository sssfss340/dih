import os
import re

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    pattern = re.compile(r'^\d+$')

    for name in os.listdir(dir_path):
        old_path = os.path.join(dir_path, name)
        if not os.path.isfile(old_path):
            continue

        base, ext = os.path.splitext(name)
        ext = ext.lstrip('.')
        if not pattern.match(ext):
            continue

        new_name = base + "." + ext + '.pcap'
        new_path = os.path.join(dir_path, new_name)

        if os.path.exists(new_path):
            i = 1
            while os.path.exists(os.path.join(dir_path, f'{base}_{i}.pcap')):
                i += 1
            new_path = os.path.join(dir_path, f'{base}_{i}.pcap')

        os.rename(old_path, new_path)
        print(f'{name} -> {os.path.basename(new_path)}')

if __name__ == '__main__':
    main()
