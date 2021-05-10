import argparse
import sec2sec_transformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform question in natural language to sparql')
    parser.parse_args('-trd', '--dataset-train', help='Training dataset.')
    parser.parse_args('-ted', '--dataset-test', help='Test dataset.')

    arg = parser.parse_args()
