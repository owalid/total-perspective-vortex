import argparse as ap
from server_stream import SocketServer


if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')

    args = parser.parse_args()

    SocketServer(args.directory_dataset).start_server()
