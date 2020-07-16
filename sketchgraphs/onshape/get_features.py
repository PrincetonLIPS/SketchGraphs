
import argparse
import json

import urllib.parse

import onshape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url')
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--enable_logging', type=bool, default=False)

    args = parser.parse_args()
    _, _, docid, _, wid, _, eid = urllib.parse.urlparse(args.url).path.split('/')

    client = onshape.Client(stack='https://cad.onshape.com',
                            logging=args.enable_logging)
    resp = client.get_features(docid, wid, eid)
    fs_json = json.loads(resp.content.decode('utf8').replace("'", '"'))
    if args.output_path:
        with open(args.output_path, 'w') as fh:
            json.dump(fs_json, fh, indent=4)
    else:
        print(json.dumps(fs_json, indent=4))


if __name__ == '__main__':
    main()
