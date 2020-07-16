"""Simple command line utilties for interacting with Onshape API.
"""
import argparse
import json
import urllib.parse

from . import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--action', required=True, choices=['add_feature', 'get_features'])
    parser.add_argument('--payload_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--enable_logging', action='store_true')

    args = parser.parse_args()
    _, _, docid, _, wid, _, eid = urllib.parse.urlparse(args.url).path.split('/')

    client = Client(stack='https://cad.onshape.com',
                            logging=args.enable_logging)

    if args.action =='add_feature':
        if not args.payload_path:
            raise ValueError("payload_path required when adding a feature")
        with open(args.payload_path, 'r') as fh:
            sketch = json.load(fh)
        client.add_feature(docid, wid, eid, payload=sketch)

    elif args.action == 'get_features':
        resp = client.get_features(docid, wid, eid)
        fs_json = json.loads(resp.content.decode('utf8').replace("'", '"'))
        if args.output_path:
            with open(args.output_path, 'w') as fh:
                json.dump(fs_json, fh, indent=4)
        else:
            print(json.dumps(fs_json, indent=4))


if __name__ == '__main__':
    main()
