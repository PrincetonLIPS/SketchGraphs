"""Simple command line utilties for interacting with Onshape API.

A sketch is considered a feature of an Onshape PartStudio. This script enables adding a sketch to a part (add_feature), retrieving all features from a part including sketches (get_features), and retrieving the possibly updated state of each sketch's entities/primitives post constraint solving (get_info).
"""
import argparse
import json
import urllib.parse

from . import Client


TEMPLATE_PATH = 'sketchgraphs/onshape/feature_template.json'


def _parse_resp(resp):
    """Parse the response of a retrieval call.
    """
    parsed_resp = json.loads(resp.content.decode('utf8').replace("'", '"'))
    return parsed_resp


def _save_or_print_resp(resp_dict, output_path=None, indent=4):
    """Saves or prints the given response dict.
    """
    if output_path:
        with open(output_path, 'w') as fh:
            json.dump(resp_dict, fh, indent=indent)
    else:
        print(json.dumps(resp_dict, indent=indent))


def _create_client(logging):
    """Creates a `Client` with the given bool value for `logging`.
    """
    client = Client(stack='https://cad.onshape.com',
                    logging=logging)
    return client


def _parse_url(url):  
    """Extracts doc, workspace, element ids from url.
    """
    _, _, docid, _, wid, _, eid = urllib.parse.urlparse(url).path.split('/')
    return docid, wid, eid


def update_template(url, logging=False):
    """Updates version identifiers in feature_template.json.

    Parameters
    ----------
    url : str
        URL of Onshape PartStudio
    logging: bool
        Whether to log API messages (default False)

    Returns
    -------
    None
    """
    # Get PartStudio features (including version IDs)
    features = get_features(url, logging)
    # Get current feature template
    with open(TEMPLATE_PATH, 'r') as fh:
        template = json.load(fh)
    for version_key in ['serializationVersion', 'sourceMicroversion', 'libraryVersion']:
        template[version_key] = features[version_key]
    # Save updated feature template
    with open(TEMPLATE_PATH, 'w') as fh:
        json.dump(template, fh, indent=4)


def add_feature(url, sketch_dict, sketch_name=None, logging=False):
    """Adds a sketch to a part.

    Parameters
    ----------
    url : str
        URL of Onshape PartStudio
    sketch_dict: dict
        A dictionary representing a `Sketch` instance with keys `entities` and `constraints`
    sketch_name: str
        Optional name for the sketch. If none provided, defaults to 'My Sketch'.
    logging: bool
        Whether to log API messages (default False)

    Returns
    -------
    None
    """
    # Get doc ids and create Client
    docid, wid, eid = _parse_url(url)
    client = _create_client(logging)
    # Get feature template
    with open(TEMPLATE_PATH, 'r') as fh:
        template = json.load(fh)
    # Add sketch's entities and constraints to the template
    template['feature']['message']['entities'] = sketch_dict['entities']
    template['feature']['message']['constraints'] = sketch_dict['constraints']
    if not sketch_name:
        sketch_name = 'My Sketch'
    template['feature']['message']['name'] = sketch_name
    # Send to Onshape
    client.add_feature(docid, wid, eid, payload=template)


def get_features(url, logging=False):
    """Retrieves features from a part.

    Parameters
    ----------
    url : str
        URL of Onshape PartStudio
    logging : bool
        Whether to log API messages (default False)

    Returns
    -------
    features : dict
        A dictionary containing the part's features
    
    """
    # Get doc ids and create Client
    docid, wid, eid = _parse_url(url)
    client = _create_client(logging)
    # Get features
    resp = client.get_features(docid, wid, eid)
    features = _parse_resp(resp)
    return features


def get_info(url, sketch_name=None, logging=False):
    """Retrieves possibly updated states of entities in a part's sketches.

    Parameters
    ----------
    url : str
        URL of Onshape PartStudio
    sketch_name : str
        If provided, only the entity info for the specified sketch will be returned. Otherwise, the full response is returned.
    logging : bool
        Whether to log API messages (default False)

    Returns
    -------
    sketch_info : dict
        A dictionary containing entity info for sketches
    
    """
    # Get doc ids and create Client
    docid, wid, eid = _parse_url(url)
    client = _create_client(logging)
    # Get features
    resp = client.sketch_information(docid, wid, eid)
    sketch_info = _parse_resp(resp)
    if sketch_name:
        sketch_found = False
        for sk in sketch_info['sketches']:
            if sk['sketch'] == sketch_name:
                sketch_info = sk
                sketch_found = True
                break
        if not sketch_found:
            raise ValueError("No sketch found with given name.")
    return sketch_info


def get_states(url, logging=False):
    """Retrieves states of sketches in a part.

    If there are no issues with a sketch, the feature state is `OK`. If there
    are issues, e.g., unsolved constraints, the state is `WARNING`. All sketches
    in the queried PartStudio must have unique names.

    Parameters
    ----------
    url : str
        URL of Onshape PartStudio
    logging : bool
        Whether to log API messages (default False)

    Returns
    -------
    sketch_states : dict
        A dictionary containing sketch names as keys and associated states as
        values
    """
    # Get features for the given part url
    features = get_features(url, logging=logging)
    # Gather all feature states
    feat_states = {f['key']:f['value']['message']['featureStatus']
        for f in features['featureStates']}
    # Gather sketch states
    sketch_states = {}
    for feat in features['features']:
        if feat['typeName'] != 'BTMSketch':
            continue
        sk_name = feat['message']['name']
        sk_id = feat['message']['featureId']
        # Check if name already encountered
        if sk_name in sketch_states:
            raise ValueError("Each sketch must have a unique name.")
        sketch_states[sk_name] = feat_states[sk_id]
    return sketch_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', 
        help='URL of Onshape PartStudio',required=True)
    parser.add_argument('--action', 
        help='The API call to perform', required=True, 
        choices=['add_feature', 'get_features', 'get_info', 'get_states',
                 'update_template'])
    parser.add_argument('--payload_path', 
        help='Path to payload being sent to Onshape', default=None)
    parser.add_argument('--output_path', 
        help='Path to save result of API call', default=None)
    parser.add_argument('--enable_logging', 
        help='Whether to log API messages', action='store_true')
    parser.add_argument('--sketch_name', 
        help='Optional name for sketch', default=None)

    args = parser.parse_args()

    # Parse the URL
    _, _, docid, _, wid, _, eid = urllib.parse.urlparse(args.url).path.split('/')

    # Create client
    client = Client(stack='https://cad.onshape.com',
                            logging=args.enable_logging)

    # Perform the specified action
    if args.action =='add_feature':
        # Add a sketch to a part
        if not args.payload_path:
            raise ValueError("payload_path required when adding a feature")
        with open(args.payload_path, 'r') as fh:
            sketch_dict = json.load(fh)
        add_feature(args.url, sketch_dict, args.sketch_name, 
            args.enable_logging)

    elif args.action == 'get_features':
        # Retrieve features from a part
        features = get_features(args.url, args.enable_logging)
        _save_or_print_resp(features, output_path=args.output_path)
    
    elif args.action == 'get_info':
        # Retrieve possibly updated states of entities in a part's sketches
        sketch_info = get_info(args.url, args.sketch_name, args.enable_logging)
        _save_or_print_resp(sketch_info, output_path=args.output_path)

    elif args.action == 'get_states':
        # Retrieve states of sketches in a part
        sketch_states = get_states(args.url, args.enable_logging)
        _save_or_print_resp(sketch_states, output_path=args.output_path)
    
    elif args.action == 'update_template':
        # Updates version identifiers in template
        update_template(args.url, args.enable_logging)


if __name__ == '__main__':
    main()