import yaml


def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def assert_required_params_list(required_params_lst, included_params_lst, header=""):
    def format(string):
        return f"{header}: {string}" if header else string
    missing = [p for p in required_params_lst if p not in included_params_lst]
    assert len(missing) == 0, format(f"missing required sections: {missing}")


def list_difference(list1, list2):
    return [item for item in list1 if item not in list2]
