import yaml

def load_dataset_paths(config_path):
    '''
    :param config_path: Path to .yml file containing the individual dataset paths
    :return: Dictionary of dataset_name --> dataset_path
    '''

    dataset_paths_dict = {}

    with open(config_path) as config_file:
        data = yaml.load(config_file, Loader=yaml.FullLoader)

        dataset_paths_dict["dSprites"]      = data['dsprites_path']
        dataset_paths_dict["cars3D"]        = data['cars3D_path']
        dataset_paths_dict["smallNorb"]     = data['smallNorb_path']
        dataset_paths_dict["shapes3d"]      = data['shapes3d_path']

    return dataset_paths_dict




