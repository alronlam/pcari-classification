import os

import Settings


def construct_path_from_project_root(path):
    return os.path.join(Settings.PROJECT_ROOT, path)