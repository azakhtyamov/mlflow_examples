import os
import sys
import git

# https://gist.github.com/GCBallesteros/6419f64400097efb0ff506aea0798c71

def _get_main_file(repo):
    path_to_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(path_to_dir, os.path.split(sys.argv[0])[-1])

    return os.path.relpath(abs_path, repo.working_tree_dir)

def _init_repo(repo_path=None):
    if repo_path is None:
        repo_path = os.path.dirname(os.path.abspath(__file__))

    repo = git.Repo(
        repo_path,
        search_parent_directories=True,
    )

    return repo

def autocommit(
        file_paths=[],
        mode='tracked',
        add_main=True,
        repo_path=None,
        message="AUTOML"
    ):
    """
    Autocommit files specified in file_paths plus additional files depending on
    the `mode` option. All commits are done on the active branch.
    If you are using jupyter notebooks you are better of passing the all files
    via the `file_paths` parameter, setting `add_main` to false and also
    passing the repo.
    :param repo_path: Path to repo or None
    :param file_paths: Other files to track with path relatives to the root of
    the git repo. lst(str)
    :param mode: tracked/all/staged (str)
    :param add_main: This is mainly for the conveniance of notebook
    users which may have trouble figuring out the filepath. (bool)
    :param message: (str)
    :return: Commit Hash
    """
    supported_modes = ["all", "tracked", "stage"]
    mode = mode.lower()
    if mode not in supported_modes:
        raise Exception('Unsupported git autocommit mode.')

    try:
        repo = _init_repo(repo_path)
    except:
        return None

    # Track __main__
    if add_main:
        file_paths.append(_get_main_file(repo))

    # Prepare arugments for commit
    commit_msg = "-m " + message
    if mode == "all":
        commit_params = ("--all", commit_msg)
    elif mode == "tracked":
        commit_params = (file_paths, commit_msg)
    elif mode == "staged":
        pass

    # Bare call to git add because otherwise filters are not apllied
    repo.git.add(file_paths)
    try:
        repo.git.commit(*commit_params)
    except:
        # Nothing to commit
        pass

    commit = repo.head.commit.hexsha
    return commit
