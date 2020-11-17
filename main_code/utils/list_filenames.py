from os import listdir
from os.path import isfile, join


def list_filenames(inner_folder_name):
    """
    1. list the files in the folder "inner_folder_name"
    """
    filenames = [f for f in listdir(inner_folder_name) if isfile(join(inner_folder_name, f))]

    return filenames


if __name__ == "__main__":
    folder = "./"
    filename_list = list_filenames(folder)
    print("filename_list: ", filename_list)
