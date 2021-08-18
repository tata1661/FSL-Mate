import paddlefsl.utils as utils


def process_root_test():
    root = '~/test'
    module_dir = utils.process_root(root, module_name='test_module')
    print(module_dir)


def download_url_test():
    background_url = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
    background_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    dir_name = '~/test'
    utils.download_url(dir_name, background_url)


def check_file_exist_test():
    dir_name = '~/test'
    file_name = 'images_background.zip'
    file_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    exist = utils.check_file_exist(dir_name, file_name, file_md5)
    print(exist)


def check_md5_test():
    file_path = '~/test/images_background.zip'
    true_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    false_md5 = 'md5'
    check1 = utils.check_md5(file_path, true_md5)
    check2 = utils.check_md5(file_path, false_md5)
    print(check1, check2)


def decompress_test():
    file_path = '~/test/images_background.zip'
    utils.decompress(file_path)


def list_dir_test():
    dir_name = '~/test'
    dirs = utils.list_dir(dir_name)
    print(dirs)


def list_files_test():
    dir_name = '~/test/test_module'
    files = utils.list_files(dir_name, '.zip')
    print(files)


def clear_file_test():
    file_path = '~/test/test.txt'
    utils.clear_file(file_path)


if __name__ == '__main__':
    process_root_test()
    download_url_test()
    check_file_exist_test()
    check_md5_test()
    decompress_test()
    list_dir_test()
    list_files_test()
    clear_file_test()
