from __future__ import print_function
import os
from subprocess import call
from builtins import input

weights_filename = 'pytorch_model.bin'
torchmoji_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_path = os.path.join(os.path.join(torchmoji_dir, 'model'), weights_filename)

weights_download_link = 'https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=0#'

MB_FACTOR = float(1 << 20)


def prompt():
    while True:
        valid = {
            'y': True,
            'ye': True,
            'yes': True,
            'n': False,
            'no': False,
        }
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print('Please respond with \'y\' or \'n\' (or \'yes\' or \'no\')')


download = True
if os.path.exists(weights_path):
    print('Weight file already exists at {}. Would you like to redownload it anyway? [y/n]'.format(weights_path))
    download = prompt()
    already_exists = True
else:
    already_exists = False

if download:
    print('About to download the pretrained weights file from {}'.format(weights_download_link))
    if not already_exists:
        print('The size of the file is roughly 85MB. Continue? [y/n]')
    else:
        os.unlink(weights_path)

    if already_exists or prompt():
        print('Downloading...')

        # downloading using wget due to issues with urlretrieve and requests
        sys_call = 'wget {} -O {}'.format(weights_download_link, os.path.abspath(weights_path))
        print("Running system call: {}".format(sys_call))
        call(sys_call, shell=True)

        if os.path.getsize(weights_path) / MB_FACTOR < 80:
            raise ValueError("Download finished, but the resulting file is too small! " +
                             "It\'s only {} bytes.".format(os.path.getsize(weights_path)))
        print('Downloaded weights to {}'.format(weights_path))
else:
    print('Exiting.')
