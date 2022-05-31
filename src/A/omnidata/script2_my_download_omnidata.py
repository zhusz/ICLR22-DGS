# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import aria2p
from subprocess import check_call, Popen
import atexit
import signal
# import urllib.request
import os
import pickle
import time
import multiprocessing as mp
import tqdm

import ipdb
from typing import Dict, Optional, Iterable


# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#     ENDC = '\033[0m'
# def notice(msg): print(f'[{bcolors.OKGREEN + bcolors.BOLD}NOTICE{bcolors.ENDC}] {msg}')
# def header(msg): print(f'[{bcolors.HEADER + bcolors.BOLD}HEADER{bcolors.ENDC}] {msg}')
# def license(msg): print(f'[{bcolors.WARNING + bcolors.BOLD}LICENSE{bcolors.ENDC}] {msg}')
# def underline(msg): print(f'{bcolors.UNDERLINE}{msg}{bcolors.ENDC}')
# def failure(msg): print(f'[{bcolors.FAIL + bcolors.BOLD}FAILURE{bcolors.ENDC}] {msg}')


PRESET_MIN = 16


def ensure_aria2_server(aria2_create_server, aria2_uri, aria2_secret, connections_total, connections_per_server_per_download, aria2_cmdline_opts, **kwargs):
  if not aria2_uri or not aria2_create_server: return None
  a2host, a2port = ":".join(aria2_uri.split(':')[:-1]), aria2_uri.split(':')[-1]
  # notice(f"Opening aria2c download daemon in background: {bcolors.WARNING}Run {bcolors.OKCYAN}'aria2p'{bcolors.WARNING} in another window{bcolors.ENDC} to view status.")
  print("Opening aria2c download daemon in background: Run 'aria2p' in another window to view status.")
  n = connections_total
  x = connections_per_server_per_download if connections_per_server_per_download is not None else connections_total
  x = min(x, PRESET_MIN)
  a2server = Popen(('aria2c --enable-rpc --rpc-listen-all --disable-ipv6 -c --auto-file-renaming=false ' +
                    # '--optimize-concurrent-downloads ' +
                    f'-s{n}  -j{n}  -x{x} {aria2_cmdline_opts}').split())
  atexit.register(os.kill, a2server.pid, signal.SIGINT)
  return aria2p.API(aria2p.Client(host=a2host, port=a2port, secret=aria2_secret))


def download_tar(url, output_dir='.', output_name=None, n=20, n_per_server=10,
  checksum=None, max_tries_per_model=3, aria2api=None, dryrun=False,
  ) -> Optional[str]:
  '''Downloads "url" to output filename. Returns downloaded fpath.'''
  fname = url.split('/')[-1] if output_name is None else output_name
  fpath = os.path.join(output_dir, fname)
  if dryrun: print(f'Downloading "{url}"" to "{fpath}"'); return fpath
  # checksum = checksum[:-3] + '000'
  # print(checksum)
  if aria2api is not None:
    options_dict = { 'out': fname, 'dir': output_dir, 'check_integrity': True}
    if checksum is not None: options_dict['checksum'] = f"md5={checksum}"
    while (max_tries_per_model := max_tries_per_model-1) > 0:
      res = aria2api.client.add_uri(uris=[url], options=options_dict)
      success = wait_on(aria2api, res)
      if success: break
    if not success: return None
  else:
    # os.makedirs(output_dir, exist_ok=True)
    # cmd = f'lftp -e "pget -n {n} {url} -o {fpath}"'
    # # print(cmd)
    # call(cmd, shell=True)
    options = f'-c --auto-file-renaming=false'
    if n_per_server is None: n_per_server = min(n, PRESET_MIN)
    options += f' -s {n} -j {n} -x {n_per_server}' # N connections
    if checksum is not None: options += f' --check-integrity=true --checksum=md5={checksum}'
    cmd = f'aria2c -k 1M -d {output_dir} -o {fname} {options} "{url}"'
    # print(cmd)
    call(cmd, shell=True)

    # os.makedirs(output_dir, exist_ok=True)
    # cmd = f'axel -q -o {fpath} -c -n {n} "{url}" '
    # success = True
    # while (max_tries_per_model := max_tries_per_model-1) > 0:
    #   call(cmd, shell=True)
    #   if checksum is not None: success = (check_output(['md5sum', fpath], encoding='UTF-8').split()[0] == checksum)
    # if not success: return None
  return fpath


def wait_on(a2api, gid, duration=0.2):
  while not (a2api.get_downloads([gid])[0].is_complete or a2api.get_downloads([gid])[0].has_failed):
    time.sleep(duration)
  success = a2api.get_downloads([gid])[0].is_complete
  a2api.remove(a2api.get_downloads([gid]))
  return success


# aria2
'''
aria2 = ensure_aria2_server(
    aria2_create_server=True,
    aria2_uri='http://localhost:6800',
    aria2_secret='',
    connections_total=32,
    connections_per_server_per_download=None,
    aria2_cmdline_opts='',
)
'''


# define dirs
folder_name = 'd2'
compress_dir = '/home/zhusz/omnidata/%s/compressed/' % folder_name
uncompress_dir = '/home/zhusz/omnidata/%s/uncompressed/' % folder_name
tag_dir = '/home/zhusz/omnidata/%s/tag/' % folder_name


def process_sample(line):
    # line = lines[j]
    flag_omnidata = 'omnidata' in line
    flag_taskonomy = 'taskonomy' in line
    assert (not flag_taskonomy) and flag_omnidata
    fn = os.path.basename(line)
    assert fn.endswith('.tar')

    ss = fn.split('-')
    assert len(ss) >= 3, (ss, fn)
    subset_name = ss[0]
    dataset_name = ss[1]
    if dataset_name in ['hm3d', 'hypersim', 'replica_gso']:
        assert len(ss) == 4
        if dataset_name == 'hm3d':
            assert len(ss[2]) == 5
            assert len(ss[3]) == 11 + 4
        scene_name = ss[2] + '-' + ss[3][:-4]
    else:
        assert len(ss) == 3, (ss, fn)
        scene_name = ss[2][:-4]

    tagFileName = tag_dir + '%s-%s-%s.pkl' % (subset_name, dataset_name, scene_name)

    if os.path.isfile(tagFileName):
        # print('Skipping %d (j1 = %d, j2 = %d' % (j, j1, j2))
        pass
    else:
        # print('script_my_download of omnidata: %d (j1 = %d, j2 = %d) - %s' %
        #       (j, j1, j2, lines[j]))
        try:

            # # wget downloading
            # wget_command_list = ['wget', '-P', compress_dir, line, '-q', '-o', '/dev/null']
            # check_call(wget_command_list)  # no timeout in case too large file

            # aria2 downloading

            '''
            flag1 = False
            flag2 = False
            if not (flag1 and flag2):  # if-pass or while-break
                time.sleep(0.5)
                fpath = download_tar(
                    url=line,
                    output_dir=compress_dir,
                    output_name=fn,
                    n=1, n_per_server=None, checksum=None,
                    max_tries_per_model=20, aria2api=aria2,
                    dryrun=False,
                )
                flag1 = bool(fpath)
                flag2 = not os.path.isfile(compress_dir + fn + '.aria2')
                if flag1 and flag2:
                    # break
                    pass
                elif flag1 and (not flag2):
                    print('Wrong Type A')
                    import ipdb
                    ipdb.set_trace()
                    print(1 + 1)
                elif (not flag1) and flag2:
                    print('Wrong Type B')
                    import ipdb
                    ipdb.set_trace()
                    print(1 + 1)
                else:
                    raise ValueError('Download Needs Restarting in Future Rollouts.')
            '''

            assert os.path.isfile(compress_dir + fn)
            # assert not os.path.isfile(compress_dir + fn + '.aria2')

            # tar -xf
            direct_dir = uncompress_dir + '%s/%s/%s' % \
                         (subset_name, dataset_name, scene_name)
            if not os.path.isdir(direct_dir):
                os.makedirs(direct_dir, exist_ok=True)
            tar_command_list = [
                'tar', '-xf', compress_dir + fn, '-C', direct_dir, '--strip-components=4'
            ]
            check_call(tar_command_list)

            # remove the compressed
            # # rm_command_list = ['rm', compress_dir + fn]
            # # check_call(rm_command_list)

            with open(tagFileName, 'wb') as f:
                pickle.dump({
                    'finishTag': 1,
                }, f)
        except:
            pass
            # Do not delete anything
            # so that relaunching can restart the downloading without losing existing downloaded part
            # '''
            # if os.path.isfile(compress_dir + fn):
            #     rm_command_list = ['rm', compress_dir + fn]
            #     check_call(rm_command_list)
            # if os.path.isfile(compress_dir + fn + '.aria2'):
            #     rm_command_list = ['rm', compress_dir + fn + '.aria2']
            #     check_call(rm_command_list)
            # '''


def main():
    for x in [compress_dir, uncompress_dir, tag_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)

    # url = 'https://datasets.epfl.ch/taskonomy/links.txt'
    # file = urllib.request.urlopen(url)
    # lines = [line.decode('utf-8').strip() for line in file]
    # lines = [line for line in lines if line.endswith('.tar')]
    with open('links_omnidata.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line.endswith('.tar')]
    lines = [line for line in lines if 'hypersim' in line]

    m = len(lines)  # 26473
    for i, line in enumerate(lines):
        assert line.endswith('.tar'), (i, line)
        assert ' ' not in line, (i, line)

    indChosen = list(range(m))

    for j in indChosen:
        print('Single thread working on %s' % lines[j])
        process_sample(lines[j])
    '''
    n_workers = 16  # When you are doing this, set J1 to be 0 and J2 to be m (or larger 10000).
    lines_to_run = [lines[x] for x in indChosen]
    with mp.Pool(n_workers) as p:
        r = list(tqdm.tqdm(p.imap(process_sample, lines_to_run), total=len(indChosen)))
    '''


if __name__ == '__main__':
    main()
