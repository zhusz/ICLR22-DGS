# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is borrowed from https://github.com/google-research/corenet

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstraction file I/O layer with support for local and GCS file systems."""

import glob
import logging
import typing as t

import fnmatch
import google.api_core.exceptions
import os
import re
from google.cloud import storage

_gcs_client = None  # type: t.Optional[storage.Client]
log = logging.getLogger(__name__)
T = t.TypeVar("T")
NUM_GCS_RETRIES = 3


def is_gs_path(p: str):
  return p.startswith("gs://")


def splitall(path: str):
  """Splits a path into all of its components."""
  result = []
  if is_gs_path(path):
    result.append(path[:5])
    path = path[5:]
  while True:
    head, tail = os.path.split(path)
    if head == path:
      result.append(head)
      break
    if tail == path:
      result.append(tail)
      break
    else:
      path = head
      result.append(tail)
  result.reverse()
  return result


def parse_gs_path(p: str):
  assert p.startswith("gs://")
  p = p[5:]
  parts = splitall(p)
  bucket, path = parts[0], "/".join(parts[1:])
  return bucket, path


def get_gcs_client():
  global _gcs_client
  if not _gcs_client:
    _gcs_client = storage.Client()

  return _gcs_client


# noinspection PyBroadException
def repeat_if_error(fn: t.Callable[[], T], num_tries, not_found_ok=False) -> T:
  for try_index in range(num_tries - 1):
    try:
      return fn()
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception as e:
      if isinstance(e, google.api_core.exceptions.NotFound) and not_found_ok:
        return None
      log.exception(f"Error in file operation, try={try_index}. Retrying ...")
  return fn()


def read_bytes(path: str) -> bytes:
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)

    def _impl():
      bucket = get_gcs_client().get_bucket(bucket_name)
      return bucket.blob(gcs_path).download_as_string()

    return repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    with open(path, "rb") as fl:
      return fl.read()


def read_text(path: str) -> str:
  return read_bytes(path).decode()


def write_bytes(path: str, contents: bytes):
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)

    def _impl():
      bucket = get_gcs_client().get_bucket(bucket_name)
      bucket.blob(gcs_path).upload_from_string(contents)

    repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    with open(path, "wb") as fl:
      fl.write(contents)


def write_text(path: str, text: str):
  write_bytes(path, text.encode())


def glob_pattern(pattern: str) -> t.Iterable[str]:
  if is_gs_path(pattern):
    bucket_name, gcs_path = parse_gs_path(pattern)

    parts = splitall(gcs_path)
    prefix = ""
    for part in parts:
      if re.match(r".*[?*\[].*", part):
        break
      prefix = os.path.join(prefix, part)

    def _impl():
      blobs = get_gcs_client().list_blobs(bucket_name, prefix=prefix)
      result = [f"gs://{bucket_name}/{v.name}" for v in blobs if
                fnmatch.fnmatch(v.name, gcs_path)]
      return result

    return repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    return glob.glob(pattern)


def unlink_file(path: str):
  if is_gs_path(path):
    bucket_name, gcs_path = parse_gs_path(path)
    return repeat_if_error(
        lambda: get_gcs_client().bucket(bucket_name).blob(gcs_path).delete(),
        NUM_GCS_RETRIES, not_found_ok=True)
  else:
    os.unlink(path)


def rename_file(old_path: str, new_path: str):
  if is_gs_path(old_path) != is_gs_path(new_path):
    log.error("Invalid rename (different file systems): "
              f"'{old_path}'->'{new_path}'")
    raise ValueError("Both files must be on the same file system")
  if is_gs_path(old_path):
    bucket_name, old_gcs_path = parse_gs_path(old_path)
    _, new_gcs_path = parse_gs_path(new_path)

    def _impl():
      bucket = get_gcs_client().bucket(bucket_name)
      bucket.rename_blob(bucket.blob(old_gcs_path), new_gcs_path)

    return repeat_if_error(_impl, NUM_GCS_RETRIES)
  else:
    os.rename(old_path, new_path)


def make_dirs(path: str):
  if is_gs_path(path):
    return
  os.makedirs(path, exist_ok=True)


def join(*args):
  if len(args) == 1:
    return args[0]
  for i, v in enumerate(args[1:]):
    if is_gs_path(v):
      return join(*args[i + 1:])
  return os.path.join(*args)


def normpath(p: str):
  if is_gs_path(p):
    return f"gs://{os.path.normpath(p[5:])}"
  return os.path.normpath(p)


def isabs(p: str):
  if is_gs_path(p):
    return True
  return os.path.isabs(p)


def dirname(p: str):
  return os.path.dirname(p)


def abspath(p: str):
  if is_gs_path(p):
    return p
  return os.path.abspath(p)


def basename(p: str):
  return os.path.basename(p)


def relpath(p: str, prefix: str) -> str:
  if is_gs_path(p) != is_gs_path(prefix):
    raise ValueError("Both paths have to be on the same storage system "
                     "(either GCS or local)")
  if is_gs_path(p):
    p = p[5:]
    prefix = prefix[5:]
  return os.path.relpath(p, prefix)


def splitext(p: str):
  return os.path.splitext(p)