# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import time

from dask.distributed import Client


def setup_dask(scheduler, retries=-1):
    if scheduler is None or scheduler == "{scheduler}":
        print("Setting up local cluster...")
        return Client()
    succeeded = False
    try_num = 0
    while not succeeded:
        try_num += 1
        if try_num == retries:
            raise Exception("Failed to connect to Dask client")
        try:
            client = Client(scheduler, timeout=60)
            succeeded = True
        except Exception as e:  # pylint: disable=broad-except
            print(e)
        time.sleep(15)

    return client
