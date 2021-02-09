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

from functools import reduce
from operator import mul


def sum_aggregation(inputs):
    validatedInputs = []
    try:
        for tens in inputs:
            tens.to("cuda:0")
            validatedInputs.append(tens)
    except Exception as e:
        print(f"The following exeption occured: {str(e)}")
    return sum(validatedInputs)


def prod_aggregation(inputs):
    validatedInputs = []
    try:
        for tens in inputs:
            tens.to("cuda:0")
            validatedInputs.append(tens) 
    except Exception as e:
        print(f"The following exeption occured: {str(e)}")
    return reduce(mul, validatedInputs, 1)


str_to_aggregation = {
    'sum': sum_aggregation,
    'prod': prod_aggregation,
}
