# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Horizon Robotics and Hobot Contributors. All Rights Reserved.
# Copyright (c) 2025 Horizon Robotics and Hobot Contributors. All Rights Reserved.
# Modified for Scale VLA RL.
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

import alf
import numpy as np
import torch
from typing import List, Union


class UTF8Encoder(object):
    """A simple utf-8 encoder that converts a string to an uint8 numpy array.
    """

    def __init__(self, max_bytes_num: int = 200):
        """
        Args:
            max_bytes_num: the maximum number of encoded bytes for each
                string. If more than this number, the byte sequence will be truncated
                without warnings. Zero bytes will be padded to shorter sequences.
        """
        assert max_bytes_num > 0
        self._max_bytes = max_bytes_num

    @property
    def instruction_spec(self):
        return alf.BoundedTensorSpec((self._max_bytes, ),
                                     maximum=255,
                                     dtype='uint8')

    def encode(self, instr: Union[str, List[str]]) -> np.ndarray:
        """Convert a string or list of strings to an unsigned bytes array.
        Args:
            instr: An instruction string or a list of instruction strings.
        Returns:
            np.ndarray:
                - If input is a single string: a 1D numpy array of type uint8 of length `self._max_bytes`.
                - If input is a list of strings: a 2D numpy array of shape (len(instr), self._max_bytes),
                  where each row is an encoded and padded/truncated byte sequence.
        """

        def encode_single(s: str) -> np.ndarray:
            s = s.lower()
            arr = np.zeros([self._max_bytes], dtype=np.uint8)
            bytes = np.array(list(s.encode(encoding='utf-8')), dtype=np.uint8)
            bytes = bytes[:self._max_bytes]
            arr[:bytes.size] = bytes
            return arr

        if isinstance(instr, str):
            return encode_single(instr)
        elif isinstance(instr, list):
            return np.stack([encode_single(s) for s in instr])
        else:
            raise TypeError("instr must be a string or a list of strings")

    @classmethod
    def decode(cls, codes: Union[torch.Tensor, np.ndarray]) -> List[str]:
        """Decode a tensor of utf-8 bytes into strings.

        A byte sequence might have been zero-padded to ``self._max_bytes``. In this
        case, zeros will be first discarded before decoding. If a sequence consists
        of only zeros, an empty string will be returned.

        Args:
            codes: a batch of utf-8 bytes with shape ``[B,self._max_bytes]``.

        Returns:
            List[str]: a string list whose length is ``B``. Each string has
                a length no greater than ``self._max_bytes``.
        """
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().numpy()
        strs = []
        for i in range(codes.shape[0]):
            s = codes[i]
            # Find out the first zero padding index
            idx, = np.where(s == 0)
            if idx.size > 0:
                s = s[:idx[0]]
            if s.size == 0:
                # Zero padding starts at the beginning -> empty string
                strs.append('')
            else:
                s = s.view(f'|S{s.size}')
                strs.append(str(s, 'utf-8'))
        return strs
