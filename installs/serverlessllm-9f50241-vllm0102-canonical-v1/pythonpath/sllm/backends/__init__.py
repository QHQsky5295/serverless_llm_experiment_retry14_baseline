# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
from importlib import import_module
from typing import TYPE_CHECKING

from .dummy_backend import DummyBackend

if TYPE_CHECKING:
    from .transformers_backend import TransformersBackend
    from .vllm_backend import VllmBackend

__all__ = ["DummyBackend", "VllmBackend", "TransformersBackend"]


def __getattr__(name):
    if name == "VllmBackend":
        return import_module(".vllm_backend", __name__).VllmBackend
    if name == "TransformersBackend":
        return import_module(".transformers_backend", __name__).TransformersBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
