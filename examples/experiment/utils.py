# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========


def create_model_urls(server_config):
    """
    Create a list of model API URLs from server config.
    Supports:
      - string (e.g., "https://openrouter.ai/api/v1")
      - dict  (e.g., {"host": "127.0.0.1", "ports": [8000]})
      - list of dicts (multiple servers)
    """
    urls = []

    # ✅ Case 1: single string
    if isinstance(server_config, str):
        # Remove possible trailing slash, then ensure ends with /v1
        if server_config.endswith("/v1"):
            urls.append(server_config)
        else:
            urls.append(server_config.rstrip("/") + "/v1")
        return urls

    # ✅ Case 2: single dict
    if isinstance(server_config, dict):
        host = server_config.get("host", "127.0.0.1")
        ports = server_config.get("ports", [8000])
        for port in ports:
            urls.append(f"http://{host}:{port}/v1")
        return urls

    # ✅ Case 3: list of dicts
    if isinstance(server_config, list):
        for server in server_config:
            host = server["host"]
            for port in server["ports"]:
                urls.append(f"http://{host}:{port}/v1")
        return urls

    raise ValueError(f"Unsupported server_config format: {type(server_config)}")
