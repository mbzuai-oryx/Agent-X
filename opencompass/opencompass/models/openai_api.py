
# import json
# import os
# import re
# import time
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
# from typing import Dict, List, Optional, Union

# import jieba
# import requests

# from opencompass.registry import MODELS
# from opencompass.utils.prompt import PromptList

# from .base_api import BaseAPIModel

# PromptType = Union[PromptList, str]
# OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


# @MODELS.register_module()
# class OpenAI(BaseAPIModel):
#     """Model wrapper around OpenAI's models.

#     Args:
#         path (str): The name of OpenAI's model.
#         max_seq_len (int): The maximum allowed sequence length of a model.
#             Note that the length of prompt + generated tokens shall not exceed
#             this value. Defaults to 2048.
#         query_per_second (int): The maximum queries allowed per second
#             between two consecutive calls of the API. Defaults to 1.
#         retry (int): Number of retires if the API call fails. Defaults to 2.
#         key (str or List[str]): OpenAI key(s). In particular, when it
#             is set to "ENV", the key will be fetched from the environment
#             variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
#             list, the keys will be used in round-robin manner. Defaults to
#             'ENV'.
#         org (str or List[str], optional): OpenAI organization(s). If not
#             specified, OpenAI uses the default organization bound to each API
#             key. If specified, the orgs will be posted with each request in
#             round-robin manner. Defaults to None.
#         meta_template (Dict, optional): The model's meta prompt template.
#         openai_api_base (str): The base url of OpenAI's API. Defaults to
#             'https://api.openai.com/v1/chat/completions'.
#         mode (str, optional): The method of input truncation when input length
#             exceeds max_seq_len. 'front','mid' and 'rear'. Defaults to 'none'.
#         temperature (float, optional): What sampling temperature to use.
#             If not None, overrides the temperature in `generate()`.
#         **gen_params: Other keyword arguments for generation payload.
#     """

#     is_api: bool = True

#     def __init__(self,
#                  path: str = 'gpt-3.5-turbo',
#                  max_seq_len: int = 4096,
#                  query_per_second: int = 1,
#                  rpm_verbose: bool = False,
#                  retry: int = 2,
#                  key: Union[str, List[str]] = 'ENV',
#                  org: Optional[Union[str, List[str]]] = None,
#                  meta_template: Optional[Dict] = None,
#                  openai_api_base: str = OPENAI_API_BASE,
#                  mode: str = 'none',
#                  temperature: Optional[float] = None,
#                  **gen_params):

#         super().__init__(path=path,
#                          max_seq_len=max_seq_len,
#                          meta_template=meta_template,
#                          query_per_second=query_per_second,
#                          rpm_verbose=rpm_verbose,
#                          retry=retry)
#         import tiktoken
#         self.tiktoken = tiktoken
#         self.temperature = temperature
#         assert mode in ['none', 'front', 'mid', 'rear']
#         self.mode = mode
#         self.gen_params = gen_params

#         if isinstance(key, str):
#             key = os.getenv('OPENAI_API_KEY') if key == 'ENV' else key
#             self.keys = [os.getenv('ALLES_API_KEY') if key == 'alles' else key]
#         else:
#             self.keys = key

#         # record invalid keys and skip them when requesting API
#         # - keys have insufficient_quota
#         self.invalid_keys = set()

#         self.key_ctr = 0
#         if isinstance(org, str):
#             self.orgs = [org]
#         else:
#             self.orgs = org
#         self.org_ctr = 0
#         self.url = openai_api_base
#         self.path = path

#     def generate(
#         self,
#         inputs: List[Union[str, PromptList]],
#         max_out_len: int = 512,
#         temperature: float = 0.7,
#     ) -> List[str]:
#         """Generate results given a list of inputs."""
#         if self.temperature is not None:
#             temperature = self.temperature

#         with ThreadPoolExecutor() as executor:
#             results = list(
#                 executor.map(self._generate, inputs,
#                              [max_out_len] * len(inputs),
#                              [temperature] * len(inputs)))
#         return results

#     def _generate(self, input: Union[str, PromptList], max_out_len: int,
#                   temperature: float) -> str:
#         """Generate result for a single input."""

#         assert isinstance(input, (str, PromptList))

#         # --- helpers for multi-modal messages ---
#         def _normalize_image_url(item) -> Optional[Dict]:
#             """Return an OpenAI image_url content part or None.

#             Accepts {'type':'image_url','image':'...'} or
#             {'type':'image_url','image_url': {'url': '...'}} or
#             {'type':'image','image': '...'}.
#             """
#             if not isinstance(item, dict):
#                 return None
#             t = item.get('type')
#             if t not in ('image_url', 'image'):
#                 return None
#             url = item.get('image')
#             if not url:
#                 img = item.get('image_url')
#                 if isinstance(img, dict):
#                     url = img.get('url')
#                 elif isinstance(img, str):
#                     url = img
#             if not url:
#                 return None
#             return {'type': 'image_url', 'image_url': {'url': url}}

#         def _to_openai_content(obj):
#             """Coerce dataset shapes to OpenAI Chat 'content' parts."""
#             # Already a list of content parts?
#             if isinstance(obj, list):
#                 parts = []
#                 for p in obj:
#                     if isinstance(p, dict):
#                         if p.get('type') == 'text':
#                             parts.append({'type': 'text', 'text': str(p.get('text', ''))})
#                         else:
#                             im = _normalize_image_url(p)
#                             if im:
#                                 parts.append(im)
#                             else:
#                                 # Unknown dict: stringify to text to avoid crashes
#                                 parts.append({'type': 'text', 'text': str(p)})
#                     else:
#                         parts.append({'type': 'text', 'text': str(p)})
#                 return parts
#             # A dict that looks like {'content': [...]}
#             if isinstance(obj, dict) and 'content' in obj:
#                 return _to_openai_content(obj['content'])
#             # A dict single part {'type':...}
#             if isinstance(obj, dict) and 'type' in obj:
#                 im = _normalize_image_url(obj)
#                 if im:
#                     return [im]
#                 if obj.get('type') == 'text':
#                     return [{'type': 'text', 'text': str(obj.get('text', ''))}]
#             # Fallback: plain text
#             return str(obj)

#         def _messages_from_promptlist(pl: PromptList) -> List[Dict]:
#             """Build OpenAI Chat 'messages' from PromptList supporting content lists."""
#             role_map = {'HUMAN': 'user', 'BOT': 'assistant', 'SYSTEM': 'system'}
#             messages = []
#             for item in pl:
#                 # expected: {'role': 'HUMAN'|'BOT'|'SYSTEM', 'prompt': <str|dict|list>}
#                 if isinstance(item, dict):
#                     role = role_map.get(item.get('role', 'HUMAN'), 'user')
#                     payload = item.get('prompt', item)
#                     content = _to_openai_content(payload)
#                     messages.append({'role': role, 'content': content})
#                 elif isinstance(item, str):
#                     messages.append({'role': 'user', 'content': item})
#                 else:
#                     # last-resort stringify
#                     messages.append({'role': 'user', 'content': str(item)})
#             return messages

#         def _text_for_token_count(messages: List[Dict]) -> str:
#             """Gather only text from message content for token estimation."""
#             buf = []
#             for m in messages:
#                 c = m.get('content', '')
#                 if isinstance(c, str):
#                     buf.append(c)
#                 elif isinstance(c, list):
#                     for part in c:
#                         if isinstance(part, dict) and part.get('type') == 'text':
#                             buf.append(part.get('text', ''))
#                 # images/other types are ignored for token estimation
#             return ' '.join(buf)

#         # max num token for gpt-3.5-turbo is 4097
#         context_window = 4096
#         if '32k' in self.path:
#             context_window = 32768
#         elif '16k' in self.path:
#             context_window = 16384
#         elif 'gpt-4' in self.path:
#             context_window = 8192

#         # will leave 100 tokens as prompt buffer, triggered if input is str
#         if isinstance(input, str) and self.mode != 'none':
#             context_window = self.max_seq_len
#             input = self.bin_trim(input, context_window - 100 - max_out_len)

#         # Build OpenAI Chat-format messages
#         if isinstance(input, str):
#             messages = [{'role': 'user', 'content': input}]
#         else:
#             messages = _messages_from_promptlist(input)

#         # Hold out 100 tokens due to potential errors in tiktoken calculation
#         prompt_text = _text_for_token_count(messages)
#         max_out_len = min(
#             max_out_len, context_window - self.get_token_len(prompt_text) - 100)
#         if max_out_len <= 0:
#             return ''

#         max_num_retries = 0
#         while max_num_retries < self.retry:
#             self.wait()

#             with Lock():
#                 if len(self.invalid_keys) == len(self.keys):
#                     raise RuntimeError('All keys have insufficient quota.')

#                 # find the next valid key
#                 while True:
#                     self.key_ctr += 1
#                     if self.key_ctr == len(self.keys):
#                         self.key_ctr = 0

#                     if self.keys[self.key_ctr] not in self.invalid_keys:
#                         break

#                 key = self.keys[self.key_ctr]

#             header = {
#                 'Authorization': f'Bearer {key}',
#                 'content-type': 'application/json',
#                 'alles-apin-token': key,
#             }

#             if self.orgs:
#                 with Lock():
#                     self.org_ctr += 1
#                     if self.org_ctr == len(self.orgs):
#                         self.org_ctr = 0
#                 header['OpenAI-Organization'] = self.orgs[self.org_ctr]

#             try:
#                 data = dict(
#                     model=self.path,
#                     messages=messages,
#                     max_tokens=max_out_len,
#                     n=1,
#                     stop=None,
#                     temperature=temperature,
#                 )
#                 data = {**data, **self.gen_params}
#                 raw_response = requests.post(self.url,
#                                              headers=header,
#                                              data=json.dumps(data))
#             except requests.ConnectionError:
#                 self.logger.error('Got connection error, retrying...')
#                 continue
#             try:
#                 response = raw_response.json()
#             except requests.JSONDecodeError:
#                 self.logger.error('JsonDecode error, got',
#                                   str(raw_response.content))
#                 continue
#             try:
#                 response = response.get('data', response)
#                 return response['choices'][0]['message']['content'].strip()
#             except KeyError:
#                 if 'error' in response:
#                     if response['error'].get('code') == 'rate_limit_exceeded':
#                         time.sleep(1)
#                         self.logger.warn('Rate limit exceeded, retrying...')
#                         continue
#                     elif response['error'].get('code') == 'insufficient_quota':
#                         self.invalid_keys.add(key)
#                         self.logger.warn(f'insufficient_quota key: {key}')
#                         continue

#                     self.logger.error('Find error message in response: ',
#                                       str(response['error']))
#             except TypeError:
#                 self.logger.error('Error response: ', str(response))
#             max_num_retries += 1

#         raise RuntimeError('Calling OpenAI failed after retrying for '
#                            f'{max_num_retries} times. Check the logs for '
#                            'details.')

#     def get_token_len(self, prompt: str) -> int:
#         """Get lengths of the tokenized string. Only English and Chinese
#         characters are counted for now. Users are encouraged to override this
#         method if more accurate length is needed.

#         Args:
#             prompt (str): Input string.

#         Returns:
#             int: Length of the input tokens
#         """
#         if self.path in self.tiktoken.model.MODEL_TO_ENCODING:
#             enc = self.tiktoken.encoding_for_model(self.path)
#         else:
#             # Defaults to use the tokenizer of GPT-4
#             enc = self.tiktoken.encoding_for_model('gpt-4')
#         return len(enc.encode(prompt))

#     def bin_trim(self, prompt: str, num_token: int) -> str:
#         """Get a suffix of prompt which is no longer than num_token tokens.

#         Args:
#             prompt (str): Input string.
#             num_token (int): The upper bound of token numbers.

#         Returns:
#             str: The trimmed prompt.
#         """
#         token_len = self.get_token_len(prompt)
#         if token_len <= num_token:
#             return prompt
#         pattern = re.compile(r'[\u4e00-\u9fa5]')
#         if pattern.search(prompt):
#             words = list(jieba.cut(prompt, cut_all=False))
#             sep = ''
#         else:
#             words = prompt.split(' ')
#             sep = ' '

#         l, r = 1, len(words)
#         while l + 2 < r:
#             mid = (l + r) // 2
#             if self.mode == 'front':
#                 cur_prompt = sep.join(words[-mid:])
#             elif self.mode == 'mid':
#                 cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
#             elif self.mode == 'rear':
#                 cur_prompt = sep.join(words[:mid])

#             if self.get_token_len(cur_prompt) <= num_token:
#                 l = mid  # noqa: E741
#             else:
#                 r = mid

#         if self.mode == 'front':
#             prompt = sep.join(words[-l:])
#         elif self.mode == 'mid':
#             prompt = sep.join(words[:l]) + sep.join(words[-l:])
#         elif self.mode == 'rear':
#             prompt = sep.join(words[:l])
#         return prompt





import json
import os
import re
import time
from uuid import uuid4
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


@MODELS.register_module()
class OpenAI(BaseAPIModel):
    """Model wrapper around OpenAI/LMDeploy OpenAI-compatible chat API.

    Adds automatic LMDeploy session cleanup for one-shot requests via `extra_body`:
      - unique `session_id` per request
      - `sequence_start: true`
      - `sequence_end: true`
      - optional `renew_session: true` every `flush_every` requests

    Args:
        path (str): Model name for the API server.
        max_seq_len (int): Max prompt+generation tokens budget (client-side).
        query_per_second (int): Throttle.
        rpm_verbose (bool): If True, prints RPS info.
        retry (int): Retries on transient failures.
        key (str|List[str]): API key(s). If 'ENV', uses $OPENAI_API_KEY.
        org (str|List[str], optional): Org(s) to round-robin.
        meta_template (Dict, optional): Optional instruction wrapper.
        openai_api_base (str): Base URL for /v1/chat/completions.
        mode (str): Input truncation strategy: 'none'|'front'|'mid'|'rear'.
        temperature (float, optional): Default temperature if provided.
        flush_every (int): Add `renew_session: true` every N requests (N>=1).
                          Use 1 to end every request cleanly (recommended).
        **gen_params: Extra JSON merged into the request body.
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 mode: str = 'none',
                 temperature: Optional[float] = None,
                 flush_every: int = 1,
                 **gen_params):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.gen_params = gen_params

        if isinstance(key, str):
            key = os.getenv('OPENAI_API_KEY') if key == 'ENV' else key
            self.keys = [os.getenv('ALLES_API_KEY') if key == 'alles' else key]
        else:
            self.keys = key

        # Track keys that hit insufficient quota to skip them
        self.invalid_keys = set()
        self.key_ctr = 0

        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0

        self.url = openai_api_base
        self.path = path

        # Session/flush accounting for LMDeploy extras
        self._flush_every = max(1, int(flush_every))
        self._req_ctr = 0
        self._ctr_lock = Lock()

    def generate(
        self,
        inputs: List[Union[str, PromptList]],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: Union[str, PromptList], max_out_len: int,
                  temperature: float) -> str:
        """Generate result for a single input."""
        assert isinstance(input, (str, PromptList))

        # --- helpers for multi-modal messages ---
        def _normalize_image_url(item) -> Optional[Dict]:
            """Return an OpenAI image_url content part or None.

            Accepts {'type':'image_url','image':'...'} or
            {'type':'image_url','image_url': {'url': '...'}} or
            {'type':'image','image': '...'}.
            """
            if not isinstance(item, dict):
                return None
            t = item.get('type')
            if t not in ('image_url', 'image'):
                return None
            url = item.get('image')
            if not url:
                img = item.get('image_url')
                if isinstance(img, dict):
                    url = img.get('url')
                elif isinstance(img, str):
                    url = img
            if not url:
                return None
            # OpenAI-compatible providers (e.g. SambaNova) require local images
            # as base64 data URIs; pass through existing data:/http(s) URLs.
            if isinstance(url, str) and not url.startswith(
                    ('data:', 'http://', 'https://')):
                import base64
                import mimetypes
                import os
                path = url[7:] if url.startswith('file://') else url
                mime = mimetypes.guess_type(path)[0] or ''
                if os.path.isfile(path) and mime.startswith('image/'):
                    with open(path, 'rb') as _imgf:
                        b64 = base64.b64encode(_imgf.read()).decode('utf-8')
                    url = f'data:{mime};base64,{b64}'
                else:
                    # Missing file or non-image (e.g. a video that was not
                    # frame-extracted): drop it rather than send an invalid
                    # image_url that the provider will reject.
                    return None
            return {'type': 'image_url', 'image_url': {'url': url}}

        def _to_openai_content(obj):
            """Coerce dataset shapes to OpenAI Chat 'content' parts."""
            # Already a list of content parts?
            if isinstance(obj, list):
                parts = []
                for p in obj:
                    if isinstance(p, dict):
                        if p.get('type') == 'text':
                            parts.append({'type': 'text', 'text': str(p.get('text', ''))})
                        else:
                            im = _normalize_image_url(p)
                            if im:
                                parts.append(im)
                            else:
                                # Unknown dict: stringify to text to avoid crashes
                                parts.append({'type': 'text', 'text': str(p)})
                    else:
                        parts.append({'type': 'text', 'text': str(p)})
                return parts
            # A dict that looks like {'content': [...]}
            if isinstance(obj, dict) and 'content' in obj:
                return _to_openai_content(obj['content'])
            # A dict single part {'type':...}
            if isinstance(obj, dict) and 'type' in obj:
                im = _normalize_image_url(obj)
                if im:
                    return [im]
                if obj.get('type') == 'text':
                    return [{'type': 'text', 'text': str(obj.get('text', ''))}]
            # Fallback: plain text
            return str(obj)

        def _messages_from_promptlist(pl: PromptList) -> List[Dict]:
            """Build OpenAI Chat 'messages' from PromptList supporting content lists."""
            role_map = {'HUMAN': 'user', 'BOT': 'assistant', 'SYSTEM': 'system'}
            messages = []
            for item in pl:
                # expected: {'role': 'HUMAN'|'BOT'|'SYSTEM', 'prompt': <str|dict|list>}
                if isinstance(item, dict):
                    role = role_map.get(item.get('role', 'HUMAN'), 'user')
                    payload = item.get('prompt', item)
                    content = _to_openai_content(payload)
                    messages.append({'role': role, 'content': content})
                elif isinstance(item, str):
                    messages.append({'role': 'user', 'content': item})
                else:
                    # last-resort stringify
                    messages.append({'role': 'user', 'content': str(item)})
            return messages

        def _text_for_token_count(messages: List[Dict]) -> str:
            """Gather only text from message content for token estimation."""
            buf = []
            for m in messages:
                c = m.get('content', '')
                if isinstance(c, str):
                    buf.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            buf.append(part.get('text', ''))
                # images/other types are ignored for token estimation
            return ' '.join(buf)

        # Heuristic model context windows
        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192

        # Optional trimming if input is a plain string and mode != 'none'
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

        # Build OpenAI Chat-format messages
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = _messages_from_promptlist(input)

        # Token budget guard (text parts only, images ignored for count)
        prompt_text = _text_for_token_count(messages)
        max_out_len = min(
            max_out_len, context_window - self.get_token_len(prompt_text) - 100)
        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0
                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break
                key = self.keys[self.key_ctr]

            header = {
                'Authorization': f'Bearer {key}',
                'content-type': 'application/json',
                'alles-apin-token': key,
            }

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            try:
                # --- Build base payload ---
                data = dict(
                    model=self.path,
                    messages=messages,
                    max_tokens=max_out_len,
                    n=1,
                    stop=None,
                    temperature=temperature,
                )
                data = {**data, **self.gen_params}

                # --- LMDeploy session controls (only when not using OpenAI) ---
                # If you're pointing at LMDeploy (e.g., http://host:12580/v1/chat/completions),
                # add session cleanup flags so KV does not accumulate.
                if "openai.com" not in self.url:
                    with self._ctr_lock:
                        self._req_ctr += 1
                        req_id = self._req_ctr

                    auto_extra = {
                        "session_id": f"oc_{os.getpid()}_{threading.get_ident()}_{uuid4()}",
                        "sequence_start": True,
                        "sequence_end": True,
                        "request_output_len": max_out_len
                    }
                    # Optional: force a renewal every N requests (if you want a "flush" boundary)
                    if req_id % self._flush_every == 0:
                        auto_extra["renew_session"] = True

                    # Respect user-provided extra_body but ensure required flags are present.
                    user_extra = {}
                    if isinstance(self.gen_params.get("extra_body"), dict):
                        user_extra = self.gen_params["extra_body"]
                    # user values take precedence
                    extra_body = {**auto_extra, **user_extra}
                    data["extra_body"] = extra_body

                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                max_num_retries += 1
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got %s',
                                  str(raw_response.content))
                max_num_retries += 1
                continue
            try:
                response = response.get('data', response)
                return response['choices'][0]['message']['content'].strip()
            except KeyError:
                if 'error' in response:
                    code = None
                    if isinstance(response['error'], dict):
                        code = response['error'].get('code')
                    if code == 'rate_limit_exceeded':
                        time.sleep(1)
                        self.logger.warn('Rate limit exceeded, retrying...')
                        max_num_retries += 1
                        continue
                    elif code == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        max_num_retries += 1
                        continue

                    self.logger.error('Find error message in response: %s',
                                      str(response['error']))
            except TypeError:
                self.logger.error('Error response: %s', str(response))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def get_token_len(self, prompt: str) -> int:
        """Estimate token length using tiktoken."""
        if self.path in self.tiktoken.model.MODEL_TO_ENCODING:
            enc = self.tiktoken.encoding_for_model(self.path)
        else:
            # Defaults to use the tokenizer of GPT-4
            enc = self.tiktoken.encoding_for_model('gpt-4')
        return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Trim a prompt to <= num_token tokens per `mode`."""
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])
            else:
                cur_prompt = prompt  # 'none'

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt
