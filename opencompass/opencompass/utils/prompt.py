# from __future__ import annotations

# import hashlib
# import json
# from copy import deepcopy
# from typing import Dict, List, Union

# from mmengine.config import ConfigDict


# def safe_format(input_str: str, **kwargs) -> str:
#     """Safely formats a string with the given keyword arguments. If a keyword
#     is not found in the string, it will be ignored.

#     Args:
#         input_str (str): The string to be formatted.
#         **kwargs: The keyword arguments to be used for formatting.

#     Returns:
#         str: The formatted string.
#     """
#     # import re
#     # segs = [input_str]
#     # for k, v in kwargs.items():
#     #     regex = re.compile(f'(?<={{{k}}})(?={{{k}}})|({{{k}}})')
#     #     segs = [regex.split(seg) for seg in segs]
#     #     segs = sum(segs, [])
#     # replace_dict = {f'{{{k}}}': str(v) for k, v in kwargs.items()}
#     # segs = [replace_dict.get(seg, seg) for seg in segs]
#     # output_str = ''.join(segs)
#     # return output_str

#     for k, v in kwargs.items():
#         input_str = input_str.replace(f'{{{k}}}', str(v))
#     return input_str


# def get_prompt_hash(dataset_cfg: Union[ConfigDict, List[ConfigDict]]) -> str:
#     """Get the hash of the prompt configuration.

#     Args:
#         dataset_cfg (ConfigDict or list[ConfigDict]): The dataset
#             configuration.

#     Returns:
#         str: The hash of the prompt configuration.
#     """
#     if isinstance(dataset_cfg, list):
#         if len(dataset_cfg) == 1:
#             dataset_cfg = dataset_cfg[0]
#         else:
#             hashes = ','.join([get_prompt_hash(cfg) for cfg in dataset_cfg])
#             hash_object = hashlib.sha256(hashes.encode())
#             return hash_object.hexdigest()
#     if 'reader_cfg' in dataset_cfg.infer_cfg:
#         # new config
#         reader_cfg = dict(type='DatasetReader',
#                           input_columns=dataset_cfg.reader_cfg.input_columns,
#                           output_column=dataset_cfg.reader_cfg.output_column)
#         dataset_cfg.infer_cfg.reader = reader_cfg
#         if 'train_split' in dataset_cfg.infer_cfg.reader_cfg:
#             dataset_cfg.infer_cfg.retriever[
#                 'index_split'] = dataset_cfg.infer_cfg['reader_cfg'][
#                     'train_split']
#         if 'test_split' in dataset_cfg.infer_cfg.reader_cfg:
#             dataset_cfg.infer_cfg.retriever[
#                 'test_split'] = dataset_cfg.infer_cfg.reader_cfg.test_split
#         for k, v in dataset_cfg.infer_cfg.items():
#             dataset_cfg.infer_cfg[k]['type'] = v['type'].split('.')[-1]
#     # A compromise for the hash consistency
#     if 'fix_id_list' in dataset_cfg.infer_cfg.retriever:
#         fix_id_list = dataset_cfg.infer_cfg.retriever.pop('fix_id_list')
#         dataset_cfg.infer_cfg.inferencer['fix_id_list'] = fix_id_list
#     d_json = json.dumps(dataset_cfg.infer_cfg.to_dict(), sort_keys=True)
#     hash_object = hashlib.sha256(d_json.encode())
#     return hash_object.hexdigest()


# class PromptList(list):
#     """An enhanced list, used for intermidate representation of a prompt."""

#     def format(self, **kwargs) -> PromptList:
#         """Replaces all instances of 'src' in the PromptList with 'dst'.

#         Args:
#             src (str): The string to be replaced.
#             dst (PromptType): The string or PromptList to replace with.

#         Returns:
#             PromptList: A new PromptList with 'src' replaced by 'dst'.

#         Raises:
#             TypeError: If 'dst' is a PromptList and 'src' is in a dictionary's
#             'prompt' key.
#         """
#         new_list = PromptList()
#         for item in self:
#             if isinstance(item, Dict):
#                 new_item = deepcopy(item)
#                 if 'prompt' in item:
#                     new_item['prompt'] = safe_format(item['prompt'], **kwargs)
#                 new_list.append(new_item)
#             else:
#                 new_list.append(safe_format(item, **kwargs))
#         return new_list

#     def replace(self, src: str, dst: Union[str, PromptList]) -> PromptList:
#         """Replaces all instances of 'src' in the PromptList with 'dst'.

#         Args:
#             src (str): The string to be replaced.
#             dst (PromptType): The string or PromptList to replace with.

#         Returns:
#             PromptList: A new PromptList with 'src' replaced by 'dst'.

#         Raises:
#             TypeError: If 'dst' is a PromptList and 'src' is in a dictionary's
#             'prompt' key.
#         """
#         new_list = PromptList()
#         for item in self:
#             if isinstance(item, str):
#                 if isinstance(dst, str):
#                     new_list.append(item.replace(src, dst))
#                 elif isinstance(dst, PromptList):
#                     split_str = item.split(src)
#                     for i, split_item in enumerate(split_str):
#                         if split_item:
#                             new_list.append(split_item)
#                         if i < len(split_str) - 1:
#                             new_list += dst
#             elif isinstance(item, Dict):
#                 new_item = deepcopy(item)
#                 if 'prompt' in item:
#                     if src in item['prompt']:
#                         if isinstance(dst, PromptList):
#                             raise TypeError(
#                                 f'Found keyword {src} in a dictionary\'s '
#                                 'prompt key. Cannot replace with a '
#                                 'PromptList.')
#                         new_item['prompt'] = new_item['prompt'].replace(
#                             src, dst)
#                 new_list.append(new_item)
#             else:
#                 new_list.append(item.replace(src, dst))
#         return new_list

#     def __add__(self, other: Union[str, PromptList]) -> PromptList:
#         """Adds a string or another PromptList to this PromptList.

#         Args:
#             other (PromptType): The string or PromptList to be added.

#         Returns:
#             PromptList: A new PromptList that is the result of the addition.
#         """
#         if not other:
#             return PromptList([*self])
#         if isinstance(other, str):
#             return PromptList(self + [other])
#         else:
#             return PromptList(super().__add__(other))

#     def __radd__(self, other: Union[str, PromptList]) -> PromptList:
#         """Implements addition when the PromptList is on the right side of the
#         '+' operator.

#         Args:
#             other (PromptType): The string or PromptList to be added.

#         Returns:
#             PromptList: A new PromptList that is the result of the addition.
#         """
#         if not other:
#             return PromptList([*self])
#         if isinstance(other, str):
#             return PromptList([other, *self])
#         else:
#             return PromptList(other + self)

#     def __iadd__(self, other: Union[str, PromptList]) -> PromptList:
#         """Implements in-place addition for the PromptList.

#         Args:
#             other (PromptType): The string or PromptList to be added.

#         Returns:
#             PromptList: The updated PromptList.
#         """
#         if not other:
#             return self
#         if isinstance(other, str):
#             self.append(other)
#         else:
#             super().__iadd__(other)
#         return self

#     def __str__(self) -> str:
#         """Converts the PromptList into a string.

#         Returns:
#             str: The string representation of the PromptList.

#         Raises:
#             TypeError: If there's an item in the PromptList that is not a
#             string or dictionary.
#         """
#         res = []
#         for item in self:
#             if isinstance(item, str):
#                 res.append(item)
#             elif isinstance(item, dict):
#                 if 'prompt' in item:
#                     res.append(item['prompt'])
#             else:
#                 raise TypeError('Invalid type in prompt list when '
#                                 'converting to string')
#         return ''.join(res)



from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import List, Union

from mmengine.config import ConfigDict


def safe_format(input_str, **kwargs) -> str:
    """Safely formats a (possibly non-string) input by first coercing it to text.
    Any "{key}" in the text will be replaced by the provided kwargs.
    Missing keys are ignored (left as-is).
    """
    s = _to_text(input_str)
    for k, v in kwargs.items():
        s = s.replace(f'{{{k}}}', str(v))
    return s


def get_prompt_hash(dataset_cfg: Union[ConfigDict, List[ConfigDict]]) -> str:
    """Get the hash of the prompt configuration.

    Args:
        dataset_cfg (ConfigDict or list[ConfigDict]): The dataset configuration.

    Returns:
        str: The hash of the prompt configuration.
    """
    if isinstance(dataset_cfg, list):
        if len(dataset_cfg) == 1:
            dataset_cfg = dataset_cfg[0]
        else:
            hashes = ','.join([get_prompt_hash(cfg) for cfg in dataset_cfg])
            hash_object = hashlib.sha256(hashes.encode())
            return hash_object.hexdigest()
    if 'reader_cfg' in dataset_cfg.infer_cfg:
        # new config
        reader_cfg = dict(type='DatasetReader',
                          input_columns=dataset_cfg.reader_cfg.input_columns,
                          output_column=dataset_cfg.reader_cfg.output_column)
        dataset_cfg.infer_cfg.reader = reader_cfg
        if 'train_split' in dataset_cfg.infer_cfg.reader_cfg:
            dataset_cfg.infer_cfg.retriever[
                'index_split'] = dataset_cfg.infer_cfg['reader_cfg'][
                    'train_split']
        if 'test_split' in dataset_cfg.infer_cfg.reader_cfg:
            dataset_cfg.infer_cfg.retriever[
                'test_split'] = dataset_cfg.infer_cfg.reader_cfg.test_split
        for k, v in dataset_cfg.infer_cfg.items():
            dataset_cfg.infer_cfg[k]['type'] = v['type'].split('.')[-1]
    # A compromise for the hash consistency
    if 'fix_id_list' in dataset_cfg.infer_cfg.retriever:
        fix_id_list = dataset_cfg.infer_cfg.retriever.pop('fix_id_list')
        dataset_cfg.infer_cfg.inferencer['fix_id_list'] = fix_id_list
    d_json = json.dumps(dataset_cfg.infer_cfg.to_dict(), sort_keys=True)
    hash_object = hashlib.sha256(d_json.encode())
    return hash_object.hexdigest()


def _to_text(item) -> str:
    """Recursively coerce nested prompt structures to a single string.

    Handles:
      - str (returned as-is)
      - dicts with a 'prompt' key (flattened)
      - dicts with multi-modal content:
          * {'type':'text','text':'...'}
          * {'type':'image_url','image':'...'} or {'type':'image_url','image_url': {'url': '...'}}
          * {'content':[ ... content items ... ]}
      - PromptList / list / tuple (concatenates their stringified elements)
      - all other types via str(item)
    """
    if isinstance(item, str):
        return item

    # PromptList subclasses list, so list handling covers PromptList too.
    if isinstance(item, (list, tuple)):
        return ''.join(_to_text(x) for x in item)

    if isinstance(item, dict):
        # {'prompt': ...}
        if 'prompt' in item:
            return _to_text(item['prompt'])

        # {'content': [ {...}, {...} ]}
        if 'content' in item:
            content = item.get('content', [])
            if isinstance(content, (list, tuple)):
                return ''.join(_to_text(x) for x in content)
            return _to_text(content)

        # Single content part
        if 'type' in item:
            t = item.get('type')
            if t == 'text':
                return str(item.get('text', ''))
            if t in ('image_url', 'image'):
                # support both {'image': url} and {'image_url': {'url': url}} shapes
                url = item.get('image') or item.get('image_url')
                if isinstance(url, dict):
                    url = url.get('url')
                # Represent images as a short token so joining never fails
                if url:
                    return f'\n[IMAGE:{url}]\n'
                else:
                    return '\n[IMAGE]\n'

        # Unknown dict form: be conservative to avoid "dict" -> str(...) noise
        return ''

    # Fallback
    return str(item)


class PromptList(list):
    """An enhanced list, used for intermediate representation of a prompt."""

    def format(self, **kwargs) -> 'PromptList':
        """Safely format placeholder variables across the structure.

        Supports nested PromptList/list/dict-with-'prompt', and multi-modal
        dicts with 'content' or single parts with 'type':'text'.
        """
        new_list = PromptList()
        for item in self:
            if isinstance(item, dict):
                new_item = deepcopy(item)

                # {'prompt': ...}
                if 'prompt' in item:
                    p = item['prompt']
                    if isinstance(p, (PromptList, list)):
                        new_item['prompt'] = PromptList(p).format(**kwargs)
                    else:
                        new_item['prompt'] = safe_format(p, **kwargs)

                # {'content': [parts ...]}
                elif 'content' in item and isinstance(item['content'], list):
                    new_content = []
                    for part in item['content']:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            part = deepcopy(part)
                            part['text'] = safe_format(part.get('text', ''), **kwargs)
                            new_content.append(part)
                        else:
                            new_content.append(part)
                    new_item['content'] = new_content

                # single part {'type':'text','text':...}
                elif item.get('type') == 'text':
                    new_item['text'] = safe_format(item.get('text', ''), **kwargs)

                new_list.append(new_item)

            elif isinstance(item, (PromptList, list)):
                new_list += PromptList(item).format(**kwargs)

            else:
                new_list.append(safe_format(item, **kwargs))
        return new_list

    def replace(self, src: str, dst: Union[str, 'PromptList']) -> 'PromptList':
        """Replaces all instances of 'src' in the PromptList with 'dst'.

        Handles nested PromptLists and multi-modal dicts gracefully.
        """
        new_list = PromptList()
        for item in self:
            if isinstance(item, str):
                if isinstance(dst, str):
                    new_list.append(item.replace(src, dst))
                elif isinstance(dst, PromptList):
                    split_str = item.split(src)
                    for i, split_item in enumerate(split_str):
                        if split_item:
                            new_list.append(split_item)
                        if i < len(split_str) - 1:
                            new_list += dst

            elif isinstance(item, dict):
                new_item = deepcopy(item)

                # {'prompt': ...}
                if 'prompt' in item:
                    p = item['prompt']
                    text_p = _to_text(p)
                    if src in text_p:
                        if isinstance(dst, PromptList):
                            pl = PromptList(p) if isinstance(p, (list, PromptList)) else PromptList([text_p])
                            new_item['prompt'] = pl.replace(src, dst)
                        else:
                            new_item['prompt'] = text_p.replace(src, dst)

                # {'content': [parts ...]}
                elif 'content' in item and isinstance(item['content'], list):
                    new_content = []
                    for part in item['content']:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            part = deepcopy(part)
                            txt = part.get('text', '')
                            if isinstance(dst, PromptList):
                                split_str = txt.split(src)
                                tmp_pl = PromptList()
                                for i, seg in enumerate(split_str):
                                    if seg:
                                        tmp_pl.append(seg)
                                    if i < len(split_str) - 1:
                                        tmp_pl += dst
                                part['text'] = str(tmp_pl)  # flattened
                            else:
                                part['text'] = txt.replace(src, dst)
                            new_content.append(part)
                        else:
                            new_content.append(part)
                    new_item['content'] = new_content

                # single part {'type':'text','text':...}
                elif item.get('type') == 'text':
                    txt = item.get('text', '')
                    if isinstance(dst, PromptList):
                        split_str = txt.split(src)
                        tmp_pl = PromptList()
                        for i, seg in enumerate(split_str):
                            if seg:
                                tmp_pl.append(seg)
                            if i < len(split_str) - 1:
                                tmp_pl += dst
                        new_item['text'] = str(tmp_pl)
                    else:
                        new_item['text'] = txt.replace(src, dst)

                new_list.append(new_item)

            elif isinstance(item, (PromptList, list)):
                new_list += PromptList(item).replace(src, dst)

            else:
                new_list.append(_to_text(item).replace(src, dst))
        return new_list

    def __add__(self, other: Union[str, 'PromptList']) -> 'PromptList':
        if not other:
            return PromptList([*self])
        if isinstance(other, str):
            return PromptList(self + [other])
        else:
            return PromptList(super().__add__(other))

    def __radd__(self, other: Union[str, 'PromptList']) -> 'PromptList':
        if not other:
            return PromptList([*self])
        if isinstance(other, str):
            return PromptList([other, *self])
        else:
            return PromptList(other + self)

    def __iadd__(self, other: Union[str, 'PromptList']) -> 'PromptList':
        if not other:
            return self
        if isinstance(other, str):
            self.append(other)
        else:
            super().__iadd__(other)
        return self

    def __str__(self) -> str:
        """Flattens the PromptList to plain text, understanding multi-modal parts.

        This prevents "TypeError: expected str instance, list found" when
        elements include lists, PromptLists, or dicts with content/image parts.
        """
        res = []
        for item in self:
            res.append(_to_text(item))
        return ''.join(res)

