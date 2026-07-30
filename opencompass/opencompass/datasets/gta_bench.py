
# import json
# import os
# from pathlib import Path
# from typing import List
# import copy
# import re
# import ast
# import subprocess  # added for ffmpeg fallback

# import numpy as np
# from datasets.arrow_dataset import Dataset
# from sentence_transformers import SentenceTransformer, util

# from opencompass.openicl.icl_evaluator import BaseEvaluator
# from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

# from .base import BaseDataset

# # ----------------------------
# # Optional: OpenAI dependency
# # ----------------------------
# # - We try legacy openai.ChatCompletion first (as in your reference file).
# # - If unavailable, we try the new SDK (client.chat.completions.create).
# # - If neither is available or OPENAI_API_KEY is empty, judge metrics are skipped per sample.
# try:
#     import openai as _openai_pkg
#     _HAS_OPENAI = True
# except Exception:
#     _openai_pkg = None
#     _HAS_OPENAI = False

# # ----------------------------
# # Optional: OpenCV dependency
# # ----------------------------
# try:
#     import cv2  # preferred for frame seeking
# except Exception:
#     cv2 = None

# from urllib.parse import urlparse, unquote
# from urllib.request import url2pathname

# # Supported video extensions
# VIDEO_EXTS = {'.mp4', '.mov', '.avi'}


# def get_all_file_paths(directory: str) -> list:
#     file_paths = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             file_paths.append(file_path)
#     return file_paths


# # ----------------------------
# # Helpers for video -> frames
# # ----------------------------
# def _is_video_path(p: str) -> bool:
#     """Return True if path (or file:// URI) has a video extension we support."""
#     if not isinstance(p, str) or not p:
#         return False
#     # Accept file:// URIs too
#     if p.startswith("file://"):
#         try:
#             parsed = urlparse(p)
#             suffix = Path(parsed.path).suffix.lower()
#             return suffix in VIDEO_EXTS
#         except Exception:
#             return False
#     return Path(p).suffix.lower() in VIDEO_EXTS


# def _from_file_uri(p: str) -> str:
#     """Convert file:// URI to a local filesystem path, else return original string."""
#     if isinstance(p, str) and p.startswith("file://"):
#         parsed = urlparse(p)
#         # Convert percent-escapes and OS-specific path
#         path = url2pathname(unquote(parsed.path))
#         # On some systems, leading slash may already be correct; keep as is
#         return path
#     return p


# def _to_abs_path(p: str, root: str) -> str:
#     """Return absolute normalized path, interpreting relative paths as under `root`."""
#     if not p:
#         return p
#     p = _from_file_uri(p)
#     return str((Path(p) if os.path.isabs(p) else Path(root) / p).resolve())


# def _extract_frames_with_opencv(video_abs: str,
#                                 out_dir: str,
#                                 base_name: str,
#                                 timestamps_s: List[int]) -> List[str]:
#     """Extract frames with OpenCV; return absolute filesystem paths of saved frames (no file://)."""
#     paths = []
#     cap = cv2.VideoCapture(video_abs)
#     if not cap.isOpened():
#         return paths

#     fps = cap.get(cv2.CAP_PROP_FPS) or 0
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
#     duration = (frame_count / fps) if fps > 0 else None

#     idx = 1
#     for t in timestamps_s:
#         # Skip timestamps beyond duration if we know it
#         if duration is not None and t > duration + 0.05:
#             continue
#         cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             continue
#         out_path = os.path.join(out_dir, f"{base_name}_frame{idx}.jpg")
#         try:
#             cv2.imwrite(out_path, frame)
#             # return absolute filesystem paths (no file://)
#             paths.append(str(Path(out_path).resolve()))
#             idx += 1
#             if idx > 4:
#                 break
#         except Exception:
#             # If write fails, stop trying further to keep behavior predictable
#             break

#     cap.release()
#     return paths


# def _extract_frames_with_ffmpeg(video_abs: str,
#                                 out_dir: str,
#                                 base_name: str,
#                                 timestamps_s: List[int]) -> List[str]:
#     """Extract frames with ffmpeg CLI; return absolute filesystem paths of saved frames (no file://)."""
#     paths = []
#     idx = 1
#     for t in timestamps_s:
#         out_path = os.path.join(out_dir, f"{base_name}_frame{idx}.jpg")
#         cmd = [
#             'ffmpeg', '-y',
#             '-ss', str(t), '-i', video_abs,
#             '-frames:v', '1', '-q:v', '2',
#             out_path
#         ]
#         try:
#             subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#             paths.append(str(Path(out_path).resolve()))
#             idx += 1
#             if idx > 4:
#                 break
#         except Exception:
#             # If ffmpeg fails at some timestamp, stop trying further
#             break
#     return paths


# def _extract_frames(video_abs: str, timestamps_s: List[int]) -> List[str]:
#     """
#     Extract up to 4 frames at the provided timestamps (seconds) from `video_abs`.
#     Returns a list of **absolute filesystem paths** (no file://) in the same directory as the video.
#     Order is preserved (0s, 10s, 20s, 30s).
#     """
#     out_dir = str(Path(video_abs).parent)
#     base_name = Path(video_abs).stem

#     # Try OpenCV first
#     if cv2 is not None:
#         paths = _extract_frames_with_opencv(video_abs, out_dir, base_name, timestamps_s)
#         if paths:
#             return paths

#     # Fallback to ffmpeg if OpenCV missing or failed
#     return _extract_frames_with_ffmpeg(video_abs, out_dir, base_name, timestamps_s)


# def _maybe_replace_video_content(content, data_root: str):
#     """
#     If `content` (dict or list) contains an 'image' block that points to a video
#     (mp4/mov/avi), return a transformed **list** that preserves existing text and
#     non-video blocks, and injects a new:
#         {"type": "video", "video": ["/abs/path/frame1.jpg", ...]}
#     Frame paths are absolute filesystem paths (no file://).

#     If no video image is found, return None (no change).
#     """
#     def _convert_to_video_block(video_path: str):
#         video_abs = _to_abs_path(video_path, data_root)
#         # Up to 4 frames at 0s, 10s, 20s, 30s (timestamps beyond duration are skipped)
#         frame_paths = _extract_frames(video_abs, timestamps_s=[0, 10, 20, 30])
#         if not frame_paths:
#             return None
#         return {"type": "video", "video": frame_paths}

#     # Case 1: content is a single dict block
#     if isinstance(content, dict):
#         if content.get("type") == "image" and _is_video_path(content.get("image")):
#             new_blocks = []
#             # Preserve any inline text field if present
#             if isinstance(content.get("text"), str) and content["text"].strip():
#                 new_blocks.append({"type": "text", "text": content["text"]})
#             vb = _convert_to_video_block(content["image"])
#             if vb is not None:
#                 new_blocks.append(vb)
#                 return new_blocks if new_blocks else None

#     # Case 2: content is a list of blocks
#     if isinstance(content, list):
#         new_blocks = []
#         converted = False
#         for blk in content:
#             # Non-dict blocks are copied through unchanged
#             if not isinstance(blk, dict):
#                 new_blocks.append(blk)
#                 continue

#             btype = blk.get("type")
#             if btype == "text":
#                 # Preserve text blocks verbatim
#                 new_blocks.append(blk)
#             elif btype == "image" and _is_video_path(blk.get("image")) and not converted:
#                 vb = _convert_to_video_block(blk["image"])
#                 if vb is not None:
#                     # Preserve any adjacent text field inside the same block (if present)
#                     if isinstance(blk.get("text"), str) and blk["text"].strip():
#                         new_blocks.append({"type": "text", "text": blk["text"]})
#                     new_blocks.append(vb)
#                     converted = True
#                 # Do NOT append the original video image block (it's replaced by `video`)
#             else:
#                 # Keep non-video images and other block types unchanged
#                 new_blocks.append(blk)

#         if converted:
#             return new_blocks

#     # No change
#     return None


# def organize_dialogs(sample: dict, path: str) -> List[dict]:
#     """
#     Build dialogs with two modifications:
#       1) Keep existing tool/tool_call handling and absolute path rewriting for tool arguments.
#       2) NEW: If a user message has content with {"type":"image","image":<video>},
#          extract up to 4 frames (0s, 10s, 20s, 30s) next to the video and **inject**
#          a {"type":"video","video":[<abs frame paths>]} block while preserving the
#          original query text and other non-video blocks.
#     """
#     dialogs = []
#     file_paths = get_all_file_paths(path)

#     for item in sample['dialogs']:
#         if item['role'] == 'tool':
#             dialog = dict(
#                 role='tool',
#                 name=item['name'],
#                 content=item['content'],
#             )
#             dialogs.append(dialog)

#         elif item['role'] == 'assistant' and 'tool_calls' in item.keys():
#             dialog = copy.deepcopy(item)
#             for name, value in dialog['tool_calls'][0]['function']['arguments'].items():
#                 if isinstance(value, str) and os.path.join(path, value) in file_paths:
#                     dialog['tool_calls'][0]['function']['arguments'][name] = os.path.join(path, value)
#             dialogs.append(dialog)

#         elif item['role'] == 'user':
#             dialog = copy.deepcopy(item)
#             transformed = _maybe_replace_video_content(dialog.get('content'), path)
#             if transformed is not None:
#                 dialog['content'] = transformed
#             dialogs.append(dialog)

#         else:
#             dialogs.append(item)

#     return dialogs


# @LOAD_DATASET.register_module()
# class GTABenchDataset(BaseDataset):
#     """GTA Benchmark."""

#     @staticmethod
#     def load(path: str):
#         data_root = Path(path)
#         data_file = data_root / 'dataset.json'
#         assert os.path.exists(data_file), f'Path {path} does not exist.'

#         data = json.load(open(data_file))
#         data_list = []
#         for idx, item in data.items():
#             idx = int(idx)
#             tools = [
#                 dict(type='tool', name=tool['name']) for tool in item['tools']
#             ]
#             files = [
#                 dict(type='file',
#                      filetype=file['type'],
#                      path=str((data_root / file['path']).absolute()))
#                 for file in item['files']
#             ]
#             gt_answer = item['gt_answer']
#             sample = {
#                 'dialogs': json.dumps(organize_dialogs(item, str(data_root.absolute()))),
#                 'resources': json.dumps(tools + files),
#                 'gt_answer': json.dumps(gt_answer)
#             }
#             data_list.append(sample)
#         dataset = Dataset.from_list(data_list)

#         return dataset


# @ICL_EVALUATORS.register_module()
# class GTABenchEvaluator(BaseEvaluator):

#     def __init__(self, mode) -> None:
#         assert mode in ['every', 'every_with_gt']
#         self.mode = mode
#         self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

#         # --------- OpenAI judge setup ----------
#         self._judge_model = os.getenv('GTA_EVAL_GPT_JUDGE', 'gpt-4o')
#         self._openai_ready = False
#         self._use_legacy = False
#         self._client = None

#         if _HAS_OPENAI:
#             # Prefer legacy style if present (to match your reference code)
#             self._use_legacy = hasattr(_openai_pkg, 'ChatCompletion')
#             api_key_env = os.getenv('OPENAI_API_KEY', '')
#             try:
#                 if self._use_legacy:
#                     # legacy SDK
#                     if getattr(_openai_pkg, 'api_key', None) in (None, '') and api_key_env:
#                         _openai_pkg.api_key = api_key_env
#                     self._openai_ready = bool(getattr(_openai_pkg, 'api_key', None))
#                 else:
#                     # new SDK layout: instantiate a client
#                     try:
#                         from openai import OpenAI as _OpenAIClient
#                         self._client = _OpenAIClient(api_key=api_key_env if api_key_env else None)
#                         self._openai_ready = self._client is not None
#                     except Exception:
#                         self._openai_ready = False
#             except Exception:
#                 self._openai_ready = False

#     # ---------------------------
#     # Similarity helper (existing)
#     # ---------------------------
#     def bert_score(self, pred: str, gt: str) -> float:
#         pred_emb = self.sentence_model.encode(pred, convert_to_tensor=True)
#         gt_emb = self.sentence_model.encode(gt, convert_to_tensor=True)
#         score = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
#         return score[0][0]

#     # ---------------------------
#     # Formatting / extraction helpers for judge metrics
#     # ---------------------------
#     @staticmethod
#     def _truncate(txt: str, max_chars: int = 1500) -> str:
#         if txt is None:
#             return ""
#         txt = str(txt)
#         return txt if len(txt) <= max_chars else (txt[:max_chars] + " ...[truncated]")

#     @staticmethod
#     def _first_or_join_aliases(ref: dict) -> str:
#         """
#         Fallback text if GT final answer is not present in dialog but provided as whitelist/blacklist.
#         Builds a readable string from 'whitelist' (and ignores blacklist).
#         """
#         try:
#             wl = ref.get('whitelist', [])
#             parts = []
#             for group in wl:
#                 if isinstance(group, list) and group:
#                     parts.append(group[0])
#             return " / ".join(parts)
#         except Exception:
#             return ""

#     @staticmethod
#     def _extract_last_user_query(dialogs: List[dict]) -> str:
#         last = ""
#         for m in dialogs:
#             if m.get('role') == 'user':
#                 last = m.get('content', '') or last
#         return last or ""

#     @staticmethod
#     def _extract_final_answer_from_dialog(dialogs: List[dict]) -> str:
#         """
#         The final assistant message with no tool_calls is considered the final answer.
#         """
#         final = ""
#         for m in dialogs:
#             if m.get('role') == 'assistant' and 'tool_calls' not in m:
#                 final = m.get('content', '') or final
#         return final or ""

#     @staticmethod
#     def _format_reasoning_trace(dialogs: List[dict], max_field_chars: int = 800) -> str:
#         """
#         Build a compact, readable, step-by-step trace:
#         - assistant(tool_calls) -> tool output pairs
#         - assistant(content) entries included as "Assistant note"
#         """
#         lines = []
#         step_id = 1
#         i = 0
#         n = len(dialogs)
#         while i < n:
#             msg = dialogs[i]
#             # Assistant tool call
#             if msg.get('role') == 'assistant' and 'tool_calls' in msg:
#                 fn = msg['tool_calls'][0].get('function', {})
#                 tname = fn.get('name', '')
#                 targs = fn.get('arguments', {})
#                 thought = msg.get('content', '')
#                 lines.append(f"Step {step_id}:")
#                 lines.append(f"  Task/Thought: {GTABenchEvaluator._truncate(thought, max_field_chars)}")
#                 lines.append(f"  Tool: {tname}")
#                 # ensure args are compact json string
#                 try:
#                     args_str = json.dumps(targs, ensure_ascii=False)
#                 except Exception:
#                     args_str = str(targs)
#                 lines.append(f"  Input: {GTABenchEvaluator._truncate(args_str, max_field_chars)}")
#                 # try to pair the immediate next tool output
#                 tool_output = None
#                 if i + 1 < n and dialogs[i + 1].get('role') in ('tool', 'function'):
#                     tool_output = dialogs[i + 1].get('content', '')
#                     i += 1  # consume tool message
#                 if tool_output is not None:
#                     lines.append(f"  Output: {GTABenchEvaluator._truncate(tool_output, max_field_chars)}")
#                 step_id += 1
#             # Assistant plain message (non-tool answer / note)
#             elif msg.get('role') == 'assistant' and 'tool_calls' not in msg:
#                 content = msg.get('content', '')
#                 if content:
#                     lines.append(f"Assistant note/answer: {GTABenchEvaluator._truncate(content, max_field_chars)}")
#             # Tool return alone (should already be paired in most cases)
#             elif msg.get('role') in ('tool', 'function'):
#                 content = msg.get('content', '')
#                 lines.append(f"Tool return (unpaired): {GTABenchEvaluator._truncate(content, max_field_chars)}")
#             # User message (kept minimal)
#             elif msg.get('role') == 'user':
#                 content = msg.get('content', '')
#                 lines.append(f"User: {GTABenchEvaluator._truncate(content, max_field_chars)}")
#             i += 1
#         return "\n".join(lines)

#     @staticmethod
#     def _build_gt_block(gt_dialogs: List[dict], ref_text_fallback: str = "") -> str:
#         query = GTABenchEvaluator._extract_last_user_query(gt_dialogs)
#         gt_final = GTABenchEvaluator._extract_final_answer_from_dialog(gt_dialogs) or ref_text_fallback
#         gt_steps = GTABenchEvaluator._format_reasoning_trace(gt_dialogs)
#         payload = {
#             "query": query,
#             "GT reasoning steps": gt_steps,
#             "GT final answer": gt_final
#         }
#         # Use a compact, Python-dict-like string (per your original prompts).
#         return str(payload)

#     # ---------------------------
#     # OpenAI judge helpers (prompts & calling)
#     # ---------------------------
#     @staticmethod
#     def _safe_parse_score_dict(text: str):
#         """
#         The judge prompts return a 'single python dictionary' as text.
#         We parse it robustly (json first, then literal_eval).
#         Returns {'Score': float or None, 'Justification': str or ''}.
#         """
#         result = {'Score': None, 'Justification': ''}
#         if not text:
#             return result
#         text = text.strip()
#         # try JSON
#         try:
#             obj = json.loads(text)
#             result.update(obj)
#         except Exception:
#             try:
#                 obj = ast.literal_eval(text)
#                 if isinstance(obj, dict):
#                     result.update(obj)
#             except Exception:
#                 # Try to heuristically extract `Score: <number>`
#                 m = re.search(r"[\"']?Score[\"']?\s*:\s*[\"']?([0-9.]+)[\"']?", text, re.IGNORECASE)
#                 if m:
#                     result['Score'] = float(m.group(1))
#                 jm = re.search(r"[\"']?Justification[\"']?\s*:\s*[\"']?(.+)[\"']?", text, re.IGNORECASE | re.DOTALL)
#                 if jm:
#                     result['Justification'] = jm.group(1).strip()
#         # normalize
#         try:
#             if result['Score'] is not None:
#                 result['Score'] = float(result['Score'])
#                 # clip to [0,1] as per prompt contract
#                 result['Score'] = max(0.0, min(1.0, result['Score']))
#         except Exception:
#             result['Score'] = None
#         if not isinstance(result.get('Justification', ''), str):
#             result['Justification'] = str(result['Justification'])
#         return result

#     def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
#         """
#         Calls OpenAI chat completion once and returns message content (string).
#         Works with legacy and new SDKs. Raises on hard failure.
#         """
#         if not self._openai_ready:
#             raise RuntimeError("OpenAI API is not configured (missing package or OPENAI_API_KEY).")

#         if self._use_legacy:
#             # legacy openai.ChatCompletion
#             resp = _openai_pkg.ChatCompletion.create(
#                 model=self._judge_model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ]
#             )
#             return resp['choices'][0]['message']['content']
#         else:
#             # new SDK client
#             msg = self._client.chat.completions.create(
#                 model=self._judge_model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ]
#             )
#             return msg.choices[0].message.content

#     _PROMPT_FAITHFULNESS = """
# You are an evaluation assistant measuring the Faithfulness Accuracy of an agent's reasoning process.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.
# You are also given the full reasoning trace produced by the agent, including each step's task, selected tool, output, and thought.

# Your task is to assess how faithful the agent's reasoning trace is to the GT — that is, whether the steps follow a logically sound plan that aligns with the structure, intent, and direction of the GT.

# Evaluation Guidelines:
# - Focus on the structure and logical flow of the agent's reasoning steps.
# - Determine whether the steps collectively form a coherent strategy to answer the query.
# - Faithfulness is about consistency with the GT's method, not necessarily correctness of individual steps.

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents the reasoning is faithful to the GT: it follows a logically sound plan that mirrors the GT in structure and direction.
# 0 — represents the reasoning is not faithful to the GT and lacks logical progression or alignment with the intended plan.

# Final output must be a single python dictionary as below:
# {'Score': '<0-1>','Justification': '< explanation for the score>'}
# """.strip()

#     _PROMPT_CONTEXT = """
# You are an evaluation assistant measuring the Context Score of an agent's reasoning step.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

# You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

# Your task is to assess whether the agent's reasoning step is grounded in the input context — and whether that context was effectively used in the reasoning process. Inputs may include image, video, text, audio, or a combination.

# Evaluation Guidelines:
# If the number of steps in the GT and the model differ:
# - Any extra model step beyond the GT steps should receive a score of 0 (hallucinated or unjustified).
# - Any GT step that the model omits should also receive a score of 0 (missing reasoning).
# The Context Score for the full query is computed as the average of the per-step scores.

# Assess the following per step:
# - Does the agent correctly use relevant parts of the input?
# - Is the input used appropriate for the tool selected and the task being attempted?
# - Does the reasoning show effective and meaningful use of the input?

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents the agent uses the input fully and appropriately.
# 0 — represents the agent ignores or misinterprets the input entirely.

# Final output must be a single python dictionary as below:
# {'Score': '<0 - 1>','Justification': '< explanation for the score>'}
# """.strip()

#     _PROMPT_FACTUAL = """
# You are an evaluation assistant measuring the factual accuracy of the agent's reasoning steps.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification.

# You are also given the full reasoning trace produced by the agent, including each ste's task, selected tool, tool input/output, and thought.

# Your task is to assess whether the agent introduces any hallucinated, fabricated, or factually incorrect information in its reasoning when compared to the GT.

# Evaluation Guidelines:
# - Compare the model's reasoning steps to the GT to identify factual hallucinations or incorrect claims.
# - Focus on whether the output or thought includes details not supported by the GT.
# - Do not penalize for minor omissions unless they lead to a factual error.

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents no factual errors or hallucinations compared to the GT.
# 0 — represents major factual errors or hallucinated content.

# Final output must be a single python dictionary as below:
# {'Score': '<0 - 1>','Justification': '< explanation for the score>'}
# """.strip()

#     _PROMPT_SEMANTIC = """
# You are an evaluation assistant measuring the Semantic Accuracy of an agent's reasoning process.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification.

# You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

# Your task is to assess whether the agent's reasoning and final output semantically align with the Ground Truth — that is, whether the agent has covered all the essential parts of the query as demonstrated in the GT.

# Evaluation Guidelines:
# - Compare the agent's reasoning trace and final answer with the GT to check whether all key components of the query are addressed.
# - Credit should be given for meaningful semantic coverage, not superficial similarity.
# - If the model ignores or misunderstands core parts of the GT reasoning or final answer, penalize accordingly.

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents the agent addresses all key components of the query, matching the GT's semantic scope.
# 0 — represents the agent's reasoning or answer omits or misrepresents major parts of the query.

# Final output must be a single python dictionary as below:
# {'Score': '<0 - 1>','Justification': '< explanation for the score>'}
# """.strip()

#     # ---------------------------
#     # Existing helpers
#     # ---------------------------
#     @staticmethod
#     def get_response_type(item):
#         if 'tool_calls' in item:
#             return 'tool', item['tool_calls'][0]['function']
#         elif item['role'] == 'assistant':
#             return 'answer', item['content']
#         else:
#             return 'tool_return', item['content']

#     @staticmethod
#     def iscorrect(pred: str, ref: dict):
#         count = 0
#         for aliases in ref['whitelist']:
#             pattern = r'\b(?:' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
#             if re.search(pattern, pred, re.IGNORECASE):
#                 count += 1
#         if not ref['blacklist']:
#             if count == len(ref['whitelist']):
#                 return True
#         else:
#             pattern_bk = r'\b(?:' + '|'.join(re.escape(alias) for aliases in ref['blacklist'] for alias in aliases) + r')\b'
#             if count == len(ref['whitelist']) and not re.search(pattern_bk, pred, re.IGNORECASE):
#                 return True
#         return False

#     def simscore(self, pred: str, ref: list):
#         max_score = 0
#         for s in ref:
#             score = self.bert_score(pred, s)
#             if score > max_score:
#                 max_score = score
#         return max_score

#     @staticmethod
#     def gettype(name: str):
#         perception = ['OCR', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox']
#         operation = ['DrawBox', 'AddText', 'GoogleSearch']
#         logic = ['Calculator', 'Solver', 'Plot', 'MathOCR', 'CountGivenObject']
#         creativity = ['TextToImage', 'ImageStylization']
#         if name in perception:
#             return 'perception'
#         elif name in operation:
#             return 'operation'
#         elif name in logic:
#             return 'logic'
#         elif name in creativity:
#             return 'creativity'
#         else:
#             return 'none'

#     # ---------------------------
#     # Main scoring
#     # ---------------------------
#     def score(self, predictions: list, references: list, gold: list):
#         if self.mode == 'every_with_gt':
#             total = {'tool': 0, 'answer': 0}
#             metrics = {
#                 'inst_align': 0,
#                 'tool_acc': 0,
#                 'arg_acc': 0,
#                 'answer_acc': 0,
#                 'tool_call': 0,
#                 'tool_call_error': 0
#             }

#             # LLM-judge metrics accumulators
#             judge_sum = {
#                 'faithfulness': 0.0,
#                 'context_score': 0.0,
#                 'factual_precision': 0.0,
#                 'semantic_accuracy': 0.0
#             }
#             judge_cnt = {
#                 'faithfulness': 0,
#                 'context_score': 0,
#                 'factual_precision': 0,
#                 'semantic_accuracy': 0
#             }

#             for preds, gts, ref in zip(predictions, gold, references):
#                 # ref is a json string per dataset creation
#                 ref_obj = json.loads(ref)
#                 if ref_obj:
#                     total['answer'] += 1

#                 # ---- Your existing per-message loop ----
#                 for pred, gt in zip(preds, gts):
#                     pred_type, pred_ = self.get_response_type(pred)
#                     gt_type, gt_ = self.get_response_type(gt)
#                     if pred_type == gt_type and 'error' not in pred:
#                         metrics['inst_align'] += 1
#                     if gt_type == 'tool':
#                         total['tool'] += 1
#                     if pred_type == 'tool':
#                         metrics['tool_call'] += 1
#                         if 'error' in pred:
#                             metrics['tool_call_error'] += 1
#                     if pred_type == gt_type == 'tool' and pred_['name'] == gt_['name']:
#                         metrics['tool_acc'] += 1
#                         if pred_['arguments'] == gt_['arguments']:
#                             metrics['arg_acc'] += 1
#                     elif pred_type == gt_type == 'answer':
#                         if isinstance(ref_obj, dict):
#                             metrics['answer_acc'] += self.iscorrect(pred_, ref_obj)
#                         elif isinstance(ref_obj, list):
#                             metrics['answer_acc'] += self.simscore(pred_, ref_obj)

#                 # ---------------------------
#                 # LLM-judged metrics (per-sample) for every_with_gt
#                 # ---------------------------
#                 try:
#                     # Build GT block + model trace according to your dialog format
#                     # preds/gts are lists of message dicts for this sample (as used above)
#                     gt_fallback_answer = ""
#                     if isinstance(ref_obj, dict):
#                         gt_fallback_answer = self._first_or_join_aliases(ref_obj)
#                     elif isinstance(ref_obj, list) and ref_obj:
#                         # pick the first string for readability
#                         if isinstance(ref_obj[0], str):
#                             gt_fallback_answer = ref_obj[0]

#                     gt_block_str = self._build_gt_block(gts, ref_text_fallback=gt_fallback_answer)
#                     pred_trace_str = self._format_reasoning_trace(preds)
#                     pred_final_answer = self._extract_final_answer_from_dialog(preds)

#                     if self._openai_ready:
#                         # Faithfulness
#                         faith_resp = self._chat_once(
#                             self._PROMPT_FAITHFULNESS,
#                             "GT: " + gt_block_str + "\n" + "agent's reasoning trace: " + pred_trace_str
#                         )

#                         faith = self._safe_parse_score_dict(faith_resp).get('Score', None)

#                         if faith is not None:
#                             judge_sum['faithfulness'] += float(faith)
#                             judge_cnt['faithfulness'] += 1

#                         # Context Score
#                         ctx_resp = self._chat_once(
#                             self._PROMPT_CONTEXT,
#                             "GT: " + gt_block_str + "\n" + "agent's reasoning_steps: " + pred_trace_str
#                         )
#                         ctx = self._safe_parse_score_dict(ctx_resp).get('Score', None)
#                         if ctx is not None:
#                             judge_sum['context_score'] += float(ctx)
#                             judge_cnt['context_score'] += 1

#                         # Factual Precision
#                         fp_resp = self._chat_once(
#                             self._PROMPT_FACTUAL,
#                             "GT: " + gt_block_str + "\n" + "agent's reasoning steps: " + pred_trace_str
#                         )
#                         fp = self._safe_parse_score_dict(fp_resp).get('Score', None)
#                         if fp is not None:
#                             judge_sum['factual_precision'] += float(fp)
#                             judge_cnt['factual_precision'] += 1

#                         # Semantic Accuracy
#                         sem_resp = self._chat_once(
#                             self._PROMPT_SEMANTIC,
#                             "GT: " + gt_block_str + "\n" +
#                             "agent's reasoning steps: " + pred_trace_str + "\n" +
#                             "agent's final answer: " + (pred_final_answer or "")
#                         )
#                         sem = self._safe_parse_score_dict(sem_resp).get('Score', None)
#                         if sem is not None:
#                             judge_sum['semantic_accuracy'] += float(sem)
#                             judge_cnt['semantic_accuracy'] += 1
#                     # If OpenAI not ready, we simply skip these for this sample (keeps run stable).
#                 except Exception:
#                     pass

#             # Avoid divide-by-zero
#             inst_align = metrics['inst_align'] / max(sum(total.values()), 1) * 100
#             tool_acc = metrics['tool_acc'] / max(total['tool'], 1) * 100
#             arg_acc = metrics['arg_acc'] / max(total['tool'], 1) * 100
#             answer_acc = metrics['answer_acc'] / max(total['answer'], 1) * 100

#             # Aggregate judge metrics to percentages
#             def avg_pct(name: str) -> float:
#                 return (judge_sum[name] / judge_cnt[name] * 100) if judge_cnt[name] > 0 else None

#             result = dict(
#                 inst_align=inst_align,
#                 tool_acc=tool_acc,
#                 arg_acc=arg_acc,
#                 answer_acc=answer_acc,
#                 tool_call=metrics['tool_call'],
#                 tool_call_error=metrics['tool_call_error'],
#                 # ---- New metrics (averaged over counted samples, 0-100 scale) ----
#                 faithfulness=avg_pct('faithfulness'),
#                 context_score=avg_pct('context_score'),
#                 factual_precision=avg_pct('factual_precision'),
#                 semantic_accuracy=avg_pct('semantic_accuracy'),
#             )
#             return result

#         elif self.mode == 'every':
#             total = {'all': 0, 'answer': 0, 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'none': 0}
#             total_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             precision = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             recall = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             f1 = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             metrics = {
#                 'answer_acc': 0, 'answer_acc_w_imggen': 0, 'tool_call': 0, 'tool_call_error': 0,
#                 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0
#             }
#             imagegen_tools = ['DrawBox', 'AddText', 'Plot', 'TextToImage', 'ImageStylization']
#             for preds, gts, ref in zip(predictions, gold, references):
#                 # breakpoint()
#                 ref = json.loads(ref)
#                 total['all'] += 1
#                 if ref:
#                     total['answer'] += 1
#                 pred_type, pred_answer = self.get_response_type(preds[0][-1])
#                 if ref and pred_type == 'answer' and pred_answer:
#                     if isinstance(ref, dict) and pred_answer:
#                         score = self.iscorrect(pred_answer, ref)
#                         metrics['answer_acc'] += score
#                         metrics['answer_acc_w_imggen'] += score
#                     elif isinstance(ref, list) and pred_answer:
#                         score = self.simscore(pred_answer, ref)
#                         metrics['answer_acc'] += score
#                         metrics['answer_acc_w_imggen'] += score
#                 # add imagegen to ansacc
#                 if not ref:
#                     pred_imagegen = dict()
#                     score = 1
#                     for pred in preds[0]:
#                         if 'tool_calls' in pred and 'error' not in pred:
#                             tool_name = pred['tool_calls'][0]['function']['name']
#                             tool_arg = pred['tool_calls'][0]['function']['arguments']
#                             if tool_name in imagegen_tools:
#                                 pred_imagegen[tool_name] = tool_arg
#                     for gt in gts[0]:
#                         if 'tool_calls' in gt:
#                             tool_name = gt['tool_calls'][0]['function']['name']
#                             tool_arg = gt['tool_calls'][0]['function']['arguments']
#                             if tool_name in imagegen_tools:
#                                 if tool_name not in pred_imagegen:
#                                     score = 0
#                                     break
#                                 else:
#                                     score = score * self.bert_score(json.dumps(tool_arg),
#                                                                     json.dumps(pred_imagegen[tool_name]))
#                     metrics['answer_acc_w_imggen'] += score

#                 for pred in preds[0]:
#                     if 'tool_calls' in pred:
#                         metrics['tool_call'] += 1
#                         if 'error' in pred:
#                             metrics['tool_call_error'] += 1

#                 pred_tool_calls = []
#                 for pred in preds[0]:
#                     if 'tool_calls' in pred:
#                         tool_type = self.gettype(pred['tool_calls'][0]['function']['name'])
#                         if tool_type in total_predict:
#                             total_predict[tool_type] += 1
#                         pred_tool_calls.append(pred['tool_calls'][0]['function']['name'])
#                 for gt in gts[0]:
#                     if 'tool_calls' in gt:
#                         tool_type = self.gettype(gt['tool_calls'][0]['function']['name'])
#                         total[tool_type] += 1
#                         if gt['tool_calls'][0]['function']['name'] in pred_tool_calls:
#                             metrics[tool_type] += 1
#             for tool_type in f1.keys():
#                 precision[tool_type] = metrics[tool_type] / (total_predict[tool_type] + 1e-5)
#                 recall[tool_type] = metrics[tool_type] / total[tool_type]
#                 f1[tool_type] = 2 * precision[tool_type] * recall[tool_type] / (precision[tool_type] + recall[tool_type] + 1e-5)
#             return dict(
#                 answer_acc=metrics['answer_acc'] / total['answer'] * 100,
#                 answer_acc_w_imggen=metrics['answer_acc_w_imggen'] / total['all'] * 100,
#                 tool_call=metrics['tool_call'],
#                 tool_call_error=metrics['tool_call_error'],
#                 p_f1=f1['perception'] * 100,
#                 o_f1=f1['operation'] * 100,
#                 l_f1=f1['logic'] * 100,
#                 c_f1=f1['creativity'] * 100,
#                 total_ansacc=metrics['answer_acc'],
#                 total_ansacc_wimggen=metrics['answer_acc_w_imggen']
#             )
#         else:
#             raise NotImplementedError(f"Unsupported mode: {self.mode}")




# import json
# import os
# from pathlib import Path
# from typing import List, Tuple
# import copy
# import re
# import ast
# import subprocess  # for ffmpeg fallback

# import numpy as np
# from datasets.arrow_dataset import Dataset
# from sentence_transformers import SentenceTransformer, util

# from opencompass.openicl.icl_evaluator import BaseEvaluator
# from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

# from .base import BaseDataset

# # ----------------------------
# # Optional: OpenAI dependency
# # ----------------------------
# try:
#     import openai as _openai_pkg
#     _HAS_OPENAI = True
# except Exception:
#     _openai_pkg = None
#     _HAS_OPENAI = False

# # ----------------------------
# # Optional: OpenCV dependency
# # ----------------------------
# try:
#     import cv2  # preferred for frame seeking
# except Exception:
#     cv2 = None

# from urllib.parse import urlparse, unquote
# from urllib.request import url2pathname

# # Supported video extensions
# VIDEO_EXTS = {'.mp4', '.mov', '.avi'}


# def get_all_file_paths(directory: str) -> list:
#     file_paths = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             file_paths.append(file_path)
#     return file_paths


# # ----------------------------
# # Helpers for video -> frames
# # ----------------------------
# def _is_video_path(p: str) -> bool:
#     """Return True if path (or file:// URI) has a video extension we support."""
#     if not isinstance(p, str) or not p:
#         return False
#     # Accept file:// URIs too
#     if p.startswith("file://"):
#         try:
#             parsed = urlparse(p)
#             suffix = Path(parsed.path).suffix.lower()
#             return suffix in VIDEO_EXTS
#         except Exception:
#             return False
#     return Path(p).suffix.lower() in VIDEO_EXTS


# def _from_file_uri(p: str) -> str:
#     """Convert file:// URI to a local filesystem path, else return original string."""
#     if isinstance(p, str) and p.startswith("file://"):
#         parsed = urlparse(p)
#         # Convert percent-escapes and OS-specific path
#         path = url2pathname(unquote(parsed.path))
#         return path
#     return p


# def _to_abs_path(p: str, root: str) -> str:
#     """Return absolute normalized path, interpreting relative paths as under `root`."""
#     if not p:
#         return p
#     p = _from_file_uri(p)
#     return str((Path(p) if os.path.isabs(p) else Path(root) / p).resolve())


# def _extract_frames_with_opencv(video_abs: str,
#                                 out_dir: str,
#                                 base_name: str,
#                                 timestamps_s: List[int]) -> List[str]:
#     """Extract frames with OpenCV; return absolute filesystem paths of saved frames (no file://)."""
#     paths = []
#     cap = cv2.VideoCapture(video_abs)
#     if not cap.isOpened():
#         return paths

#     fps = cap.get(cv2.CAP_PROP_FPS) or 0
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
#     duration = (frame_count / fps) if fps > 0 else None

#     idx = 1
#     for t in timestamps_s:
#         # Skip timestamps beyond duration if we know it
#         if duration is not None and t > duration + 0.05:
#             continue
#         cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             continue
#         out_path = os.path.join(out_dir, f"{base_name}_frame{idx}.jpg")
#         try:
#             cv2.imwrite(out_path, frame)
#             paths.append(str(Path(out_path).resolve()))
#             idx += 1
#             if idx > 4:
#                 break
#         except Exception:
#             # If write fails, stop trying further to keep behavior predictable
#             break

#     cap.release()
#     return paths


# def _extract_frames_with_ffmpeg(video_abs: str,
#                                 out_dir: str,
#                                 base_name: str,
#                                 timestamps_s: List[int]) -> List[str]:
#     """Extract frames with ffmpeg CLI; return absolute filesystem paths of saved frames (no file://)."""
#     paths = []
#     idx = 1
#     for t in timestamps_s:
#         out_path = os.path.join(out_dir, f"{base_name}_frame{idx}.jpg")
#         cmd = [
#             'ffmpeg', '-y',
#             '-ss', str(t), '-i', video_abs,
#             '-frames:v', '1', '-q:v', '2',
#             out_path
#         ]
#         try:
#             subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#             paths.append(str(Path(out_path).resolve()))
#             idx += 1
#             if idx > 4:
#                 break
#         except Exception:
#             # If ffmpeg fails at some timestamp, stop trying further
#             break
#     return paths


# def _extract_frames(video_abs: str, timestamps_s: List[int]) -> List[str]:
#     """
#     Extract up to 4 frames at the provided timestamps (seconds) from `video_abs`.
#     Returns a list of **absolute filesystem paths** (no file://) in the same directory as the video.
#     Order is preserved (0s, 10s, 20s, 30s).
#     """
#     out_dir = str(Path(video_abs).parent)
#     base_name = Path(video_abs).stem

#     # Try OpenCV first
#     if cv2 is not None:
#         paths = _extract_frames_with_opencv(video_abs, out_dir, base_name, timestamps_s)
#         if paths:
#             return paths

#     # Fallback to ffmpeg if OpenCV missing or failed
#     return _extract_frames_with_ffmpeg(video_abs, out_dir, base_name, timestamps_s)


# def _maybe_replace_video_content(content, data_root: str) -> Tuple[object, List[str]]:
#     """
#     Detect a user content that references a video inside an 'image' block and:
#       - Extract up to 4 frames (0s, 10s, 20s, 30s).
#       - DELETE the original video file (best-effort) if at least one frame was saved.
#       - Replace that video 'image' block by **multiple** 'image' blocks (one per frame).
#       - Preserve other non-video blocks and any inline 'text' fields.

#     Return:
#       (transformed_content_or_None, frame_paths_list)

#       * transformed_content_or_None:
#           - If a video was found and frames were extracted: a **list** of blocks where
#             the video block is replaced by multiple {"type":"image","image":<abs_frame>} blocks.
#           - If no video was found or extraction failed: None (content unchanged).
#       * frame_paths_list: list of absolute frame file paths extracted (empty if none).
#     """
#     def _convert_to_image_blocks(video_path: str) -> Tuple[List[dict], List[str]]:
#         """Extract frames & delete video; return ([image blocks], [frame paths])."""
#         video_abs = _to_abs_path(video_path, data_root)
#         frame_paths = _extract_frames(video_abs, timestamps_s=[0, 10, 20, 30])
#         image_blocks: List[dict] = []
#         if frame_paths:
#             # Delete original video (best-effort)
#             try:
#                 if os.path.exists(video_abs):
#                     os.remove(video_abs)
#             except Exception:
#                 pass
#             # Build image blocks for each frame
#             for fp in frame_paths:
#                 image_blocks.append({"type": "image", "image": fp})
#         return image_blocks, frame_paths

#     # Case 1: content is a single dict block
#     if isinstance(content, dict):
#         if content.get("type") == "image" and _is_video_path(content.get("image")):
#             new_blocks = []
#             # Preserve any inline text field if present
#             if isinstance(content.get("text"), str) and content["text"].strip():
#                 new_blocks.append({"type": "text", "text": content["text"]})
#             img_blocks, frame_paths = _convert_to_image_blocks(content["image"])
#             if img_blocks:
#                 new_blocks.extend(img_blocks)
#                 return new_blocks, frame_paths
#             # Extraction failed or yielded nothing → unchanged
#             return None, []

#     # Case 2: content is a list of blocks
#     if isinstance(content, list):
#         new_blocks = []
#         converted = False
#         collected_frames: List[str] = []
#         for blk in content:
#             # Non-dict blocks are copied through unchanged
#             if not isinstance(blk, dict):
#                 new_blocks.append(blk)
#                 continue

#             btype = blk.get("type")
#             if btype == "text":
#                 new_blocks.append(blk)
#             elif btype == "image" and _is_video_path(blk.get("image")) and not converted:
#                 img_blocks, frame_paths = _convert_to_image_blocks(blk["image"])
#                 if img_blocks:
#                     if isinstance(blk.get("text"), str) and blk["text"].strip():
#                         new_blocks.append({"type": "text", "text": blk["text"]})
#                     new_blocks.extend(img_blocks)
#                     collected_frames.extend(frame_paths)
#                     converted = True
#                 else:
#                     # Extraction failed; keep original video image block unchanged
#                     new_blocks.append(blk)
#             else:
#                 # Keep non-video images and other block types unchanged
#                 new_blocks.append(blk)

#         if converted:
#             return new_blocks, collected_frames

#     # No change
#     return None, []


# def organize_dialogs(sample: dict, path: str) -> Tuple[List[dict], List[str]]:
#     """
#     Build dialogs with two modifications:
#       1) Keep existing tool/tool_call handling and absolute path rewriting for tool arguments.
#          (Now uses os.path.exists on absolute candidates so newly created frames are recognized.)
#       2) If a user message has content with {"type":"image","image":<video>},
#          extract up to 4 frames (0s, 10s, 20s, 30s) next to the video, DELETE the video,
#          and **inject multiple {"type":"image","image":<abs frame path>} blocks** while
#          preserving the original query text and other non-video blocks.

#     Returns:
#       dialogs: normalized dialog list
#       extracted_frames: list of absolute frame paths extracted across the whole dialog
#     """
#     dialogs: List[dict] = []
#     extracted_frames_all: List[str] = []

#     for item in sample['dialogs']:
#         if item['role'] == 'tool':
#             dialog = dict(
#                 role='tool',
#                 name=item['name'],
#                 content=item['content'],
#             )
#             dialogs.append(dialog)

#         elif item['role'] == 'assistant' and 'tool_calls' in item.keys():
#             dialog = copy.deepcopy(item)
#             # Rewrite only top-level string args of the first tool call
#             try:
#                 func_args = dialog['tool_calls'][0]['function']['arguments']
#                 if isinstance(func_args, dict):
#                     for name, value in list(func_args.items()):
#                         if isinstance(value, str):
#                             # Build absolute candidate (supports file:// too)
#                             abs_candidate = _to_abs_path(value, path)
#                             if os.path.exists(abs_candidate):
#                                 dialog['tool_calls'][0]['function']['arguments'][name] = str(Path(abs_candidate).resolve())
#             except Exception:
#                 # Be robust; if arguments shape is unexpected, leave as-is
#                 pass
#             dialogs.append(dialog)

#         elif item['role'] == 'user':
#             dialog = copy.deepcopy(item)
#             transformed, frames = _maybe_replace_video_content(dialog.get('content'), path)
#             if transformed is not None:
#                 dialog['content'] = transformed
#                 extracted_frames_all.extend(frames)
#             dialogs.append(dialog)

#         else:
#             dialogs.append(item)

#     return dialogs, extracted_frames_all


# @LOAD_DATASET.register_module()
# class GTABenchDataset(BaseDataset):
#     """GTA Benchmark with video-to-frames support and frame listing in resources."""

#     @staticmethod
#     def load(path: str):
#         data_root = Path(path)
#         data_file = data_root / 'dataset.json'
#         assert os.path.exists(data_file), f'Path {path} does not exist.'

#         data = json.load(open(data_file))
#         data_list = []
#         for idx, item in data.items():
#             idx = int(idx)

#             # First, normalize dialogs (this will extract frames & delete videos)
#             dialogs, frame_paths = organize_dialogs(item, str(data_root.absolute()))

#             # Tools from dataset.json
#             tools = [
#                 dict(type='tool', name=tool['name']) for tool in item['tools']
#             ]

#             # Files from dataset.json (omit videos; keep others)
#             files = []
#             for file in item['files']:
#                 abs_path = str((data_root / file['path']).absolute())
#                 suffix = Path(abs_path).suffix.lower()
#                 if suffix in VIDEO_EXTS:
#                     # Skip adding the original video to resources since it is deleted
#                     continue
#                 files.append(
#                     dict(
#                         type='file',
#                         filetype=file.get('type', 'unknown'),
#                         path=abs_path
#                     )
#                 )

#             # Add extracted frames to resources so tools can access them
#             for fp in frame_paths:
#                 files.append(
#                     dict(
#                         type='file',
#                         filetype='image',
#                         path=str(Path(fp).resolve())
#                     )
#                 )

#             gt_answer = item['gt_answer']
#             sample = {
#                 'dialogs': json.dumps(dialogs),
#                 'resources': json.dumps(tools + files),
#                 'gt_answer': json.dumps(gt_answer)
#             }
#             data_list.append(sample)

#         dataset = Dataset.from_list(data_list)
#         return dataset


# @ICL_EVALUATORS.register_module()
# class GTABenchEvaluator(BaseEvaluator):

#     def __init__(self, mode) -> None:
#         assert mode in ['every', 'every_with_gt']
#         self.mode = mode
#         self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

#         # --------- OpenAI judge setup ----------
#         self._judge_model = os.getenv('GTA_EVAL_GPT_JUDGE', 'gpt-4o')
#         self._openai_ready = False
#         self._use_legacy = False
#         self._client = None

#         if _HAS_OPENAI:
#             # Prefer legacy style if present (to match your reference code)
#             self._use_legacy = hasattr(_openai_pkg, 'ChatCompletion')
#             api_key_env = os.getenv('OPENAI_API_KEY', '')
#             try:
#                 if self._use_legacy:
#                     # legacy SDK
#                     if getattr(_openai_pkg, 'api_key', None) in (None, '') and api_key_env:
#                         _openai_pkg.api_key = api_key_env
#                     self._openai_ready = bool(getattr(_openai_pkg, 'api_key', None))
#                 else:
#                     # new SDK layout: instantiate a client
#                     try:
#                         from openai import OpenAI as _OpenAIClient
#                         self._client = _OpenAIClient(api_key=api_key_env if api_key_env else None)
#                         self._openai_ready = self._client is not None
#                     except Exception:
#                         self._openai_ready = False
#             except Exception:
#                 self._openai_ready = False

#     # ---------------------------
#     # Similarity helper (existing)
#     # ---------------------------
#     def bert_score(self, pred: str, gt: str) -> float:
#         pred_emb = self.sentence_model.encode(pred, convert_to_tensor=True)
#         gt_emb = self.sentence_model.encode(gt, convert_to_tensor=True)
#         score = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
#         return score[0][0]

#     # ---------------------------
#     # Formatting / extraction helpers for judge metrics
#     # ---------------------------
#     @staticmethod
#     def _truncate(txt: str, max_chars: int = 1500) -> str:
#         if txt is None:
#             return ""
#         txt = str(txt)
#         return txt if len(txt) <= max_chars else (txt[:max_chars] + " ...[truncated]")

#     @staticmethod
#     def _first_or_join_aliases(ref: dict) -> str:
#         try:
#             wl = ref.get('whitelist', [])
#             parts = []
#             for group in wl:
#                 if isinstance(group, list) and group:
#                     parts.append(group[0])
#             return " / ".join(parts)
#         except Exception:
#             return ""

#     @staticmethod
#     def _extract_last_user_query(dialogs: List[dict]) -> str:
#         last = ""
#         for m in dialogs:
#             if m.get('role') == 'user':
#                 last = m.get('content', '') or last
#         return last or ""

#     @staticmethod
#     def _extract_final_answer_from_dialog(dialogs: List[dict]) -> str:
#         final = ""
#         for m in dialogs:
#             if m.get('role') == 'assistant' and 'tool_calls' not in m:
#                 final = m.get('content', '') or final
#         return final or ""

#     @staticmethod
#     def _format_reasoning_trace(dialogs: List[dict], max_field_chars: int = 800) -> str:
#         lines = []
#         step_id = 1
#         i = 0
#         n = len(dialogs)
#         while i < n:
#             msg = dialogs[i]
#             # Assistant tool call
#             if msg.get('role') == 'assistant' and 'tool_calls' in msg:
#                 fn = msg['tool_calls'][0].get('function', {})
#                 tname = fn.get('name', '')
#                 targs = fn.get('arguments', {})
#                 thought = msg.get('content', '')
#                 lines.append(f"Step {step_id}:")
#                 lines.append(f"  Task/Thought: {GTABenchEvaluator._truncate(thought, max_field_chars)}")
#                 lines.append(f"  Tool: {tname}")
#                 try:
#                     args_str = json.dumps(targs, ensure_ascii=False)
#                 except Exception:
#                     args_str = str(targs)
#                 lines.append(f"  Input: {GTABenchEvaluator._truncate(args_str, max_field_chars)}")
#                 tool_output = None
#                 if i + 1 < n and dialogs[i + 1].get('role') in ('tool', 'function'):
#                     tool_output = dialogs[i + 1].get('content', '')
#                     i += 1  # consume tool message
#                 if tool_output is not None:
#                     lines.append(f"  Output: {GTABenchEvaluator._truncate(tool_output, max_field_chars)}")
#                 step_id += 1
#             elif msg.get('role') == 'assistant' and 'tool_calls' not in msg:
#                 content = msg.get('content', '')
#                 if content:
#                     lines.append(f"Assistant note/answer: {GTABenchEvaluator._truncate(content, max_field_chars)}")
#             elif msg.get('role') in ('tool', 'function'):
#                 content = msg.get('content', '')
#                 lines.append(f"Tool return (unpaired): {GTABenchEvaluator._truncate(content, max_field_chars)}")
#             elif msg.get('role') == 'user':
#                 content = msg.get('content', '')
#                 lines.append(f"User: {GTABenchEvaluator._truncate(content, max_field_chars)}")
#             i += 1
#         return "\n".join(lines)

#     @staticmethod
#     def _build_gt_block(gt_dialogs: List[dict], ref_text_fallback: str = "") -> str:
#         query = GTABenchEvaluator._extract_last_user_query(gt_dialogs)
#         gt_final = GTABenchEvaluator._extract_final_answer_from_dialog(gt_dialogs) or ref_text_fallback
#         gt_steps = GTABenchEvaluator._format_reasoning_trace(gt_dialogs)
#         payload = {
#             "query": query,
#             "GT reasoning steps": gt_steps,
#             "GT final answer": gt_final
#         }
#         return str(payload)

#     # ---------------------------
#     # OpenAI judge helpers (prompts & calling)
#     # ---------------------------
#     @staticmethod
#     def _safe_parse_score_dict(text: str):
#         result = {'Score': None, 'Justification': ''}
#         if not text:
#             return result
#         text = text.strip()
#         # try JSON
#         try:
#             obj = json.loads(text)
#             result.update(obj)
#         except Exception:
#             try:
#                 obj = ast.literal_eval(text)
#                 if isinstance(obj, dict):
#                     result.update(obj)
#             except Exception:
#                 m = re.search(r"[\"']?Score[\"']?\s*:\s*[\"']?([0-9.]+)[\"']?", text, re.IGNORECASE)
#                 if m:
#                     result['Score'] = float(m.group(1))
#                 jm = re.search(r"[\"']?Justification[\"']?\s*:\s*[\"']?(.+)[\"']?", text, re.IGNORECASE | re.DOTALL)
#                 if jm:
#                     result['Justification'] = jm.group(1).strip()
#         try:
#             if result['Score'] is not None:
#                 result['Score'] = float(result['Score'])
#                 result['Score'] = max(0.0, min(1.0, result['Score']))
#         except Exception:
#             result['Score'] = None
#         if not isinstance(result.get('Justification', ''), str):
#             result['Justification'] = str(result['Justification'])
#         return result

#     def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
#         if not self._openai_ready:
#             raise RuntimeError("OpenAI API is not configured (missing package or OPENAI_API_KEY).")

#         if self._use_legacy:
#             resp = _openai_pkg.ChatCompletion.create(
#                 model=self._judge_model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ]
#             )
#             return resp['choices'][0]['message']['content']
#         else:
#             msg = self._client.chat.completions.create(
#                 model=self._judge_model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ]
#             )
#             return msg.choices[0].message.content

#     _PROMPT_FAITHFULNESS = """
# You are an evaluation assistant measuring the Faithfulness Accuracy of an agent's reasoning process.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.
# You are also given the full reasoning trace produced by the agent, including each step's task, selected tool, output, and thought.

# Your task is to assess how faithful the agent's reasoning trace is to the GT — that is, whether the steps follow a logically sound plan that aligns with the structure, intent, and direction of the GT.

# Evaluation Guidelines:
# - Focus on the structure and logical flow of the agent's reasoning steps.
# - Determine whether the steps collectively form a coherent strategy to answer the query.
# - Faithfulness is about consistency with the GT's method, not necessarily correctness of individual steps.

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents the reasoning is faithful to the GT: it follows a logically sound plan that mirrors the GT in structure and direction.
# 0 — represents the reasoning is not faithful to the GT and lacks logical progression or alignment with the intended plan.

# Final output must be a single python dictionary as below:
# {'Score': '<0-1>','Justification': '< explanation for the score>'}
# """.strip()

#     _PROMPT_CONTEXT = """
# You are an evaluation assistant measuring the Context Score of an agent's reasoning step.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

# You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

# Your task is to assess whether the agent's reasoning step is grounded in the input context — and whether that context was effectively used in the reasoning process. Inputs may include image, video, text, audio, or a combination.

# Evaluation Guidelines:
# If the number of steps in the GT and the model differ:
# - Any extra model step beyond the GT steps should receive a score of 0 (hallucinated or unjustified).
# - Any GT step that the model omits should also receive a score of 0 (missing reasoning).
# The Context Score for the full query is computed as the average of the per-step scores.

# Assess the following per step:
# - Does the agent correctly use relevant parts of the input?
# - Is the input used appropriate for the tool selected and the task being attempted?
# - Does the reasoning show effective and meaningful use of the input?

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents the agent uses the input fully and appropriately.
# 0 — represents the agent ignores or misinterprets the input entirely.

# Final output must be a single python dictionary as below:
# {'Score': '<0 - 1>','Justification': '< explanation for the score>'}
# """.strip()

#     _PROMPT_FACTUAL = """
# You are an evaluation assistant measuring the factual accuracy of the agent's reasoning steps.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification.

# You are also given the full reasoning trace produced by the agent, including each ste's task, selected tool, tool input/output, and thought.

# Your task is to assess whether the agent introduces any hallucinated, fabricated, or factually incorrect information in its reasoning when compared to the GT.

# Evaluation Guidelines:
# - Compare the model's reasoning steps to the GT to identify factual hallucinations or incorrect claims.
# - Focus on whether the output or thought includes details not supported by the GT.
# - Do not penalize for minor omissions unless they lead to a factual error.

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents no factual errors or hallucinations compared to the GT.
# 0 — represents major factual errors or hallucinated content.

# Final output must be a single python dictionary as below:
# {'Score': '<0 - 1>','Justification': '< explanation for the score>'}
# """.strip()

#     _PROMPT_SEMANTIC = """
# You are an evaluation assistant measuring the Semantic Accuracy of an agent's reasoning process.

# You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification.

# You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

# Your task is to assess whether the agent's reasoning and final output semantically align with the Ground Truth — that is, whether the agent has covered all the essential parts of the query as demonstrated in the GT.

# Evaluation Guidelines:
# - Compare the agent's reasoning trace and final answer with the GT to check whether all key components of the query are addressed.
# - Credit should be given for meaningful semantic coverage, not superficial similarity.
# - If the model ignores or misunderstands core parts of the GT reasoning or final answer, penalize accordingly.

# Scoring Criteria: Output the scores between the range 0 and 1 such that:
# 1 — represents the agent addresses all key components of the query, matching the GT's semantic scope.
# 0 — represents the agent's reasoning or answer omits or misrepresents major parts of the query.

# Final output must be a single python dictionary as below:
# {'Score': '<0 - 1>','Justification': '< explanation for the score>'}
# """.strip()

#     # ---------------------------
#     # Existing helpers
#     # ---------------------------
#     @staticmethod
#     def get_response_type(item):
#         if 'tool_calls' in item:
#             return 'tool', item['tool_calls'][0]['function']
#         elif item['role'] == 'assistant':
#             return 'answer', item['content']
#         else:
#             return 'tool_return', item['content']

#     @staticmethod
#     def iscorrect(pred: str, ref: dict):
#         count = 0
#         for aliases in ref['whitelist']:
#             pattern = r'\b(?:' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
#             if re.search(pattern, pred, re.IGNORECASE):
#                 count += 1
#         if not ref['blacklist']:
#             if count == len(ref['whitelist']):
#                 return True
#         else:
#             pattern_bk = r'\b(?:' + '|'.join(re.escape(alias) for aliases in ref['blacklist'] for alias in aliases) + r')\b'
#             if count == len(ref['whitelist']) and not re.search(pattern_bk, pred, re.IGNORECASE):
#                 return True
#         return False

#     def simscore(self, pred: str, ref: list):
#         max_score = 0
#         for s in ref:
#             score = self.bert_score(pred, s)
#             if score > max_score:
#                 max_score = score
#         return max_score

#     @staticmethod
#     def gettype(name: str):
#         perception = ['OCR', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox']
#         operation = ['DrawBox', 'AddText', 'GoogleSearch']
#         logic = ['Calculator', 'Solver', 'Plot', 'MathOCR', 'CountGivenObject']
#         creativity = ['TextToImage', 'ImageStylization']
#         if name in perception:
#             return 'perception'
#         elif name in operation:
#             return 'operation'
#         elif name in logic:
#             return 'logic'
#         elif name in creativity:
#             return 'creativity'
#         else:
#             return 'none'

#     # ---------------------------
#     # Main scoring
#     # ---------------------------
#     def score(self, predictions: list, references: list, gold: list):
#         if self.mode == 'every_with_gt':
#             total = {'tool': 0, 'answer': 0}
#             metrics = {
#                 'inst_align': 0,
#                 'tool_acc': 0,
#                 'arg_acc': 0,
#                 'answer_acc': 0,
#                 'tool_call': 0,
#                 'tool_call_error': 0
#             }

#             # LLM-judge metrics accumulators
#             judge_sum = {
#                 'faithfulness': 0.0,
#                 'context_score': 0.0,
#                 'factual_precision': 0.0,
#                 'semantic_accuracy': 0.0
#             }
#             judge_cnt = {
#                 'faithfulness': 0,
#                 'context_score': 0,
#                 'factual_precision': 0,
#                 'semantic_accuracy': 0
#             }

#             for preds, gts, ref in zip(predictions, gold, references):
#                 ref_obj = json.loads(ref)
#                 if ref_obj:
#                     total['answer'] += 1

#                 for pred, gt in zip(preds, gts):
#                     pred_type, pred_ = self.get_response_type(pred)
#                     gt_type, gt_ = self.get_response_type(gt)
#                     if pred_type == gt_type and 'error' not in pred:
#                         metrics['inst_align'] += 1
#                     if gt_type == 'tool':
#                         total['tool'] += 1
#                     if pred_type == 'tool':
#                         metrics['tool_call'] += 1
#                         if 'error' in pred:
#                             metrics['tool_call_error'] += 1
#                     if pred_type == gt_type == 'tool' and pred_['name'] == gt_['name']:
#                         metrics['tool_acc'] += 1
#                         if pred_['arguments'] == gt_['arguments']:
#                             metrics['arg_acc'] += 1
#                     elif pred_type == gt_type == 'answer':
#                         if isinstance(ref_obj, dict):
#                             metrics['answer_acc'] += self.iscorrect(pred_, ref_obj)
#                         elif isinstance(ref_obj, list):
#                             metrics['answer_acc'] += self.simscore(pred_, ref_obj)

#                 # ---------------------------
#                 # LLM-judged metrics (per-sample)
#                 # ---------------------------
#                 try:
#                     gt_fallback_answer = ""
#                     if isinstance(ref_obj, dict):
#                         gt_fallback_answer = self._first_or_join_aliases(ref_obj)
#                     elif isinstance(ref_obj, list) and ref_obj:
#                         if isinstance(ref_obj[0], str):
#                             gt_fallback_answer = ref_obj[0]

#                     gt_block_str = self._build_gt_block(gts, ref_text_fallback=gt_fallback_answer)
#                     pred_trace_str = self._format_reasoning_trace(preds)
#                     pred_final_answer = self._extract_final_answer_from_dialog(preds)

#                     if self._openai_ready:
#                         faith_resp = self._chat_once(
#                             self._PROMPT_FAITHFULNESS,
#                             "GT: " + gt_block_str + "\n" + "agent's reasoning trace: " + pred_trace_str
#                         )
#                         faith = self._safe_parse_score_dict(faith_resp).get('Score', None)
#                         if faith is not None:
#                             judge_sum['faithfulness'] += float(faith)
#                             judge_cnt['faithfulness'] += 1

#                         ctx_resp = self._chat_once(
#                             self._PROMPT_CONTEXT,
#                             "GT: " + gt_block_str + "\n" + "agent's reasoning_steps: " + pred_trace_str
#                         )
#                         ctx = self._safe_parse_score_dict(ctx_resp).get('Score', None)
#                         if ctx is not None:
#                             judge_sum['context_score'] += float(ctx)
#                             judge_cnt['context_score'] += 1

#                         fp_resp = self._chat_once(
#                             self._PROMPT_FACTUAL,
#                             "GT: " + gt_block_str + "\n" + "agent's reasoning steps: " + pred_trace_str
#                         )
#                         fp = self._safe_parse_score_dict(fp_resp).get('Score', None)
#                         if fp is not None:
#                             judge_sum['factual_precision'] += float(fp)
#                             judge_cnt['factual_precision'] += 1

#                         sem_resp = self._chat_once(
#                             self._PROMPT_SEMANTIC,
#                             "GT: " + gt_block_str + "\n" +
#                             "agent's reasoning steps: " + pred_trace_str + "\n" +
#                             "agent's final answer: " + (pred_final_answer or "")
#                         )
#                         sem = self._safe_parse_score_dict(sem_resp).get('Score', None)
#                         if sem is not None:
#                             judge_sum['semantic_accuracy'] += float(sem)
#                             judge_cnt['semantic_accuracy'] += 1
#                 except Exception:
#                     pass

#             inst_align = metrics['inst_align'] / max(sum(total.values()), 1) * 100
#             tool_acc = metrics['tool_acc'] / max(total['tool'], 1) * 100
#             arg_acc = metrics['arg_acc'] / max(total['tool'], 1) * 100
#             answer_acc = metrics['answer_acc'] / max(total['answer'], 1) * 100

#             def avg_pct(name: str) -> float:
#                 return (judge_sum[name] / judge_cnt[name] * 100) if judge_cnt[name] > 0 else None

#             result = dict(
#                 inst_align=inst_align,
#                 tool_acc=tool_acc,
#                 arg_acc=arg_acc,
#                 answer_acc=answer_acc,
#                 tool_call=metrics['tool_call'],
#                 tool_call_error=metrics['tool_call_error'],
#                 faithfulness=avg_pct('faithfulness'),
#                 context_score=avg_pct('context_score'),
#                 factual_precision=avg_pct('factual_precision'),
#                 semantic_accuracy=avg_pct('semantic_accuracy'),
#             )
#             return result

#         elif self.mode == 'every':
#             total = {'all': 0, 'answer': 0, 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'none': 0}
#             total_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             precision = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             recall = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             f1 = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
#             metrics = {
#                 'answer_acc': 0, 'answer_acc_w_imggen': 0, 'tool_call': 0, 'tool_call_error': 0,
#                 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0
#             }
#             imagegen_tools = ['DrawBox', 'AddText', 'Plot', 'TextToImage', 'ImageStylization']
#             for preds, gts, ref in zip(predictions, gold, references):
#                 ref = json.loads(ref)
#                 total['all'] += 1
#                 if ref:
#                     total['answer'] += 1
#                 pred_type, pred_answer = self.get_response_type(preds[0][-1])
#                 if ref and pred_type == 'answer' and pred_answer:
#                     if isinstance(ref, dict) and pred_answer:
#                         score = self.iscorrect(pred_answer, ref)
#                         metrics['answer_acc'] += score
#                         metrics['answer_acc_w_imggen'] += score
#                     elif isinstance(ref, list) and pred_answer:
#                         score = self.simscore(pred_answer, ref)
#                         metrics['answer_acc'] += score
#                         metrics['answer_acc_w_imggen'] += score
#                 # add imagegen to ansacc
#                 if not ref:
#                     pred_imagegen = dict()
#                     score = 1
#                     for pred in preds[0]:
#                         if 'tool_calls' in pred and 'error' not in pred:
#                             tool_name = pred['tool_calls'][0]['function']['name']
#                             tool_arg = pred['tool_calls'][0]['function']['arguments']
#                             if tool_name in imagegen_tools:
#                                 pred_imagegen[tool_name] = tool_arg
#                     for gt in gts[0]:
#                         if 'tool_calls' in gt:
#                             tool_name = gt['tool_calls'][0]['function']['name']
#                             tool_arg = gt['tool_calls'][0]['function']['arguments']
#                             if tool_name in imagegen_tools:
#                                 if tool_name not in pred_imagegen:
#                                     score = 0
#                                     break
#                                 else:
#                                     score = score * self.bert_score(json.dumps(tool_arg),
#                                                                     json.dumps(pred_imagegen[tool_name]))
#                     metrics['answer_acc_w_imggen'] += score

#                 for pred in preds[0]:
#                     if 'tool_calls' in pred:
#                         metrics['tool_call'] += 1
#                         if 'error' in pred:
#                             metrics['tool_call_error'] += 1

#                 pred_tool_calls = []
#                 for pred in preds[0]:
#                     if 'tool_calls' in pred:
#                         tool_type = self.gettype(pred['tool_calls'][0]['function']['name'])
#                         if tool_type in total_predict:
#                             total_predict[tool_type] += 1
#                         pred_tool_calls.append(pred['tool_calls'][0]['function']['name'])
#                 for gt in gts[0]:
#                     if 'tool_calls' in gt:
#                         tool_type = self.gettype(gt['tool_calls'][0]['function']['name'])
#                         total[tool_type] += 1
#                         if gt['tool_calls'][0]['function']['name'] in pred_tool_calls:
#                             metrics[tool_type] += 1
#             for tool_type in f1.keys():
#                 precision[tool_type] = metrics[tool_type] / (total_predict[tool_type] + 1e-5)
#                 recall[tool_type] = metrics[tool_type] / (total[tool_type] if total[tool_type] > 0 else 1)
#                 f1[tool_type] = 2 * precision[tool_type] * recall[tool_type] / (precision[tool_type] + recall[tool_type] + 1e-5)
#             return dict(
#                 answer_acc=metrics['answer_acc'] / total['answer'] * 100 if total['answer'] > 0 else 0.0,
#                 answer_acc_w_imggen=metrics['answer_acc_w_imggen'] / total['all'] * 100 if total['all'] > 0 else 0.0,
#                 tool_call=metrics['tool_call'],
#                 tool_call_error=metrics['tool_call_error'],
#                 p_f1=f1['perception'] * 100,
#                 o_f1=f1['operation'] * 100,
#                 l_f1=f1['logic'] * 100,
#                 c_f1=f1['creativity'] * 100,
#                 total_ansacc=metrics['answer_acc'],
#                 total_ansacc_wimggen=metrics['answer_acc_w_imggen']
#             )
#         else:
#             raise NotImplementedError(f"Unsupported mode: {self.mode}")



import json
import os
from pathlib import Path
from typing import List, Tuple
import copy
import re
import ast
import subprocess  # for ffmpeg fallback

import numpy as np
from datasets.arrow_dataset import Dataset
from sentence_transformers import SentenceTransformer, util

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

# ----------------------------
# Optional: OpenAI dependency
# ----------------------------
try:
    import openai as _openai_pkg
    _HAS_OPENAI = True
except Exception:
    _openai_pkg = None
    _HAS_OPENAI = False

# ----------------------------
# Optional: OpenCV dependency
# ----------------------------
try:
    import cv2  # preferred for frame seeking
except Exception:
    cv2 = None

from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

# Supported video extensions
VIDEO_EXTS = {'.mp4', '.mov', '.avi'}


def get_all_file_paths(directory: str) -> list:
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


# ----------------------------
# Helpers for video -> frames
# ----------------------------
def _is_video_path(p: str) -> bool:
    """Return True if path (or file:// URI) has a video extension we support."""
    if not isinstance(p, str) or not p:
        return False
    # Accept file:// URIs too
    if p.startswith("file://"):
        try:
            parsed = urlparse(p)
            suffix = Path(parsed.path).suffix.lower()
            return suffix in VIDEO_EXTS
        except Exception:
            return False
    return Path(p).suffix.lower() in VIDEO_EXTS


def _from_file_uri(p: str) -> str:
    """Convert file:// URI to a local filesystem path, else return original string."""
    if isinstance(p, str) and p.startswith("file://"):
        parsed = urlparse(p)
        # Convert percent-escapes and OS-specific path
        path = url2pathname(unquote(parsed.path))
        return path
    return p


def _to_abs_path(p: str, root: str) -> str:
    """Return absolute normalized path, interpreting relative paths as under `root`."""
    if not p:
        return p
    p = _from_file_uri(p)
    return str((Path(p) if os.path.isabs(p) else Path(root) / p).resolve())


def _extract_frames_with_opencv(video_abs: str,
                                out_dir: str,
                                base_name: str,
                                timestamps_s: List[int]) -> List[str]:
    """Extract frames with OpenCV; return absolute filesystem paths of saved frames (no file://)."""
    paths = []
    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        return paths

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (frame_count / fps) if fps > 0 else None

    idx = 1
    for t in timestamps_s:
        # Skip timestamps beyond duration if we know it
        if duration is not None and t > duration + 0.05:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = os.path.join(out_dir, f"{base_name}_frame{idx}.jpg")
        try:
            cv2.imwrite(out_path, frame)
            paths.append(str(Path(out_path).resolve()))
            idx += 1
            if idx > 4:
                break
        except Exception:
            # If write fails, stop trying further to keep behavior predictable
            break

    cap.release()
    return paths


def _extract_frames_with_ffmpeg(video_abs: str,
                                out_dir: str,
                                base_name: str,
                                timestamps_s: List[int]) -> List[str]:
    """Extract frames with ffmpeg CLI; return absolute filesystem paths of saved frames (no file://)."""
    paths = []
    idx = 1
    for t in timestamps_s:
        out_path = os.path.join(out_dir, f"{base_name}_frame{idx}.jpg")
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(t), '-i', video_abs,
            '-frames:v', '1', '-q:v', '2',
            out_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            paths.append(str(Path(out_path).resolve()))
            idx += 1
            if idx > 4:
                break
        except Exception:
            # If ffmpeg fails at some timestamp, stop trying further
            break
    return paths


def _extract_frames(video_abs: str, timestamps_s: List[int]) -> List[str]:
    """
    Extract up to 4 frames at the provided timestamps (seconds) from `video_abs`.
    Returns a list of **absolute filesystem paths** (no file://) in the same directory as the video.
    Order is preserved (0s, 10s, 20s, 30s).
    """
    out_dir = str(Path(video_abs).parent)
    base_name = Path(video_abs).stem

    # Try OpenCV first
    if cv2 is not None:
        paths = _extract_frames_with_opencv(video_abs, out_dir, base_name, timestamps_s)
        if paths:
            return paths

    # Fallback to ffmpeg if OpenCV missing or failed
    return _extract_frames_with_ffmpeg(video_abs, out_dir, base_name, timestamps_s)


def _maybe_replace_video_content(content, data_root: str) -> Tuple[object, List[str]]:
    """
    Normalize any 'image' block to an **absolute path** (using data_root) and,
    if the 'image' is a video, also:
      - extract up to 4 frames (0s, 10s, 20s, 30s) next to the video,
      - DELETE the original video file (best-effort) if at least one frame was saved,
      - REPLACE the video block with multiple {"type":"image","image":<abs_frame>} blocks.

    Returns:
      (transformed_content_or_None, frame_paths_list)
        - transformed_content_or_None: the updated block(s) if any path was normalized
          or a video was converted; otherwise None to keep original content.
        - frame_paths_list: absolute paths of extracted frames (empty if none).
    """
    def _convert_to_image_blocks(video_path: str) -> Tuple[List[dict], List[str]]:
        """Extract frames & delete video; return ([image blocks], [frame paths])."""
        video_abs = _to_abs_path(video_path, data_root)
        frame_paths = _extract_frames(video_abs, timestamps_s=[0, 10, 20, 30])
        image_blocks: List[dict] = []
        if frame_paths:
            # NOTE (suite fork): the original code deleted the source video here
            # (os.remove), which made runs non-reproducible — a second run would
            # find the video gone and drop the task to text-only. Disabled so the
            # video survives and every run can re-extract frames.
            # try:
            #     if os.path.exists(video_abs):
            #         os.remove(video_abs)
            # except Exception:
            #     pass
            # Build image blocks for each frame
            for fp in frame_paths:
                image_blocks.append({"type": "image", "image": fp})
        return image_blocks, frame_paths

    # Case 1: content is a single dict block
    if isinstance(content, dict):
        if content.get("type") == "image":
            img_path = content.get("image")
            # Video: convert to frames
            if _is_video_path(img_path):
                new_blocks = []
                # Preserve any inline text field if present
                if isinstance(content.get("text"), str) and content["text"].strip():
                    new_blocks.append({"type": "text", "text": content["text"]})
                img_blocks, frame_paths = _convert_to_image_blocks(img_path)
                if img_blocks:
                    new_blocks.extend(img_blocks)
                    return new_blocks, frame_paths
                # Extraction failed or yielded nothing → fall through to absolute path normalization

            # Non-video (or video extraction failed): normalize to absolute path
            try:
                abs_img = _to_abs_path(img_path, data_root)
                if abs_img and abs_img != img_path:
                    new_dict = dict(content)
                    new_dict["image"] = str(Path(abs_img).resolve())
                    return new_dict, []
            except Exception:
                pass
            # No change
            return None, []

    # Case 2: content is a list of blocks
    if isinstance(content, list):
        new_blocks = []
        changed_any = False            # did we change anything at all?
        converted_video = False        # ensure we only convert the FIRST video to frames (preserve prior behavior)
        collected_frames: List[str] = []
        for blk in content:
            # Non-dict blocks are copied through unchanged
            if not isinstance(blk, dict):
                new_blocks.append(blk)
                continue

            btype = blk.get("type")
            if btype == "text":
                new_blocks.append(blk)
            elif btype == "image":
                img_path = blk.get("image")
                if _is_video_path(img_path) and not converted_video:
                    # Try to convert video to frames
                    img_blocks, frame_paths = _convert_to_image_blocks(img_path)
                    if img_blocks:
                        if isinstance(blk.get("text"), str) and blk["text"].strip():
                            new_blocks.append({"type": "text", "text": blk["text"]})
                        new_blocks.extend(img_blocks)
                        collected_frames.extend(frame_paths)
                        converted_video = True
                        changed_any = True
                    else:
                        # Could not extract frames; at least normalize to absolute path
                        try:
                            abs_img = _to_abs_path(img_path, data_root)
                            if abs_img and abs_img != img_path:
                                blk2 = dict(blk)
                                blk2["image"] = str(Path(abs_img).resolve())
                                new_blocks.append(blk2)
                                changed_any = True
                            else:
                                new_blocks.append(blk)
                        except Exception:
                            new_blocks.append(blk)
                else:
                    # Non-video image (or subsequent videos we aren't converting): normalize to absolute path
                    try:
                        abs_img = _to_abs_path(img_path, data_root)
                        if abs_img and abs_img != img_path:
                            blk2 = dict(blk)
                            blk2["image"] = str(Path(abs_img).resolve())
                            new_blocks.append(blk2)
                            changed_any = True
                        else:
                            new_blocks.append(blk)
                    except Exception:
                        new_blocks.append(blk)
            else:
                # Keep other block types unchanged
                new_blocks.append(blk)

        if changed_any:
            return new_blocks, collected_frames

    # No change
    return None, []


def organize_dialogs(sample: dict, path: str) -> Tuple[List[dict], List[str]]:
    """
    Build dialogs with two modifications:
      1) Keep existing tool/tool_call handling and absolute path rewriting for tool arguments.
         (Now uses os.path.exists on absolute candidates so newly created frames are recognized.)
      2) Normalize any {"type":"image","image":<path>} in **user** and plain **assistant** content
         to an **absolute** path (when the dataset JSON only provides a relative path like
         "image/AgentX_13.jpg"). If the path points to a video, extract frames as before and
         replace the block with image frames.

    Returns:
      dialogs: normalized dialog list
      extracted_frames: list of absolute frame paths extracted across the whole dialog
    """
    dialogs: List[dict] = []
    extracted_frames_all: List[str] = []

    for item in sample['dialogs']:
        if item['role'] == 'tool':
            dialog = dict(
                role='tool',
                name=item['name'],
                content=item['content'],
            )
            dialogs.append(dialog)

        elif item['role'] == 'assistant' and 'tool_calls' in item.keys():
            dialog = copy.deepcopy(item)
            # Rewrite only top-level string args of the first tool call
            try:
                func_args = dialog['tool_calls'][0]['function']['arguments']
                if isinstance(func_args, dict):
                    for name, value in list(func_args.items()):
                        if isinstance(value, str):
                            # Build absolute candidate (supports file:// too)
                            abs_candidate = _to_abs_path(value, path)
                            if os.path.exists(abs_candidate):
                                dialog['tool_calls'][0]['function']['arguments'][name] = str(Path(abs_candidate).resolve())
            except Exception:
                # Be robust; if arguments shape is unexpected, leave as-is
                pass
            dialogs.append(dialog)

        elif item['role'] == 'user':
            dialog = copy.deepcopy(item)
            transformed, frames = _maybe_replace_video_content(dialog.get('content'), path)
            if transformed is not None:
                dialog['content'] = transformed
                extracted_frames_all.extend(frames)
            dialogs.append(dialog)

        elif item['role'] == 'assistant':
            # Assistant messages without tool calls: also normalize image paths / convert videos
            dialog = copy.deepcopy(item)
            transformed, frames = _maybe_replace_video_content(dialog.get('content'), path)
            if transformed is not None:
                dialog['content'] = transformed
                extracted_frames_all.extend(frames)
            dialogs.append(dialog)

        else:
            dialogs.append(item)

    return dialogs, extracted_frames_all


@LOAD_DATASET.register_module()
class GTABenchDataset(BaseDataset):
    """GTA Benchmark with video-to-frames support and frame listing in resources.

    IMPORTANT: dataset.json may store ONLY relative file paths like:
      - image/AgentX_13.jpg
      - image/AgentX_20.mp4
    This loader automatically resolves them to absolute paths under `path`.
    """

    @staticmethod
    def load(path: str):
        data_root = Path(path)
        data_file = data_root / 'dataset.json'
        assert os.path.exists(data_file), f'Path {path} does not exist.'

        data = json.load(open(data_file))
        data_list = []
        for idx, item in data.items():
            idx = int(idx)

            # First, normalize dialogs (this will also extract frames & delete videos)
            dialogs, frame_paths = organize_dialogs(item, str(data_root.absolute()))

            # Tools from dataset.json
            tools = [
                dict(type='tool', name=tool['name']) for tool in item['tools']
            ]

            # Files from dataset.json (omit videos; keep others) — resolve relative -> absolute
            files = []
            for file in item['files']:
                abs_path = str((data_root / file['path']).absolute())
                suffix = Path(abs_path).suffix.lower()
                if suffix in VIDEO_EXTS:
                    # Skip adding the original video to resources since it is deleted
                    continue
                files.append(
                    dict(
                        type='file',
                        filetype=file.get('type', 'unknown'),
                        path=abs_path
                    )
                )

            # Add extracted frames to resources so tools can access them
            for fp in frame_paths:
                files.append(
                    dict(
                        type='file',
                        filetype='image',
                        path=str(Path(fp).resolve())
                    )
                )

            gt_answer = item['gt_answer']
            sample = {
                'dialogs': json.dumps(dialogs),
                'resources': json.dumps(tools + files),
                'gt_answer': json.dumps(gt_answer)
            }
            data_list.append(sample)

        dataset = Dataset.from_list(data_list)
        return dataset


@ICL_EVALUATORS.register_module()
class GTABenchEvaluator(BaseEvaluator):

    def __init__(self, mode) -> None:
        assert mode in ['every', 'every_with_gt']
        self.mode = mode
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

        # --------- OpenAI judge setup ----------
        self._judge_model = os.getenv('GTA_EVAL_GPT_JUDGE', 'gpt-4o')
        self._openai_ready = False
        self._use_legacy = False
        self._client = None

        if _HAS_OPENAI:
            # Prefer legacy style if present (to match your reference code)
            self._use_legacy = hasattr(_openai_pkg, 'ChatCompletion')
            api_key_env = os.getenv('OPENAI_API_KEY', '')
            try:
                if self._use_legacy:
                    # legacy SDK
                    if getattr(_openai_pkg, 'api_key', None) in (None, '') and api_key_env:
                        _openai_pkg.api_key = api_key_env
                    self._openai_ready = bool(getattr(_openai_pkg, 'api_key', None))
                else:
                    # new SDK layout: instantiate a client
                    try:
                        from openai import OpenAI as _OpenAIClient
                        self._client = _OpenAIClient(api_key=api_key_env if api_key_env else None)
                        self._openai_ready = self._client is not None
                    except Exception:
                        self._openai_ready = False
            except Exception:
                self._openai_ready = False

    # ---------------------------
    # Similarity helper (existing)
    # ---------------------------
    def bert_score(self, pred: str, gt: str) -> float:
        pred_emb = self.sentence_model.encode(pred, convert_to_tensor=True)
        gt_emb = self.sentence_model.encode(gt, convert_to_tensor=True)
        score = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
        return score[0][0]

    # ---------------------------
    # Formatting / extraction helpers for judge metrics
    # ---------------------------
    @staticmethod
    def _truncate(txt: str, max_chars: int = 1500) -> str:
        if txt is None:
            return ""
        txt = str(txt)
        return txt if len(txt) <= max_chars else (txt[:max_chars] + " ...[truncated]")

    @staticmethod
    def _first_or_join_aliases(ref: dict) -> str:
        try:
            wl = ref.get('whitelist', [])
            parts = []
            for group in wl:
                if isinstance(group, list) and group:
                    parts.append(group[0])
            return " / ".join(parts)
        except Exception:
            return ""

    @staticmethod
    def _extract_last_user_query(dialogs: List[dict]) -> str:
        last = ""
        for m in dialogs:
            if m.get('role') == 'user':
                last = m.get('content', '') or last
        return last or ""

    @staticmethod
    def _extract_final_answer_from_dialog(dialogs: List[dict]) -> str:
        final = ""
        for m in dialogs:
            if m.get('role') == 'assistant' and 'tool_calls' not in m:
                final = m.get('content', '') or final
        return final or ""

    @staticmethod
    def _format_reasoning_trace(dialogs: List[dict], max_field_chars: int = 800) -> str:
        lines = []
        step_id = 1
        i = 0
        n = len(dialogs)
        while i < n:
            msg = dialogs[i]
            # Assistant tool call
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                fn = msg['tool_calls'][0].get('function', {})
                tname = fn.get('name', '')
                targs = fn.get('arguments', {})
                thought = msg.get('content', '')
                lines.append(f"Step {step_id}:")
                lines.append(f"  Task/Thought: {GTABenchEvaluator._truncate(thought, max_field_chars)}")
                lines.append(f"  Tool: {tname}")
                try:
                    args_str = json.dumps(targs, ensure_ascii=False)
                except Exception:
                    args_str = str(targs)
                lines.append(f"  Input: {GTABenchEvaluator._truncate(args_str, max_field_chars)}")
                tool_output = None
                if i + 1 < n and dialogs[i + 1].get('role') in ('tool', 'function'):
                    tool_output = dialogs[i + 1].get('content', '')
                    i += 1  # consume tool message
                if tool_output is not None:
                    lines.append(f"  Output: {GTABenchEvaluator._truncate(tool_output, max_field_chars)}")
                step_id += 1
            elif msg.get('role') == 'assistant' and 'tool_calls' not in msg:
                content = msg.get('content', '')
                if content:
                    lines.append(f"Assistant note/answer: {GTABenchEvaluator._truncate(content, max_field_chars)}")
            elif msg.get('role') in ('tool', 'function'):
                content = msg.get('content', '')
                lines.append(f"Tool return (unpaired): {GTABenchEvaluator._truncate(content, max_field_chars)}")
            elif msg.get('role') == 'user':
                content = msg.get('content', '')
                lines.append(f"User: {GTABenchEvaluator._truncate(content, max_field_chars)}")
            i += 1
        return "\n".join(lines)

    @staticmethod
    def _build_gt_block(gt_dialogs: List[dict], ref_text_fallback: str = "") -> str:
        query = GTABenchEvaluator._extract_last_user_query(gt_dialogs)
        gt_final = GTABenchEvaluator._extract_final_answer_from_dialog(gt_dialogs) or ref_text_fallback
        gt_steps = GTABenchEvaluator._format_reasoning_trace(gt_dialogs)
        payload = {
            "query": query,
            "GT reasoning steps": gt_steps,
            "GT final answer": gt_final
        }
        return str(payload)

    # ---------------------------
    # OpenAI judge helpers (prompts & calling)
    # ---------------------------
    @staticmethod
    def _safe_parse_score_dict(text: str):
        result = {'Score': None, 'Justification': ''}
        if not text:
            return result
        text = text.strip()
        # try JSON
        try:
            obj = json.loads(text)
            result.update(obj)
        except Exception:
            try:
                obj = ast.literal_eval(text)
                if isinstance(obj, dict):
                    result.update(obj)
            except Exception:
                m = re.search(r"[\"']?Score[\"']?\s*:\s*[\"']?([0-9.]+)[\"']?", text, re.IGNORECASE)
                if m:
                    result['Score'] = float(m.group(1))
                jm = re.search(r"[\"']?Justification[\"']?\s*:\s*[\"']?(.+)[\"']?", text, re.IGNORECASE | re.DOTALL)
                if jm:
                    result['Justification'] = jm.group(1).strip()
        try:
            if result['Score'] is not None:
                result['Score'] = float(result['Score'])
                result['Score'] = max(0.0, min(1.0, result['Score']))
        except Exception:
            result['Score'] = None
        if not isinstance(result.get('Justification', ''), str):
            result['Justification'] = str(result['Justification'])
        return result

    def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
        if not self._openai_ready:
            raise RuntimeError("OpenAI API is not configured (missing package or OPENAI_API_KEY).")

        if self._use_legacy:
            resp = _openai_pkg.ChatCompletion.create(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return resp['choices'][0]['message']['content']
        else:
            msg = self._client.chat.completions.create(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return msg.choices[0].message.content

    _PROMPT_FAITHFULNESS = """
You are an evaluation assistant measuring the Faithfulness Accuracy of an agent's reasoning process.

You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.
You are also given the full reasoning trace produced by the agent, including each step's task, selected tool, output, and thought.

Your task is to assess how faithful the agent's reasoning trace is to the GT — that is, whether the steps follow a logically sound plan that aligns with the structure, intent, and direction of the GT.

Evaluation Guidelines:
- Focus on the structure and logical flow of the agent's reasoning steps.
- Determine whether the steps collectively form a coherent strategy to answer the query.
- Faithfulness is about consistency with the GT's method, not necessarily correctness of individual steps.

Scoring Criteria: Output the scores between the range 0 and 1 such that:
1 — represents the reasoning is faithful to the GT: it follows a logically sound plan that mirrors the GT in structure and direction.
0 — represents the reasoning is not faithful to the GT and lacks logical progression or alignment with the intended plan.

Final output must be a single python dictionary as below:
{'Score': '<0-1>','Justification': '< explanation for the score>'}
""".strip()

    _PROMPT_CONTEXT = """
You are an evaluation assistant measuring the Context Score of an agent's reasoning step.

You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

Your task is to assess whether the agent's reasoning step is grounded in the input context — and whether that context was effectively used in the reasoning process. Inputs may include image, video, text, audio, or a combination.

Evaluation Guidelines:
If the number of steps in the GT and the model differ:
- Any extra model step beyond the GT steps should receive a score of 0 (hallucinated or unjustified).
- Any GT step that the model omits should also receive a score of 0 (missing reasoning).
The Context Score for the full query is computed as the average of the per-step scores.

Assess the following per step:
- Does the agent correctly use relevant parts of the input?
- Is the input used appropriate for the tool selected and the task being attempted?
- Does the reasoning show effective and meaningful use of the input?

Scoring Criteria: Output the scores between the range 0 and 1 such that:
1 — represents the agent uses the input fully and appropriately.
0 — represents the agent ignores or misinterprets the input entirely.

Final output must be a single python dictionary as below:
{'Score': '<0 - 1>','Justification': '< explanation for the score>'}
""".strip()

    _PROMPT_FACTUAL = """
You are an evaluation assistant measuring the factual accuracy of the agent's reasoning steps.

You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification.

You are also given the full reasoning trace produced by the agent, including each ste's task, selected tool, tool input/output, and thought.

Your task is to assess whether the agent introduces any hallucinated, fabricated, or factually incorrect information in its reasoning when compared to the GT.

Evaluation Guidelines:
- Compare the model's reasoning steps to the GT to identify factual hallucinations or incorrect claims.
- Focus on whether the output or thought includes details not supported by the GT.
- Do not penalize for minor omissions unless they lead to a factual error.

Scoring Criteria: Output the scores between the range 0 and 1 such that:
1 — represents no factual errors or hallucinations compared to the GT.
0 — represents major factual errors or hallucinated content.

Final output must be a single python dictionary as below:
{'Score': '<0 - 1>','Justification': '< explanation for the score>'}
""".strip()

    _PROMPT_SEMANTIC = """
You are an evaluation assistant measuring the Semantic Accuracy of an agent's reasoning process.

You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification.

You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

Your task is to assess whether the agent's reasoning and final output semantically align with the Ground Truth — that is, whether the agent has covered all the essential parts of the query as demonstrated in the GT.

Evaluation Guidelines:
- Compare the agent's reasoning trace and final answer with the GT to check whether all key components of the query are addressed.
- Credit should be given for meaningful semantic coverage, not superficial similarity.
- If the model ignores or misunderstands core parts of the GT reasoning or final answer, penalize accordingly.

Scoring Criteria: Output the scores between the range 0 and 1 such that:
1 — represents the agent addresses all key components of the query, matching the GT's semantic scope.
0 — represents the agent's reasoning or answer omits or misrepresents major parts of the query.

Final output must be a single python dictionary as below:
{'Score': '<0 - 1>','Justification': '< explanation for the score>'}
""".strip()

    # ---------------------------
    # Existing helpers
    # ---------------------------
    @staticmethod
    def get_response_type(item):
        if 'tool_calls' in item:
            return 'tool', item['tool_calls'][0]['function']
        elif item['role'] == 'assistant':
            return 'answer', item['content']
        else:
            return 'tool_return', item['content']

    @staticmethod
    def iscorrect(pred: str, ref: dict):
        count = 0
        for aliases in ref['whitelist']:
            pattern = r'\b(?:' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
            if re.search(pattern, pred, re.IGNORECASE):
                count += 1
        if not ref['blacklist']:
            if count == len(ref['whitelist']):
                return True
        else:
            pattern_bk = r'\b(?:' + '|'.join(re.escape(alias) for aliases in ref['blacklist'] for alias in aliases) + r')\b'
            if count == len(ref['whitelist']) and not re.search(pattern_bk, pred, re.IGNORECASE):
                return True
        return False

    def simscore(self, pred: str, ref: list):
        max_score = 0
        for s in ref:
            score = self.bert_score(pred, s)
            if score > max_score:
                max_score = score
        return max_score

    @staticmethod
    def gettype(name: str):
        perception = ['OCR', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox']
        operation = ['DrawBox', 'AddText', 'GoogleSearch']
        logic = ['Calculator', 'Solver', 'Plot', 'MathOCR', 'CountGivenObject']
        creativity = ['TextToImage', 'ImageStylization']
        if name in perception:
            return 'perception'
        elif name in operation:
            return 'operation'
        elif name in logic:
            return 'logic'
        elif name in creativity:
            return 'creativity'
        else:
            return 'none'

    # ---------------------------
    # Main scoring
    # ---------------------------
    def score(self, predictions: list, references: list, gold: list):
        if self.mode == 'every_with_gt':
            total = {'tool': 0, 'answer': 0}
            metrics = {
                'inst_align': 0,
                'tool_acc': 0,
                'arg_acc': 0,
                'answer_acc': 0,
                'tool_call': 0,
                'tool_call_error': 0
            }

            # LLM-judge metrics accumulators
            judge_sum = {
                'faithfulness': 0.0,
                'context_score': 0.0,
                'factual_precision': 0.0,
                'semantic_accuracy': 0.0
            }
            judge_cnt = {
                'faithfulness': 0,
                'context_score': 0,
                'factual_precision': 0,
                'semantic_accuracy': 0
            }

            for preds, gts, ref in zip(predictions, gold, references):
                ref_obj = json.loads(ref)
                if ref_obj:
                    total['answer'] += 1

                for pred, gt in zip(preds, gts):
                    pred_type, pred_ = self.get_response_type(pred)
                    gt_type, gt_ = self.get_response_type(gt)
                    if pred_type == gt_type and 'error' not in pred:
                        metrics['inst_align'] += 1
                    if gt_type == 'tool':
                        total['tool'] += 1
                    if pred_type == 'tool':
                        metrics['tool_call'] += 1
                        if 'error' in pred:
                            metrics['tool_call_error'] += 1
                    if pred_type == gt_type == 'tool' and pred_['name'] == gt_['name']:
                        metrics['tool_acc'] += 1
                        if pred_['arguments'] == gt_['arguments']:
                            metrics['arg_acc'] += 1
                    elif pred_type == gt_type == 'answer':
                        if isinstance(ref_obj, dict):
                            metrics['answer_acc'] += self.iscorrect(pred_, ref_obj)
                        elif isinstance(ref_obj, list):
                            metrics['answer_acc'] += self.simscore(pred_, ref_obj)

                # ---------------------------
                # LLM-judged metrics (per-sample)
                # ---------------------------
                try:
                    gt_fallback_answer = ""
                    if isinstance(ref_obj, dict):
                        gt_fallback_answer = self._first_or_join_aliases(ref_obj)
                    elif isinstance(ref_obj, list) and ref_obj:
                        if isinstance(ref_obj[0], str):
                            gt_fallback_answer = ref_obj[0]

                    gt_block_str = self._build_gt_block(gts, ref_text_fallback=gt_fallback_answer)
                    pred_trace_str = self._format_reasoning_trace(preds)
                    pred_final_answer = self._extract_final_answer_from_dialog(preds)

                    if self._openai_ready:
                        faith_resp = self._chat_once(
                            self._PROMPT_FAITHFULNESS,
                            "GT: " + gt_block_str + "\n" + "agent's reasoning trace: " + pred_trace_str
                        )
                        faith = self._safe_parse_score_dict(faith_resp).get('Score', None)
                        if faith is not None:
                            judge_sum['faithfulness'] += float(faith)
                            judge_cnt['faithfulness'] += 1

                        ctx_resp = self._chat_once(
                            self._PROMPT_CONTEXT,
                            "GT: " + gt_block_str + "\n" + "agent's reasoning_steps: " + pred_trace_str
                        )
                        ctx = self._safe_parse_score_dict(ctx_resp).get('Score', None)
                        if ctx is not None:
                            judge_sum['context_score'] += float(ctx)
                            judge_cnt['context_score'] += 1

                        fp_resp = self._chat_once(
                            self._PROMPT_FACTUAL,
                            "GT: " + gt_block_str + "\n" + "agent's reasoning steps: " + pred_trace_str
                        )
                        fp = self._safe_parse_score_dict(fp_resp).get('Score', None)
                        if fp is not None:
                            judge_sum['factual_precision'] += float(fp)
                            judge_cnt['factual_precision'] += 1

                        sem_resp = self._chat_once(
                            self._PROMPT_SEMANTIC,
                            "GT: " + gt_block_str + "\n" +
                            "agent's reasoning steps: " + pred_trace_str + "\n" +
                            "agent's final answer: " + (pred_final_answer or "")
                        )
                        sem = self._safe_parse_score_dict(sem_resp).get('Score', None)
                        if sem is not None:
                            judge_sum['semantic_accuracy'] += float(sem)
                            judge_cnt['semantic_accuracy'] += 1
                except Exception:
                    pass

            inst_align = metrics['inst_align'] / max(sum(total.values()), 1) * 100
            tool_acc = metrics['tool_acc'] / max(total['tool'], 1) * 100
            arg_acc = metrics['arg_acc'] / max(total['tool'], 1) * 100
            answer_acc = metrics['answer_acc'] / max(total['answer'], 1) * 100

            def avg_pct(name: str) -> float:
                return (judge_sum[name] / judge_cnt[name] * 100) if judge_cnt[name] > 0 else None

            result = dict(
                inst_align=inst_align,
                tool_acc=tool_acc,
                arg_acc=arg_acc,
                answer_acc=answer_acc,
                tool_call=metrics['tool_call'],
                tool_call_error=metrics['tool_call_error'],
                faithfulness=avg_pct('faithfulness'),
                context_score=avg_pct('context_score'),
                factual_precision=avg_pct('factual_precision'),
                semantic_accuracy=avg_pct('semantic_accuracy'),
            )
            return result

        elif self.mode == 'every':
            total = {'all': 0, 'answer': 0, 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'none': 0}
            total_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            precision = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            recall = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            f1 = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0}
            metrics = {
                'answer_acc': 0, 'answer_acc_w_imggen': 0, 'tool_call': 0, 'tool_call_error': 0,
                'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0
            }
            imagegen_tools = ['DrawBox', 'AddText', 'Plot', 'TextToImage', 'ImageStylization']
            for preds, gts, ref in zip(predictions, gold, references):
                ref = json.loads(ref)
                total['all'] += 1
                if ref:
                    total['answer'] += 1
                pred_type, pred_answer = self.get_response_type(preds[0][-1])
                if ref and pred_type == 'answer' and pred_answer:
                    if isinstance(ref, dict) and pred_answer:
                        score = self.iscorrect(pred_answer, ref)
                        metrics['answer_acc'] += score
                        metrics['answer_acc_w_imggen'] += score
                    elif isinstance(ref, list) and pred_answer:
                        score = self.simscore(pred_answer, ref)
                        metrics['answer_acc'] += score
                        metrics['answer_acc_w_imggen'] += score
                # add imagegen to ansacc
                if not ref:
                    pred_imagegen = dict()
                    score = 1
                    for pred in preds[0]:
                        if 'tool_calls' in pred and 'error' not in pred:
                            tool_name = pred['tool_calls'][0]['function']['name']
                            tool_arg = pred['tool_calls'][0]['function']['arguments']
                            if tool_name in imagegen_tools:
                                pred_imagegen[tool_name] = tool_arg
                    for gt in gts[0]:
                        if 'tool_calls' in gt:
                            tool_name = gt['tool_calls'][0]['function']['name']
                            tool_arg = gt['tool_calls'][0]['function']['arguments']
                            if tool_name in imagegen_tools:
                                if tool_name not in pred_imagegen:
                                    score = 0
                                    break
                                else:
                                    score = score * self.bert_score(json.dumps(tool_arg),
                                                                    json.dumps(pred_imagegen[tool_name]))
                    metrics['answer_acc_w_imggen'] += score

                for pred in preds[0]:
                    if 'tool_calls' in pred:
                        metrics['tool_call'] += 1
                        if 'error' in pred:
                            metrics['tool_call_error'] += 1

                pred_tool_calls = []
                for pred in preds[0]:
                    if 'tool_calls' in pred:
                        tool_type = self.gettype(pred['tool_calls'][0]['function']['name'])
                        if tool_type in total_predict:
                            total_predict[tool_type] += 1
                        pred_tool_calls.append(pred['tool_calls'][0]['function']['name'])
                for gt in gts[0]:
                    if 'tool_calls' in gt:
                        tool_type = self.gettype(gt['tool_calls'][0]['function']['name'])
                        total[tool_type] += 1
                        if gt['tool_calls'][0]['function']['name'] in pred_tool_calls:
                            metrics[tool_type] += 1
            for tool_type in f1.keys():
                precision[tool_type] = metrics[tool_type] / (total_predict[tool_type] + 1e-5)
                recall[tool_type] = metrics[tool_type] / (total[tool_type] if total[tool_type] > 0 else 1)
                f1[tool_type] = 2 * precision[tool_type] * recall[tool_type] / (precision[tool_type] + recall[tool_type] + 1e-5)
            return dict(
                answer_acc=metrics['answer_acc'] / total['answer'] * 100 if total['answer'] > 0 else 0.0,
                answer_acc_w_imggen=metrics['answer_acc_w_imggen'] / total['all'] * 100 if total['all'] > 0 else 0.0,
                tool_call=metrics['tool_call'],
                tool_call_error=metrics['tool_call_error'],
                p_f1=f1['perception'] * 100,
                o_f1=f1['operation'] * 100,
                l_f1=f1['logic'] * 100,
                c_f1=f1['creativity'] * 100,
                total_ansacc=metrics['answer_acc'],
                total_ansacc_wimggen=metrics['answer_acc_w_imggen']
            )
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")
