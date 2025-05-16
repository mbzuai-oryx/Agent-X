import argparse
import random
import socket
import time
import traceback

import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer
from threading import Thread
from multiprocessing import Process, Queue

EOS_FLAG = "<EOS>"
SEPARATOR = "<SEP>"


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Streamer(object):

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.queue = Queue(maxsize=1024)
        self.stop_signal = EOS_FLAG

    def put(self, value):
        self.queue.put(value)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            value = self.queue.get(timeout=self.timeout)
        except:
            raise StopIteration()

        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class VideoLLaMA3PlainClient(object):

    def __init__(self, host="localhost", port=16666):
        self.host = host
        self.port = port

        self.input_buffer = Queue(maxsize=1024)
        self.streamers = dict()
        self.logger = get_logger("videollama3.client")

        client_thread = Thread(target=self._client_worker)
        client_thread.deamon = True
        client_thread.start()        

    def _receive_worker(self, server_socket):
        try:
            while True:
                data = server_socket.recv(8192)
                if not data:
                    self.logger.info(f"Connection has been terminated.")
                    for streamer in self.streamers.values():
                        streamer.put(streamer.stop_signal)
                    break

                for sub_data in data.decode("utf-8").split(SEPARATOR):
                    if len(sub_data) == 0:
                        continue

                    try:
                        sub_data = json.loads(sub_data)
                    except:
                        self.logger.info(f"Failed to parse data: {sub_data}")
                        continue

                    self.logger.info(f"Received: {sub_data['data']}")
                    self.streamers[sub_data["id"]].put(sub_data["data"])

                    if sub_data["data"] == EOS_FLAG:
                        self.streamers.pop(sub_data["id"])

        except ConnectionResetError:
            self.logger.info(f"Connection has been terminated.")

    def _send_worker(self, server_socket):
        while True:
            request_id, conversation = self.input_buffer.get()
            data = json.dumps({"id": request_id, "data": conversation}) + SEPARATOR
            server_socket.sendall(data.encode("utf-8"))
            self.logger.info(f"Sent: {data}")

    def _client_worker(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            while True:
                try:
                    server_socket.connect((self.host, self.port))
                    break
                except ConnectionRefusedError:
                    self.logger.info("Waiting for the server to start...")
                    time.sleep(1)
                    continue

            self.logger.info("Connected to server.")
            receive_thread = Thread(target=self._receive_worker, args=(server_socket,))
            receive_thread.daemon = True
            receive_thread.start()

            send_thread = Thread(target=self._send_worker, args=(server_socket,))
            send_thread.daemon = True
            send_thread.start()

            receive_thread.join()

    def submit(self, conversation):
        request_id = random.randint(0, 4294967295)
        streamer = Streamer()
        self.streamers[request_id] = streamer
        self.input_buffer.put((request_id, conversation))
        return streamer


class VideoLLaMA3PlainServer(object):

    def __init__(
        self,
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_processes=1,
        buffer_size=2,
        host="localhost",
        port=16666,
    ):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.num_processes = num_processes
        self.buffer_size = buffer_size

        self.host = host
        self.port = port

    def _model_worker(self, input_buffer, output_buffer, device_map, rank):
        logger = get_logger(f"videollama3.server.worker_{rank}")
        logger.info(f"Loading model from {self.model_path}...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        logger.info(f"Successfully loaded model.")

        while True:
            logger.info("Waiting for input...")
            request_id, data = input_buffer.get()
            try:
                inputs = processor(
                    conversation=data["conversation"],
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(f"cuda:{rank}") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = {
                    **inputs,
                    **data["generation_config"],
                    "streamer": streamer,
                }

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.deamon = True
                thread.start()

                for token in streamer:
                    output_buffer.put((request_id, token))
                output_buffer.put((request_id, EOS_FLAG))

            except:
                logger.error(f"An error occurred: {traceback.format_exc()}")
                output_buffer.put((request_id, "Server error! Please check the server logs and retry."))
                output_buffer.put((request_id, EOS_FLAG))

    def _receive_worker(self, logger, input_buffer, client_socket, client_address):
        try:
            while True:
                data = client_socket.recv(8192)
                if not data:
                    logger.info(f"Connection from {client_address} has been terminated.")
                    break

                for sub_data in data.decode("utf-8").split(SEPARATOR):
                    if len(sub_data) == 0:
                        continue

                    try:
                        sub_data = json.loads(sub_data)
                    except:
                        logger.info(f"Failed to parse data: {sub_data}")
                        continue

                    logger.info(f"Received from {client_address}: {sub_data}")
                    input_buffer.put((sub_data["id"], sub_data["data"]))

        except ConnectionResetError:
            logger.info(f"Connection from {client_address} has been terminated.")

    def _send_worker(self, logger, output_buffer, client_socket, client_address):
        try:
            while True:
                request_id, token = output_buffer.get()
                data = json.dumps({"id": request_id, "data": token}) + SEPARATOR
                client_socket.sendall(data.encode("utf-8"))

        except ConnectionResetError:
            logger.info(f"Connection from {client_address} has been terminated.")

    def launch(self):
        logger = get_logger(f"videollama3.server.controller")

        input_buffer = Queue(maxsize=self.num_processes * self.buffer_size)
        output_buffer = Queue(maxsize=self.num_processes * 1024)

        for i in range(self.num_processes):
            device_map = {"": f"cuda:{i}"}
            process = Process(target=self._model_worker, args=(input_buffer, output_buffer, device_map, i))
            process.start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            logger.info("Waiting for connection...")

            while True:
                client_socket, client_address = server_socket.accept()
                logger.info(f"Connected to {client_address}.")

                receive_thread = Thread(target=self._receive_worker, args=(logger, input_buffer, client_socket, client_address))
                receive_thread.deamon = True
                receive_thread.start()

                send_thread = Thread(target=self._send_worker, args=(logger, output_buffer, client_socket, client_address))
                send_thread.deamon = True
                send_thread.start()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--port", type=int, default=16666)
    args = parser.parse_args()

    server = VideoLLaMA3PlainServer(
        model_path=args.model_path,
        num_processes=args.nproc,
        port=args.port,
    )
    server.launch()
