#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import pyaudio
import rclpy
from rclpy.node import Node
from faster_whisper import WhisperModel
from openai import OpenAI, APIError
import time  

from erasers_kachaka_common.tts import TTS  # TTS モジュール（ROS Node 必須）

def record_audio(seconds=5, rate=16000, chunk=1024):
    """音声録音"""
    print("🎙️ 録音開始...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    for _ in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("🛑 録音終了")
    return np.concatenate(frames, axis=0)

def transcribe_audio(audio_data):
    """Whisperによる文字起こし"""
    print("🧠 Whisperによる文字起こし中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("large", device=device, compute_type="float32")

    segments, _ = model.transcribe(audio_data, beam_size=5, language="ja")
    return " ".join([seg.text for seg in segments])

def query_chatgpt(message):
    """ChatGPTとの対話"""
    print("🤖 ChatGPTに問い合わせ中...")
    client = OpenAI()
    messages = [
        {"role": "system", "content": "端的に簡素な返事をするロボット"},
        {"role": "user", "content": message}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content.strip()
    except APIError as e:
        print(f"❌ ChatGPT API error: {e}")
        return "すみません、エラーが発生しました。"

def main():
    rclpy.init()
    node = Node("tts_node_dummy")
    tts = TTS(node)

    try:
        audio_data = record_audio(seconds=5)
        transcription = transcribe_audio(audio_data)

        print(f"📝 認識結果: {transcription}")

        if len(transcription.strip()) < 3 or "ご視聴ありがとうございました" in transcription:
            print("⚠️ 無効な認識結果のため終了します。")
            return

        reply = query_chatgpt(transcription)
        print(f"🤖 ChatGPTの応答: {reply}")

        print("🔊 TTS 発話中...")
        tts.say(reply)
        print("✅ 完了しました。")

    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
