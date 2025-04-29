#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from faster_whisper import WhisperModel
from openai import OpenAI, APIError
import time  

import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from erasers_kachaka_common.tts import TTS  # TTS モジュール（ROS Node 必須）

class HumanTracker(Node):
    """人物追跡クラス"""
    def __init__(self):
        super().__init__("human_tracker")
        self.bridge = CvBridge()
        self.face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")


        # ロボット制御コマンドを送信するパブリッシャーのQoS設定
        self.cmd_pub = self.create_publisher(Twist, "/er_kachaka/manual_control/cmd_vel", 10)
        
        # サブスクライバーのQoS設定 (BestEffortで受信)
        qos_profile = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/er_kachaka/front_camera/image_raw", self.get_image, qos_profile)

    def get_image(self, msg):
        """画像を取得し人物追跡を開始"""
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.track_person(x, y, w, h, image.shape[1])

    def track_person(self, x, y, w, h, frame_width):
        """顔の位置をもとにロボットを移動"""
        cmd = Twist()
        center_x = x + w / 2
        cmd.angular.z = 0.2 if center_x < frame_width / 2 - 30 else -0.2 if center_x > frame_width / 2 + 30 else 0.0
        self.cmd_pub.publish(cmd)

    def start_tracking(self):
        """追跡を開始"""
        rclpy.spin(self)

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
    tracker = HumanTracker()  # 🔥 追跡クラスをインスタンス化

    try:
        # 音声ではなく手動入力を受け取る
        transcription = input("何をして欲しいですか？: ")

        print(f"📝 入力結果: {transcription}")

        if len(transcription.strip()) < 3 or "ご視聴ありがとうございました" in transcription:
            print("⚠️ 無効な認識結果のため終了します。")
            return

        if "ついてきて" in transcription:
            print("🚶‍♂️ 追跡モードを直接起動！")
            tts.say("追跡します")
            tracker.start_tracking()  # 🔥 クラスを直接呼び出す
            return
        
        if "カチャカスタート" in transcription:
            tts.say("ねえ、カチャカ")
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
