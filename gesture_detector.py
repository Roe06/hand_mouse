"""
手势检测器模块
使用 MediaPipe 检测手部关键点并识别预设手势
"""

import cv2
import numpy as np
import time
import config


class GestureDetector:
    def __init__(self):
        # 初始化 MediaPipe 手部检测
        # 动态导入以兼容不同版本
        import mediapipe as mp
        
        # 尝试访问 solutions
        if hasattr(mp, 'solutions'):
            mp_hands_module = mp.solutions.hands
            mp_drawing_module = mp.solutions.drawing_utils
        else:
            raise ImportError(
                "MediaPipe 版本不兼容。请运行: pip install mediapipe --upgrade"
            )
        
        self.mp_hands_module = mp_hands_module
        self.mp_drawing = mp_drawing_module
        self.mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
        
        self.hands = mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,  # 只检测一只手
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 手势状态跟踪
        self.prev_open_hand_y = None  # 追踪五指张开时的Y轴位置
        self.prev_index_pos = None
        self.last_click_time = 0
        self.last_right_click_time = 0
        self.last_scroll_time = 0
        
        # 手势状态记录
        self.is_pinching = False  # 是否正在捏合（用于持续左键）
        self.pinch_start_time = 0  # 捏合开始时间
        
        # 手势识别状态
        self.current_gesture = "None"
        
    def detect_hand_landmarks(self, frame):
        """
        检测手部关键点
        返回: landmarks 对象和处理后的 RGB 图像
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results, rgb_frame
    
    def draw_landmarks(self, frame, results):
        """在图像上绘制手部关键点"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands_connections
                )
        return frame
    
    def get_finger_tip_positions(self, landmarks):
        """
        获取手指尖端位置
        返回: 字典包含拇指、食指、中指、无名指、小指的坐标
        """
        return {
            'thumb': landmarks.landmark[4],      # 拇指尖
            'index': landmarks.landmark[8],      # 食指尖
            'middle': landmarks.landmark[12],    # 中指尖
            'ring': landmarks.landmark[16],      # 无名指尖
            'pinky': landmarks.landmark[20],     # 小指尖
            'wrist': landmarks.landmark[0],      # 手腕
            'palm': landmarks.landmark[9]        # 手掌中心
        }
    
    def calculate_distance(self, point1, point2):
        """计算两个关键点之间的欧氏距离"""
        return np.sqrt(
            (point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2 +
            (point1.z - point2.z) ** 2
        )
    
    def detect_pinch(self, positions):
        """
        检测拇指和食指捏合手势（模拟鼠标左键）
        不管其他手指状态，只要拇指和食指捏合就触发
        """
        # 拇指和食指距离
        thumb_index_dist = self.calculate_distance(positions['thumb'], positions['index'])
        
        # 捏合条件：拇指食指很近即可
        is_pinch = thumb_index_dist < config.PINCH_THRESHOLD
        
        return is_pinch
    

    def detect_only_index_up(self, positions):
        """
        检测是否只有食指伸出（用于鼠标移动）
        其他手指应该收拢或半收拢
        """
        palm = positions['palm']
        
        # 食指距离（应该伸直）
        index_dist = self.calculate_distance(positions['index'], palm)
        # 中指距离（应该收拢）
        middle_dist = self.calculate_distance(positions['middle'], palm)
        # 无名指和小指
        ring_dist = self.calculate_distance(positions['ring'], palm)
        pinky_dist = self.calculate_distance(positions['pinky'], palm)
        
        # 只有食指伸出（放宽食指阈值，严格其他手指）
        return (index_dist > 0.14 and 
                middle_dist < 0.12 and 
                ring_dist < 0.12 and 
                pinky_dist < 0.12)
    
    def detect_thumb_index_open(self, positions):
        """
        检测食指和拇指是否都伸出张开（不是捏合）
        """
        palm = positions['palm']
        
        # 食指和拇指都应该伸直
        thumb_dist = self.calculate_distance(positions['thumb'], palm)
        index_dist = self.calculate_distance(positions['index'], palm)
        
        # 拇指和食指之间的距离（应该较远，表示张开）
        thumb_index_dist = self.calculate_distance(positions['thumb'], positions['index'])
        
        # 食指和拇指都伸出，且距离较远（张开状态）
        both_extended = (thumb_dist > 0.12 and index_dist > 0.15)
        not_pinching = thumb_index_dist > config.PINCH_THRESHOLD
        
        return both_extended and not_pinching
    
    def detect_open_hand(self, positions):
        """
        检测手掌是否张开（五指伸展）
        必须五指都伸展才识别为张开手掌
        严格检测，避免与单食指混淆
        """
        palm = positions['palm']
        
        # 计算所有手指到手掌的距离
        thumb_dist = self.calculate_distance(positions['thumb'], palm)
        index_dist = self.calculate_distance(positions['index'], palm)
        middle_dist = self.calculate_distance(positions['middle'], palm)
        ring_dist = self.calculate_distance(positions['ring'], palm)
        pinky_dist = self.calculate_distance(positions['pinky'], palm)
        
        # 所有手指都必须伸展（更严格的条件，避免单食指误触）
        all_extended = (
            thumb_dist > 0.13 and
            index_dist > 0.15 and
            middle_dist > 0.14 and
            ring_dist > 0.14 and
            pinky_dist > 0.14
        )
        
        return all_extended
    
    def detect_fist(self, positions):
        """
        检测握拳手势（模拟鼠标右键）
        必须五指紧握才触发，特别注意避免与拇指食指张开混淆
        """
        palm = positions['palm']
        
        # 计算所有指尖到手掌的距离
        distances = [
            self.calculate_distance(positions['thumb'], palm),
            self.calculate_distance(positions['index'], palm),
            self.calculate_distance(positions['middle'], palm),
            self.calculate_distance(positions['ring'], palm),
            self.calculate_distance(positions['pinky'], palm)
        ]
        avg_distance = np.mean(distances)
        max_distance = max(distances)
        
        # 握拳条件：平均距离很小且所有手指都收拢（更严格）
        # 如果食指或拇指伸出，则不是握拳
        thumb_dist = distances[0]
        index_dist = distances[1]
        
        is_fist = (avg_distance < config.FIST_THRESHOLD and 
                   max_distance < config.FIST_THRESHOLD * 2.0 and
                   thumb_dist < 0.10 and  # 拇指必须收拢
                   index_dist < 0.10)      # 食指必须收拢
        
        return is_fist
    
    def detect_open_hand_swipe(self, positions, frame_height):
        """
        检测五指张开时大幅度上下扫动
        返回: 'scroll_up', 'scroll_down', 或 None
        注意：手向下扫→向上滚动，手向上扫→向下滚动
        """
        # 首先检查是否是五指张开姿势
        if not self.detect_open_hand(positions):
            self.prev_open_hand_y = None
            return None
        
        # 使用手掌中心的Y坐标
        current_y = positions['palm'].y * frame_height
        
        if self.prev_open_hand_y is None:
            self.prev_open_hand_y = current_y
            return None
        
        y_diff = self.prev_open_hand_y - current_y  # Y轴向下为正
        
        # 降低阈值，使滚动更灵敏
        threshold = frame_height * config.SWIPE_THRESHOLD
        
        result = None
        # 手向上移动（y_diff > 0）-> 触发向下滚动
        if y_diff > threshold:
            result = 'scroll_down'
            # 持续更新位置以支持连续滚动
            self.prev_open_hand_y = current_y
        # 手向下移动（y_diff < 0）-> 触发向上滚动
        elif y_diff < -threshold:
            result = 'scroll_up'
            # 持续更新位置以支持连续滚动
            self.prev_open_hand_y = current_y
        
        return result
    
    def get_index_position(self, positions, frame_width, frame_height):
        """
        获取食指位置用于控制鼠标移动
        返回: (x, y) 归一化坐标
        """
        index = positions['index']
        return (index.x * frame_width, index.y * frame_height)
    
    def recognize_gesture(self, results, frame_shape):
        """
        主要手势识别函数
        返回: 字典包含手势类型和相关数据
        """
        if not results.multi_hand_landmarks:
            self.current_gesture = "None"
            self.prev_open_hand_y = None
            self.is_pinching = False
            return {
                'type': 'none',
                'data': None
            }
        
        landmarks = results.multi_hand_landmarks[0]
        positions = self.get_finger_tip_positions(landmarks)
        frame_height, frame_width = frame_shape[:2]
        
        current_time = time.time()
        
        # 新逻辑优先级:
        # 1. 捏合 → 点击左键或长按1秒+移动
        # 2. 握拳 → 右键点击
        # 3. 单食指 → 移动鼠标（提高优先级，避免被五指误识别）
        # 4. 食指+拇指张开 → 无操作
        # 5. 五指张开滚动（大幅度扫动）
        
        # 1. 检测捏合手势（左键点击或长按1秒后拖拽）
        is_pinching = self.detect_pinch(positions)
        if is_pinching:
            if not self.is_pinching:
                # 刚开始捏合 - 触发按下
                if current_time - self.last_click_time > config.CLICK_COOLDOWN:
                    self.is_pinching = True
                    self.pinch_start_time = current_time
                    self.last_click_time = current_time
                    self.current_gesture = "Pinch Start (Left Press)"
                    return {
                        'type': 'left_press',
                        'data': None
                    }
            else:
                # 持续捏合 - 检查时长
                pinch_duration = current_time - self.pinch_start_time
                if pinch_duration > 1.0:  # 捏合超过1秒，允许拖拽
                    # 获取食指位置用于拖拽
                    index_pos = self.get_index_position(positions, frame_width, frame_height)
                    self.current_gesture = "Pinch Hold + Move (Dragging)"
                    return {
                        'type': 'pinch_drag',
                        'data': index_pos
                    }
                else:
                    # 捏合但未超过1秒 - 只保持按下，不移动鼠标
                    self.current_gesture = "Pinch Hold (Holding)"
                    return {
                        'type': 'left_hold',
                        'data': None
                    }
        else:
            # 松开捏合
            if self.is_pinching:
                self.is_pinching = False
                self.current_gesture = "Pinch Release"
                return {
                    'type': 'left_release',
                    'data': None
                }
        
        # 2. 检测握拳手势（鼠标右键）
        if self.detect_fist(positions):
            if current_time - self.last_right_click_time > config.CLICK_COOLDOWN:
                self.current_gesture = "Fist (Right Click)"
                self.last_right_click_time = current_time
                return {
                    'type': 'right_click',
                    'data': None
                }
            else:
                # 握拳但在冷却中，不移动鼠标
                self.current_gesture = "Fist (Cooldown)"
                return {
                    'type': 'none',
                    'data': None
                }
        
        # 3. 只在单食指伸出时移动鼠标（优先于五指张开）
        if self.detect_only_index_up(positions):
            index_pos = self.get_index_position(positions, frame_width, frame_height)
            self.current_gesture = "Move (Index Only)"
            return {
                'type': 'move',
                'data': index_pos
            }
        
        # 4. 检测食指和拇指张开（无操作）
        if self.detect_thumb_index_open(positions):
            self.current_gesture = "Thumb + Index Open (No Action)"
            return {
                'type': 'none',
                'data': None
            }
        
        # 5. 检测五指张开大幅扫动（滚动）
        if self.detect_open_hand(positions):
            swipe_result = self.detect_open_hand_swipe(positions, frame_height)
            if swipe_result and current_time - self.last_scroll_time > config.SCROLL_COOLDOWN:
                self.last_scroll_time = current_time
                if swipe_result == 'scroll_up':
                    self.current_gesture = "Open Hand Swipe Down (Scroll UP)"
                    return {
                        'type': 'scroll',
                        'data': 'up'
                    }
                elif swipe_result == 'scroll_down':
                    self.current_gesture = "Open Hand Swipe Up (Scroll DOWN)"
                    return {
                        'type': 'scroll',
                        'data': 'down'
                    }
            else:
                # 五指张开但没有大幅移动 - 不移动鼠标
                self.current_gesture = "Open Hand Ready"
                return {
                    'type': 'none',
                    'data': None
                }
        
        # 2. 检测食指和拇指张开（无操作）
        if self.detect_thumb_index_open(positions):
            self.current_gesture = "Thumb + Index Open (No Action)"
            return {
                'type': 'none',
                'data': None
            }
        
        # 3. 检测捏合手势（左键点击或长按1秒后拖拽）
        is_pinching = self.detect_pinch(positions)
        if is_pinching:
            if not self.is_pinching:
                # 刚开始捏合 - 触发按下
                if current_time - self.last_click_time > config.CLICK_COOLDOWN:
                    self.is_pinching = True
                    self.pinch_start_time = current_time
                    self.last_click_time = current_time
                    self.current_gesture = "Pinch Start (Left Press)"
                    return {
                        'type': 'left_press',
                        'data': None
                    }
            else:
                # 持续捏合 - 检查时长
                pinch_duration = current_time - self.pinch_start_time
                if pinch_duration > 1.0:  # 捏合超过1秒，允许拖拽
                    # 获取食指位置用于拖拽
                    index_pos = self.get_index_position(positions, frame_width, frame_height)
                    self.current_gesture = "Pinch Hold + Move (Dragging)"
                    return {
                        'type': 'pinch_drag',
                        'data': index_pos
                    }
                else:
                    # 捏合但未超过1秒 - 只保持按下，不移动鼠标
                    self.current_gesture = "Pinch Hold (Holding)"
                    return {
                        'type': 'left_hold',
                        'data': None
                    }
        else:
            # 松开捏合
            if self.is_pinching:
                self.is_pinching = False
                self.current_gesture = "Pinch Release"
                return {
                    'type': 'left_release',
                    'data': None
                }
        
        # 2. 检测握拳手势（鼠标右键）
        if self.detect_fist(positions):
            if current_time - self.last_right_click_time > config.CLICK_COOLDOWN:
                self.current_gesture = "Fist (Right Click)"
                self.last_right_click_time = current_time
                return {
                    'type': 'right_click',
                    'data': None
                }
            else:
                # 握拳但在冷却中，不移动鼠标
                self.current_gesture = "Fist (Cooldown)"
                return {
                    'type': 'none',
                    'data': None
                }
        
        # 3. 只在单食指伸出时移动鼠标（优先于五指张开）
        if self.detect_only_index_up(positions):
            index_pos = self.get_index_position(positions, frame_width, frame_height)
            self.current_gesture = "Move (Index Only)"
            return {
                'type': 'move',
                'data': index_pos
            }
        
        # 4. 检测食指和拇指张开（无操作）
        if self.detect_thumb_index_open(positions):
            self.current_gesture = "Thumb + Index Open (No Action)"
            return {
                'type': 'none',
                'data': None
            }
        
        # 5. 检测五指张开大幅扫动（滚动）
        if self.detect_open_hand(positions):
            if not self.is_pinching:
                # 刚开始捏合 - 触发按下
                if current_time - self.last_click_time > config.CLICK_COOLDOWN:
                    self.is_pinching = True
                    self.pinch_start_time = current_time
                    self.last_click_time = current_time
                    self.current_gesture = "Pinch Start (Left Press)"
                    return {
                        'type': 'left_press',
                        'data': None
                    }
            else:
                # 持续捏合 - 检查时长
                pinch_duration = current_time - self.pinch_start_time
                if pinch_duration > 1.0:  # 捏合超过1秒，允许拖拽
                    # 获取食指位置用于拖拽
                    index_pos = self.get_index_position(positions, frame_width, frame_height)
                    self.current_gesture = "Pinch Hold + Move (Dragging)"
                    return {
                        'type': 'pinch_drag',
                        'data': index_pos
                    }
                else:
                    # 捏合但未超过1秒 - 只保持按下，不移动鼠标
                    self.current_gesture = "Pinch Hold (Holding)"
                    return {
                        'type': 'left_hold',
                        'data': None
                    }
        else:
            # 松开捏合
            if self.is_pinching:
                self.is_pinching = False
                self.current_gesture = "Pinch Release"
                return {
                    'type': 'left_release',
                    'data': None
                }
        
        # 4. 检测握拳手势（鼠标右键）
        if self.detect_fist(positions):
            if current_time - self.last_right_click_time > config.CLICK_COOLDOWN:
                self.current_gesture = "Fist (Right Click)"
                self.last_right_click_time = current_time
                return {
                    'type': 'right_click',
                    'data': None
                }
            else:
                # 握拳但在冷却中，不移动鼠标
                self.current_gesture = "Fist (Cooldown)"
                return {
                    'type': 'none',
                    'data': None
                }
        
        # 5. 只在单食指伸出时移动鼠标
        if self.detect_only_index_up(positions):
            index_pos = self.get_index_position(positions, frame_width, frame_height)
            self.current_gesture = "Move (Index Only)"
            return {
                'type': 'move',
                'data': index_pos
            }
        
        # 其他情况 - 不移动鼠标
        self.current_gesture = "Idle"
        return {
            'type': 'none',
            'data': None
        }
    
    def get_current_gesture_text(self):
        """获取当前识别的手势文本（用于显示）"""
        return self.current_gesture
    
    def release(self):
        """释放资源"""
        self.hands.close()
