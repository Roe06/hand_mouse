"""
鼠标控制器模块
使用 PyAutoGUI 控制鼠标移动、点击和滚轮
"""

import pyautogui
import numpy as np
import config


class MouseController:
    def __init__(self):
        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        
        # PyAutoGUI 安全设置
        pyautogui.FAILSAFE = True  # 鼠标移到屏幕角落时抛出异常以停止程序
        pyautogui.PAUSE = 0.01     # 每次操作后的暂停时间
        
        # 平滑移动相关
        self.prev_mouse_x = None
        self.prev_mouse_y = None
        
        # 坐标映射范围（减去边缘padding）
        self.map_width = self.screen_width - 2 * config.SCREEN_PADDING
        self.map_height = self.screen_height - 2 * config.SCREEN_PADDING
        
        print(f"屏幕分辨率: {self.screen_width} x {self.screen_height}")
    
    def map_coordinates(self, hand_x, hand_y, frame_width, frame_height):
        """
        将摄像头坐标映射到屏幕坐标
        hand_x, hand_y: 手部在摄像头画面中的坐标
        返回: 屏幕坐标 (screen_x, screen_y)
        """
        # 归一化到 0-1 范围
        norm_x = hand_x / frame_width
        norm_y = hand_y / frame_height
        
        # 注意：frame 已经在主程序中进行了镜像翻转，这里不需要再次翻转
        
        # 映射到屏幕坐标（加上padding）
        screen_x = config.SCREEN_PADDING + norm_x * self.map_width
        screen_y = config.SCREEN_PADDING + norm_y * self.map_height
        
        # 限制在屏幕范围内
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return int(screen_x), int(screen_y)
    
    def smooth_coordinates(self, current_x, current_y):
        """
        使用指数移动平均平滑鼠标坐标
        减少抖动，使移动更流畅
        """
        if self.prev_mouse_x is None or self.prev_mouse_y is None:
            self.prev_mouse_x = current_x
            self.prev_mouse_y = current_y
            return current_x, current_y
        
        # 指数移动平均
        smooth_x = (config.SMOOTH_FACTOR * self.prev_mouse_x + 
                   (1 - config.SMOOTH_FACTOR) * current_x)
        smooth_y = (config.SMOOTH_FACTOR * self.prev_mouse_y + 
                   (1 - config.SMOOTH_FACTOR) * current_y)
        
        self.prev_mouse_x = smooth_x
        self.prev_mouse_y = smooth_y
        
        return int(smooth_x), int(smooth_y)
    
    def move_mouse(self, hand_x, hand_y, frame_width, frame_height):
        """
        移动鼠标到指定位置
        """
        try:
            # 映射坐标
            screen_x, screen_y = self.map_coordinates(
                hand_x, hand_y, frame_width, frame_height
            )
            
            # 应用平滑
            smooth_x, smooth_y = self.smooth_coordinates(screen_x, screen_y)
            
            # 移动鼠标（直接使用坐标，不再额外乘以灵敏度）
            pyautogui.moveTo(
                smooth_x,
                smooth_y,
                duration=0
            )
        except pyautogui.FailSafeException:
            print("检测到紧急停止（鼠标移到屏幕角落）")
            raise
    
    def left_click(self):
        """执行鼠标左键点击"""
        try:
            pyautogui.click(button='left')
            print("执行: 左键点击")
        except Exception as e:
            print(f"左键点击失败: {e}")
    
    def left_press(self):
        """按下鼠标左键（不释放）"""
        try:
            pyautogui.mouseDown(button='left')
            print("执行: 左键按下")
        except Exception as e:
            print(f"左键按下失败: {e}")
    
    def left_release(self):
        """释放鼠标左键"""
        try:
            pyautogui.mouseUp(button='left')
            print("执行: 左键释放")
        except Exception as e:
            print(f"左键释放失败: {e}")
    
    def right_click(self):
        """执行鼠标右键点击"""
        try:
            pyautogui.click(button='right')
            print("执行: 右键点击")
        except Exception as e:
            print(f"右键点击失败: {e}")
    
    def scroll(self, direction):
        """
        执行鼠标滚轮滚动
        direction: 'up' 或 'down'
        """
        try:
            scroll_amount = int(config.SCROLL_SENSITIVITY * 3)
            if direction == 'up':
                pyautogui.scroll(scroll_amount)
                print(f"执行: 向上滚动 {scroll_amount}")
            elif direction == 'down':
                pyautogui.scroll(-scroll_amount)
                print(f"执行: 向下滚动 {scroll_amount}")
        except Exception as e:
            print(f"滚动失败: {e}")
    
    def execute_gesture(self, gesture_result, frame_width, frame_height):
        """
        根据手势识别结果执行相应的鼠标操作
        gesture_result: gesture_detector 返回的字典
        """
        gesture_type = gesture_result['type']
        gesture_data = gesture_result['data']
        
        if gesture_type == 'move' and gesture_data:
            # 单食指移动鼠标
            hand_x, hand_y = gesture_data
            self.move_mouse(hand_x, hand_y, frame_width, frame_height)
            
        elif gesture_type == 'left_press':
            # 捏合开始 - 鼠标左键按下
            self.left_press()
        
        elif gesture_type == 'left_hold':
            # 捏合持续 - 保持按下状态（不移动）
            pass
        
        elif gesture_type == 'pinch_drag' and gesture_data:
            # 捏合拖拽 - 左键按下且移动鼠标
            hand_x, hand_y = gesture_data
            self.move_mouse(hand_x, hand_y, frame_width, frame_height)
        
        elif gesture_type == 'left_release':
            # 捏合松开 - 鼠标左键释放
            self.left_release()
            
        elif gesture_type == 'right_click':
            # 握拳 - 鼠标右键点击
            self.right_click()
            
        elif gesture_type == 'scroll' and gesture_data:
            # 五指张开扫动 - 鼠标滚轮滚动
            self.scroll(gesture_data)
    
    def reset_smoothing(self):
        """重置平滑参数（当手势中断后重新开始）"""
        self.prev_mouse_x = None
        self.prev_mouse_y = None
