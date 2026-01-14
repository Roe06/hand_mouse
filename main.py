# -*- coding: utf-8 -*-
"""
手势控制鼠标主程序
通过摄像头捕捉手势，实时控制鼠标操作
"""

import cv2
import sys
import time
import os

# 设置标准输出编码为 UTF-8（Windows 兼容）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from gesture_detector import GestureDetector
from mouse_controller import MouseController
import config


def main():
    print("=" * 60)
    print("手势控制鼠标程序 - Gesture Mouse Control")
    print("=" * 60)
    print("\n手势操作说明:")
    print("  1. 单食指伸出         → 移动鼠标")
    print("  2. 食指+拇指张开      → 无操作（不移动鼠标）")
    print("  3. 食指+拇指捏合      → 点击左键（不移动鼠标）")
    print("  4. 捏合保持1秒+移动   → 拖拽（长按左键+移动）")
    print("  5. 五指握拳           → 右键点击")
    print("  6. 五指张开向下扫     → 向上滚动（连续扫动连续滚动）")
    print("  7. 五指张开向上扫     → 向下滚动（连续扫动连续滚动）")
    print("\n操作提示:")
    print("  - 将手放在摄像头前 30-60cm 处")
    print("  - 移动鼠标：只伸出食指，其他手指收拢")
    print("  - 点击左键：食指拇指捏合后快速松开")
    print("  - 拖拽：捏合保持1秒以上，然后移动手指")
    print("  - 滚动：五指全部张开，大幅度上下扫动")
    print("  - 按 ESC 或 Q 键退出，或直接关闭窗口")
    print("=" * 60)
    
    # 初始化摄像头
    print("\n正在初始化摄像头...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        print("请检查:")
        print("  1. 摄像头是否正确连接")
        print("  2. 摄像头权限是否开启")
        print("  3. 是否有其他程序正在使用摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    # 初始化手势检测器和鼠标控制器
    print("正在初始化手势检测器...")
    gesture_detector = GestureDetector()
    
    print("正在初始化鼠标控制器...")
    mouse_controller = MouseController()
    
    print("\n[OK] 初始化完成！程序开始运行...")
    print("=" * 60)
    
    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("警告: 无法读取摄像头画面")
                break
            
            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测手部关键点
            results, rgb_frame = gesture_detector.detect_hand_landmarks(frame)
            
            # 绘制手部关键点
            if config.SHOW_HAND_LANDMARKS:
                frame = gesture_detector.draw_landmarks(frame, results)
            
            # 识别手势
            gesture_result = gesture_detector.recognize_gesture(
                results, 
                frame.shape
            )
            
            # 执行鼠标操作
            if gesture_result['type'] != 'none':
                try:
                    mouse_controller.execute_gesture(
                        gesture_result,
                        config.FRAME_WIDTH,
                        config.FRAME_HEIGHT
                    )
                except Exception as e:
                    if "FailSafeException" in str(type(e)):
                        print("\n紧急停止触发！程序退出。")
                        break
                    else:
                        print(f"鼠标操作错误: {e}")
            else:
                # 没有检测到手时，重置平滑参数
                mouse_controller.reset_smoothing()
            
            # 显示摄像头画面
            if config.SHOW_CAMERA_FEED:
                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = 30 / elapsed_time
                    start_time = time.time()
                
                # 在画面上显示信息
                gesture_text = gesture_detector.get_current_gesture_text()
                cv2.putText(
                    frame, 
                    f"Gesture: {gesture_text}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                
                cv2.putText(
                    frame,
                    "Press ESC or Q to exit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
                
                # 显示窗口
                cv2.imshow('Gesture Mouse Control', frame)
            
            # 检测退出键（ESC 或 Q）和窗口关闭
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n用户按下 ESC 键，程序退出。")
                break
            elif key == ord('q') or key == ord('Q'):  # Q 键
                print("\n用户按下 Q 键，程序退出。")
                break
            
            # 检测窗口是否被关闭
            try:
                if cv2.getWindowProperty('Gesture Mouse Control', cv2.WND_PROP_VISIBLE) < 1:
                    print("\n窗口已关闭，程序退出。")
                    break
            except:
                print("\n窗口已关闭，程序退出。")
                break
    
    except KeyboardInterrupt:
        print("\n用户中断程序（Ctrl+C），程序退出。")
    
    except Exception as e:
        print(f"\n程序发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 释放资源
        print("\n正在释放资源...")
        cap.release()
        cv2.destroyAllWindows()
        gesture_detector.release()
        print("资源释放完成。")
        print("=" * 60)
        print("程序已安全退出。感谢使用！")
        print("=" * 60)


if __name__ == "__main__":
    main()
