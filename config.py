# 手势鼠标控制配置文件

# 鼠标控制灵敏度设置
MOUSE_SENSITIVITY = 1.0  # 鼠标移动灵敏度倍数
SCROLL_SENSITIVITY = 8   # 滚轮滚动灵敏度（增加滚动距离）
GESTURE_THRESHOLD = 0.05 # 手势识别阈值

# 手势触发间隔（秒）
CLICK_COOLDOWN = 0.15    # 点击冷却时间（降低以提高灵敏度）
SCROLL_COOLDOWN = 0.05   # 滚动冷却时间（降低以提高灵敏度）
SWIPE_COOLDOWN = 0.2     # 滑动冷却时间

# 摄像头设置
CAMERA_INDEX = 0         # 摄像头索引（0为默认摄像头）
FRAME_WIDTH = 640        # 视频帧宽度
FRAME_HEIGHT = 480       # 视频帧高度
FPS = 30                 # 帧率

# 显示设置
SHOW_CAMERA_FEED = True  # 是否显示摄像头画面
SHOW_HAND_LANDMARKS = True  # 是否显示手部关键点

# 手势识别参数
PINCH_THRESHOLD = 0.05   # 捏合手势阈值（拇指和食指距离，放宽以更灵敏）
FIST_THRESHOLD = 0.08    # 握拳手势阈值（需要握得非常紧）
SWIPE_THRESHOLD = 0.04   # 五指张开滑动阈值（降低以更灵敏）

# 鼠标移动平滑参数
SMOOTH_FACTOR = 0.5      # 平滑系数（降低平滑增加跟手性）

# 屏幕映射区域（将手部检测区域映射到屏幕）
SCREEN_PADDING = 50      # 屏幕边缘预留像素（减少padding增加可用区域）
