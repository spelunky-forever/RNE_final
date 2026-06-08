from car_control_pkg.nav2_utils import (
    cal_distance,
    calculate_diff_angle,
)
from action_interface.action import NavGoal
import time

class NavigationController:
    def __init__(self, car_control_node):
        self.car_control_node = car_control_node
        self.nav_end_flag = 0 
        self.STATE_NAV2_TRACKING = "NAV2_TRACKING"       
        self.STATE_VISUAL_SERVO_BEAR = "VISUAL_SERVO"   
        self.STATE_CROSS_BRIDGE = "CROSS_BRIDGE"         
        self.STATE_ARRIVED_STOPPING = "ARRIVED_STOPPING" 
        self.current_state = self.STATE_NAV2_TRACKING   

    # ==========================================
    # 核心新增 1：差速驅動底層轉換 (Kinematic Model)
    # ==========================================
    def cmd_vel_to_wheels(self, v, w):
        """
        將線速度 (v) 與角速度 (w) 轉換為四輪轉速並放大至 Unity 適用區間
        """
        # ==========================================
        # 【核心修正】物理單位與 Unity 數值縮放
        # ==========================================
        v_scale = 18.0  # 【調參點】放大前進速度，讓它接近 manual 的 10.0
        w_scale = 15.0  # 【調參點】放大轉向速度，讓車子有足夠力矩轉向不撞牆
        
        v_unity = v * v_scale
        w_unity = w * w_scale

        track_factor = 1.2 # 調整軸距係數
        
        left_speed = v_unity - w_unity * track_factor
        right_speed = v_unity + w_unity * track_factor
        
        # 限制最大速度不超過手動限制
        max_limit = 15.0
        left_speed = max(-max_limit, min(max_limit, left_speed))
        right_speed = max(-max_limit, min(max_limit, right_speed))
        
        # 對應順序 [rear_left, rear_right, front_left, front_right]
        velocities = [left_speed, right_speed, left_speed, right_speed]
        
        # 呼叫你在 ros_communicator 寫好的新函式
        self.car_control_node.publish_raw_car_control(velocities)


    # ==========================================
    # 核心新增 2：P-Controller 與感測器融合循跡
    # ==========================================
    def continuous_nav2_tracking(self, targets):
        """
        全面依賴 Nav2 的局部規劃器 (Local Planner) 進行防撞導航，
        並加入 YOLO 橋樑 (Bridge) 視覺排斥力，捨棄無效的 Road 標籤。
        """
        result = self.check_prerequisites()
        if isinstance(result, NavGoal.Result):
            return result

        car_position, car_orientation, path_points, goal_pose = result
        car_position, car_orientation, goal_pose = self.data_init(car_position, car_orientation, goal_pose)

        # 1. 判斷是否抵達終點
        target_distance = cal_distance(car_position, goal_pose)
        if target_distance < 0.5:
            self.nav_end_flag = 1
            self.cmd_vel_to_wheels(0.0, 0.0) # 煞車
            return NavGoal.Result(success=True, message="Navigation goal reached successfully.")

        # ==========================================
        # 深度交互 Nav2: 直接採用 Local Planner 的安全指令
        # ==========================================
        cmd_vel_msg = self.car_control_node.latest_cmd_vel
        if cmd_vel_msg is not None:
            # 這是 Nav2 經過 Costmap 運算後，保證不撞牆的建議前進與轉向速度
            v_nav = cmd_vel_msg.linear.x
            w_nav = cmd_vel_msg.angular.z
        else:
            # 若暫時沒收到，保持停止避免暴衝
            v_nav, w_nav = 0.0, 0.0

        # ==========================================
        # YOLO Bridge 視覺避障 (只在不需要上橋時觸發遠離)
        # ==========================================
        w_bridge_avoid = 0.0
        v_bridge_slowdown = 1.0 # 減速係數
        
        # 1 代表 Bridge 的 Class ID
        if self.current_state != self.STATE_CROSS_BRIDGE and 1 in targets and len(targets[1]) > 0:
            bridge_data = targets[1][0] 
            bridge_depth = bridge_data['depth']
            bridge_offset = bridge_data['delta_x']
            
            # 如果橋樑進入視野警戒範圍 (例如深度小於 2.5 公尺)
            if 0.0 < bridge_depth < 2.5:
                kp_avoid = 0.005  # 【調參點】橋樑排斥力強度
                
                # 若橋在畫面右偏 (offset > 0)，為了遠離它，我們要向左轉 (w 為正)
                # 若橋在畫面左偏 (offset < 0)，我們要向右轉 (w 為負)
                w_bridge_avoid = -bridge_offset * kp_avoid
                
                # 看到橋就稍微減速，給予車子轉向避障的時間
                v_bridge_slowdown = 0.6 

        # ==========================================
        # 速度融合與底層輸出
        # ==========================================
        # 將基礎速度乘上減速係數
        v_final = v_nav * v_bridge_slowdown
        
        # 將 Nav2 規劃的安全轉向 加上 橋樑的排斥力
        w_final = w_nav + w_bridge_avoid

        # 發布平滑速度給四輪
        self.cmd_vel_to_wheels(v_final, w_final)
        
        return None

    # ==========================================
    # 核心新增 3：視覺伺服平滑跟隨
    # ==========================================
    def continuous_visual_servo(self, y_offset, object_depth, state_type):
        """
        針對熊或橋樑的視覺連續鎖定
        """
        kp_yaw = 0.005 # 【調參點】視覺追蹤靈敏度
        
        # offset > 0 (目標在右側) -> 車子需向右轉 (負 w)
        w = -y_offset * kp_yaw 
        v = 0.0

        # 解耦邏輯：判斷 X 軸偏移量是否夠小
        # 假設畫面中心容忍誤差為 40 pixels (需依實際相機解析度微調)
        alignment_tolerance = 40.0 

        if abs(y_offset) > alignment_tolerance:
            # 狀態 A：還沒對齊，只給角速度 (原地旋轉)
            # 限制旋轉速度上下限，避免轉太快或轉不動
            w = max(-1.2, min(1.2, w)) 
            v = 0.0
        else:
            # 狀態 B：已經對齊，給予前進速度，並保留微幅的角速度修正
            if state_type == self.STATE_CROSS_BRIDGE:
                v_base = 3.5  # 上橋需要動力
            else: # STATE_VISUAL_SERVO_BEAR
                # 距離越近越慢，但不低於 0.4 避免卡死
                v_base = max(0.4, object_depth * 0.8) 
            
            v = v_base
            w = max(-0.3, min(0.3, w)) # 對齊後限制轉向幅度

        self.cmd_vel_to_wheels(v, w)


    # 以下為你的狀態機邏輯 (使用新的連續控制取代舊版)
    def customize_nav(self):
        raw_coordinate = self.car_control_node.get_latest_yolo_info()
        targets = self.parse_yolo_array(raw_coordinate)
        
        # ==================== 狀態 1：Nav2 大範圍導航狀態 ====================
        if self.current_state == self.STATE_NAV2_TRACKING:
            if 1 in targets and len(targets[1]) > 0:
                bridge = targets[1][0]
                if bridge['depth'] < 1.8 and abs(bridge['delta_x']) < 0.35:
                    self.car_control_node.get_logger().info("【狀態切換】Nav2 導航正對橋樑，視覺接管過橋！")
                    self.current_state = self.STATE_CROSS_BRIDGE
                    return self.customize_nav()
                
            if 0 in targets and len(targets[0]) > 0:
                bear = targets[0][0]
                if bear['depth'] < 1.5 and abs(bear['delta_x']) < 0.4:
                    self.car_control_node.get_logger().info("【狀態切換】正前方發現目標熊！切換至視覺伺服。")
                    self.current_state = self.STATE_VISUAL_SERVO_BEAR
                    self.nav_end_flag = 0
                    return self.customize_nav()
                
            self.nav_end_flag = 0
            # 呼叫新的平滑導航取代 manual_nav
            return self.continuous_nav2_tracking(targets)

        # ==================== 狀態 2：YOLO 視覺伺服精密鎖定熊 ====================
        elif self.current_state == self.STATE_VISUAL_SERVO_BEAR:
            if 0 not in targets or len(targets[0]) == 0:
                self.car_control_node.get_logger().warn("【狀態警告】目標熊丟失，切回 Nav2。")
                self.current_state = self.STATE_NAV2_TRACKING
                self.cmd_vel_to_wheels(0.0, 0.0)
                return None
                
            closest_bear = targets[0][0]
            object_depth = closest_bear['depth']
            y_offset = closest_bear['delta_x']
            
            if object_depth <= 0.31:
                self.current_state = self.STATE_ARRIVED_STOPPING
                return self.customize_nav()
                
            # 呼叫新的平滑視覺伺服
            self.continuous_visual_servo(y_offset, object_depth, self.current_state)
            return None

        # ==================== 狀態 3：視覺強行修正上橋/過橋模式 ====================
        elif self.current_state == self.STATE_CROSS_BRIDGE:
            if 0 in targets and len(targets[0]) > 0 and targets[0][0]['depth'] < 1.3:
                self.car_control_node.get_logger().info("【狀態切換】登頂並鎖定橋上的熊！")
                self.current_state = self.STATE_VISUAL_SERVO_BEAR
                return self.customize_nav()

            if 1 not in targets or len(targets[1]) == 0:
                self.car_control_node.get_logger().info("【過橋盲區】維持強行衝刺。")
                self.cmd_vel_to_wheels(3.5, 0.0) # 強行直走
                return None
                
            closest_bridge = targets[1][0]
            bridge_depth = closest_bridge['depth']
            bridge_offset = closest_bridge['delta_x']
            
            self.continuous_visual_servo(bridge_offset, bridge_depth, self.current_state)
            return None

        # ==================== 狀態 4：精準煞車與環境清理狀態 ====================
        elif self.current_state == self.STATE_ARRIVED_STOPPING:
            self.car_control_node.get_logger().info("執行精密煞車中...")
            for _ in range(10):
                self.cmd_vel_to_wheels(0.0, 0.0)
                time.sleep(0.03)
                
            self.car_control_node.clear_plan()
            self.car_control_node.clear_goal_pose()
            self.current_state = self.STATE_NAV2_TRACKING
            
            return NavGoal.Result(success=True, message="導航成功結束！")

    # (保留原有的 check_prerequisites, data_init, parse_yolo_array, get_next_target_point, reset_index)
    def check_prerequisites(self):
        car_position, car_orientation = self.car_control_node.get_car_position_and_orientation()
        path_points = self.car_control_node.get_path_points()
        goal_pose = self.car_control_node.get_goal_pose()

        if not car_position or not path_points or not goal_pose:
            message = ("Cannot obtain car position data" if not car_position else 
                      ("No path points available" if not path_points else "No goal pose"))
            return NavGoal.Result(success=False, message=message)
        else:
            return car_position, car_orientation, path_points, goal_pose

    def data_init(self, car_position, car_orientation, goal_pose):
        return ([car_position.x, car_position.y], [car_orientation.z, car_orientation.w], [goal_pose.x, goal_pose.y])

    def reset_index(self):
        self.index = 0

    def parse_yolo_array(self, raw_data):
        targets = {}
        if not raw_data or len(raw_data) % 3 != 0:
            return targets
        for i in range(0, len(raw_data), 3):
            class_id, depth, delta_x = int(raw_data[i]), float(raw_data[i+1]), float(raw_data[i+2])
            if depth <= 0.0: continue
            if class_id not in targets: targets[class_id] = []
            targets[class_id].append({'depth': depth, 'delta_x': delta_x})
        for cid in targets:
            targets[cid] = sorted(targets[cid], key=lambda k: k['depth'])
        return targets

    def get_next_target_point(self, car_position, path_points, min_required_distance=0.5):
        logger = self.car_control_node.get_logger()
        if not path_points: return None, None
        if not hasattr(self, "index"): self.index = 0

        for idx in range(self.index, len(path_points)):
            point = path_points[idx]
            try:
                pos, orient = point["position"], point["orientation"]
                target_x, target_y = pos[0], pos[1]
                orientation_x, orientation_y = orient[0], orient[1]
            except Exception: continue

            distance_to_target = cal_distance(car_position, (target_x, target_y))
            if distance_to_target >= min_required_distance:
                self.index = idx
                return [target_x, target_y], [orientation_x, orientation_y]

        try:
            last_point = path_points[-1]
            pos, orient = last_point["position"], last_point["orientation"]
            self.index = len(path_points) - 1
            return [pos[0], pos[1]], [orient[0], orient[1]]
        except Exception: return None, None