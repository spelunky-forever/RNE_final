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
        # 【新增】：定義自駕核心有限狀態機狀態 (FSM)
        self.STATE_NAV2_TRACKING = "NAV2_TRACKING"       # 狀態 1：跟隨 Nav2 全域路徑大範圍行駛
        self.STATE_VISUAL_SERVO_BEAR = "VISUAL_SERVO"   # 狀態 2：發現熊目標，視覺伺服精密對齊
        self.STATE_CROSS_BRIDGE = "CROSS_BRIDGE"         # 狀態 3：發現橋樑，無視 Nav2 視覺強行衝刺過橋
        self.STATE_ARRIVED_STOPPING = "ARRIVED_STOPPING" # 狀態 4：到達目標精準煞車距離
        self.current_state = self.STATE_NAV2_TRACKING   # 預設為 Nav2 導航模式

    def check_prerequisites(self):
        """Check if all prerequisites for navigation are met"""
        # Check if we have position data
        car_position, car_orientation = (
            self.car_control_node.get_car_position_and_orientation()
        )
        path_points = self.car_control_node.get_path_points()
        goal_pose = self.car_control_node.get_goal_pose()

        # Check data validity
        if not car_position or not path_points or not goal_pose:
            # Determine the specific error message based on what's missing
            message = (
                "Cannot obtain car position data"
                if not car_position
                else (
                    "No path points available for navigation"
                    if not path_points
                    else "No goal pose defined for navigation"
                )
            )

            return NavGoal.Result(success=False, message=message)
        else:
            # All prerequisites are met
            return car_position, car_orientation, path_points, goal_pose

    def data_init(self, car_position, car_orientation, goal_pose):
        return (
            [car_position.x, car_position.y],
            [car_orientation.z, car_orientation.w],
            [goal_pose.x, goal_pose.y],
        )

    def reset_index(self):
        self.index = 0

    def parse_yolo_array(self, raw_data):
        """
        將 YOLO 傳來的 1D 陣列轉換為分組字典，並按照距離由近到遠排序。
        格式: [Class_ID_1, Depth_1, DeltaX_1, Class_ID_2, Depth_2, DeltaX_2...]
        類別對應: 0: bear, 1: bridge, 2: knob
        """
        targets = {}
        if not raw_data or len(raw_data) % 3 != 0:
            return targets

        for i in range(0, len(raw_data), 3):
            class_id = int(raw_data[i])
            depth = float(raw_data[i+1])
            delta_x = float(raw_data[i+2])

            if depth <= 0.0:
                continue

            if class_id not in targets:
                targets[class_id] = []
                
            targets[class_id].append({
                'depth': depth,
                'delta_x': delta_x
            })

        # 依據深度進行升序排序，確保 targets[class_id][0] 永遠是最近的目標
        for cid in targets:
            targets[cid] = sorted(targets[cid], key=lambda k: k['depth'])
            
        return targets

    def customize_nav(self):
        """期末專案核心自動控制狀態機 (修復路過撞橋 Bug 版)"""
        raw_coordinate = self.car_control_node.get_latest_yolo_info()
        targets = self.parse_yolo_array(raw_coordinate)
        
        # ==================== 狀態 1：Nav2 大範圍導航狀態 ====================
        if self.current_state == self.STATE_NAV2_TRACKING:
            
            # 【修復 1】：刪除 road (ID 3) 輔助，避免與 Nav2 避障路徑衝突
            
            # 【修復 2：防止成為「尋橋飛彈」】
            # 只有當橋樑位於畫面中央附近 (abs(delta_x) < 0.35)，代表 Nav2 的路線真的是直指橋面，才觸發過橋。
            # 如果橋在畫面邊緣 (abs(delta_x) >= 0.35)，代表只是路過，無視它。
            if 1 in targets and len(targets[1]) > 0:
                bridge = targets[1][0]
                if bridge['depth'] < 1.8 and abs(bridge['delta_x']) < 0.35:
                    self.car_control_node.get_logger().info("【狀態切換】Nav2 導航正對橋樑，視覺接管過橋！")
                    self.current_state = self.STATE_CROSS_BRIDGE
                    return self.customize_nav()
                
            # 【夾取目標熊 (ID 0) 邏輯】：同樣加入 delta_x 限制，避免鎖定到遠處其他任務的熊
            if 0 in targets and len(targets[0]) > 0:
                bear = targets[0][0]
                if bear['depth'] < 1.5 and abs(bear['delta_x']) < 0.4:
                    self.car_control_node.get_logger().info("【狀態切換】正前方發現目標熊！切換至視覺伺服。")
                    self.current_state = self.STATE_VISUAL_SERVO_BEAR
                    self.nav_end_flag = 0
                    return self.customize_nav()
                
            # 沒有觸發視覺接管，老老實實聽 Foxglove 給的 Nav2 路線
            prereq_result = self.check_prerequisites()
            if isinstance(prereq_result, NavGoal.Result):
                self.car_control_node.publish_control("STOP")
                return None
            
            self.nav_end_flag = 0
            return self.manual_nav()

        # ==================== 狀態 2：YOLO 視覺伺服精密鎖定熊 ====================
        elif self.current_state == self.STATE_VISUAL_SERVO_BEAR:
            if 0 not in targets or len(targets[0]) == 0:
                self.car_control_node.get_logger().warn("【狀態警告】目標熊丟失，切回 Nav2。")
                self.current_state = self.STATE_NAV2_TRACKING
                self.car_control_node.publish_control("STOP")
                return None
                
            closest_bear = targets[0][0]
            object_depth = closest_bear['depth']
            y_offset = closest_bear['delta_x']
            
            if object_depth <= 0.31:
                self.current_state = self.STATE_ARRIVED_STOPPING
                return self.customize_nav()
                
            action = self.choose_action_y_offset(y_offset, object_depth)
            self.car_control_node.publish_control(action)
            return None

        # ==================== 狀態 3：視覺強行修正上橋/過橋模式 ====================
        elif self.current_state == self.STATE_CROSS_BRIDGE:
            if 0 in targets and len(targets[0]) > 0 and targets[0][0]['depth'] < 1.3:
                self.car_control_node.get_logger().info("【狀態切換】登頂並鎖定橋上的熊！")
                self.current_state = self.STATE_VISUAL_SERVO_BEAR
                return self.customize_nav()

            if 1 not in targets or len(targets[1]) == 0:
                self.car_control_node.get_logger().info("【過橋盲區】維持強行 FORWARD 衝刺。")
                self.car_control_node.publish_control("FORWARD")
                return None
                
            closest_bridge = targets[1][0]
            bridge_depth = closest_bridge['depth']
            bridge_offset = closest_bridge['delta_x']
            
            action = self.choose_action_y_offset(bridge_offset, bridge_depth)
            if action == "FORWARD_SLOW":
                action = "FORWARD" # 確保爬坡動力
            self.car_control_node.publish_control(action)
            return None

        # ==================== 狀態 4：精準煞車與環境清理狀態 ====================
        elif self.current_state == self.STATE_ARRIVED_STOPPING:
            self.car_control_node.get_logger().info("執行精密煞車中...")
            for _ in range(10):
                self.car_control_node.publish_control("STOP")
                time.sleep(0.03)
                
            self.car_control_node.clear_plan()
            self.car_control_node.clear_goal_pose()
            self.current_state = self.STATE_NAV2_TRACKING
            
            return NavGoal.Result(
                success=True,
                message="導航成功結束！",
            )
            
    def choose_action_y_offset(self, y_offset, object_depth):
        if object_depth >= 0.5:
            limit = 0.5
        elif object_depth <= 0.5:
            limit = 0.1
        if y_offset > -limit and y_offset < limit:
            return "FORWARD_SLOW"
            self.car_control_node.publish_control("FORWARD_SLOW")
        elif y_offset >= limit: # 物體在左
            return "COUNTERCLOCKWISE_ROTATION_SLOW"
            self.car_control_node.publish_control("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif y_offset <= -limit:
            return "CLOCKWISE_ROTATION_SLOW"
            self.car_control_node.publish_control("CLOCKWISE_ROTATION_SLOW")

    def manual_nav(self):
        result = self.check_prerequisites()

        if isinstance(result, NavGoal.Result):
            # 有錯誤就直接回傳結果，不繼續導航流程
            return result

        # 正常情況才解包
        car_position, car_orientation, path_points, goal_pose = result
        car_position, car_orientation, goal_pose = self.data_init(
            car_position, car_orientation, goal_pose
        )

        target_distance = cal_distance(car_position, goal_pose)
        if target_distance < 0.5:
            self.nav_end_flag = 1
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=True,
                message="Navigation goal reached successfully. Final distance",
            )
        else:
            target_points, orientation_points = self.get_next_target_point(
                car_position=car_position, path_points=path_points
            )
            diff_angle = calculate_diff_angle(
                car_position, car_orientation, target_points
            )
            action_key = self.choose_action(diff_angle)
            self.car_control_node.publish_control(action_key)

    def choose_action(self, diff_angle):
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def get_next_target_point(
        self, car_position, path_points, min_required_distance=0.5
    ):
        """
        Get the next target point along the path that is at least min_required_distance away
        from the car_position. Returns a tuple of ([target_x, target_y], [orientation_x, orientation_y])
        or (None, None) if no valid target is found.
        """
        logger = self.car_control_node.get_logger()

        if not path_points:
            logger.error("Error: No path points available!")
            return None, None

        # Ensure self.index is initialized
        if not hasattr(self, "index"):
            self.index = 0

        # Iterate over the remaining path points starting from the current index
        for idx in range(self.index, len(path_points)):
            point = path_points[idx]
            try:
                pos = point["position"]
                orient = point["orientation"]
                target_x, target_y = pos[0], pos[1]
                orientation_x, orientation_y = orient[0], orient[1]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Invalid path point format at index {idx}: {e}")
                continue

            distance_to_target = cal_distance(car_position, (target_x, target_y))
            if distance_to_target >= min_required_distance:
                # Update self.index to current valid point index for future calls
                self.index = idx
                logger.debug(
                    f"Found valid target point at index {idx} with distance {distance_to_target:.2f}"
                )
                return [target_x, target_y], [orientation_x, orientation_y]
            else:
                logger.debug(
                    f"Skipping point at index {idx}: distance {distance_to_target:.2f} is less than required {min_required_distance}"
                )

        # If no intermediate point meets the criteria, return the final point regardless of distance.
        try:
            last_point = path_points[-1]
            pos = last_point["position"]
            orient = last_point["orientation"]
            last_x, last_y = pos[0], pos[1]
            last_ox, last_oy = orient[0], orient[1]
            logger.info(
                "No point met the minimum distance requirement; using the last point as target."
            )
            self.index = len(path_points) - 1
            return [last_x, last_y], [last_ox, last_oy]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Invalid format for last path point: {e}")

        logger.warning("No valid target point found.")
        return None, None
