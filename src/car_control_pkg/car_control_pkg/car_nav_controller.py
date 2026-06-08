# -*- coding: utf-8 -*-
from action_interface.action import NavGoal
from car_control_pkg.nav2_utils import (
    cal_distance,
    calculate_diff_angle,
)
import time
import math

# YOLO 標籤定義
CLASS_BEAR = 0
CLASS_BRIDGE = 1
CLASS_DOOR = 2

class NavigationController:
    def __init__(self, car_control_node):
        self.car_control_node = car_control_node
        self.index = 0
        self.nav_end_flag = 0 
        
        # --- 狀態機狀態定義 ---
        self.STATE_EXPLORE_FIND_BEAR = "EXPLORE_FIND_BEAR"   # 主動探索找第一個熊
        self.STATE_VISUAL_SERVO_BEAR = "VISUAL_SERVO_BEAR"   # 視覺伺服精密鎖定熊
        self.STATE_AVOID_BRIDGE_WALL = "AVOID_BRIDGE_WALL"   # 緊急避障橋樑側壁
        self.STATE_ARRIVED_STOPPING = "ARRIVED_STOPPING"     # 精密煞車與原地停等
        
        # 初始狀態設定為：主動探索找熊
        self.current_state = self.STATE_EXPLORE_FIND_BEAR
        
        # 任務控制計時器
        self.action_start_time = None
        self.avoidance_end_time = 0.0

    def cmd_vel_to_wheels(self, v, w):
        """
        將線速度 (v) 與角速度 (w) 轉換為左右輪速，並透過 BaseCarControlNode 的
        publish_control 接口發布。此處完美銜接 your car_control_common.py 的設計。
        """
        # 根據 Unity 物理特性進行數值縮放與力矩微調
        v_scale = 15.0  
        w_scale = 12.0  
        
        v_unity = v * v_scale
        w_unity = w * w_scale
        track_factor = 1.0 
        
        left_speed = v_unity - w_unity * track_factor
        right_speed = v_unity + w_unity * track_factor
        
        # 限制極值，防止暴衝或馬達過載
        max_limit = 12.0
        left_speed = max(-max_limit, min(max_limit, left_speed))
        right_speed = max(-max_limit, min(max_limit, right_speed))
        
        # 透過 list 形式傳遞，會自動被 car_control_common 解析為四輪轉速
        self.car_control_node.publish_control([left_speed, right_speed])

    def customize_nav(self):
        """
        全自動主控決策迴圈（無省略完整版）
        """
        # 1. 取得最新經由物體檢測發布的一維動態陣列
        raw_coordinate = self.car_control_node.get_latest_yolo_info()
        targets = self.parse_yolo_array(raw_coordinate)
        
        # 2. 基本安全前置檢查
        car_position, car_orientation, path_points, goal_pose = self.check_and_unpack_data()
        if car_position is None:
            # 資料尚未補齊時，原地保持安全靜止
            self.cmd_vel_to_wheels(0.0, 0.0)
            return None

        # =========================================================================
        # 全局最高優先級：橋樑側壁安全排斥機制（防止撞擊立體三角形非入口區域）
        # =========================================================================
        # 只要在探索或跟隨熊的過程中，前方出現橋樑且距離過近，立刻中斷當前行為進行強制避障
        if self.current_state != self.STATE_AVOID_BRIDGE_WALL:
            if CLASS_BRIDGE in targets and len(targets[CLASS_BRIDGE]) > 0:
                closest_bridge = targets[CLASS_BRIDGE][0]
                # 關鍵防撞閾值：2.3公尺（立著的三角形側面盲區大，需拉長警戒距離）
                if closest_bridge['depth'] < 2.3:
                    self.car_control_node.get_logger().warn(
                        f"🚨 檢測到橋樑側壁危險靠近！距離: {closest_bridge['depth']}m，切換至緊急避障狀態！"
                    )
                    self.current_state = self.STATE_AVOID_BRIDGE_WALL
                    self.avoidance_end_time = time.time() + 1.5 # 強制執行避障動作 1.5 秒
                    return self.customize_nav()

        # =========================================================================
        # 狀態機核心決策分支
        # =========================================================================
        
        # 狀態 1：主動探索找附近的第一隻熊
        if self.current_state == self.STATE_EXPLORE_FIND_BEAR:
            # 檢查視野中是否捕捉到熊
            if CLASS_BEAR in targets and len(targets[CLASS_BEAR]) > 0:
                self.car_control_node.get_logger().info("🐻 YOLO 成功鎖定附近的第一隻熊！轉入視覺伺服追蹤。")
                self.current_state = self.STATE_VISUAL_SERVO_BEAR
                return self.customize_nav()
            
            # 主動探索控制邏輯：採用低速原地打轉 + 微幅前進的螺旋式搜索
            # 解析 latest_cmd_vel (格式為 [v_left, v_right] 的 List)
            cmd_vel_list = self.car_control_node.latest_cmd_vel
            if cmd_vel_list is not None and len(cmd_vel_list) == 2:
                v_left, v_right = cmd_vel_list[0], cmd_vel_list[1]
                # 從左右輪速反推算 Nav2 建議的線速度(v)與角速度(w)，假設輪距為 0.5
                wheel_distance = 0.5
                v_nav = (v_left + v_right) / 2.0
                w_nav = (v_right - v_left) / wheel_distance
                
                w_explore = 0.5 if w_nav == 0.0 else w_nav
                v_explore = max(0.1, min(0.3, v_nav))
            else:
                v_explore = 0.1  # 緩慢微幅前進打破死點
                w_explore = 0.6  # 原地旋轉掃描
            
            self.cmd_vel_to_wheels(v_explore, w_explore)
            return None

        # 狀態 2：視覺伺服精密鎖定熊
        elif self.current_state == self.STATE_VISUAL_SERVO_BEAR:
            # 確保目標熊依然存在於視野中
            if CLASS_BEAR not in targets or len(targets[CLASS_BEAR]) == 0:
                self.car_control_node.get_logger().warn("⚠️ 目標熊從視野丟失，重新回到主動探索狀態。")
                self.current_state = self.STATE_EXPLORE_FIND_BEAR
                self.cmd_vel_to_wheels(0.0, 0.0)
                return None
            
            closest_bear = targets[CLASS_BEAR][0]
            bear_depth = closest_bear['depth']
            bear_offset = closest_bear['delta_x']
            
            # 抵達判定條件（Task 1 規定 N units，此處設 0.45m 作為精準抓取觀測點）
            if bear_depth <= 0.45:
                self.car_control_node.get_logger().info("🎯 已抵達第一隻熊的觀測範圍，切換至精密煞車與原地停等狀態。")
                self.current_state = self.STATE_ARRIVED_STOPPING
                self.action_start_time = time.time()
                return self.customize_nav()
            
            # 視覺伺服 P-Controller 比例控制
            kp_yaw = 0.0045  # 依像素偏差調整轉向增益
            w_control = -bear_offset * kp_yaw
            
            # 狀態解耦：如果橫向偏差過大（大於45像素），原地轉向對齊，不給前進速度，防止側向偏離撞牆
            alignment_tolerance = 45.0
            if abs(bear_offset) > alignment_tolerance:
                v_control = 0.0
                w_control = max(-0.8, min(0.8, w_control)) # 限制最大旋轉角速度
            else:
                # 已基本對齊，開始穩步接近，距離越近速度越慢（平滑減速機制）
                v_control = max(0.3, min(0.8, bear_depth * 0.4))
                w_control = max(-0.2, min(0.2, w_control)) # 對齊後限制轉向擺動幅度
                
            self.cmd_vel_to_wheels(v_control, w_control)
            return None

        # 狀態 3：緊急避障橋樑側壁
        elif self.current_state == self.STATE_AVOID_BRIDGE_WALL:
            # 強制避障時間檢查
            if time.time() > self.avoidance_end_time:
                self.car_control_node.get_logger().info("🔄 離開緊急避障狀態，重新進入找熊主動探索。")
                self.current_state = self.STATE_EXPLORE_FIND_BEAR
                return self.customize_nav()
                
            # 避障運動規劃：策略是「倒車 + 向橋樑相反側強旋轉」
            # 檢查當前畫面中橋的相對位置來決定反向切舵方向
            if CLASS_BRIDGE in targets and len(targets[CLASS_BRIDGE]) > 0:
                bridge_offset = targets[CLASS_BRIDGE][0]['delta_x']
                if bridge_offset > 0:
                    # 橋在右側 -> 倒車並往左猛轉 (正 w)
                    self.cmd_vel_to_wheels(-0.4, 1.0)
                else:
                    # 橋在左側 -> 倒車並往右猛轉 (負 w)
                    self.cmd_vel_to_wheels(-0.4, -1.0)
            else:
                # 若視野中突然失去橋的軌跡，採取安全退後並左轉的盲退策略
                self.cmd_vel_to_wheels(-0.4, 0.8)
            return None

        # 狀態 4：精密煞車與環境清理狀態（實現 TASK 1 的 STATIONARY FOR 5+ SECONDS）
        elif self.current_state == self.STATE_ARRIVED_STOPPING:
            # 強制介入連續清空發布速度，確保物理底盤在滑行後立刻停死
            self.cmd_vel_to_wheels(0.0, 0.0)
            
            if self.action_start_time is None:
                self.action_start_time = time.time()
                
            elapsed_time = time.time() - self.action_start_time
            self.car_control_node.get_logger().info(
                f"⏱️ 原地安全停等中... 目前已停等: {elapsed_time:.1f} 秒 / 目標: 5.0 秒", 
                throttle_duration_sec=1.0
            )
            
            if elapsed_time >= 5.2:
                self.car_control_node.get_logger().info("🎉 【TASK 1 成功完成】已成功找到第一隻熊並原地停留超過 5 秒！")
                
                # 任務結束，依照系統規範清除全局目標與路徑，避免殘留干擾
                self.car_control_node.clear_plan()
                self.car_control_node.clear_goal_pose()
                self.current_state = self.STATE_EXPLORE_FIND_BEAR # 狀態機重設
                self.action_start_time = None
                
                return NavGoal.Result(success=True, message="Task 1 (Locate & Observe Bear) completed perfectly.")
            return None

        return None

    # =========================================================================
    # 基礎底層輔助函數（與架構完全相依，維持不省略）
    # =========================================================================
    def check_and_unpack_data(self):
        """檢查並解包 ROS2 節點訂閱到的關鍵里程計與全局路徑資料"""
        car_position, car_orientation = self.car_control_node.get_car_position_and_orientation()
        path_points = self.car_control_node.get_path_points()
        goal_pose = self.car_control_node.get_goal_pose()

        if not car_position:
            return None, None, None, None
            
        return car_position, car_orientation, path_points, goal_pose

    def parse_yolo_array(self, raw_data):
        """解析 object_detect.py 傳過來的一維 Float32MultiArray 陣列"""
        targets = {}
        if not raw_data or len(raw_data) % 3 != 0:
            return targets
            
        for i in range(0, len(raw_data), 3):
            class_id = int(raw_data[i])
            depth = float(raw_data[i+1])
            delta_x = float(raw_data[i+2])
            
            # 過濾掉無效的深度異常值
            if depth <= 0.0: 
                continue
                
            if class_id not in targets: 
                targets[class_id] = []
                
            targets[class_id].append({'depth': depth, 'delta_x': delta_x})
            
        # 針對每一種類別的目標，依照距離由近到遠排序（优先處理最近的危險物/目標）
        for cid in targets:
            targets[cid] = sorted(targets[cid], key=lambda k: k['depth'])
            
        return targets

    def manual_nav(self):
        """保留原系統架構之手動巡航接口（防呆預留）"""
        self.cmd_vel_to_wheels(0.0, 0.0)
        return NavGoal.Result(success=True, message="Manual Navigation IDLE.")

    def reset_index(self):
        """重置全域路徑索引值"""
        self.index = 0

    def get_next_target_point(self, car_position, path_points, min_required_distance=0.5):
        """全局路徑追蹤點計算函數（保留完整算法）"""
        if not path_points: 
            return None, None
            
        for idx in range(self.index, len(path_points)):
            point = path_points[idx]
            try:
                pos = point["position"]
                orient = point["orientation"]
                target_x, target_y = pos[0], pos[1]
                orientation_x, orientation_y = orient[0], orient[1]
            except (KeyError, IndexError, TypeError): 
                continue

            distance_to_target = cal_distance(car_position, (target_x, target_y))
            if distance_to_target >= min_required_distance:
                self.index = idx
                return [target_x, target_y], [orientation_x, orientation_y]

        try:
            last_point = path_points[-1]
            pos = last_point["position"]
            orient = last_point["orientation"]
            self.index = len(path_points) - 1
            return [pos[0], pos[1]], [orient[0], orient[1]]
        except (KeyError, IndexError, TypeError): 
            return None, None