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
        publish_control 接口發布。
        """
        v_scale = 30.0
        w_scale = 25.0
        
        v_unity = v * v_scale
        w_unity = w * w_scale
        track_factor = 1.0 
        
        left_speed = v_unity - w_unity * track_factor
        right_speed = v_unity + w_unity * track_factor
        
        # 限制極值，防止暴衝或馬達過載
        max_limit = 25.0
        left_speed = max(-max_limit, min(max_limit, left_speed))
        right_speed = max(-max_limit, min(max_limit, right_speed))
        
        # 【新增盲除錯日誌】：每秒印出一次目前實際送給輪子的數值
        self.car_control_node.get_logger().info(
            f"🛞 速度發布中 -> 狀態: [{self.current_state}] | 輸入 v: {v:.2f}, w: {w:.2f} | 輸出給底盤輪速 L: {left_speed:.2f}, R: {right_speed:.2f}",
            throttle_duration_sec=1.0
        )
        
        # 透過 list 形式傳遞，會自動被 car_control_common 解析為四輪轉速
        self.car_control_node.publish_control([left_speed, right_speed])

    def customize_nav(self):
        """
        全自動主控決策迴圈
        """
        # 1. 取得最新經由物體檢測發布的一維動態陣列
        raw_coordinate = self.car_control_node.get_latest_yolo_info()
        targets = self.parse_yolo_array(raw_coordinate)
        
        # 2. 基本安全前置檢查
        car_position, car_orientation, path_points, goal_pose = self.check_and_unpack_data()
        
        # 【新增常駐狀態日誌】：每兩秒監控一次核心資料流，抓出斷流兇手
        bear_count = len(targets.get(CLASS_BEAR, []))
        bridge_count = len(targets.get(CLASS_BRIDGE, []))
        self.car_control_node.get_logger().info(
            f"🔍 [決策迴圈監視] 狀態機: {self.current_state} | 定位: {'OK' if car_position else '遺失!'} | 視野內熊隻數: {bear_count} | 橋樑數: {bridge_count}",
            throttle_duration_sec=2.0
        )

        if car_position is None:
            # 資料尚未補齊時，原地保持安全靜止
            self.cmd_vel_to_wheels(0.0, 0.0)
            return None

        # =========================================================================
        # 全局最高優先級：橋樑側壁安全排斥機制
        # =========================================================================
        if self.current_state != self.STATE_AVOID_BRIDGE_WALL:
            if CLASS_BRIDGE in targets and len(targets[CLASS_BRIDGE]) > 0:
                closest_bridge = targets[CLASS_BRIDGE][0]
                if closest_bridge['depth'] < 2.3:
                    self.car_control_node.get_logger().warn(
                        f"🚨 檢測到橋樑側壁危險靠近！距離: {closest_bridge['depth']}m，切換至緊急避障狀態！"
                    )
                    self.current_state = self.STATE_AVOID_BRIDGE_WALL
                    self.avoidance_end_time = time.time() + 1.5 
                    return self.customize_nav()

        # =========================================================================
        # 狀態機核心決策分支
        # =========================================================================
        
        # 狀態 1：主動探索找附近的第一隻熊
        if self.current_state == self.STATE_EXPLORE_FIND_BEAR:
            if CLASS_BEAR in targets and len(targets[CLASS_BEAR]) > 0:
                self.car_control_node.get_logger().info("🐻 YOLO 成功鎖定附近的第一隻熊！轉入視覺伺服追蹤。")
                self.current_state = self.STATE_VISUAL_SERVO_BEAR
                return self.customize_nav()
            
            # 主動探索控制邏輯
            cmd_vel_list = self.car_control_node.latest_cmd_vel
            if cmd_vel_list is not None and len(cmd_vel_list) == 2:
                v_left, v_right = cmd_vel_list[0], cmd_vel_list[1]
                wheel_distance = 0.5
                v_nav = (v_left + v_right) / 2.0
                w_nav = (v_right - v_left) / wheel_distance
                
                w_explore = 0.5 if w_nav == 0.0 else w_nav
                v_explore = max(0.1, min(0.3, v_nav))
            else:
                v_explore = 0.1  
                w_explore = 0.6  
            
            self.cmd_vel_to_wheels(v_explore, w_explore)
            return None

        # 狀態 2：視覺伺服精密鎖定熊
        elif self.current_state == self.STATE_VISUAL_SERVO_BEAR:
            if CLASS_BEAR not in targets or len(targets[CLASS_BEAR]) == 0:
                self.car_control_node.get_logger().warn("⚠️ 目標熊從視野丟失，重新回到主動探索狀態。")
                self.current_state = self.STATE_EXPLORE_FIND_BEAR
                self.cmd_vel_to_wheels(0.0, 0.0)
                return None
            
            closest_bear = targets[CLASS_BEAR][0]
            bear_depth = closest_bear['depth']
            bear_offset = closest_bear['delta_x']
            
            if bear_depth <= 0.45:
                self.car_control_node.get_logger().info("🎯 已抵達第一隻熊的觀測範圍，切換至精密煞車與原地停等狀態。")
                self.current_state = self.STATE_ARRIVED_STOPPING
                self.action_start_time = time.time()
                return self.customize_nav()
            
            kp_yaw = 0.0045  
            w_control = -bear_offset * kp_yaw
            
            alignment_tolerance = 45.0
            if abs(bear_offset) > alignment_tolerance:
                v_control = 0.0
                w_control = max(-0.8, min(0.8, w_control)) 
            else:
                v_control = max(0.3, min(0.8, bear_depth * 0.4))
                w_control = max(-0.2, min(0.2, w_control)) 
                
            self.cmd_vel_to_wheels(v_control, w_control)
            return None

        # 狀態 3：緊急避障橋樑側壁
        elif self.current_state == self.STATE_AVOID_BRIDGE_WALL:
            if time.time() > self.avoidance_end_time:
                self.car_control_node.get_logger().info("🔄 離開緊急避障狀態，重新進入找熊主動探索。")
                self.current_state = self.STATE_EXPLORE_FIND_BEAR
                return self.customize_nav()
                
            if CLASS_BRIDGE in targets and len(targets[CLASS_BRIDGE]) > 0:
                bridge_offset = targets[CLASS_BRIDGE][0]['delta_x']
                if bridge_offset > 0:
                    self.cmd_vel_to_wheels(-0.4, 1.0)
                else:
                    self.cmd_vel_to_wheels(-0.4, -1.0)
            else:
                self.cmd_vel_to_wheels(-0.4, 0.8)
            return None

        # 狀態 4：精密煞車與原地停等狀態 (TASK 1 規定 STATIONARY FOR 5+ SECONDS)
        elif self.current_state == self.STATE_ARRIVED_STOPPING:
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
                
                self.car_control_node.clear_plan()
                self.car_control_node.clear_goal_pose()
                self.current_state = self.STATE_EXPLORE_FIND_BEAR 
                self.action_start_time = None
                
                return NavGoal.Result(success=True, message="Task 1 (Locate & Observe Bear) completed perfectly.")
            return None

        return None

    def check_and_unpack_data(self):
        car_position, car_orientation = self.car_control_node.get_car_position_and_orientation()
        path_points = self.car_control_node.get_path_points()
        goal_pose = self.car_control_node.get_goal_pose()
        if not car_position:
            return None, None, None, None
        return car_position, car_orientation, path_points, goal_pose

    def parse_yolo_array(self, raw_data):
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
                
            targets[class_id].append({'depth': depth, 'delta_x': delta_x})
            
        for cid in targets:
            targets[cid] = sorted(targets[cid], key=lambda k: k['depth'])
            
        return targets

    def manual_nav(self):
        self.cmd_vel_to_wheels(0.0, 0.0)
        return NavGoal.Result(success=True, message="Manual Navigation IDLE.")

    def reset_index(self):
        self.index = 0