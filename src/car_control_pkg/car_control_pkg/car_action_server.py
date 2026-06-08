import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from action_interface.action import NavGoal
from car_control_pkg.car_control_common import BaseCarControlNode
import functools
from car_control_pkg.car_nav_controller import NavigationController


class NavigationActionServer(Node):
    def __init__(self, car_control_node):
        super().__init__("navigation_action_server_node")
        self._action_server = ActionServer(
            self,
            NavGoal,
            "nav_action_server",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.car_control_node = car_control_node
        self.nav_controller = NavigationController(self.car_control_node)
        self.get_logger().info("✅ Navigation Action Server 成功初始化")

    def goal_callback(self, goal_request):
        requested_mode = goal_request.mode
        self.get_logger().info(f"📡 收到導航目標請求，請求模式為: [{requested_mode}]")
        
        car_position, _ = self.car_control_node.get_car_position_and_orientation()
        
        # 【關鍵解耦修改】：如果是主動探索自定義導航，不需要等 Nav2 畫出藍色路線
        if requested_mode == "Customize_Nav":
            if not car_position:
                self.get_logger().error(f"❌ 無法啟動 {requested_mode}: 缺少 AMCL 定位數據！請先在地圖上給予 Initial Pose。")
                return GoalResponse.REJECT
            self.get_logger().info(f"🚀 {requested_mode} 驗證成功：主動探索不依賴全局規劃路線，直接放行啟動！")
            return GoalResponse.ACCEPT
            
        # 其他常規導航模式（如原本的巡航模式）仍維持原判斷，必須等到藍色路線出現
        else:
            path_points = self.car_control_node.get_path_points(include_orientation=True)
            if not car_position or not path_points:
                self.get_logger().error(f"❌ 無法啟動 {requested_mode}: 缺少藍色規劃路線或定位數據！(請點擊 2D Nav Goal)")
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("🛑 收到取消請求，正在強制停止車輛...")
        self.car_control_node.publish_control("STOP")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Navigation action callback"""
        result = NavGoal.Result()
        mode = goal_handle.request.mode
        self.get_logger().info(f"🎬 開始進入 Action 執行核心，當前模式: [{mode}]")
        rate = self.create_rate(10)
        self.nav_controller.reset_index()
        
        while rclpy.ok():
            # 先給 executor 時間處理回調函數數據
            rate.sleep()
            
            if goal_handle.is_cancel_requested:
                self.get_logger().info("🧭 導航已被使用者手動取消")
                self.car_control_node.publish_control("STOP")
                result = NavGoal.Result(success=False, message="Navigation canceled")
                goal_handle.canceled()
                break

            car_auto_method = self._select_car_auto_method(mode)
            if car_auto_method is None:
                self.get_logger().error(f"💥 致命錯誤：無法執行！找不到對應的控制方法，模式字串 [{mode}] 可能不匹配或打錯字！")
                self.car_control_node.publish_control("STOP")
                result = NavGoal.Result(success=False, message="Unknown mode mapping")
                goal_handle.abort()
                break

            # 執行決策核心
            nav_result = car_auto_method()
            
            if isinstance(nav_result, NavGoal.Result):
                if nav_result.success:
                    self.get_logger().info(f"🎉 導航任務圓滿完成: {nav_result.message}")
                    goal_handle.succeed()
                else:
                    self.get_logger().error(f"💥 導航任務失敗: {nav_result.message}")
                    goal_handle.abort()
                result = nav_result
                break

            # 正常執行中，持續發布進度反饋
            feedback_msg = NavGoal.Feedback()
            feedback_msg.distance_to_goal = float(0.0)
            goal_handle.publish_feedback(feedback_msg)

        return result

    def _select_car_auto_method(self, mode: str):
        """根據模式選擇對應的方法，加入字串清洗防呆"""
        cleaned_mode = mode.strip()
        if cleaned_mode == "Manual_Nav":
            return self.nav_controller.manual_nav
        elif cleaned_mode == "Customize_Nav":
            return self.nav_controller.customize_nav
        else:
            return None