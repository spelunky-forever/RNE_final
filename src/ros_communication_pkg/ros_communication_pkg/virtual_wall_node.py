import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import math

class VirtualWallNode(Node):
    def __init__(self):
        super().__init__('virtual_wall_node')
        
        # 建立發布虛擬牆的 Publisher
        self.wall_pub = self.create_publisher(LaserScan, '/bridge_virtual_scan', 10)
        
        # 訂閱 YOLO 的辨識結果
        self.yolo_sub = self.create_subscription(
            Float32MultiArray, '/yolo/target_info', self.yolo_callback, 10)
        
        self.latest_bridge_depth = float('inf')
        
        # 依據架構書要求：必須以固定頻率(10Hz)持續發布 
        self.timer = self.create_timer(0.1, self.publish_virtual_wall)
        
        self.get_logger().info("虛擬牆發布節點 (Virtual Wall Node) 已啟動，頻率 10Hz")

    def yolo_callback(self, msg):
        raw_data = msg.data
        min_depth = float('inf')
        
        # 解析 yolo_target_info [class_id, depth, delta_x, ...]
        if raw_data:
            for i in range(0, len(raw_data), 3):
                class_id = int(raw_data[i])
                depth = float(raw_data[i+1])
                
                # 如果看到橋樑 (ID: 1)，且有有效深度
                if class_id == 1 and depth > 0.1:
                    if depth < min_depth:
                        min_depth = depth
                        
        self.latest_bridge_depth = min_depth

    def publish_virtual_wall(self):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        # 綁定在車體中心，這樣牆壁會跟著車頭轉動
        scan.header.frame_id = 'base_footprint' 
        
        # 模擬一個前方 90 度的雷達視角 (-45度 到 +45度)
        scan.angle_min = -0.785
        scan.angle_max = 0.785
        scan.angle_increment = 0.0174 # 約 1 度一個點
        scan.range_min = 0.1
        scan.range_max = 5.0
        
        num_rays = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
        
        if self.latest_bridge_depth == float('inf'):
            # 沒看到橋：發布充滿 inf 的陣列，讓 Nav2 清除 Costmap 上的牆壁 
            scan.ranges = [float('inf')] * num_rays
        else:
            # 看到橋：在前方產生一堵距離為 latest_bridge_depth 的牆
            # 為了確保 Nav2 絕對會繞路，我們把這個 90 度的扇形全部填滿深度
            # 這樣在 Costmap 上看起來就是正前方有一面無法穿越的巨大弧形牆
            scan.ranges = [self.latest_bridge_depth] * num_rays
            
        self.wall_pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = VirtualWallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()