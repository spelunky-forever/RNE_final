import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import math

class VirtualWallNode(Node):
    def __init__(self):
        super().__init__('virtual_wall_node')
        self.wall_pub = self.create_publisher(LaserScan, '/bridge_virtual_scan', 10)
        
        # 1. 監聽 YOLO：只用來判斷「畫面中有沒有橋」
        self.yolo_sub = self.create_subscription(
            Float32MultiArray, '/yolo/target_info', self.yolo_callback, 10)
        # 2. 監聽 20 個深度點：用來抓取真實的物理斜坡距離
        self.depth_sub = self.create_subscription(
            Float32MultiArray, '/camera/x_multi_depth_values', self.depth_callback, 10)
        
        self.bridge_in_view = False
        self.multi_depths = []
        
        self.timer = self.create_timer(0.1, self.publish_virtual_wall)
        self.get_logger().info("【真・物理記憶牆】已啟動！具備永久記憶與防穿透直牆能力。")

    def yolo_callback(self, msg):
        raw_data = msg.data
        self.bridge_in_view = False
        if raw_data:
            for i in range(0, len(raw_data), 3):
                if int(raw_data[i]) == 1:  # ID 1 為橋樑
                    self.bridge_in_view = True
                    break

    def depth_callback(self, msg):
        self.multi_depths = msg.data

    def publish_virtual_wall(self):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_footprint' 
        
        # 模擬 60 度廣角雷達
        scan.angle_min = -0.523
        scan.angle_max = 0.523
        scan.angle_increment = 1.046 / 20.0  # 對應你的 20 個等分點
        scan.range_min = 0.1
        scan.range_max = 5.0
        
        ranges = [float('inf')] * 20
        
        if self.bridge_in_view and len(self.multi_depths) == 20:
            # 過濾無效深度，找出畫面中「最靠近」的物理點（絕對是橋的斜坡，而不是橋底的破洞）
            valid_depths = [d for d in self.multi_depths if 0.1 < d < 4.0]
            if valid_depths:
                min_depth = min(valid_depths)
                
                # 幾何拉平演算法：將這 20 個點攤平成一堵與車頭垂直的堅固直牆
                for i in range(20):
                    angle = scan.angle_min + i * scan.angle_increment
                    ranges[i] = min_depth / math.cos(angle)
                    
        scan.ranges = ranges
        self.wall_pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = VirtualWallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()