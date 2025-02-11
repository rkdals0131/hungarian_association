import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment

# ***TO DO***
# YOLOv8이 ROS2 에서 작동할 때 송출하는 메시지 타입, cone_detection 패키지가 퍼블리시하는 /sorted_cones 메시지 여기서 정의하고 받아와야 함
# 보내는 좌표는 타입을 뭘로 하지? 받는거랑 다른 타입으로 또 만들어야 하나?
from your_msgs.msg import YoloBboxes, ConePoints

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class YoloLidarFusion(Node):
    def __init__(self):
        super().__init__('hungarian_association_node')
        
        # QoS profile for better real-time performance
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher for fused coordinates
        self.coord_pub = self.create_publisher(
            ConePoints, 
            'fused_cone_coordinates',
            qos_profile
        )
        
        # Subscribers
        self.yolo_sub = message_filters.Subscriber(
            self,
            YoloBboxes,
            'yolo_bboxes_topic',
            qos_profile=qos_profile
        )
        
        self.cone_sub = message_filters.Subscriber(
            self,
            ConePoints,
            'cone_points_topic',
            qos_profile=qos_profile
        )
        
        # Approximate time synchronization
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [self.yolo_sub, self.cone_sub],
            queue_size=10,
            slop=0.1
        )
        
        self.ats.registerCallback(self.hungarian_callback)
        
        # 비용함수에서 사용할, 이미지 상에서의 두 점 사이 ec distance threshold
        self.max_matching_distance = 5.0  
        self.get_logger().info('YoloLidarFusion node initialized')

    @staticmethod
    def convert_yolo_msg_to_array(yolo_msg):
        """Convert YoloBboxes message to numpy array."""
        boxes = []
        for bbox in yolo_msg.bboxes:
            boxes.append([
                bbox.x_min,
                bbox.y_min,
                bbox.x_max,
                bbox.y_max
            ])
        return np.array(boxes)

    @staticmethod
    def convert_cone_msg_to_array(cone_msg):
        """Convert ConePoints message to numpy array."""
        points = []
        for point in cone_msg.points:
            points.append([point.x, point.y, point.z])
        return np.array(points)

    def compute_cost_matrix(self, yolo_bboxes, cone_points):
        num_boxes = yolo_bboxes.shape[0]
        num_cones = cone_points.shape[0]
        cost_matrix = np.zeros((num_boxes, num_cones))
        
        # Fill the cost matrix with the Euclidean distances
        for i in range(num_boxes):
            # Calculate the center of the i-th bounding box
            center_x = (yolo_bboxes[i, 0] + yolo_bboxes[i, 2]) / 2.0
            center_y = (yolo_bboxes[i, 1] + yolo_bboxes[i, 3]) / 2.0
            for j in range(num_cones):
                distance = np.linalg.norm([
                    center_x - cone_points[j, 0],
                    center_y - cone_points[j, 1]
                ])
                # Penalize matches beyond maximum distance
                cost_matrix[i, j] = distance if distance < self.max_matching_distance else 1e6
        
        # Pad the cost matrix to make it square
        if num_boxes < num_cones:
            dummy_rows = np.full((num_cones - num_boxes, num_cones), 1e6)
            cost_matrix = np.vstack((cost_matrix, dummy_rows))
        elif num_boxes > num_cones:
            dummy_cols = np.full((num_boxes, num_boxes - num_cones), 1e6)
            cost_matrix = np.hstack((cost_matrix, dummy_cols))
        
        return cost_matrix

    def hungarian_callback(self, yolo_msg, cone_msg):
        """Process synchronized YOLO and LiDAR cone detections."""
        try:
            # Convert messages to NumPy arrays
            yolo_bboxes = self.convert_yolo_msg_to_array(yolo_msg)
            cone_points = self.convert_cone_msg_to_array(cone_msg)
            
            if len(yolo_bboxes) == 0 or len(cone_points) == 0:
                self.get_logger().warn('ZERO detections in one or both sensors')
                return
            
            # Compute cost matrix and find optimal assignment
            cost_matrix = self.compute_cost_matrix(yolo_bboxes, cone_points)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Create message for matched coordinates
            matched_msg = ConePoints()
            matched_msg.header = cone_msg.header
            
            # Process matches
            for r, c in zip(row_ind, col_ind):
                if r < len(yolo_bboxes) and c < len(cone_points):
                    cost = cost_matrix[r, c]
                    if cost < self.max_matching_distance:
                        # Add matched cone point to output message
                        matched_msg.points.append(cone_msg.points[c])
                        
                        # Log matching information
                        self.get_logger().debug(
                            f'Matched YOLO bbox {r} with cone point {c}, '
                            f'ec distance: {cost:.2f}m'
                        )
            
            # Publish matched coordinates
            self.coord_pub.publish(matched_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    hungarian_association_node = YoloLidarFusion()
    try:
        rclpy.spin(hungarian_association_node)
    except KeyboardInterrupt:
        pass
    finally:
        hungarian_association_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()