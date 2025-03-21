import os
import cv2
import yaml
import numpy as np
import rclpy
from typing import Tuple, List, Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.optimize import linear_sum_assignment
from hungarian_association.config_utils import load_hungarian_config

from yolo_msgs.msg import DetectionArray
from custom_interface.msg import ModifiedFloat32MultiArray




def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    return T

def load_camera_calibration(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs

class YoloLidarFusion(Node):
    def __init__(self):
        super().__init__('hungarian_association_node')
        
        # Load the configuration from hungarian_association package
        self.config = load_hungarian_config()
        if self.config is None:
            self.get_logger().error("Failed to load hungarian_association configuration.")
            return
            
        # Get parameters from config with defaults as fallback
        hungarian_config = self.config.get('hungarian_association', {})
        
        # Initialize parameters from config
        self.declare_parameter('cone_z_offset', 
                              hungarian_config.get('cone_z_offset', -0.6))
        self.cone_z_offset = self.get_parameter('cone_z_offset').value
        self.get_logger().info(f"Using cone z offset: {self.cone_z_offset} meters")
        
        # Set up max matching distance
        self.max_matching_distance = hungarian_config.get('max_matching_distance', 5.0)
        self.get_logger().info(f"Max matching distance: {self.max_matching_distance}")

        # Add on set parameters callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Get calibration file paths
        calib_config = hungarian_config.get('calibration', {})
        config_folder = calib_config.get('config_folder', '')
        extrinsic_file = calib_config.get('camera_extrinsic_calibration', '')
        intrinsic_file = calib_config.get('camera_intrinsic_calibration', '')
        
        # Load extrinsic and intrinsic calibrations
        extrinsic_yaml = os.path.join(config_folder, extrinsic_file)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = os.path.join(config_folder, intrinsic_file)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        # Get topic names from config
        cones_topic = hungarian_config.get('cones_topic', "/sorted_cones_time")
        boxes_topic = hungarian_config.get('boxes_topic', "/detections")
        output_topic = hungarian_config.get('output_topic', "/fused_sorted_cones")
        
        self.get_logger().info(f"Subscribing to cones topic: {cones_topic}")
        self.get_logger().info(f"Subscribing to boxes topic: {boxes_topic}")

        # QoS settings from config
        qos_config = hungarian_config.get('qos', {})
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_config.get('history_depth', 1)
        )

        # message_filters를 이용해 2개 토픽을 동기화
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, cones_topic, qos_profile=best_effort_qos)
        self.boxes_sub = Subscriber(self, DetectionArray, boxes_topic, qos_profile=best_effort_qos)

        # Approximate time synchronization
        self.ats = ApproximateTimeSynchronizer(
            [self.cones_sub, self.boxes_sub],
            queue_size=qos_config.get('sync_queue_size', 10),
            slop=qos_config.get('sync_slop', 0.1)
        )
        
        self.ats.registerCallback(self.hungarian_callback)

        # Color mapping for visualization (BGR format)
        self.color_mapping = {
            "Crimson Cone": (0, 0, 255),   # Red
            "Yellow Cone":  (0, 255, 255), # Yellow
            "Blue Cone":    (255, 0, 0),   # Blue
            "Unknown":      (0, 255, 0)    # Green (default)
        }

        # Publisher for fused coordinates
        self.coord_pub = self.create_publisher(
            ModifiedFloat32MultiArray, 
            output_topic,
            qos_profile=best_effort_qos
        )

        self.get_logger().info('YoloLidarFusion node initialized')

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'cone_z_offset':
                self.cone_z_offset = param.value
                self.get_logger().info(f"Updated cone z offset: {self.cone_z_offset} meters")

    @staticmethod
    def convert_yolo_msg_to_array(yolo_msg):
        """Convert DetectionArray message to numpy array."""
        boxes = []
        for detection in yolo_msg.detections:
            boxes.append([
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
                detection.bbox.size.x,
                detection.bbox.size.y
            ])
        return np.array(boxes)

    def convert_cone_msg_to_array(self, cone_msg):
        """Convert ModifiedFloat32MultiArray message to numpy array and project to image plane."""
        # Process cones message
        cone_data = np.array(cone_msg.data, dtype=np.float32)
        
        if cone_data.size == 0:
            self.get_logger().warn("Empty cones data.")
            return np.array([]), np.array([])
        
        # Get number of points from layout
        num_points = cone_msg.layout.dim[0].size
        if num_points * 2 != cone_data.size:
            self.get_logger().error(f"Cone data size ({cone_data.size}) does not match layout dimensions ({num_points}*2).")
            return np.array([]), np.array([])
        
        # Reshape to (N,2) array
        cones_xy = cone_data.reshape(num_points, 2)
        
        # Add z coordinate using parameter value
        cones_xyz = np.hstack((cones_xy, np.ones((num_points, 1), dtype=np.float32) * self.cone_z_offset))
        
        # Convert to homogeneous coordinates
        cones_xyz_h = np.hstack((cones_xyz, np.ones((cones_xyz.shape[0], 1), dtype=np.float32)))
        
        # Transform from LiDAR to camera coordinate system
        cones_cam_h = cones_xyz_h @ self.T_lidar_to_cam.T
        cones_cam = cones_cam_h[:, :3]  # Extract 3D coordinates from homogeneous
        
        # Create array of original indices (all points)
        original_indices = np.arange(num_points)
        
        # Project to image plane
        if num_points > 0:
            rvec = np.zeros((3,1), dtype=np.float64)
            tvec = np.zeros((3,1), dtype=np.float64)
            cone_image_points, _ = cv2.projectPoints(
                cones_cam.astype(np.float64),
                rvec, tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            cone_image_points = cone_image_points.reshape(-1, 2)
            
            self.get_logger().debug(f"Projected {len(cone_image_points)} cones to image plane")
            
            return cone_image_points, original_indices
        
        return np.array([]), np.array([])

    def compute_cost_matrix(self, yolo_bboxes, cone_points):
        num_boxes = yolo_bboxes.shape[0]
        num_cones = cone_points.shape[0]
        cost_matrix = np.zeros((num_boxes, num_cones))
        
        # Fill the cost matrix with the Euclidean distances
        for i in range(num_boxes):
            # Calculate the center of the i-th bounding box
            center_x = yolo_bboxes[i, 0]
            center_y = yolo_bboxes[i, 1]
            for j in range(num_cones):
                distance = np.linalg.norm([
                    center_x - cone_points[j, 0],
                    center_y - cone_points[j, 1]
                ])
                # Penalize matches beyond maximum distance
                cost_matrix[i, j] = distance if distance < self.max_matching_distance else 1e6
        
        # Pad the cost matrix to make it square
        if num_boxes < num_cones:
            # Set cost to high value for dummy YOLO boxes (we prefer to keep LiDAR points as "Unknown")
            dummy_rows = np.full((num_cones - num_boxes, num_cones), 0.0)
            cost_matrix = np.vstack((cost_matrix, dummy_rows))
        elif num_boxes > num_cones:
            # Set cost to 0 for dummy LiDAR points (be "forgiving" for YOLO boxes without LiDAR match)
            dummy_cols = np.full((num_boxes, num_boxes - num_cones), 0.0)
            cost_matrix = np.hstack((cost_matrix, dummy_cols))
        
        return cost_matrix

    def hungarian_callback(self, cone_msg, yolo_msg):
        """Process synchronized YOLO and LiDAR cone detections."""
        try:
            # Convert messages to NumPy arrays
            yolo_bboxes = self.convert_yolo_msg_to_array(yolo_msg)
            
            # Project cone points to image plane and get original indices
            cone_image_points, original_indices = self.convert_cone_msg_to_array(cone_msg)
            
            # 매칭된 콘만 저장할 새로운 메시지 생성
            filtered_msg = ModifiedFloat32MultiArray()
            filtered_msg.header = cone_msg.header
            filtered_msg.layout = cone_msg.layout
            filtered_msg.class_names = []
            filtered_msg.data = []
            
            if len(yolo_bboxes) == 0 or len(cone_image_points) == 0:
                self.get_logger().warn('ZERO detections in one or both sensors')
                self.coord_pub.publish(filtered_msg)  # 빈 메시지 발행
                return
            
            # 코스트 매트릭스 계산 및 매칭
            cost_matrix = self.compute_cost_matrix(yolo_bboxes, cone_image_points)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 매칭된 포인트만 필터링
            for i, j in zip(row_ind, col_ind):
                # YOLO 박스와 매칭되고 최대 거리 이내인 경우만 선택
                if (i < len(yolo_bboxes) and j < len(cone_image_points) and 
                    cost_matrix[i, j] < self.max_matching_distance):
                    
                    original_idx = original_indices[j]
                    # 원본 데이터에서 해당 콘의 x, y 좌표 추출
                    original_x = cone_msg.data[original_idx * 2]
                    original_y = cone_msg.data[original_idx * 2 + 1]
                    
                    # 필터링된 메시지에 추가
                    filtered_msg.data.extend([original_x, original_y])
                    filtered_msg.class_names.append(yolo_msg.detections[i].class_name)
            
            # 레이아웃 크기 업데이트
            filtered_msg.layout.dim[0].size = len(filtered_msg.class_names)
            
            # 필터링된 메시지 발행
            self.coord_pub.publish(filtered_msg)
            
            self.get_logger().info(
                f'Published {len(filtered_msg.class_names)} matched cones out of '
                f'{len(cone_image_points)} LiDAR detections and {len(yolo_bboxes)} YOLO detections'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

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