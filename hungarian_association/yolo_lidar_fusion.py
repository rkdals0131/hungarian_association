import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment

# ***TO DO***
from your_msgs.msg import YoloBboxes, ConePoints
from sensor_msgs import PointCloud2

class YoloLidarFusion (Node):
    def __init__(self):
        super().__init__('hungarian_association_node')

        #Publisher
        self.coord_sub

        #Subscriber
        self.yolo_sub = message_filters.Subscriber('yolo_bboxes_topic', YoloBboxes)
        self.cone_sub = message_filters.Subscriber('cone_points_topic', ConePoints)
        self.ats = message_filters.ApproximateTimeSynchronizer([self.yolo_sub, self.cone_sub], queue_size=10, slop=0.1)        
        self.ats.registerCallback(self.callback)
        

    def compute_cost_matrix(yolo_bboxes, cone_points):
        
        num_boxes = yolo_bboxes.shape[0]  # N: number of YOLO bounding boxes
        num_cones = cone_points.shape[0]  # M: number of cone points
        cost_matrix = np.zeros((num_boxes, num_cones))
        
        # Fill the cost matrix with the Euclidean distances.
        for i in range(num_boxes):
            # Calculate the center of the i-th bounding box.
            center_x = (yolo_bboxes[i, 0] + yolo_bboxes[i, 2]) / 2.0
            center_y = (yolo_bboxes[i, 1] + yolo_bboxes[i, 3]) / 2.0
            for j in range(num_cones):
                cost_matrix[i, j] = np.linalg.norm([center_x - cone_points[j, 0],
                                                    center_y - cone_points[j, 1]])
        
        # Pad the cost matrix to make it square.
        if num_boxes < num_cones:
            # Add dummy rows: number of rows to add = num_cones - num_boxes.
            dummy_rows = np.zeros((num_cones - num_boxes, num_cones))
            cost_matrix = np.vstack((cost_matrix, dummy_rows))
        elif num_boxes > num_cones:
            # Add dummy columns: number of columns to add = num_boxes - num_cones.
            dummy_cols = np.zeros((num_boxes, num_boxes - num_cones))
            cost_matrix = np.hstack((cost_matrix, dummy_cols))
        
        return cost_matrix

    def callback(yolo_msg, cone_msg):
        # Convert messages to NumPy arrays
        yolo_bboxes = convert_yolo_msg_to_array(yolo_msg)
        cone_points = convert_cone_msg_to_array(cone_msg)
        cost_matrix = compute_cost_matrix(yolo_bboxes, cone_points)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # *****todo*****
        # Process the matching results here 
    
def main(args=None):
    rclpy.init(args=args)
    hungarian_association_node = YoloLidarFusion()
    try:
        rclpy.spin(hungarian_association_node)
    finally:
        hungarian_association_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
