import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from ArducamDepthCamera import ArducamCamera, TOFConnect, TOFDeviceType, TOFOutput, TOFControl, DepthData, ArducamInfo

GRAD_THRESH = 300
fx = 240 / (2 * math.tan(0.5 * math.pi * 64.3 / 180));
fy = 180 / (2 * math.tan(0.5 * math.pi * 50.4 / 180));

rclpy.init()
node = rclpy.create_node('tof')
img_pub = node.create_publisher(Image, "depth_image", 1)
img_pub2 = node.create_publisher(Image, "edge_image", 1)
lines_pub = node.create_publisher(Marker, "struct_lines", 1);

tof = ArducamCamera()
ret = 0
ret = tof.open(TOFConnect.CSI, 0)
if not ret:
    print("Failed to open camera. Error code:", ret)
    exit()
ret = tof.start(TOFOutput.DEPTH)
if ret != 0:
    print("Failed to start camera. Error code:", ret)
    tof.close()
    exit()
tof.setControl(TOFControl.RANGE, 2)
tof.setControl(TOFControl.FRAME_RATE, 5)
#tof.setControl(TOFControl.AUTO_FRAME_RATE, 0)

skip_c = 0;
print("start");

while rclpy.ok():
    frame = tof.requestFrame(200)
    if frame is not None and isinstance(frame, DepthData):
        skip_c += 1
        if skip_c > 5:
            skip_c = 0
            depth_buf = frame.getDepthData()
            confidence_buf = frame.getConfidenceData()
            tof.releaseFrame(frame)

            depth_buf[confidence_buf < 60] = 2000
            depth_buf[depth_buf > 2000] = 2000
            depth_u16 = depth_buf.astype(np.uint16)

            img = Image()
            img.header.stamp = node.get_clock().now().to_msg()
            img.height = 180
            img.width = 240
            img.is_bigendian = 0
            img.encoding = "mono16"
            img.step = 240*2
            img.data = depth_u16.ravel().view(np.uint8)
            img_pub.publish(img)

            edge_img = np.zeros((180, 240, 3), dtype=np.uint8)

            # detect vertical structures
            grad = cv2.Sobel(depth_u16, cv2.CV_16S, 1, 0, -1)
            ret, grad_thresh = cv2.threshold(grad, GRAD_THRESH, 255, cv2.THRESH_BINARY)
            grad_u8 = grad_thresh.astype(np.uint8)
            lines_x_p = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 50, 5)
            if lines_x_p is not None:
                for line in lines_x_p:
                    l = line[0]
                    cv2.line(edge_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_8)
            ret, grad_thresh = cv2.threshold(grad, -GRAD_THRESH, 255, cv2.THRESH_BINARY_INV);
            grad_u8 = grad_thresh.astype(np.uint8)
            lines_x_n = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 50, 5)
            if lines_x_n is not None:
                for line in lines_x_n:
                    l = line[0]
                    cv2.line(edge_img, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1, cv2.LINE_8)
            # only select vertical lines which postive & negative edges close enough
            vert_struct = None
            if lines_x_p is not None and lines_x_n is not None:
                max_len_sq = 0
                for line_x_p in lines_x_p:
                    pl = line_x_p[0]
                    for line_x_n in lines_x_n:
                        nl = line_x_n[0]
                        if pl[0]-nl[0]<20 and pl[0]-nl[0]>2:
                            dx = pl[0]-pl[2]
                            dy = pl[1]-pl[3]
                            len_sq = dx*dx+dy*dy;
                            if len_sq > max_len_sq:
                                max_len_sq = len_sq
                                vert_struct = pl
                            break
            #print("vert struct", vert_struct)

            # detect horizontal structures
            grad = cv2.Sobel(depth_u16, cv2.CV_16S, 0, 1, -1)
            ret, grad_thresh = cv2.threshold(grad, GRAD_THRESH, 255, cv2.THRESH_BINARY)
            grad_u8 = grad_thresh.astype(np.uint8)
            lines_y = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 80, 5)
            # find the horizontal line with max length
            hori_struct = None
            if lines_y is not None:
                max_len_sq = 0
                for line in lines_y:
                    l = line[0]
                    dx = l[0]-l[2]
                    dy = l[1]-l[3]
                    len_sq = dx*dx+dy*dy
                    if len_sq > max_len_sq:
                        max_len_sq = len_sq
                        hori_struct = l
                    cv2.line(edge_img, (l[0], l[1]), (l[2], l[3]), (255,0,0), 1, cv2.LINE_8)

            img.header.stamp = node.get_clock().now().to_msg()
            img.encoding = "bgr8"
            img.step = 240*3
            img.data = edge_img.ravel().view(np.uint8)
            img_pub2.publish(img)

            line_list = Marker()
            line_list.header.frame_id = "body"
            line_list.header.stamp = node.get_clock().now().to_msg()
            line_list.action = Marker.ADD
            line_list.type = Marker.LINE_LIST
            line_list.pose.orientation.w = 1.0
            line_list.ns = "verti_struct"
            line_list.scale.x = 0.01
            line_list.color.r = 1.0
            line_list.color.a = 1.0

            if vert_struct is not None:
                v = vert_struct;
                x1 = v[0]-2
                x2 = v[2]-2
                if x1 > 0 and x2 > 0:
                    if v[1] == 0:
                        my = 0
                    elif v[1] == 179:
                        my = 177
                    else:
                        my = v[1] - 1
                    pp = np.sort(depth_u16[my:my+3, x1-1:x1+2], axis=None)
                    d = pp[4] * 0.001
                    p = Point()
                    p.x = d
                    p.y = (120 - x1) / fx * d
                    p.z = (90 - v[1]) / fy * d
                    line_list.points.append(p)

                    if v[3] == 0:
                        my = 0
                    elif v[3] == 179:
                        my = 177
                    else:
                        my = v[3] - 1
                    pp = np.sort(depth_u16[my:my+3, x2-1:x2+2], axis=None)
                    d = pp[4] * 0.001
                    p = Point()
                    p.x = d
                    p.y = (120 - x2) / fx * d
                    p.z = (90 - v[3]) / fy * d
                    line_list.points.append(p)

                    lines_pub.publish(line_list)
        else:
            tof.releaseFrame(frame)

tof.stop()
tof.close()

rclpy.shutdown()
