import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from ArducamDepthCamera import ArducamCamera, TOFConnect, TOFDeviceType, TOFOutput, TOFControl, DepthData, ArducamInfo

GRAD_THRESH = 300
fx = 240 / (2 * math.tan(0.5 * math.pi * 64.3 / 180));
fy = 180 / (2 * math.tan(0.5 * math.pi * 50.4 / 180));

rclpy.init()
node = rclpy.create_node('tof')
img_pub = node.create_publisher(Image, "depth_image", 1)
img_pub2 = node.create_publisher(Image, "edge_image", 1)
lines_pub = node.create_publisher(Marker, "struct_lines", 1)
pp_pub = node.create_publisher(PointCloud2, "point_cloud", 1)

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

            header = Header()
            header.frame_id = "body"
            header.stamp = node.get_clock().now().to_msg()
            img = Image()
            img.header = header
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
            #verti_mask = np.zeros((180, 240), np.uint16)
            if lines_x_p is not None and lines_x_n is not None:
                max_len_sq = 0
                for line_x_p in lines_x_p:
                    pl = line_x_p[0]
                    if pl[1] > pl[3]:
                        tx = pl[0]
                        ty = pl[1]
                        pl[0] = pl[2]
                        pl[1] = pl[3]
                        pl[2] = tx
                        pl[3] = ty
                    for line_x_n in lines_x_n:
                        nl = line_x_n[0]
                        if nl[1] > nl[3]:
                            tx = nl[0]
                            ty = nl[1]
                            nl[0] = nl[2]
                            nl[1] = nl[3]
                            nl[2] = tx
                            nl[3] = ty
                        if pl[0]-nl[0]<20 and pl[0]-nl[0]>2 and abs(pl[1]-nl[1])<10:
                            dx = pl[0]-pl[2]
                            dy = pl[1]-pl[3]
                            len_sq = dx*dx+dy*dy;
                            if len_sq > max_len_sq:
                                max_len_sq = len_sq
                                #vert_struct = pl
                                vert_struct = (pl, nl)
                            #cv2.fillConvexPoly(verti_mask, np.array([[pl[0],pl[1]], [pl[2],pl[3]], [nl[2],nl[3]], [nl[0],nl[1]]]), 65535)
                            #vert_structs.append((pl[0]-2, pl[1], pl[2]-2, pl[3]))
                            break
            #print("vert struct", vert_struct)
            #verti_mask = np.bitwise_and(depth_u16, verti_mask)
            #img.data = verti_mask.ravel().view(np.uint8)
            #img_pub.publish(img)

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

            img.header = header
            img.encoding = "bgr8"
            img.step = 240*3
            img.data = edge_img.ravel().view(np.uint8)
            img_pub2.publish(img)

#            if vert_struct is not None:
#                pl, nl = vert_struct
#                xx = (pl[0], pl[2], nl[0], nl[2])
#                yy = (pl[1], pl[3], nl[1], nl[3])
#                points = []
#                for x in range(min(xx), max(xx)):
#                    for y in range(min(yy), max(yy)):
#                        d = depth_u16[y, x] * 0.001
#                        p = (d, (120 - x) / fx * d, (90 - y) / fy * d)
#                        points.append(p)
#                pp_pub.publish(point_cloud2.create_cloud_xyz32(header, points))

#            points = []
#            for x in range(0, 240):
#                for y in range(0, 180):
#                    d = depth_u16[y, x]
#                    if d < 2000:
#                        d = d * 0.001
#                        p = (d, (120 - x) / fx * d, (90 - y) / fy * d)
#                        points.append(p)
#            pp_pub.publish(point_cloud2.create_cloud_xyz32(header, points))

            line_list = Marker()
            line_list.header = header
            line_list.action = Marker.ADD
            line_list.type = Marker.LINE_LIST
            line_list.id = 1
#            line_list.pose.position.x = 0
#            line_list.pose.position.y = 0
#            line_list.pose.position.z = 0
#            line_list.pose.orientation.x = 0
#            line_list.pose.orientation.y = 0
#            line_list.pose.orientation.z = 0
            line_list.pose.orientation.w = 1.0
            line_list.ns = "verti_struct"
            line_list.scale.x = 0.01
            line_list.color.r = 1.0
            line_list.color.a = 1.0

            if vert_struct is not None:
                pl, nl = vert_struct
                pp = np.linspace(np.array([pl[0]-2, pl[1]]), np.array([pl[2]-2, pl[3]]), num=30).astype(np.int32)
                pp_3d = []
                for p in pp:
                    d = depth_u16[p[1], p[0]]
                    if d < 2000:
                        d = d * 0.001
                        pp_3d.append((d, (120 - p[0]) / fx * d, (90 - p[1]) / fy * d))
                pp = np.linspace(np.array([nl[0]+2, nl[1]]), np.array([nl[2]+2, nl[3]]), num=30).astype(np.int32)
                for p in pp:
                    d = depth_u16[p[1], p[0]]
                    if d < 2000:
                        d = d * 0.001
                        pp_3d.append((d, (120 - p[0]) / fx * d, (90 - p[1]) / fy * d))
                pp_pub.publish(point_cloud2.create_cloud_xyz32(header, pp_3d))
                l = cv2.fitLine(np.array(pp_3d), cv2.DIST_L2, 0, 0.01, 0.01)
                x = l[3].item(0)
                y = l[4].item(0)
                z = l[5].item(0)
                vx = l[0].item(0)
                vy = l[1].item(0)
                vz = l[2].item(0)
                p = Point()
                p.x = x - vx
                p.y = y - vy
                p.z = z - vz
                line_list.points.append(p)
                p = Point()
                p.x = x + vx
                p.y = y + vy
                p.z = z + vz
                line_list.points.append(p)
#            p = Point()
#            p.x = 1.0
#            p.y = 1.0
#            p.z = 0
#            line_list.points.append(p)
#            p = Point()
#            p.x = 1.0
#            p.y = 1.0
#            p.z = 10.0
#            line_list.points.append(p)
            lines_pub.publish(line_list)
#
#            if vert_struct is not None:
#                pl, nl = vert_struct
#                sx = (pl[0]+nl[0])//2
#                sy = (pl[1]+nl[1])//2
#                ex = (pl[2]+nl[2])//2
#                ey = (pl[3]+nl[3])//2
#                d = depth_u16[sy, sx] * -0.001
#                p = Point()
#                p.x = d
#                p.y = (120 - sx) / fx * d
#                p.z = (90 - sy) / fy * d
#                line_list.points.append(p)
#                d = depth_u16[ey, ex] * -0.001
#                p = Point()
#                p.x = d
#                p.y = (120 - ex) / fx * d
#                p.z = (90 - ey) / fy * d
#                line_list.points.append(p)

#                pp = ((pl[0], pl[1]), (pl[2], pl[3]), (nl[0], nl[1]), (nl[2], nl[3]))
#                for x, y in pp:
#                    d = depth_u16[y, x] * -0.001
#                    p = Point()
#                    p.x = d
#                    p.y = (120 - x) / fx * d
#                    p.z = (90 - y) / fy * d
#                    line_list.points.append(p)

#                lines_pub.publish(line_list)

#            if vert_struct is not None:
#                v = vert_struct[0];
#                x1 = v[0]-2
#                x2 = v[2]-2
#                if x1 > 0 and x2 > 0:
#                    if v[1] == 0:
#                        my = 0
#                    elif v[1] == 179:
#                        my = 177
#                    else:
#                        my = v[1] - 1
#                    pp = np.sort(depth_u16[my:my+3, x1-1:x1+2], axis=None)
#                    d = pp[4] * -0.001
#                    p = Point()
#                    p.x = d
#                    p.y = (120 - x1) / fx * d
#                    p.z = (90 - v[1]) / fy * d
#                    #print("start", p.x, p.y, p.z)
#                    line_list.points.append(p)
#
#                    if v[3] == 0:
#                        my = 0
#                    elif v[3] == 179:
#                        my = 177
#                    else:
#                        my = v[3] - 1
#                    pp = np.sort(depth_u16[my:my+3, x2-1:x2+2], axis=None)
#                    d = pp[4] * -0.001
#                    p = Point()
#                    p.x = d
#                    p.y = (120 - x2) / fx * d
#                    p.z = (90 - v[3]) / fy * d
#                    #print("end", p.x, p.y, p.z)
#                    line_list.points.append(p)
#                    lines_pub.publish(line_list)
        else:
            tof.releaseFrame(frame)

tof.stop()
tof.close()

rclpy.shutdown()
