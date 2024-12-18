import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from ArducamDepthCamera import ArducamCamera, TOFConnect, TOFDeviceType, TOFOutput, TOFControl, DepthData, ArducamInfo

GRAD_THRESH = 300

rclpy.init()
node = rclpy.create_node('tof')
img_pub = node.create_publisher(Image, 'depth_image', 1)
img_pub2 = node.create_publisher(Image, 'edge_image', 1)

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
            #depth_buf = depth_buf.astype('uint8')
            #print(type(depth_buf), depth_buf.ndim, depth_buf.dtype)

            depth_buf[confidence_buf < 60] = 2000
            depth_buf[depth_buf > 2000] = 2000
            #depth_buf = depth_buf * 0.1
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

            #depth_buf = depth_buf * 0.1
            #edge_img = cv2.cvtColor(depth_buf.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            grad = cv2.Sobel(depth_u16, cv2.CV_16S, 0, 1, -1);
            ret, grad_thresh = cv2.threshold(grad, GRAD_THRESH, 255, cv2.THRESH_BINARY);
            grad_u8 = grad_thresh.astype(np.uint8);
            lines_y = cv2.HoughLinesP(grad_u8, 1, np.pi/180, 50, None, 80, 5)
            edge_img = np.zeros((180, 240, 3), dtype=np.uint8)
            if lines_y is not None:
                for line in lines_y:
                    l = line[0]
                    cv2.line(edge_img, (l[0], l[1]), (l[2], l[3]), (255,0,0), 1, cv2.LINE_8)

            #img = Image()
            img.header.stamp = node.get_clock().now().to_msg()
            #img.height = 180
            #img.width = 240
            #img.is_bigendian = 0
            img.encoding = "bgr8"
            img.step = 240*3
            img.data = edge_img.ravel().view(np.uint8)
            img_pub2.publish(img)

        tof.releaseFrame(frame)

tof.stop()
tof.close()

rclpy.shutdown()
