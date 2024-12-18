import rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from ArducamDepthCamera import ArducamCamera, TOFConnect, TOFDeviceType, TOFOutput, TOFControl, DepthData, ArducamInfo

rclpy.init()
node = rclpy.create_node('tof')
img_pub = node.create_publisher(Image, 'depth_image', 1)

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
tof.setControl(TOFControl.FRAME_RATE, 10)

while rclpy.ok():
    frame = tof.requestFrame(200)
    if frame is not None and isinstance(frame, DepthData):
        depth_buf = frame.getDepthData()
        confidence_buf = frame.getConfidenceData()
        #depth_buf = depth_buf.astype('uint8')
        #print(type(depth_buf), depth_buf.ndim, depth_buf.dtype)

        depth_buf[confidence_buf < 60] = 2000
        depth_buf = depth_buf * 0.1
        depth_buf = depth_buf.astype('uint8')

        img = Image()
        img.header.stamp = node.get_clock().now().to_msg()
        img.height = 180
        img.width = 240
        img.encoding = "mono8"
        img.step = 240
        img.is_bigendian = 0
        img.data = depth_buf.ravel().tolist()
        img_pub.publish(img)

        tof.releaseFrame(frame)

tof.stop()
tof.close()

rclpy.shutdown()
