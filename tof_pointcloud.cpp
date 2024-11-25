#include <signal.h>
#include <stdio.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/PointStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ArducamTOFCamera.hpp>

bool gogogo = true;

void abort_handler(int signum) {
    gogogo = false;
}

int main(int argc, char *argv[])
{
    struct sigaction abort_act;
    abort_act.sa_handler = abort_handler;
    sigemptyset(&abort_act.sa_mask);
    abort_act.sa_flags = 0;
    sigaction(SIGINT, &abort_act, NULL);

    sensor_msgs::PointCloud2 point_cloud;
    point_cloud.header.frame_id = "body";
    point_cloud.width = 240;
    point_cloud.height = 180;
    point_cloud.fields.resize(3);
    point_cloud.fields[0].name = "x";
    point_cloud.fields[1].name = "y";
    point_cloud.fields[2].name = "z";
    point_cloud.fields[0].offset = 0;
    point_cloud.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    point_cloud.fields[0].count  = 1;
    point_cloud.fields[1].offset = 4;
    point_cloud.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    point_cloud.fields[1].count  = 1;
    point_cloud.fields[2].offset = 8;
    point_cloud.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    point_cloud.fields[2].count  = 1;
    point_cloud.point_step = 12;
    point_cloud.is_bigendian = false;
    point_cloud.is_dense = false;
    point_cloud.data.resize(12*240*180);
    point_cloud.row_step = 12*240;

    float fx = 240 / (2 * tan(0.5 * M_PI * 64.3 / 180));
    float fy = 180 / (2 * tan(0.5 * M_PI * 50.4 / 180));
    Arducam::ArducamFrameBuffer* frame;
    float* depth_ptr;
    float* confidence_ptr;
    Arducam::ArducamTOFCamera tof;

    if (tof.open(Arducam::Connection::CSI))
    {
        printf("initialize fail\n");
        exit(-1);
    }

    if (tof.start())
    {
        printf("start fail\n");
        exit(-1);
    }
    tof.setControl(Arducam::CameraCtrl::RANGE, 2);
    tof.setControl(Arducam::CameraCtrl::FRAME_RATE, 5);

    ros::init(argc, argv, "tof", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh("~");
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud", 1);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise("depth_image", 1);
    image_transport::Publisher img_pub2 = it.advertise("edge_image", 1);

    cv::Mat grad_x(180, 240, CV_8U);
    cv::Mat grad_y(180, 240, CV_32F);
    cv::Mat abs_grad_x(180, 240, CV_8U);
    cv::Mat abs_grad_y(180, 240, CV_8U);
    cv::Mat result_frame(180, 240, CV_8U);
    cv::Mat lines_frame(180, 240, CV_8U);

    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector();

    while (gogogo) {
        frame = tof.requestFrame(200);
        if (frame != nullptr) {
            point_cloud.header.stamp = ros::Time::now();
            depth_ptr = (float*)frame->getData(Arducam::FrameType::DEPTH_FRAME);
            confidence_ptr = (float*)frame->getData(Arducam::FrameType::CONFIDENCE_FRAME);
            //unsigned int pos = 0;
            //float max_conf = 0;
            sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud, "z");
            for (int row_idx = 0; row_idx < 180; row_idx++) {
                for (int col_idx = 1; col_idx < 239; col_idx++) {
                    unsigned int pos = row_idx * 240 + col_idx;
                    //if (confidence_ptr[pos] > max_conf) max_conf = confidence_ptr[pos];
                    if (confidence_ptr[pos] > 80 && confidence_ptr[pos-1] > 80 && confidence_ptr[pos+1] > 80) {
                        float zz = depth_ptr[pos] * 0.001f;
                        *iter_x = zz;
                        *iter_y = -(((120 - col_idx)) / fx) * zz;
                        *iter_z = -((90 - row_idx) / fy) * zz;
                    } else {
                        *iter_x = std::numeric_limits<float>::quiet_NaN();
                        *iter_y = std::numeric_limits<float>::quiet_NaN();
                        *iter_z = std::numeric_limits<float>::quiet_NaN();
                    }
                    ++iter_x;
                    ++iter_y;
                    ++iter_z;
                }
            }
            pub.publish(point_cloud);
            //printf("max confidence %f\n", max_conf);

            cv::Mat depth_frame(180, 240, CV_32F, depth_ptr);
            cv::Mat confidence_frame(180, 240, CV_32F, confidence_ptr);
            depth_frame.setTo(2000, confidence_frame < 60);
            cv::flip(depth_frame, depth_frame, -1);
            //double max;
            //cv::minMaxLoc(depth_frame, NULL, &max);
            //printf("max dist %f\n", max);
            depth_frame.convertTo(result_frame, CV_8U, 255.0 / 2000.0);
            //cv::convertScaleAbs(depth_frame, result_frame);
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", result_frame).toImageMsg();
            img_pub2.publish(img_msg);
            //cv::flip(result_frame, result_frame, -1);
            //GaussianBlur(result_frame, result_frame, cv::Size(3, 3), 0);
            cv::Sobel(result_frame, grad_x, CV_8U, 0, 1, -1);
            //cv::threshold(grad_x, grad_x, 250, 255, cv::THRESH_BINARY);
            //cv::Sobel(depth_frame, grad_y, -1, 0, 1);
            //cv::convertScaleAbs(grad_x, abs_grad_x);
            //cv::convertScaleAbs(grad_y, abs_grad_y);
            //addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, result_frame);
            //cv::applyColorMap(result_frame, result_frame, cv::COLORMAP_RAINBOW);
            //result_frame.setTo(cv::Scalar(0, 0, 0), confidence_frame < 80);

            lines_frame = cv::Mat::zeros(lines_frame.size(), CV_8U);
            std::vector<cv::Vec4f> lines_std;
            lsd->detect(grad_x, lines_std);
            lsd->drawSegments(lines_frame, lines_std);
#if 0
            // Probabilistic Line Transform
            std::vector<cv::Vec4i> linesP; // will hold the results of the detection
            cv::HoughLinesP(grad_x, linesP, 1, CV_PI/180, 50, 50, 100); // runs the actual detection
            //printf("%d lines detected\n", linesP.size());
            // Draw the lines
            for(size_t i = 0; i < linesP.size(); i++ )
            {
                cv::Vec4i l = linesP[i];
                line(lines_frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 255, 1, cv::LINE_AA);
            }
#endif

            img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", lines_frame).toImageMsg();
            img_pub.publish(img_msg);
        }
        tof.releaseFrame(frame);
    }

    tof.stop();
    tof.close();

    printf("byebye\n");

    return 0;
}
