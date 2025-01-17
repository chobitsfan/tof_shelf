#include <signal.h>
#include <stdio.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ArducamTOFCamera.hpp>

//see https://stackoverflow.com/questions/44081873/what-are-the-units-and-limits-of-gradient-magnitude
#define GRAD_THRESH 300
//#define HORI_STRUCT_W_CM 6
#define PUB_PP

bool gogogo = true;

void abort_handler(int signum) {
    gogogo = false;
}

bool line_compare(const cv::Vec4i& a, const cv::Vec4i& b) {
    return (a[0]-a[2])*(a[0]-a[2])+(a[1]-a[3])*(a[1]-a[3])>(b[0]-b[2])*(b[0]-b[2])+(b[1]-b[3])*(b[1]-b[3]);
}

int main(int argc, char *argv[])
{
    struct sigaction abort_act;
    abort_act.sa_handler = abort_handler;
    sigemptyset(&abort_act.sa_mask);
    abort_act.sa_flags = 0;
    sigaction(SIGINT, &abort_act, NULL);
#ifdef PUB_PP
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
#endif
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
    //image_transport::Publisher img_pub3 = it.advertise("struct_image", 1);
    ros::Publisher lines_pub = nh.advertise<visualization_msgs::Marker>("struct_lines", 1);

    cv::Mat grad_x(180, 240, CV_16S);
    cv::Mat grad_x_thresh(180, 240, CV_16S);
    cv::Mat grad_8u(180, 240, CV_8U);
    cv::Mat depth_8u(180, 240, CV_8U);
    cv::Mat lines_frame(180, 240, CV_8UC3);

    while (gogogo) {
        frame = tof.requestFrame(200);
        if (frame != nullptr) {
            depth_ptr = (float*)frame->getData(Arducam::FrameType::DEPTH_FRAME);
            confidence_ptr = (float*)frame->getData(Arducam::FrameType::CONFIDENCE_FRAME);
#ifdef PUB_PP
            point_cloud.header.stamp = ros::Time::now();
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
                        *iter_y = (120 - col_idx) / fx * zz;
                        *iter_z = (90 - row_idx) / fy * zz;
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
#endif

            cv::Mat depth_frame(180, 240, CV_32F, depth_ptr);
            cv::Mat confidence_frame(180, 240, CV_32F, confidence_ptr);
            depth_frame.setTo(2000, confidence_frame < 60);
            cv::threshold(depth_frame, depth_frame, 2000, 0, cv::THRESH_TRUNC);
            //cv::flip(depth_frame, depth_frame, -1);
            depth_frame.convertTo(depth_8u, CV_8U, 0.1);
            //double min;
            //cv::minMaxLoc(gray_frame, &min, NULL);
            //printf("min dist %f\n", min);
            //cv::convertScaleAbs(depth_frame, result_frame);
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", depth_8u).toImageMsg();
            img_pub.publish(img_msg);
            //cv::flip(result_frame, result_frame, -1);
            //GaussianBlur(result_frame, result_frame, cv::Size(3, 3), 0);

            //detect vertical structures
            cv::Sobel(depth_8u, grad_x, CV_16S, 1, 0, -1);
            cv::threshold(grad_x, grad_x_thresh, GRAD_THRESH, 255, cv::THRESH_BINARY);
            grad_x_thresh.convertTo(grad_8u, CV_8U);
            // Probabilistic Line Transform
            std::vector<cv::Vec4i> lines_x_p; // will hold the results of the detection
            cv::HoughLinesP(grad_8u, lines_x_p, 1, CV_PI/180, 80, 50, 5); // runs the actual detection
            // Draw the lines
            lines_frame = cv::Mat::zeros(lines_frame.size(), CV_8UC3);
            for (const cv::Vec4i& l:lines_x_p)
            {
                cv::line(lines_frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 1, cv::LINE_AA);
            }
            cv::threshold(grad_x, grad_x_thresh, -GRAD_THRESH, 255, cv::THRESH_BINARY_INV);
            grad_x_thresh.convertTo(grad_8u, CV_8U);
            std::vector<cv::Vec4i> lines_x_n; // will hold the results of the detection
            cv::HoughLinesP(grad_8u, lines_x_n, 1, CV_PI/180, 80, 50, 5); // runs the actual detection
            // Draw the lines
            for (const cv::Vec4i& l:lines_x_n)
            {
                cv::line(lines_frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,255,0), 1, cv::LINE_AA);
            }

            //detect horizontal structures
            cv::Sobel(depth_8u, grad_x, CV_16S, 0, 1, -1);
            cv::threshold(grad_x, grad_x_thresh, GRAD_THRESH, 255, cv::THRESH_BINARY);
            grad_x_thresh.convertTo(grad_8u, CV_8U);
            std::vector<cv::Vec4i> lines_y;
            cv::HoughLinesP(grad_8u, lines_y, 1, CV_PI/180, 80, 80, 5); // runs the actual detection
            int max_len_sq = 0;
            cv::Vec4i* hori_struct = NULL;
            // find the horizontal line with max length
            for (cv::Vec4i& l:lines_y)
            {
                // Draw the lines
                cv::line(lines_frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,0,0), 1, cv::LINE_AA);

                int dx = l[0]-l[2];
                int dy = l[1]-l[3];
                int len_sq = dx*dx+dy*dy;
                if (len_sq > max_len_sq) {
                    max_len_sq = len_sq;
                    hori_struct = &l;
                }
            }
            img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", lines_frame).toImageMsg();
            img_pub2.publish(img_msg);

            //only keep vertical lines which postive & negative edges are close enough
            //std::vector<cv::Vec4i> vert_structs;
            cv::Vec4i* vert_struct = NULL;
            max_len_sq = 0;
            for (cv::Vec4i& pl:lines_x_p) {
                /*bool dup_lines = false;
                for (const cv::Vec4i& v:vert_structs) {
                    if (abs(pl[0]-v[0]) < 3) {
                        dup_lines = true;
                        break;
                    }
                }
                if (dup_lines) continue;*/
                for (const cv::Vec4i& nl:lines_x_n) {
                    if (pl[0]-nl[0]<20 && pl[0]-nl[0]>2) {
                        //vert_structs.emplace_back(nl[0], nl[1], nl[2], nl[3]);
                        //vert_structs.emplace_back(pl[0], pl[1], pl[2], pl[3]);
                        int dx = pl[0]-pl[2];
                        int dy = pl[1]-pl[3];
                        int len_sq = dx*dx+dy*dy;
                        if (len_sq > max_len_sq) {
                            max_len_sq = len_sq;
                            vert_struct = &pl;
                        }
                        break;
                    }
                }
            }

            visualization_msgs::Marker line_list;
            line_list.header.frame_id = "body";
            line_list.header.stamp = ros::Time::now();
            line_list.action = visualization_msgs::Marker::ADD;
            line_list.type = visualization_msgs::Marker::LINE_LIST;
            line_list.pose.orientation.w = 1;
            line_list.ns = "structures";
            line_list.scale.x = 0.01;
            line_list.color.r = 1.0;
            line_list.color.a = 1.0;

            //printf("vert structures:");
            //for (const cv::Vec4i& v:vert_structs)
            if (vert_struct != NULL) {
                const cv::Vec4i& v = *vert_struct;
                //cv::line(lines_frame, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), cv::Scalar(255,255,255), 1, cv::LINE_AA);
                //printf("[%d,%d:%d->%d,%d:%d] ", v[0], v[1], depth_8u.at<uchar>(v[1],v[0]), v[2], v[3], depth_8u.at<uchar>(v[3],v[2]));
                int x1 = v[0]-2;
                int x2 = v[2]-2;
                if (x1 > 0 && x2 > 0) {
                    int my;
                    if (v[1] == 0) my = 0; else if (v[1] == 179) my = 177; else my = v[1] - 1;
                    cv::Mat sub_d = depth_frame(cv::Rect(x1-1, my, 3, 3)).clone();
                    float* pp = sub_d.ptr<float>();
                    std::sort(pp, pp+9);
                    float d = pp[4] * 0.001f;
                    geometry_msgs::Point p;
                    p.x = d;
                    p.y = (120 - x1) / fx * d;
                    p.z = (90 - v[1]) / fy * d;
                    line_list.points.push_back(p);

                    if (v[3] == 0) my = 0; else if (v[3] == 179) my = 177; else my = v[3] - 1;
                    sub_d = depth_frame(cv::Rect(x2-1, my, 3, 3)).clone();
                    pp = sub_d.ptr<float>();
                    std::sort(pp, pp+9);
                    d = pp[4] * 0.001f;
                    p.x = d;
                    p.y = (120 - x2) / fx * d;
                    p.z = (90 - v[3]) / fy * d;
                    line_list.points.push_back(p);
                }
            }

            //printf("hori structures:");
            //float dd[5];
            //for (const cv::Vec4i& l:lines_y)
            if (hori_struct != NULL) {
                const cv::Vec4i& l = *hori_struct;
                //cv::line(lines_frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,255,255), 1, cv::LINE_AA);
                //printf("[%d,%d:%d->%d,%d:%d] ", l[0], l[1], depth_8u.at<uchar>(l[1]-2,l[0]), l[2], l[3], depth_8u.at<uchar>(l[3]-2,l[2]));
                int y1 = l[1]-3;
                int y2 = l[3]-3;
                if (y1 > 0 && y2 > 0) {
                    int mx;
                    if (l[0] == 0) mx = 0; else if (l[0] == 239) mx = 237; else mx = l[0] - 1;
                    cv::Mat sub_d = depth_frame(cv::Rect(mx, y1-1, 3, 3)).clone();
                    float* pp = sub_d.ptr<float>();
                    std::sort(pp, pp+9);
                    //for (int i=0;i<5;i++) printf("%f ", dd[i]);
                    //printf("\n");
                    //float d = depth_frame.at<float>(y1,l[0]) * 0.001;
                    float d = pp[4]*0.001f;
                    geometry_msgs::Point p;
                    p.x = d;
                    p.y = (120 - l[0]) / fx * d;
                    p.z = (90 - y1) / fy * d;
                    line_list.points.push_back(p);

                    if (l[2] == 0) mx = 0; else if (l[2] == 239) mx = 237; else mx = l[2] - 1;
                    sub_d = depth_frame(cv::Rect(mx, y2-1, 3, 3)).clone();
                    pp = sub_d.ptr<float>();
                    std::sort(pp, pp+9);
                    //pp = depth_frame.ptr<float>(y2,l[2]-5);
                    //memcpy(dd, pp, sizeof(float)*5);
                    //std::sort(dd, dd+5);
                    //for (int i=0;i<5;i++) printf("%f ", dd[i]);
                    //printf("\n");
                    //d = depth_frame.at<float>(y2,l[2]) * 0.001;
                    d = pp[4]*0.001f;
                    p.x = d;
                    p.y = (120 - l[2]) / fx * d;
                    p.z = (90 - y2) / fy * d;
                    line_list.points.push_back(p);
                }
            }
            //printf("\n");
            //lines_frame = cv::Mat::zeros(lines_frame.size(), CV_8UC3);
            //img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", lines_frame).toImageMsg();
            //img_pub3.publish(img_msg);
            lines_pub.publish(line_list);
        }
        tof.releaseFrame(frame);
    }

    tof.stop();
    tof.close();

    printf("byebye\n");

    return 0;
}
