#include <signal.h>
#include <stdio.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/PointStamped.h>
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
    tof.setControl(Arducam::CameraCtrl::RANGE, 4);
    tof.setControl(Arducam::CameraCtrl::FRAME_RATE, 10);

    ros::init(argc, argv, "tof", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh("~");
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud", 2);

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
        }
        tof.releaseFrame(frame);
    }

    tof.stop();
    tof.close();

    printf("byebye\n");

    return 0;
}
