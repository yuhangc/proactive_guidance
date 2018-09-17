#include <string>
#include <sstream>

#include "proactive_guidance/serial_manager.h"

// ============================================================================
SerialManager::SerialManager(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh)
{
    std::string device_port;
    pnh.param<std::string>("device_port", device_port, "/dev/ttyACM0");

    // setup ros interfaces
    haptic_ctrl_sub_ = nh_.subscribe<std_msgs::String>("/haptic_control", 1,
                                                       &SerialManager::haptic_callback, this);
    imu_data_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/human_rotation", 1);


    // initialize the arduino device
    arduino_ = new CArduinoDevice(device_port, CArduinoDevice::BAUD_115200);
    if (!arduino_->connect()) {
        ROS_ERROR("Cannot connect to input device!");
    } else {
        ROS_INFO("Input device connected");
    }
}

// ============================================================================
SerialManager::~SerialManager()
{
    delete arduino_;
}

// ============================================================================
void SerialManager::run(const double freq)
{
    ros::Rate rate(freq);

    while (!ros::isShuttingDown()) {
        // check if Arduino is connected
        if (!arduino_->isConnected()) {
            ROS_WARN("Input device not connected! Try to connect again");
            if (arduino_->connect()) {
                ROS_INFO("Input device connected");
            } else {
                ROS_ERROR("Cannot connect to input device!");
                return;
            }
        }

        // get raw message
        std::string message;
        int read_n = arduino_->read(message);

        // parse and publish
        if (read_n > 10) {
            std_msgs::Float32MultiArray imu_msg;
            std::stringstream ss(message);
            float val;

            for (int i = 0; i < 7; i++) {
                ss >> val;
                imu_msg.data.push_back(val);

                if (i < 4) {
                    ss.ignore(2);
                }
            }

            imu_data_pub_.publish(imu_msg);
        }

        ros::spinOnce();
        rate.sleep();
    }
}

// ============================================================================
void SerialManager::haptic_callback(const std_msgs::String::ConstPtr &haptic_msg)
{
    // some sanity check before hand?
    arduino_->write(haptic_msg->data);
}


// ============================================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "serial_manager");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    SerialManager serial(nh, pnh);

    serial.run(50);
}