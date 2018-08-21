#ifndef SERIAL_MANAGER_H
#define SERIAL_MANAGER_H

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>

#include "CArduinoDevice.h"

class SerialManager
{
public:
    // constructor
    SerialManager(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // destructor
    ~SerialManager();

    // main update function
    void run(const double freq);

private:
    // node handler
    ros::NodeHandle nh_;

    // subscriber and publishers
    ros::Subscriber haptic_ctrl_sub_;
    ros::Publisher imu_data_pub_;

    // arduino device
    CArduinoDevice* arduino_;

    // callback functions
    void haptic_callback(const std_msgs::String::ConstPtr& haptic_msg);
};

#endif