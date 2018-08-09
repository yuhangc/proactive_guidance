
// Pantograph.ino

// Yuhang Che
// Adapted from Julie Walker
// Created on July 27, 2018

#include <Servo.h>

#include <ros.h>
#include <std_msgs/String.h>

#include <PantographDevice.h>

// ros node handler
ros::NodeHandle nh;

// device configurations
static const float a1 = 15.00;     // proximal motor upper arm [mm]
static const float a2 = 20.00;     // proximal motor lower arm
static const float a3 = 20.00;     // distal motor lower arm
static const float a4 = 15.00;     // distal motor upper arm
static const float a5 = 20.00;     // spacing between motors

static const int power_ctrl_pin = 35;

static const float servo_offset_left = 29;
static const float servo_offset_right = 33;

static const int servo_pin_left = 10;
static const int servo_pin_right = 9;

static const float goal_tol = 1;      // mm
static const float rate_loop = 50;
static const int dt_loop = 20;        // ms
static const float rate_moving = 50;

// global flags to control program behavior
static const bool resetOffsets = false;
static const bool flag_input_from_ros = true;

bool flag_input_updated;
bool flag_action;

// state control
enum {
    Idle,
    Moving,
    Pausing,
    Resetting
} device_state;

int t_pause_start;

// workspace guides
static const float r_max = 10;
float x_center;
float y_center;
float xI;
float yI;

bool badCoords;

// push maginitude and pause
float mag = 0.4;
float pause = 0.2;

const float dmag = 1.0;
const float dpause = 0.2;
const float mag_range[2] = {2.0, 8.0};
const float pause_range[2] = {0.0, 1.0};
const float rot_corr = -1;

// a pantograph device pointer
PantographDevice device(a1, a2, a3, a4, a5, servo_pin_left, servo_pin_right,
                        servo_offset_left, servo_offset_right, power_ctrl_pin);
//PantographDevice* device;

// a publisher
std_msgs::String pub_msg;
ros::Publisher test_pub("test_topic", &pub_msg);

//----------------------------- callback functions ------------------------------
String cmd_dir;
void ctrl_callback(const std_msgs::String& msg) {
    cmd_dir = msg.data;
    flag_input_updated = true;
}

ros::Subscriber<std_msgs::String> sub("haptic_control", &ctrl_callback);

//----------------------------- get input ------------------------------
// block version
char get_input_block() {
    char dir_val;
    
    // input is either from keyboard or ros
    if (flag_input_from_ros) {
        while (!flag_input_updated) {
            nh.spinOnce();
            delay(10);
        }
        
        dir_val = cmd_dir[0];
        flag_input_updated = false;
    }
    else {
        // check for user input
        while (Serial.available() == 0) { }
    
        // use only most recent variable
        while (Serial.available() > 0) {
            dir_val = Serial.read();
        }
    }
    
    if (dir_val == 'w' || dir_val == 's' || dir_val == 'a' || dir_val == 'd') {
        flag_action = false;
    }
    else {
        flag_action = true;
    }
    
    return dir_val;
}

// none block version
char get_input() {
    char dir_val;
    
    if (flag_input_from_ros) {
        if (!flag_input_updated)
            return 'n';

        dir_val = cmd_dir[0];
        flag_input_updated = false;
    }
    else {
        if (!Serial.available())
            return 'n';

        while (Serial.available()) {
            dir_val = Serial.read();
        }
    }

    if (dir_val == 'w' || dir_val == 's' || dir_val == 'a' || dir_val == 'd') {
        flag_action = false;
    }
    else {
        flag_action = true;
    }
    
    return dir_val;
}

//----------------------------- helper functions ------------------------------
void clip(float& x, const float x_min, const float x_max) {
    if (x < x_min)
        x = x_min;
    else if (x > x_max)
        x = x_max;
}

float dist(float x1, float y1, float x2, float y2) {
    float x_diff = x1 - x2;
    float y_diff = y1 - y2;
    
    return sqrt(x_diff * x_diff + y_diff * y_diff);
}

//----------------------------- main setup ------------------------------
void setup() {
    if (flag_input_from_ros) {
        nh.initNode();
        nh.subscribe(sub);
        nh.advertise(test_pub);
    }
    else {
        // use serial monitor
        Serial.begin(38400);
    }
    
    // setup the device
    device.Setup(goal_tol, rate_loop, rate_moving);
    
    device_state = Idle;

    // servo reset can only be adjusted through Serial Monitor
//    if (!flag_input_from_ros && resetOffsets) {
//        servo_base_right_0 = servoOffset(servo_base_right) - 90;
//        servo_base_left_0 = servoOffset(servo_base_left) - 90;
//    }
    
    if (!flag_input_from_ros) {
        Serial.print("servo_base_right Offset: ");
        Serial.print(servo_offset_right);
        Serial.print("   servo_base_left Offset: ");
        Serial.println(servo_offset_left);
    }
    
    device.GetPos(xI, yI);
    x_center = xI;
    y_center = yI;
    
    if (!flag_input_from_ros) {
        Serial.print("x_center: ");
        Serial.print(xI);
        Serial.print("   y_center: ");
        Serial.println(yI);
    }
    
    if (!flag_input_from_ros) {
        Serial.println("Choose a direction:");
        Serial.println("     I: Forward");
        Serial.println("     ,: Back");
        Serial.println("     J: Left");
        Serial.println("     L: Right");
        Serial.println("     K: Reset to center");
    }

    flag_input_updated = false;
}

//----------------------------- main control ------------------------------
//void execute_control() {
//    int delta = 15;
//    
//    // calculate new motor commands
//    if ( (xI - x_center) * (xI - x_center) + (yI - y_center) * (yI - y_center) > r_max) {
//        badCoords = true;
//    }
//    
////    inverseKinematics(xI, yI, newTheta_left, newTheta_right);
//    servo_base_right_pos = newTheta_left;
//    servo_base_left_pos = newTheta_right;
//    
//    // write motor commands (offset by starting position)
//    if ( !badCoords) {
//        coordinatedMovement(servo_base_right, servo_base_left, delta, 
//            servo_base_right_pos + servo_base_right_0, 
//            servo_base_left_pos + servo_base_left_0);
//    }
//    else {
//        if (!flag_input_from_ros) {
//            Serial.println("Bad Coordinates");
//        }
//    }
//    
//    // pause for 0.2s
//    delay(pause*1000);
//    
//    // return to center
//    yI = y_center;
//    xI = x_center;
//    
////    inverseKinematics(xI, yI, newTheta_left, newTheta_right);
//    servo_base_right_pos = newTheta_left;
//    servo_base_left_pos = newTheta_right;
//    
//    coordinatedMovement(servo_base_right, servo_base_left, delta, 
//            servo_base_right_pos + servo_base_right_0, 
//            servo_base_left_pos + servo_base_left_0);
//}

//----------------------------- state machine helpers ------------------------------
void adjust_param(char dir_val) {
    switch (dir_val) {
    case 'w':
        mag += dmag;
        break;
    
    case 's':
        mag -= dmag;
        break;
        
    case 'a':
        pause -= dpause;
        break;
        
    case 'd':
        pause += dpause;
        break;
    default:
        break;
    }
    
    // clip mag and pause
    clip(mag, mag_range[0], mag_range[1]);
    clip(pause, pause_range[0], pause_range[1]);
}

void adjust_goal(char dir_val) {
    const float mag_diag = mag * 0.8;
    
    switch (dir_val) {
    case 'I':                 // Forward
    case 'i':
    case '1':
        xI = xI - mag * rot_corr;
        break;
    
    case ',':                 // Backward
        xI = xI + mag * rot_corr;
        break;
    
    case 'L':                 // Left
    case 'l':
    case '3':
        yI = yI + mag * rot_corr;
        break;
    
    case 'J':               // Right
    case 'j':
    case '4':
        yI = yI - mag * rot_corr;
        break;
    
    case 'u':
    case 'U':
        xI = xI - mag_diag * rot_corr;
        yI = yI - mag_diag * rot_corr;
        break;
        
    case 'o':
    case 'O':
        xI = xI - mag_diag * rot_corr;
        yI = yI + mag_diag * rot_corr;
        break;
        
    case 'm':
    case 'M':
        xI = xI + mag_diag * rot_corr;
        yI = yI - mag_diag * rot_corr;
        break;
        
    case '.':
        xI = xI + mag_diag * rot_corr;
        yI = yI + mag_diag * rot_corr;
        break;
    }
}

//----------------------------- state machine ------------------------------
void state_machine(char dir_val) {
    switch (device_state) {
        case Idle:
            if (dir_val != 'n') {
                if (flag_action) {
                    adjust_goal(dir_val);
                    
                    device.SetGoal(xI, yI);
                    device.ExecuteControl();
                    
                    device_state = Moving;
                }
                else {
                    adjust_param(dir_val);
                }
            }
            break;
        case Moving:
            // execute control
            device.ExecuteControl();
            
            // check goal reached
            if (device.GoalReached()) {
                t_pause_start = millis();
                device_state = Pausing;
            }
            break;
        case Pausing:
            int t_pause = millis() - t_pause_start;
            if (t_pause >= pause * 1000) {
                xI = x_center;
                yI = y_center;
                
                device.SetGoal(xI, yI);
                device.ExecuteControl();
                
                device_state = Resetting;
            }
            break;
        case Resetting:
            // execute control
            device.ExecuteControl();
            
            // check goal reached
            if (device.GoalReached()) {
                device_state = Idle;
            }
            
            break;
    }
}

//----------------------------- main loop ------------------------------
void loop() {
    int t_loop_start = millis();
    
    char directionVal;
    directionVal = get_input();
    
    state_machine(directionVal);
    
    int t_loop = millis() - t_loop_start;
    delay(dt_loop - t_loop);
}

