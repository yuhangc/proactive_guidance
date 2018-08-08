

// Pantograph.ino

// Yuhang Che
// Adapted from Julie Walker
// Created on July 27, 2018

#include <Servo.h>

#include <ros.h>
#include <std_msgs/String.h>

ros::NodeHandle nh;

// global flags to control program behavior
static const bool resetOffsets = false;
static const bool flag_input_from_ros = true;

bool flag_input_updated;
bool flag_action;

// arm lengths
static const float a1 = 15.00;     // proximal motor upper arm [mm]
static const float a2 = 20.00;     // proximal motor lower arm
static const float a3 = 20.00;     // distal motor lower arm
static const float a4 = 15.00;     // distal motor upper arm
static const float a5 = 20.00;     // spacing between motors

Servo servo_base_right;
Servo servo_base_left;

static const int power_ctrl_pin = 35;

float newTheta_left;
float newTheta_right;
int servo_base_right_pos;
int servo_base_left_pos;

// center positions
int servo_base_right_0; // = 35;
int servo_base_left_0; // = 32;

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

// a publisher
std_msgs::String pub_msg;
ros::Publisher test_pub("test_topic", &pub_msg);

char test_str[] = "this is test";

//----------------------------- callback functions ------------------------------
String cmd_dir;
void ctrl_callback(const std_msgs::String& msg) {
    cmd_dir = msg.data;
    flag_input_updated = true;
}

ros::Subscriber<std_msgs::String> sub("haptic_control", &ctrl_callback);


//----------------------------- get input ------------------------------
char get_input() {
    char dir_val;
    
    // input is either from keyboard or ros
    if (flag_input_from_ros) {
        while (!flag_input_updated) {
            pub_msg.data = test_str;
            test_pub.publish(&pub_msg);
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

//----------------------------- helper functions ------------------------------
void clip(float& x, const float x_min, const float x_max) {
    if (x < x_min)
        x = x_min;
    else if (x > x_max)
        x = x_max;
}

void dist(float x1, float y1, float x2, float y2) {
    float x_diff = x1 - x2;
    float y_diff = y1 - y2;
    
    return sqrt(x_diff * x_diff + y_diff * y_diff);
}

//----------------------------- main setup ------------------------------
void setup() {
    pinMode(power_ctrl_pin, OUTPUT);
    // set power to off first
    digitalWrite(power_ctrl_pin, LOW);
    
    if (flag_input_from_ros) {
        nh.initNode();
        nh.subscribe(sub);
        nh.advertise(test_pub);
    }
    else {
        // use serial monitor
        Serial.begin(38400);
    }
    
    // connect servos
    servo_base_right.attach(9);
    servo_base_left.attach(10);
    
    // servo reset can only be adjusted through Serial Monitor
    if (!flag_input_from_ros && resetOffsets) {
        servo_base_right_0 = servoOffset(servo_base_right) - 90;
        servo_base_left_0 = servoOffset(servo_base_left) - 90;
    }
    else {
        servo_base_right_0 = 33; //15;
        servo_base_left_0 = 29;
    }
    
    if (!flag_input_from_ros) {
        Serial.print("servo_base_right Offset: ");
        Serial.print(servo_base_right_0);
        Serial.print("   servo_base_left Offset: ");
        Serial.println(servo_base_left_0);
    }
    
    x_center = -a5 / 2;
    y_center = 3.0 * sqrt((a1 + a2) * (a1 + a2) - (0.5 * a5) * (0.5 * a5)) / 4.0;
    
    if (!flag_input_from_ros) {
        Serial.print("x_center: ");
        Serial.print(x_center);
        Serial.print("   y_center: ");
        Serial.println(y_center);
    }
    
    xI = x_center;
    yI = y_center;
    
    inverseKinematics(xI, yI, newTheta_left, newTheta_right);
    servo_base_right_pos = newTheta_left;
    servo_base_left_pos = newTheta_right;
    
    // start at center position
    servo_base_right.write(servo_base_right_pos + servo_base_right_0);
    servo_base_left.write(servo_base_left_pos + servo_base_left_0);
    
    // enable servo power after writing the position
    digitalWrite(power_ctrl_pin, HIGH);
    
    //  Serial.println("Starting values: ");
    //  Serial.print("Index dist: ");
    //  Serial.println(servo_base_right.read());
    //  Serial.print("Index prox: ");
    //  Serial.println(servo_base_left.read());
    
    
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
void execute_control() {
    int delta = 15;
    
    // calculate new motor commands
    if ( (xI - x_center) * (xI - x_center) + (yI - y_center) * (yI - y_center) > r_max) {
        badCoords = true;
    }
    
    inverseKinematics(xI, yI, newTheta_left, newTheta_right);
    servo_base_right_pos = newTheta_left;
    servo_base_left_pos = newTheta_right;
    
    // write motor commands (offset by starting position)
    if ( !badCoords) {
        coordinatedMovement(servo_base_right, servo_base_left, delta, 
            servo_base_right_pos + servo_base_right_0, 
            servo_base_left_pos + servo_base_left_0);
    }
    else {
        if (!flag_input_from_ros) {
            Serial.println("Bad Coordinates");
        }
    }
    
    // pause for 0.2s
    delay(pause*1000);
    
    // return to center
    yI = y_center;
    xI = x_center;
    
    inverseKinematics(xI, yI, newTheta_left, newTheta_right);
    servo_base_right_pos = newTheta_left;
    servo_base_left_pos = newTheta_right;
    
    coordinatedMovement(servo_base_right, servo_base_left, delta, 
            servo_base_right_pos + servo_base_right_0, 
            servo_base_left_pos + servo_base_left_0);
}

//----------------------------- main loop ------------------------------
void loop() {
    char directionVal;
    float mag_diag = mag * 0.8;
    float rot_corr = -1.0;
    
    directionVal = get_input();
    
    switch (directionVal) {                     //  apply new command
    
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
    
    case 'K':               // Center
    case 'k':
    case '9':
      yI = y_center;
      xI = x_center;
      break;
      
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
      
    }
    
    // clip mag and pause
    clip(mag, mag_range[0], mag_range[1]);
    clip(pause, pause_range[0], pause_range[1]);
    
    if (flag_action) {
        execute_control();
    }
}

