
// Pantograph.ino

// Yuhang Che
// Adapted from Julie Walker
// Created on July 27, 2018

#include <PWMServo.h>

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>

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

//static const float servo_offset_left = 29;
//static const float servo_offset_right = 33;
static const float servo_offset_left = 3;
static const float servo_offset_right = -1;

//static const int servo_pin_left = 10;
//static const int servo_pin_right = 9;

static const int servo_pin_left = SERVO_PIN_A;
static const int servo_pin_right = SERVO_PIN_B;

static const float goal_tol = 1;      // mm
static const float rate_loop = 80;
static const float dt_loop = 1000.0/rate_loop;        // ms
static const float rate_moving = 100;
static const int imu_reading_nskip = 1;

// global flags to control program behavior
static const bool flag_using_ros = false;
static const bool flag_print_debug = false;

bool flag_input_updated;
bool flag_action;

// state control
enum {
    Idle,
    Starting,
    Moving,
    Pausing,
    Resetting
} device_state;

unsigned long t_pause_start;
unsigned long t_start;
unsigned long t_last;

float t_next;
int nskipped;

// workspace guides
static const float r_max = 10;
float x_center;
float y_center;
float xI;
float yI;

bool badCoords;

// push maginitude and pause
float dir = 0;
float mag = 4.0;
float pause = 0.2;

const float rot_corr = -1;
const char dir_char_map[] = {'i', 'o', 'l', '.', ',', 'm', 'j', 'u'};
const float mag_map[] = {2, 4, 5};
const float pause_map[] = {0.1, 0.2, 0.3};

// a pantograph device pointer
PantographDevice device(a1, a2, a3, a4, a5, servo_pin_left, servo_pin_right,
                        servo_offset_left, servo_offset_right, power_ctrl_pin);
                        
// a IMU class
Adafruit_BNO055 bno = Adafruit_BNO055(55);

// a publisher
std_msgs::Float32MultiArray rot_msg;
ros::Publisher rot_pub("human_rotation", &rot_msg);
float rot_val[3];

std_msgs::String ctrl_received_msg;
ros::Publisher ctrl_received_pub("ctrl_received", &ctrl_received_msg);

//----------------------------- callback functions ------------------------------
void ctrl_callback(const std_msgs::Float32MultiArray& msg) {
    dir = msg.data[0];
    mag = msg.data[1];
    pause = msg.data[2];
    flag_input_updated = true;
    
    static int count = 0;
    ++count;
    String msg_data = "received" + String(count);
    ctrl_received_msg.data = msg_data.c_str();
    ctrl_received_pub.publish(&ctrl_received_msg);
}

ros::Subscriber<std_msgs::Float32MultiArray> sub("haptic_control", &ctrl_callback);

//----------------------------- get input ------------------------------
char get_input() {
    char dir_val;
    
    if (flag_using_ros) {
        if (!flag_input_updated)
            return 'n';

        dir_val = dir_char_map[(int)dir];
        flag_input_updated = false;
    }
    else {
        if (!Serial.available())
            return 'n';

        // first char is dir, second is mag, third is pause
        char val = Serial.read();
        dir_val = dir_char_map[val - '0'];
        
        val = Serial.read();
        mag = mag_map[val - '0'];
        
        val = Serial.read();
        pause = pause_map[val - '0'];
        
        // clear the rest of input buffer
        while (Serial.available()) {
            val = Serial.read();
        }
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
    if (flag_using_ros) {
        nh.initNode();
        nh.subscribe(sub);
        nh.advertise(rot_pub);
//        nh.advertise(ctrl_received_pub);
    }
    else {
        // use serial monitor
        Serial.begin(115200);
    }
    
    // setup the device
    device.Setup(goal_tol, rate_loop, rate_moving);
    
    device_state = Idle;
    
    if (flag_print_debug) {
        Serial.print("servo_base_right Offset: ");
        Serial.print(servo_offset_right);
        Serial.print("   servo_base_left Offset: ");
        Serial.println(servo_offset_left);
    }
    
    device.GetPos(xI, yI);
    x_center = xI;
    y_center = yI;
    
    if (flag_print_debug) {
        Serial.print("x_center: ");
        Serial.print(xI);
        Serial.print("   y_center: ");
        Serial.println(yI);
    }
    
    if (flag_print_debug) {
        Serial.println("Choose a direction:");
        Serial.println("     I: Forward");
        Serial.println("     ,: Back");
        Serial.println("     J: Left");
        Serial.println("     L: Right");
        Serial.println("     K: Reset to center");
    }
    
    if (flag_print_debug) {
        Serial.println("In state idle");
    }

    flag_input_updated = false;
    
    // setup the IMU
    if (!bno.begin()) {
        if (flag_print_debug) {
            Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
        }
        while(1);
    }
    
    /* Use external crystal for better accuracy */
    bno.setExtCrystalUse(true);
   
    /* Display some basic information on this sensor */
    if (flag_print_debug) {
        sensor_t sensor;
        bno.getSensor(&sensor);
        Serial.println("------------------------------------");
        Serial.print  ("Sensor:       "); Serial.println(sensor.name);
        Serial.print  ("Driver Ver:   "); Serial.println(sensor.version);
        Serial.print  ("Unique ID:    "); Serial.println(sensor.sensor_id);
        Serial.print  ("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" xxx");
        Serial.print  ("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" xxx");
        Serial.print  ("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" xxx");
        Serial.println("------------------------------------");
        Serial.println("");
        delay(500);
    }
    
    // ros message type
    rot_msg.data_length = 3;
    rot_msg.data = rot_val;
    
    // t start
    t_next = millis();
    nskipped = 0;
}

//----------------------------- state machine helpers ------------------------------
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
    
    case 'L':                 // Right
    case 'l':
    case '3':
        yI = yI - mag * rot_corr;
        break;
    
    case 'J':               // Left
    case 'j':
    case '4':
        yI = yI + mag * rot_corr;
        break;
    
    case 'u':
    case 'U':
        xI = xI - mag_diag * rot_corr;
        yI = yI + mag_diag * rot_corr;
        break;
        
    case 'o':
    case 'O':
        xI = xI - mag_diag * rot_corr;
        yI = yI - mag_diag * rot_corr;
        break;
        
    case 'm':
    case 'M':
        xI = xI + mag_diag * rot_corr;
        yI = yI + mag_diag * rot_corr;
        break;
        
    case '.':
        xI = xI + mag_diag * rot_corr;
        yI = yI - mag_diag * rot_corr;
        break;
    }
}

//----------------------------- state machine ------------------------------
void state_machine(char dir_val) {
    float th_left, th_right;
    
    switch (device_state) {
        case Idle:
            if (dir_val != 'n') {
                adjust_goal(dir_val);
                    
                device.SetGoal(xI, yI);
                device.SetOn();
                    
                device_state = Starting;
                t_start = millis();
                    
                if (flag_print_debug) {
                    Serial.print("Goal is: ");
                    Serial.print(xI);
                    Serial.print(", ");
                    Serial.println(yI);
                        
                    Serial.println("Switching to state moving");
                }
            }
            break;
        case Starting:
            if (millis() - t_start >= 160) {
                device_state = Moving;
            }
            break;
        case Moving:
            // execute control
            if (flag_print_debug) {
                Serial.println("in state moving");
            }
            
            device.ExecuteControl();
            
            // check goal reached
            if (device.GoalReached()) {
                t_pause_start = millis();
                device_state = Pausing;
                
                if (flag_print_debug) {
                    Serial.println("Switching to state pausing");
                }
            }
            break;
        case Pausing:
            if (millis() - t_pause_start >= pause * 1000) {
                xI = x_center;
                yI = y_center;
                
                device.SetGoal(xI, yI);
                device.ExecuteControl();
                
                device_state = Resetting;
                
                if (flag_print_debug) {
                    Serial.print("Goal is: ");
                        Serial.print(xI);
                        Serial.print(", ");
                        Serial.println(yI);
                        
                    Serial.println("Switching to state resetting");
                }
            }
            break;
        case Resetting:
            // execute control
            device.ExecuteControl();
            
            // check goal reached
            if (device.GoalReached()) {
                device.SetOff();
                device_state = Idle;
                
                if (flag_print_debug) {
                    Serial.println("Goal reached, now going to idel");
                }
            }
            
            break;
    }
}

//----------------------------- main loop ------------------------------
void loop() {
    unsigned long t_curr = millis();
    
    if (t_curr >= t_next) {
        char directionVal;
        directionVal = get_input();
    
        state_machine(directionVal);
    
        // update sensor reading every n loops
        /* Get a new sensor event */
        if (nskipped >= imu_reading_nskip) {
            sensors_event_t event;
            bno.getEvent(&event);

            if (flag_using_ros) {
                rot_msg.data[0] = (float)event.orientation.x;
                rot_msg.data[1] = (float)event.orientation.y;
                rot_msg.data[2] = (float)event.orientation.z;

                rot_pub.publish(&rot_msg);
            }
            else {
                if (flag_print_debug) {
                    Serial.print(F("Orientation: "));
                }
                Serial.print((float)event.orientation.x);
                Serial.print(F(", "));
                Serial.print((float)event.orientation.y);
                Serial.print(F(", "));
                Serial.println((float)event.orientation.z);
            }

            nskipped = 0;
        } else {
            nskipped += 1;
        }

        // spin ros
        if (flag_using_ros) {
            nh.spinOnce();
        }
        
        t_next += dt_loop;
    }
}

