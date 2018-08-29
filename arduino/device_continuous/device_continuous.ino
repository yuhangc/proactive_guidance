
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

static const float servo_offset_left = 3;
static const float servo_offset_right = -1;

static const int servo_pin_left = SERVO_PIN_A;
static const int servo_pin_right = SERVO_PIN_B;

static const float goal_tol = 1;      // mm
static const float rate_loop = 80;
static const float dt_loop = 1000.0/rate_loop;        // ms
static const float rate_moving = 100;
static const int imu_reading_nskip = 1;

// global flags to control program behavior
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
const float mag_map[] = {3, 4, 6};
const float pause_map[] = {0.1, 0.2, 0.3};

// a pantograph device pointer
PantographDevice device(a1, a2, a3, a4, a5, servo_pin_left, servo_pin_right,
                        servo_offset_left, servo_offset_right, power_ctrl_pin);
                        
// a IMU class
Adafruit_BNO055 bno = Adafruit_BNO055(55);

//----------------------------- get input ------------------------------
bool get_input() {
    if (!Serial.available())
        return false;

    // first n char is dir in degrees, n+1 is ',', n+2 is mag/pause
    char val = Serial.read();
    String num = "";

    while (val != ',') {
        num = num + val;
        val = Serial.read();
    }

    dir = (float) num.toInt();
    dir = dir * 3.1415926 / 180.0;

    val = Serial.read();
    mag = mag_map[val - '0'];
    pause = pause_map[val - '0'];

    // clear the rest of input buffer
    while (Serial.available()) {
        val = Serial.read();
    }

    return true;
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
    Serial.begin(115200);
    
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
    
    // t start
    t_next = millis();
    nskipped = 0;
}

//----------------------------- state machine ------------------------------
void state_machine() {
    float th_left, th_right;
    
    switch (device_state) {
        case Idle:
            if (flag_input_updated) {
                flag_input_updated = false;

                xI = x_center + mag * sin(dir);
                yI = y_center + mag * cos(dir);
                    
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
        if (get_input()) {
            flag_input_updated = true;
        }
    
        state_machine();
    
        // update sensor reading every n loops
        /* Get a new sensor event */
        if (nskipped >= imu_reading_nskip) {
            sensors_event_t event;
            bno.getEvent(&event);

            if (flag_print_debug) {
                Serial.print(F("Orientation: "));
            }
            Serial.print((float)event.orientation.x);
            Serial.print(F(", "));
            Serial.print((float)event.orientation.y);
            Serial.print(F(", "));
            Serial.println((float)event.orientation.z);

            nskipped = 0;
        } else {
            nskipped += 1;
        }
        
        t_next += dt_loop;
    }
}

