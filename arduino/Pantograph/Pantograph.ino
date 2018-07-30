

// Pantograph.ino

// Yuhang Che
// Adapted from Julie Walker
// Created on July 27, 2018

#include <Servo.h>

#include <ros.h>
#include <std_msgs/String.h>

ros::NodeHandle nh;

bool resetOffsets = false;

// arm lengths
float a1 = 15.00;     // proximal motor upper arm [mm]
float a2 = 20.00;     // proximal motor lower arm
float a3 = 20.00;     // distal motor lower arm
float a4 = 15.00;     // distal motor upper arm
float a5 = 20.00;     // spacing between motors

Servo servo_base_right;
Servo servo_base_left;

float newTheta_left;
float newTheta_right;
int servo_base_right_pos;
int servo_base_left_pos;

// center positions
//float x0 = -10.00;  // [mm]
//float y0 = 28.00;   // [mm]
int servo_base_right_0; // = 35;
int servo_base_left_0; // = 32;

// workspace guides
float r_max = 10;
float x_center;
float y_center;
float xI;
float yI;

bool badCoords;


void ctrl_callback(const std_msgs::String& msg) {
    
}

// MAIN SETUP ///////////////////////////////////////////////////////////////////
void setup() {
    //Serial communication
    Serial.begin(38400);
    
    // connect servos
    servo_base_right.attach(9);
    servo_base_left.attach(10);
    
    if (resetOffsets) {
    servo_base_right_0 = servoOffset(servo_base_right) - 90;
    servo_base_left_0 = servoOffset(servo_base_left) - 90;
    }
    else {
    servo_base_right_0 = 33; //15;
    servo_base_left_0 = 29;
    }
    Serial.print("servo_base_right Offset: ");
    Serial.print(servo_base_right_0);
    Serial.print("   servo_base_left Offset: ");
    Serial.println(servo_base_left_0);
    
    x_center = -a5 / 2;
    y_center = 3.0 * sqrt((a1 + a2) * (a1 + a2) - (0.5 * a5) * (0.5 * a5)) / 4.0;
    
    Serial.print("x_center: ");
    Serial.print(x_center);
    Serial.print("   y_center: ");
    Serial.println(y_center);
    
    xI = x_center;
    yI = y_center;
    
    inverseKinematics(xI, yI);
    servo_base_right_pos = newTheta_left;
    servo_base_left_pos = newTheta_right;
    
    // start at center position
    servo_base_right.write(servo_base_right_pos + servo_base_right_0);
    servo_base_left.write(servo_base_left_pos + servo_base_left_0);
    
    //  Serial.println("Starting values: ");
    //  Serial.print("Index dist: ");
    //  Serial.println(servo_base_right.read());
    //  Serial.print("Index prox: ");
    //  Serial.println(servo_base_left.read());
    
    Serial.println("Choose a direction:");
    Serial.println("     I: Forward");
    Serial.println("     M: Back");
    Serial.println("     J: Clockwise");
    Serial.println("     L: Counterclockwise");
    Serial.println("     K: Reset to center");

}


// MAIN LOOP /////////////////////////////////////////////////////////////////
void loop() {
    int delta = 15;
    char directionVal;
    int push = 4;
    
    // check for user input
    while (Serial.available() == 0) { }
    
    // use only most recent variable
    while (Serial.available() > 0) {
        directionVal = Serial.read();
    }
    
    switch (directionVal) {                     //  apply new command
    
    case 'I':                 // Up
    case 'i':
    case '1':
      yI = yI - push;  // [mm]
      break;
    
    case 'M':                 // Down
    case 'm':
    case '2':
      yI = yI + push;
      break;
    
    case 'L':                 // Back
    case 'l':
    case '3':
      xI = xI + push;
      break;
    
    case 'J':               // Forward
    case 'j':
    case '4':
      xI = xI - push;
      break;
    
    case 'H':               // Twist Left
    case 'h':
    case '5':
      xI = xI + push;
      break;
    
    case ';':               // Twist Right
    case '6':
      xI = xI - push;
      break;
    
    case 'U':               // Tilt Left
    case 'u':
    case '7':
      yI = yI - push;
      break;
    
    case 'O':             // Tilt Right
    case 'o':
    case '8':
      yI = yI + push;
      break;
    
    
    case 'K':               // Center
    case 'k':
    case '9':
      yI = y_center;
      xI = x_center;
      break;
    
    case 'Y':
    case 'y':
    case '0':
      float r = 1.5;
      for (float t = PI / 4; t < PI / 2; t += PI / 64) {
        xI = x_center + r * cos(t);
        yI = (y_center - r) + r * sin(t);
        
        if ( (xI - x_center) * (xI - x_center) + (yI - y_center) * (yI - y_center) > r_max) {
          badCoords = true;
        }
    
        inverseKinematics(xI, yI);
        servo_base_right_pos = newTheta_left;
        servo_base_left_pos = newTheta_right;
        if ( !badCoords) {
          coordinatedMovement(servo_base_right, servo_base_left, delta, 
                servo_base_right_pos + servo_base_right_0, 
                servo_base_left_pos + servo_base_left_0);
        }
        delay(2);
      }
    
    
    }
    
    // calculate new motor commands
    if ( (xI - x_center) * (xI - x_center) + (yI - y_center) * (yI - y_center) > r_max) {
        badCoords = true;
    }
    
    inverseKinematics(xI, yI);
    servo_base_right_pos = newTheta_left;
    servo_base_left_pos = newTheta_right;
    
    // write motor commands (offset by starting position)
    if ( !badCoords) {
        coordinatedMovement(servo_base_right, servo_base_left, delta, 
            servo_base_right_pos + servo_base_right_0, 
            servo_base_left_pos + servo_base_left_0);

    //    Serial.print("Index dist: ");
    //    Serial.println(servo_base_right.read());
    //    Serial.print("Index prox: ");
    //    Serial.println(servo_base_left.read());
    //    delay(20);
    }
    else {
        Serial.println("Bad Coordinates");
    }
}

