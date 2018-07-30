

// Pantograph.ino

// Yuhang Che
// Adapted from Julie Walker
// Created on July 27, 2018

#include <Servo.h>

bool resetOffsets = false;

// arm lengths
float a1 = 15.00;     // proximal motor upper arm [mm]
float a2 = 20.00;     // proximal motor lower arm
float a3 = 20.00;     // distal motor lower arm
float a4 = 15.00;     // distal motor upper arm
float a5 = 20.00;     // spacing between motors

Servo thumbDist;
Servo thumbProx;
Servo servo_base_right;
Servo servo_base_left;


float newTheta_left;
float newTheta_right;
int thumbDist_pos;
int thumbProx_pos;
int servo_base_right_pos;
int servo_base_left_pos;

// center positions
//float x0 = -10.00;  // [mm]
//float y0 = 28.00;   // [mm]
int thumbDist_0; // = 31;
int thumbProx_0; // = 27;
int servo_base_right_0; // = 35;
int servo_base_left_0; // = 32;

// workspace guides
float r_max = 10;
float x_center;
float y_center;
float xI;
float xT;
float yI;
float yT;

bool badCoords;


// MAIN SETUP ///////////////////////////////////////////////////////////////////
void setup() {
  //Serial communication
  Serial.begin(38400);

  // connect servos
  thumbDist.attach(5);
  thumbProx.attach(6);
  servo_base_right.attach(9);
  servo_base_left.attach(10);

  if (resetOffsets) {
    servo_base_right_0 = servoOffset(servo_base_right) - 90;
    servo_base_left_0 = servoOffset(servo_base_left) - 90;
    thumbDist_0 = servoOffset(thumbDist) - 90;
    thumbProx_0 = servoOffset(thumbProx) - 90;
  }
  else {
    servo_base_right_0 = 33; //15;
    servo_base_left_0 = 29;
    thumbDist_0 = -26;
    thumbProx_0 = 35;
  }
  Serial.print("servo_base_right Offset: ");
  Serial.print(servo_base_right_0);
  Serial.print("   servo_base_left Offset: ");
  Serial.println(servo_base_left_0);
  Serial.print("thumbDist Offset: ");
  Serial.print(thumbDist_0);
  Serial.print("   thumbProx Offset: ");
  Serial.println(thumbProx_0);

  x_center = -a5 / 2;
  y_center = 3.0 * sqrt((a1 + a2) * (a1 + a2) - (0.5 * a5) * (0.5 * a5)) / 4.0;

  Serial.print("x_center: ");
  Serial.print(x_center);
  Serial.print("   y_center: ");
  Serial.println(y_center);

  xI = x_center;
  xT = x_center;
  yI = y_center;
  yT = y_center;

  inverseKinematics(xT, yT);
  thumbDist_pos = newTheta_right;
  thumbProx_pos = newTheta_left;
  inverseKinematics(xI, yI);
  servo_base_right_pos = newTheta_left;
  servo_base_left_pos = newTheta_right;

  // start at center position
  thumbDist.write(thumbDist_pos + thumbDist_0);
  thumbProx.write(thumbProx_pos + thumbProx_0);
  servo_base_right.write(servo_base_right_pos + servo_base_right_0);
  servo_base_left.write(servo_base_left_pos + servo_base_left_0);

  //  Serial.println("Starting values: ");
  //  Serial.print("Thumb dist: ");
  //  Serial.println(thumbDist.read());
  //  Serial.print("Thumb prox: ");
  //  Serial.println(thumbProx.read());
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
      yT = yT - push;  // [mm]
      break;

    case 'M':                 // Down
    case 'm':
    case '2':
      yI = yI + push;
      yT = yT + push;
      break;

    case 'L':                 // Back
    case 'l':
    case '3':
      xI = xI + push;
      xT = xT - push;
      break;

    case 'J':               // Forward
    case 'j':
    case '4':
      xI = xI - push;
      xT = xT + push;
      break;

    case 'H':               // Twist Left
    case 'h':
    case '5':
      xI = xI + push;
      xT = xT + push;
      break;

    case ';':               // Twist Right
    case '6':
      xI = xI - push;
      xT = xT - push;
      break;

    case 'U':               // Tilt Left
    case 'u':
    case '7':
      yI = yI - push;
      yT = yT + push;
      break;

    case 'O':             // Tilt Right
    case 'o':
    case '8':
      yI = yI + push;
      yT = yT - push;
      break;


    case 'K':               // Center
    case 'k':
    case '9':
      yI = y_center;
      yT = y_center;
      xI = x_center;
      xT = x_center;

      break;

    case 'Y':
    case 'y':
    case '0':
      float r = 1.5;
      for (float t = PI / 4; t < PI / 2; t += PI / 64) {
        xI = x_center + r * cos(t);
        yI = (y_center - r) + r * sin(t);
        xT = x_center - r * cos(t);
        yT = (y_center - r) - r * sin(t);
        
        if ( (xI - x_center) * (xI - x_center) + (yI - y_center) * (yI - y_center) > r_max || (xT - x_center) * (xT - x_center) + (yT - y_center) * (yT - y_center) > r_max) {
          badCoords = true;
        }

        inverseKinematics(xT, yT);
        thumbDist_pos = newTheta_right;
        thumbProx_pos = newTheta_left;
        inverseKinematics(xI, yI);
        servo_base_right_pos = newTheta_left;
        servo_base_left_pos = newTheta_right;
        if ( !badCoords) {
          coordinatedMovement(thumbDist, thumbProx, servo_base_right, servo_base_left, delta, thumbDist_pos + thumbDist_0, thumbProx_pos + thumbProx_0, servo_base_right_pos + servo_base_right_0, servo_base_left_pos + servo_base_left_0);
        }
        delay(2);
      }


  }

  // calculate new motor commands
  if ( (xI - x_center) * (xI - x_center) + (yI - y_center) * (yI - y_center) > r_max || (xT - x_center) * (xT - x_center) + (yT - y_center) * (yT - y_center) > r_max) {
    badCoords = true;
  }

  inverseKinematics(xT, yT);
  thumbDist_pos = newTheta_right;
  thumbProx_pos = newTheta_left;
  inverseKinematics(xI, yI);
  servo_base_right_pos = newTheta_left;
  servo_base_left_pos = newTheta_right;

  // write motor commands (offset by starting position)
  if ( !badCoords) {
    //        thumbDist.write(thumbDist_pos + thumbDist_0);
    //        thumbProx.write(thumbProx_pos + thumbProx_0);
    //        servo_base_right.write(servo_base_right_pos + servo_base_right_0);
    //        servo_base_left.write(servo_base_left_pos + servo_base_left_0);
    //

    coordinatedMovement(thumbDist, thumbProx, servo_base_right, servo_base_left, delta, thumbDist_pos + thumbDist_0, thumbProx_pos + thumbProx_0, servo_base_right_pos + servo_base_right_0, servo_base_left_pos + servo_base_left_0);

    //    Serial.print("Thumb dist: ");
    //    Serial.println(thumbDist.read());
    //    Serial.print("Thumb prox: ");
    //    Serial.println(thumbProx.read());
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

