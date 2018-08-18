#include <Servo.h>

static const int servo_pin_left = 12;
static const int servo_pin_right = 11;
static const int power_pin = 35;

Servo servo_left;
Servo servo_right;

void setup() {
    // use serial monitor
    Serial.begin(57600);
    
    servo_left.attach(servo_pin_left);
    servo_right.attach(servo_pin_right);
    
    pinMode(power_pin, OUTPUT);
    digitalWrite(power_pin, HIGH);
    
    servoOffset(servo_left);
    servoOffset(servo_right);
    
    digitalWrite(power_pin, LOW);
}

// calibrate the offset of servos
int servoOffset(Servo s) {
  Serial.println("Press H,J,L,; to move servo arm to straight down. Press A when done");

  bool setPointFound = false;
  char userInput = 'v';
  int pos = s.read();

  while (!setPointFound) {
    while (Serial.available() == 0) { }

    // use only most recent variable
    while (Serial.available() > 0) {
      userInput = Serial.read();
    }

    switch (userInput) {                     // then apply new command
      case 'J':
      case 'j':
        pos = pos - 1;  // [mm]
        break;
      case 'H':
      case 'h':
        pos = pos - 10;
        break;

      case 'l':
      case 'L':
        pos = pos + 1;
        break;

      case ';':
        pos = pos + 10;
        break;

      case 'A':
      case 'a':
        setPointFound = true;
        return (pos);
        break;
    }
    Serial.println(pos);
    s.write(pos);              // tell servo to go to position in variable 'pos'
  }
}

void loop() {
    
}
