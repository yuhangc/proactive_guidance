
// STARTUP OFFSET ADJUSTMENT ///////////////////////////////////////////////////////////////////
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

