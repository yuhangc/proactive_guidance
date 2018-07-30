
void coordinatedMovement(Servo s1, Servo s2, Servo s3, Servo s4, int delta, int p1, int p2, int p3, int p4) {

  float p1_0 = s1.read();
  float p2_0 = s2.read();
  float p3_0 = s3.read();
  float p4_0 = s4.read();

  float delta1 = p1 - p1_0;
  float delta2 = p2 - p2_0;
  float delta3 = p3 - p3_0;
  float delta4 = p4 - p4_0;

  int nSteps = max(abs(delta1), max(abs(delta2), max(abs(delta3), abs(delta4))));

  if (nSteps > 0) {
    for ( int i = 0; i <= nSteps; i++) {
      s1.write(p1_0 + i * delta1 / nSteps);
      s2.write(p2_0 + i * delta2 / nSteps );
      s3.write(p3_0 + i * delta3 / nSteps );
      s4.write(p4_0 + i * delta4 / nSteps );

      delay(delta);
    }
  }

}
