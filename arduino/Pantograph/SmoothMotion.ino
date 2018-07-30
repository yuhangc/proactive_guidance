
void coordinatedMovement(Servo s1, Servo s2, int delta, int p1, int p2) {

  float p1_0 = s1.read();
  float p2_0 = s2.read();

  float delta1 = p1 - p1_0;
  float delta2 = p2 - p2_0;

  int nSteps = max(abs(delta1), abs(delta2));

  if (nSteps > 0) {
    for ( int i = 0; i <= nSteps; i++) {
      s1.write(p1_0 + i * delta1 / nSteps);
      s2.write(p2_0 + i * delta2 / nSteps );

      delay(delta);
    }
  }

}
