
// INVERSE KINEMATICS ///////////////////////////////////////////////////////////////////
void inverseKinematics(float x, float y, float& new_th_left, float& new_th_right) {
    if (!flag_input_from_ros) {
        Serial.print("x: ");
        Serial.print(x);
        Serial.print("    y: ");
        Serial.println(y);
    }

  // Distances to x,y from each motor hub
  float  D1 = sqrt(x * x + y * y);
  float D2 = sqrt((x + a5) * (x + a5) + y * y);

  if ( D1 > (a1 + a2) || D2 > (a3 + a4)) {
    badCoords = true;
  }
  else {
    badCoords = false;
  }

  // intermediate terms
  float alpha1 = acos( (a2 * a2 - a1 * a1 - D1 * D1) / (-2 * a1 * D1) );
  float beta1 = atan2(y, -x);
  float beta5 = acos( (a3 * a3 - a4 * a4 - D2 * D2) / (-2 * a4 * D2) );
  float alpha5 = atan2(y, x + a5);

  // solved motor angles
  new_th_left = (PI - alpha1 - beta1) * 180 / PI;
  new_th_right = (alpha5 +  beta5) * 180 / PI;

}

