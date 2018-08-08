class PantographDevice {
public:
    PantographDevice(float a1, float a2, float a3, float a4, float a5, 
                     int s_left, int s_right, int power_pin): 
                     a1(a1), a2(a2), a3(a3), a4(a4), a5(a5) {
                         
        servo_base_right.attach(s_left);
        servo_base_right.attach(s_right);
        
        // setup power pin
        pinMode(power_ctrl_pin, OUTPUT);
        digitalWrite(power_ctrl_pin, LOW);
        
        // compute and move to center
        x_goal = -a5 / 2;
        y_goal = 3.0 * sqrt((a1 + a2) * (a1 + a2) - (0.5 * a5) * (0.5 * a5)) / 4.0;
        
        MoveToGoal(x_goal, y_goal);
        
    };
    
    void SetGoalReachingTol(float tol) {
        goal_reaching_tol = tol;
    }
    
    void SetLoopRate(float rate, float rate_moving) {
        // rate in Hz, rate_moving in rad(deg/tick?)/s
        loop_dt = 1000.0 / rate;
        dth = rate_moving * loop_dt / 1000.0;
    }
    
    void SetGoal(float x_goal_new, float y_goal_new);
    
    void ExecuteControl();        // for one time step
    void MoveToGoal(float x_goal_new, float y_goal_new);    // blocking until goal is reached
    
private:
    // link lengths
    float a1;
    float a2;
    float a3;
    float a4;
    float a5;
    
    // servo motor
    Servo servo_base_right;
    Servo servo_base_left;
    
    // workspace guides
    float x;
    float y;
    float x_goal;
    float y_goal;
    
    // joint space
    float th_left;
    float th_right;
    float th_goal_left;
    float th_goal_right;
    
    // tolerances
    float goal_reaching_tol;
    
    // loop control
    float loop_dt;
    float dth;
    
    // control flags
    bool flag_new_goal_set;
    bool flag_goal_reached;
    
    // functions
    // void forwardKinematics(float& x_new, float& y_new);
    void inverseKinematics(float xd, float yd, float& new_th_left, float& new_th_right);
}

void PantographgDevice::SetGoal(float x_goal_new, float y_goal_new) {
    x_goal = x_goal_new;
    y_goal = y_goal_new;
    
    // perform inverse kinmatics
    inverseKinematics(x_goal, y_goal, th_left_goal, th_right_goal);
    
    // reset flags
    flag_goal_reached = false;
    flag_new_goal_set = true;
}

void PantographDevice::inverseKinematics(float xd, float yd, float& new_th_left, float& new_th_right) {
    // Distances to x,y from each motor hub
    float  D1 = sqrt(xd * xd + yd * yd);
    float D2 = sqrt((xd + a5) * (xd + a5) + yd * yd);

    if ( D1 > (a1 + a2) || D2 > (a3 + a4)) {
        badCoords = true;
    }
    else {
        badCoords = false;
    }

    // intermediate terms
    float alpha1 = acos( (a2 * a2 - a1 * a1 - D1 * D1) / (-2 * a1 * D1) );
    float beta1 = atan2(yd, -xd);
    float beta5 = acos( (a3 * a3 - a4 * a4 - D2 * D2) / (-2 * a4 * D2) );
    float alpha5 = atan2(yd, xd + a5);

    // solved motor angles
    new_th_left = (PI - alpha1 - beta1) * 180 / PI;
    new_th_right = (alpha5 +  beta5) * 180 / PI;
}

void PantographDevice::ExecuteControl() {
    // do nothing if goal already reached
    if (flag_goal_reached)
        return;
    
    // update position first
    th_left = servo_base_left.read();
    th_right = servo_base_right.read();
    
    // compute the max error
    float th_err_left = th_left - th_left_goal;
    float th_err_right = th_right - th_right_goal;
    float th_err = max(abs(th_err_left), abs(th_err_right));
    
    // check first if goal is reached
    if (th_err < goal_reaching_tol) {
        flag_goal_reached = true;
        return;
    }
    
    // attempt to move to goal
    float th_cmd_left, th_cmd_right;
    
    if (abs(th_err_left) < dth) {
        th_cmd_left = th_left_goal;
    }
    else {
        if (th_left_goal > th_left) {
            th_cmd_left += dth;
        }
        else {
            th_cmd_left -= dth;
        }
    }
    
    if (abs(th_err_right) < dth) {
        th_cmd_right = th_left_goal;
    }
    else {
        if (th_right_goal > th_right) {
            th_cmd_right += dth;
        }
        else {
            th_cmd_right -= dth;
        }
    }
    
    // write output
    servo_base_left.write(th_cmd_left);
    servo_base_right.write(th_cmd_right);
}

void PantographDevice::MoveToGoal(float x_goal_new, float y_goal_new) {
    SetGoal(x_goal_new, y_goal_new);
    
    while (!flag_goal_reached) {
        ExecuteControl();
        delay(loop_dt);
    }
}
