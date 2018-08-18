#include "Arduino.h"
#include "PantographDevice.h"

PantographDevice::PantographDevice(float a1, float a2, float a3, float a4, float a5, 
                     int s_left, int s_right, float offset_left, float offset_right, int power_pin): 
                     a1(a1), a2(a2), a3(a3), a4(a4), a5(a5), pin_left(s_left), pin_right(s_right),
					 th_offset_left(offset_left), th_offset_right(offset_right), power_pin(power_pin) {
	// setup power pin
    pinMode(power_pin, OUTPUT);    
	digitalWrite(power_pin, LOW);
};

void PantographDevice::Setup(float tol, float rate, float rate_moving) {
	goal_reaching_tol = tol;
	
	// rate in Hz, rate_moving in rad(deg/tick?)/s
    loop_dt = 1000.0 / rate;
    dth = rate_moving * loop_dt / 1000.0;

	// should not attach in constructor
	servo_base_left.attach(pin_left);
    servo_base_right.attach(pin_right);

	// compute and move to center
    x_goal = -a5 / 2;
    y_goal = 3.0 * sqrt((a1 + a2) * (a1 + a2) - (0.5 * a5) * (0.5 * a5)) / 4.0 + 1.0;
	x = x_goal;
	y = y_goal;

    InverseKinematics(x_goal, y_goal, th_goal_left, th_goal_right);
    
    servo_base_left.write(th_goal_left);
    servo_base_right.write(th_goal_right);

    // enable the motor power
    digitalWrite(power_pin, HIGH);

    // give a little time for it to stabilize
    delay(200);

	// set power back off
	digitalWrite(power_pin, LOW);
}

void PantographDevice::SetGoal(float x_goal_new, float y_goal_new) {
    x_goal = x_goal_new;
    y_goal = y_goal_new;
    
    // perform inverse kinmatics
    InverseKinematics(x_goal, y_goal, th_goal_left, th_goal_right);
    
    // reset flags
    flag_goal_reached = false;
    flag_new_goal_set = true;
}

bool PantographDevice::InverseKinematics(float xd, float yd, float& new_th_left, float& new_th_right) {
    // Distances to x,y from each motor hub
    float  D1 = sqrt(xd * xd + yd * yd);
    float D2 = sqrt((xd + a5) * (xd + a5) + yd * yd);

    if ( D1 > (a1 + a2) || D2 > (a3 + a4)) {
        return false;
    }

    // intermediate terms
    float alpha1 = acos( (a2 * a2 - a1 * a1 - D1 * D1) / (-2 * a1 * D1) );
    float beta1 = atan2(yd, -xd);
    float beta5 = acos( (a3 * a3 - a4 * a4 - D2 * D2) / (-2 * a4 * D2) );
    float alpha5 = atan2(yd, xd + a5);

    // solved motor angles
    new_th_left = (PI - alpha1 - beta1) * 180 / PI + th_offset_left;
    new_th_right = (alpha5 +  beta5) * 180 / PI + th_offset_right;

	return true;
}

void PantographDevice::ExecuteControl() {
    // do nothing if goal already reached
    if (flag_goal_reached)
        return;
    
    // update position first
    th_left = servo_base_left.read();
    th_right = servo_base_right.read();
    
    // compute the max error
    float th_err_left = th_left - th_goal_left;
    float th_err_right = th_right - th_goal_right;
    float th_err = max(abs(th_err_left), abs(th_err_right));
    
    // check first if goal is reached
    if (th_err < goal_reaching_tol) {
        flag_goal_reached = true;
        return;
    }
    
    // attempt to move to goal
	float th_cmd_left = th_left;
	float th_cmd_right = th_right;
    
    if (abs(th_err_left) < dth) {
        th_cmd_left = th_goal_left;
    }
    else {
        if (th_goal_left > th_left) {
            th_cmd_left += dth;
        }
        else {
            th_cmd_left -= dth;
        }
    }
    
    if (abs(th_err_right) < dth) {
        th_cmd_right = th_goal_right;
    }
    else {
        if (th_goal_right > th_right) {
            th_cmd_right += dth;
        }
        else {
            th_cmd_right -= dth;
        }
    }

/*
	Serial.print("Commands are: ");
	Serial.print(th_cmd_left);
	Serial.print(", ");
	Serial.println(th_cmd_right);*/
    
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
