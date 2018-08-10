#include "Arduino.h"
#include "Servo.h"

class PantographDevice {
public:
    PantographDevice(float a1, float a2, float a3, float a4, float a5, 
                     int s_left, int s_right, float offset_left, float offset_right, int power_pin);

    // set methods
	void Setup(float tol, float rate, float rate_moving);
    
    void SetGoal(float x_goal_new, float y_goal_new);

    // control
    void ExecuteControl();        // for one time step
    void MoveToGoal(float x_goal_new, float y_goal_new);    // blocking until goal is reached

    // get methods
    bool GoalReached() {
        return flag_goal_reached;
    }

    void GetPos(float& x_out, float& y_out) {
        x_out = x;
        y_out = y;
    }

	void GetPosJoint(float& left, float& right) {
		left = th_left;
		right = th_right;
	}

	void __DirectControl(int left, int right) {
		servo_base_left.write(left);
		servo_base_right.write(right);
	}

	void __DirectRead(int& left, int& right) {
		left = servo_base_left.read();
		right = servo_base_right.read();
	}
    
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

	// power pin
	int power_pin;
	int pin_left;
	int pin_right;
    
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
    float th_offset_left;
    float th_offset_right;
    
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
    bool InverseKinematics(float xd, float yd, float& new_th_left, float& new_th_right);
};

