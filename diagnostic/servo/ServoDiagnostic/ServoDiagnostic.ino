#include <Servo.h>

Servo motor;

// Default servo pin
int servoPin = 9;

// Speed (percentage 0–100)
int speedPercent = 100;

void setup() {
  Serial.begin(9600);
  Serial.println("=== CONTINUOUS ROTATION SERVO CONTROL ===");
  Serial.println("Commands:");
  Serial.println("  right  - rotate clockwise");
  Serial.println("  left   - rotate counter-clockwise");
  Serial.println("  stop   - stop movement");
  Serial.println("  speedX - set speed 0-100");
  Serial.println("  pinX   - change servo pin");
  Serial.println("=========================================");

  motor.attach(servoPin);
  motor.write(90); // stop by default
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toLowerCase();

    if (cmd == "right") {
      rotateRight();
    }
    else if (cmd == "left") {
      rotateLeft();
    }
    else if (cmd == "stop") {
      stopMotor();
    }
    else if (cmd.startsWith("speed")) {
      int value = cmd.substring(5).toInt();
      setSpeed(value);
    }
    else if (cmd.startsWith("pin")) {
      int newPin = cmd.substring(3).toInt();
      changePin(newPin);
    }
    else {
      Serial.println("Invalid command");
    }
  }
}

void rotateRight() {
  int pwm = map(speedPercent, 0, 100, 90, 0);
  Serial.print("→ RIGHT (PWM ");
  Serial.print(pwm);
  Serial.println(")");
  motor.write(pwm);
}

void rotateLeft() {
  int pwm = map(speedPercent, 0, 100, 90, 180);
  Serial.print("← LEFT (PWM ");
  Serial.print(pwm);
  Serial.println(")");
  motor.write(pwm);
}

void stopMotor() {
  Serial.println("■ STOP");
  motor.write(90);
}

void setSpeed(int s) {
  if (s < 0) s = 0;
  if (s > 100) s = 100;
  speedPercent = s;
  Serial.print("Speed set to: ");
  Serial.print(speedPercent);
  Serial.println("%");
}

void changePin(int p) {
  int validPins[] = {3,5,6,9,10,11};
  bool ok = false;
  for (int i=0; i<6; i++)
    if (p == validPins[i]) ok = true;

  if (!ok) {
    Serial.println("Invalid pin! Use 3,5,6,9,10,11");
    return;
  }

  motor.detach();
  delay(100);

  servoPin = p;
  motor.attach(servoPin);
  motor.write(90);

  Serial.print("Changed servo pin to ");
  Serial.println(servoPin);
}
