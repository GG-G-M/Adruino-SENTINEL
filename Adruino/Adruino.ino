#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

// ============================================
// HARDWARE SETUP
// ============================================
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo gateServo;

#define SERVO_PIN 9
#define BUZZER_PIN 7
#define TRIG_PIN 3
#define ECHO_PIN 2

// ============================================
// DOOR CONFIGURATION - MODIFY THESE SETTINGS
// ============================================

// DOOR MECHANISM SETTINGS
// Set to true if your door opens clockwise when viewed from the servo side
// Set to false if your door opens counter-clockwise
const bool DOOR_OPENS_CLOCKWISE = true;

// SPEED SETTINGS (0-100%)
const int OPEN_SPEED = 60;           // Speed for opening rotation (0-100%)
const int CLOSE_SPEED = 60;          // Speed for closing rotation (0-100%)

// TIMING SETTINGS (milliseconds)
const int DOOR_OPEN_DURATION = 5000;     // How long door stays open (5000ms = 5 seconds)
const int DOOR_CLOSE_DELAY = 2000;       // Delay before closing starts (2000ms = 2 seconds)
const int ROTATION_TIME = 3000;          // Time for complete door rotation (3000ms = 3 seconds)

// SERVO CONTROL SETTINGS
const int SERVO_NEUTRAL = 90;            // Stop position for continuous servo
const int SPEED_SMOOTHING = 10;          // Smooth speed transitions (0-50)

// Motor stop delay to prevent rapid direction changes
const int MOTOR_COOLDOWN = 500;          // Minimum time between direction changes

String currentCommand = "";

// ============================================
// SERVO CONTROL FUNCTIONS
// ============================================

void setup() {
  Serial.begin(9600);

  // LCD Setup
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Security System");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");

  // Buzzer Setup
  pinMode(BUZZER_PIN, OUTPUT);

  // Sonar Setup
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Attach continuous servo
  gateServo.attach(SERVO_PIN);
  gateServo.write(SERVO_NEUTRAL); // Stop the continuous servo

  // Startup beep
  tone(BUZZER_PIN, 1500, 200);
  delay(1000);

  // Ready message
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  Serial.println("Arduino Ready - Configurable Continuous Servo Control");
  Serial.println("Configuration:");
  Serial.print("  Door opens clockwise: ");
  Serial.println(DOOR_OPENS_CLOCKWISE ? "YES" : "NO");
  Serial.print("  Open speed: ");
  Serial.print(OPEN_SPEED);
  Serial.println("%");
  Serial.print("  Close speed: ");
  Serial.print(CLOSE_SPEED);
  Serial.println("%");
  Serial.print("  Open duration: ");
  Serial.print(DOOR_OPEN_DURATION / 1000);
  Serial.println("s");
}

void loop() {
  // Listen for commands from Python
  if (Serial.available() > 0) {
    currentCommand = Serial.readStringUntil('\n');
    currentCommand.trim();
    currentCommand.toLowerCase();

    // Process commands
    if (currentCommand == "granted") {
      accessGranted();
    } else if (currentCommand == "lock") {
      lockDoor();
    } else if (currentCommand == "denied") {
      accessDenied();
    } else if (currentCommand == "ready") {
      systemReady();
    } else if (currentCommand == "face_verified") {
      faceVerified();
    } else if (currentCommand == "gesture_required") {
      gestureRequired();
    } else if (currentCommand == "test") {
      testComponents();
    } else if (currentCommand == "test_sonar") {
      testSonar();
    } else if (currentCommand == "right") {
      rotateRight();
    } else if (currentCommand == "left") {
      rotateLeft();
    } else if (currentCommand == "stop") {
      stopMotor();
    } else if (currentCommand.startsWith("speed")) {
      int speed = currentCommand.substring(5).toInt();
      setSpeed(speed);
    }
  }
}

// ============================================
// DOOR CONTROL FUNCTIONS
// ============================================

void accessGranted() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ACCESS GRANTED");
  lcd.setCursor(0, 1);
  lcd.print("Welcome!");

  // Success tone
  tone(BUZZER_PIN, 2000, 800);
  delay(800);

  // Open door
  openDoor();

  // Send signal to Python that door is open
  Serial.println("DOOR_OPEN");

  // Display door open countdown
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Door Open");

  // Wait for configured duration
  int countdown = DOOR_OPEN_DURATION / 1000;
  for (int i = countdown; i > 0; i--) {
    lcd.setCursor(0, 1);
    lcd.print("Closing in ");
    lcd.print(i);
    lcd.print("s ");
    delay(1000);
  }

  // Close door
  closeDoor();

  // Update LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Door Locked");
  delay(1000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  // Send signal to Python that door is locked
  Serial.println("DOOR_LOCKED");
}

void openDoor() {
  Serial.println("Opening door...");
  
  // Determine rotation direction based on configuration
  if (DOOR_OPENS_CLOCKWISE) {
    Serial.println("  Opening clockwise");
    rotateClockwise();
  } else {
    Serial.println("  Opening counter-clockwise");
    rotateCounterClockwise();
  }
  
  // Run for the configured rotation time
  delay(ROTATION_TIME);
  stopMotor();
  
  Serial.println("Door opened");
}

void closeDoor() {
  Serial.println("Closing door...");
  
  // Determine rotation direction based on configuration (opposite of opening)
  if (DOOR_OPENS_CLOCKWISE) {
    Serial.println("  Closing counter-clockwise");
    rotateCounterClockwise();
  } else {
    Serial.println("  Closing clockwise");
    rotateClockwise();
  }
  
  // Run for the configured rotation time
  delay(ROTATION_TIME);
  stopMotor();
  
  Serial.println("Door closed");
}

void lockDoor() {
  closeDoor();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  Serial.println("DOOR_LOCKED");
}

// ============================================
// SERVO MOTOR CONTROL
// ============================================

void rotateClockwise() {
  Serial.println("→ Rotating CLOCKWISE");
  int pwm = map(OPEN_SPEED, 0, 100, SERVO_NEUTRAL, 0);
  gateServo.write(pwm);
  Serial.print("PWM: ");
  Serial.println(pwm);
}

void rotateCounterClockwise() {
  Serial.println("← Rotating COUNTER-CLOCKWISE");
  int pwm = map(CLOSE_SPEED, 0, 100, SERVO_NEUTRAL, 180);
  gateServo.write(pwm);
  Serial.print("PWM: ");
  Serial.println(pwm);
}

void rotateRight() {
  Serial.println("→ Rotating RIGHT (clockwise)");
  int pwm = map(OPEN_SPEED, 0, 100, SERVO_NEUTRAL, 0);
  gateServo.write(pwm);
  Serial.print("PWM: ");
  Serial.println(pwm);
}

void rotateLeft() {
  Serial.println("← Rotating LEFT (counter-clockwise)");
  int pwm = map(CLOSE_SPEED, 0, 100, SERVO_NEUTRAL, 180);
  gateServo.write(pwm);
  Serial.print("PWM: ");
  Serial.println(pwm);
}

void stopMotor() {
  Serial.println("■ STOPPING motor");
  gateServo.write(SERVO_NEUTRAL); // Stop the continuous servo
  delay(MOTOR_COOLDOWN); // Cool down period
}

void setSpeed(int speed) {
  Serial.print("Speed set to: ");
  Serial.print(speed);
  Serial.println("%");
  // Note: Speed is used in rotation functions via the configuration constants
}

// ============================================
// SYSTEM FUNCTIONS
// ============================================

void accessDenied() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ACCESS DENIED!");
  lcd.setCursor(0, 1);
  lcd.print("Unauthorized");

  // Alert beeps
  for (int i = 0; i < 3; i++) {
    tone(BUZZER_PIN, 4000, 500);
    delay(600);
  }

  delay(2000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
}

void systemReady() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  tone(BUZZER_PIN, 1500, 300);
}

void faceVerified() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Face Verified!");
  lcd.setCursor(0, 1);
  lcd.print("Show Gesture...");

  // Double beep
  tone(BUZZER_PIN, 1500, 200);
  delay(300);
  tone(BUZZER_PIN, 1500, 200);
}

void gestureRequired() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Gesture");
  lcd.setCursor(0, 1);
  lcd.print("Verification...");

  for (int i = 0; i < 2; i++) {
    tone(BUZZER_PIN, 1500, 100);
    delay(200);
  }
}

void testComponents() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing...");
  lcd.setCursor(0, 1);
  lcd.print("Components");

  delay(500);

  // Test continuous servo functionality
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Servo");

  Serial.println("Testing continuous servo...");
  
  // Test rotating right
  Serial.println("Testing right rotation...");
  lcd.setCursor(0, 1);
  lcd.print("Right...");
  rotateRight();
  delay(2000);
  stopMotor();
  delay(500);
  
  // Test rotating left
  Serial.println("Testing left rotation...");
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Servo");
  lcd.setCursor(0, 1);
  lcd.print("Left...");
  rotateLeft();
  delay(2000);
  stopMotor();
  delay(500);
  
  // Test stop
  Serial.println("Testing stop...");
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Servo");
  lcd.setCursor(0, 1);
  lcd.print("Stop...");
  stopMotor();
  delay(1000);

  // Test buzzer
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Buzzer");
  tone(BUZZER_PIN, 2000, 500);
  delay(1000);

  // Complete
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Test Complete");
  Serial.println("Test Complete");
  delay(2000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
}

// ============================================
// SONAR SENSOR
// ============================================
float getDistance() {
  // Clear trigger
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  // Send pulse
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);  // 30ms timeout

  // Calculate distance
  float distance = duration * 0.0343 / 2;

  // Return -1 if timeout
  if (duration == 0) {
    return -1;
  }

  return distance;
}

void testSonar() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Sonar...");

  unsigned long startTime = millis();

  // Test for 5 seconds
  while (millis() - startTime < 5000) {
    float distance = getDistance();

    // Send to serial
    Serial.print("DISTANCE:");
    if (distance < 0) {
      Serial.println("ERROR");
    } else {
      Serial.print(distance);
      Serial.println("cm");
    }

    // Display on LCD
    lcd.setCursor(0, 1);
    if (distance < 0) {
      lcd.print("No Echo         ");
    } else {
      lcd.print(distance);
      lcd.print(" cm      ");
    }

    delay(500);
  }

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Sonar Test Done");
  delay(1000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
}