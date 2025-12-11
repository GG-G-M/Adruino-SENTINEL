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
// SERVO SETTINGS (User configurable)
// ============================================
const int SERVO_CLOSED_ANGLE = 0;   // Closed/locked position
const int SERVO_OPEN_ANGLE = 90;    // Open/unlocked position
const int SERVO_MOVE_DELAY = 1500;  // Delay for servo to reach position (ms)

// ============================================
// SONAR SETTINGS
// ============================================
const int DETECTION_DISTANCE = 50;  // Distance in cm to trigger detection

String currentCommand = "";

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

  // ⚠️ CRITICAL: DO NOT attach servo in setup!
  // This prevents the startup spin/twitch issue
  // Servo will only be attached when needed (during GRANTED command)

  // Buzzer Setup
  pinMode(BUZZER_PIN, OUTPUT);

  // Sonar Setup
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Startup beep
  tone(BUZZER_PIN, 1500, 200);
  delay(1000);

  // Ready message
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  Serial.println("Arduino Ready - Servo will attach only on access grant");
}

void loop() {
  // Listen for commands from Python
  if (Serial.available() > 0) {
    currentCommand = Serial.readStringUntil('\n');
    currentCommand.trim();

    // Process commands
    if (currentCommand == "GRANTED") {
      accessGranted();
    } else if (currentCommand == "LOCK") {
      lockDoor();
    } else if (currentCommand == "DENIED") {
      accessDenied();
    } else if (currentCommand == "READY") {
      systemReady();
    } else if (currentCommand == "FACE_VERIFIED") {
      faceVerified();
    } else if (currentCommand == "GESTURE_REQUIRED") {
      gestureRequired();
    } else if (currentCommand == "TEST") {
      testComponents();
    } else if (currentCommand == "TEST_SONAR") {
      testSonar();
    } else if (currentCommand.startsWith("UNLOCK:")) {
      // Custom unlock duration from Python
      // Format: UNLOCK:5 (for 5 seconds)
      int duration = currentCommand.substring(7).toInt();
      accessGrantedCustomDuration(duration);
    }
  }
}

// ============================================
// ACCESS GRANTED - Python controls timing
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

  // ✅ Attach servo ONLY when needed
  gateServo.attach(SERVO_PIN);
  delay(50);  // Small delay for servo to initialize

  // Open gate
  gateServo.write(SERVO_OPEN_ANGLE);
  delay(SERVO_MOVE_DELAY);

  // Send signal to Python that door is open
  Serial.println("DOOR_OPEN");

  // Python will send "LOCK" command when ready to close
  // Don't close automatically - let Python control timing
}

// ============================================
// LOCK DOOR (Called by Python)
// ============================================
void lockDoor() {
  // Close gate
  gateServo.write(SERVO_CLOSED_ANGLE);
  delay(SERVO_MOVE_DELAY);

  // ✅ Detach servo to prevent jitter/spin
  gateServo.detach();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  Serial.println("DOOR_LOCKED");
}

// ============================================
// CUSTOM DURATION ACCESS (Optional)
// ============================================
void accessGrantedCustomDuration(int seconds) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ACCESS GRANTED");
  lcd.setCursor(0, 1);
  lcd.print("Welcome!");

  tone(BUZZER_PIN, 2000, 800);
  delay(800);

  // Attach and open
  gateServo.attach(SERVO_PIN);
  delay(50);
  gateServo.write(SERVO_OPEN_ANGLE);
  delay(SERVO_MOVE_DELAY);

  // Keep open for specified duration
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Door Open");

  for (int i = seconds; i > 0; i--) {
    lcd.setCursor(0, 1);
    lcd.print("Closing in ");
    lcd.print(i);
    lcd.print("s ");
    delay(1000);
  }

  // Close and detach
  gateServo.write(SERVO_CLOSED_ANGLE);
  delay(SERVO_MOVE_DELAY);
  gateServo.detach();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
}

// ============================================
// ACCESS DENIED
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

// ============================================
// SYSTEM READY
// ============================================
void systemReady() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");

  tone(BUZZER_PIN, 1500, 300);
}

// ============================================
// FACE VERIFIED
// ============================================
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

// ============================================
// GESTURE REQUIRED
// ============================================
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

// ============================================
// TEST COMPONENTS
// ============================================
void testComponents() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing...");
  lcd.setCursor(0, 1);
  lcd.print("Components");

  delay(500);

  // Test servo
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Servo");

  gateServo.attach(SERVO_PIN);
  delay(50);

  // Open
  gateServo.write(SERVO_OPEN_ANGLE);
  delay(SERVO_MOVE_DELAY);

  // Close
  gateServo.write(SERVO_CLOSED_ANGLE);
  delay(SERVO_MOVE_DELAY);

  // Detach
  gateServo.detach();

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