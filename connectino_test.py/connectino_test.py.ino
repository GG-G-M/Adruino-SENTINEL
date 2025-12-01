#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

// ===========================
// 🔧 USER-MODIFIABLE SETTINGS
// ===========================

// --- SERVO SETTINGS ---
int servoStartAngle = 0;    // Closed position
int servoOpenAngle = 90;    // Open position
int servoMoveDelay = 1500;  // delay after servo movements

// --- BUZZER (PASSIVE) SETTINGS ---
int accessGrantedTone = 2000;   // Pleasant tone for access granted
int accessDeniedTone = 4000;    // High pitch for access denied
int notificationTone = 1500;    // Medium tone for notifications

// ===========================
// 🔧 HARDWARE DEFINITIONS
// ===========================
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo gateServo;
#define SERVO_PIN 9
#define BUZZER_PIN 7

// ===========================
// 🔧 STATE VARIABLES
// ===========================
String currentCommand = "";

void setup() {
  // Initialize Serial
  Serial.begin(9600);
  
  // LCD Setup
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Security System");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  
  // Servo is NOT attached in setup to prevent twitching

  // Buzzer Setup
  pinMode(BUZZER_PIN, OUTPUT);

  // Single quick beep to indicate ready (NO SERVO MOVEMENT)
  tone(BUZZER_PIN, notificationTone, 200);
  delay(500);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
  
  Serial.println("Arduino Ready - Waiting for commands...");
}

void loop() {
  if (Serial.available() > 0) {
    currentCommand = Serial.readStringUntil('\n');
    currentCommand.trim();
    
    Serial.print("Received: ");
    Serial.println(currentCommand);
    
    processCommand(currentCommand);
  }
}

void processCommand(String command) {
  if (command == "READY") systemReady();
  else if (command == "FACE_VERIFIED") faceVerified();
  else if (command == "GESTURE_REQUIRED") gestureRequired();
  else if (command == "GRANTED") accessGranted();  // ✅ SERVO MOVES ONLY HERE
  else if (command == "DENIED") accessDenied();    // ❌ No servo movement
  else if (command == "TEST") testAllComponents(); // ✅ Manual test moves servo
}

// ------------------------
// SYSTEM STATES / COMMANDS
// ------------------------

void systemReady() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
  
  // Short confirmation beep (NO SERVO MOVEMENT)
  tone(BUZZER_PIN, notificationTone, 300);
  delay(500);
}

void faceVerified() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Face Verified!");
  lcd.setCursor(0, 1);
  lcd.print("Show Gesture...");
  
  // Double beep (NO SERVO MOVEMENT)
  tone(BUZZER_PIN, notificationTone, 200);
  delay(300);
  tone(BUZZER_PIN, notificationTone, 200);
}

void gestureRequired() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Gesture");
  lcd.setCursor(0, 1);
  lcd.print("Verification...");
  
  for(int i = 0; i < 3; i++) {
    tone(BUZZER_PIN, notificationTone, 100);
    delay(200);
  }
}

void accessGranted() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ACCESS GRANTED");
  lcd.setCursor(0, 1);
  lcd.print("Welcome!");
  
  // Play tone
  tone(BUZZER_PIN, accessGrantedTone, 800);
  delay(800);

  // ✅ SERVO MOVES ONLY HERE
  gateServo.attach(SERVO_PIN);       // Attach right before moving
  gateServo.write(servoOpenAngle);    // Open gate
  delay(servoMoveDelay);
  delay(5000);                        // Keep open
  gateServo.write(servoStartAngle);   // Close gate
  delay(servoMoveDelay);
  gateServo.detach();                 // Detach to prevent twitch

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
}

void accessDenied() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ACCESS DENIED!");
  lcd.setCursor(0, 1);
  lcd.print("Unauthorized");
  
  for(int i = 0; i < 3; i++) {
    tone(BUZZER_PIN, accessDeniedTone, 500);
    delay(600);
  }
  
  delay(3000);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  lcd.setCursor(0, 1);
  lcd.print("Awaiting Auth...");
}

void testAllComponents() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing All");
  lcd.setCursor(0, 1);
  lcd.print("Components...");
  delay(1000);

  // ✅ Manual test moves servo
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Servo");

  gateServo.attach(SERVO_PIN);       // Attach for test
  gateServo.write(servoOpenAngle);
  delay(servoMoveDelay);
  gateServo.write(servoStartAngle);
  delay(servoMoveDelay);
  gateServo.detach();                 // Detach after test

  // Test Buzzer
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Testing Buzzer");
  tone(BUZZER_PIN, notificationTone, 1000);
  delay(1500);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Test Complete");
  lcd.setCursor(0, 1);
  lcd.print("System Ready");
  delay(2000);
}
