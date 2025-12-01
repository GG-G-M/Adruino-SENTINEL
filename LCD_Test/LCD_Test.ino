#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

// ===========================
// 🔧 USER-MODIFIABLE SETTINGS
// ===========================

// --- SERVO SETTINGS ---
int servoStartAngle = 0;
int servoMiddleAngle = 90;
int servoFullAngle   = 180;
int servoMoveDelay   = 1500;   // delay after servo movements

// --- BUZZER (PASSIVE) SETTINGS ---
int longBeepDuration  = 1500;   // milliseconds
int shortBeepDuration = 150;    // milliseconds
int shortBeepPause    = 120;    // pause between short beeps
int shortBeepCount    = 8;      // number of short beeps

int highPitch = 4000;           // high frequency in Hz

// ===========================
// 🔧 HARDWARE DEFINITIONS
// ===========================

LiquidCrystal_I2C lcd(0x27, 16, 2);

Servo gateServo;
#define SERVO_PIN 9

#define BUZZER_PIN 7

void setup() {
  // LCD Setup
  lcd.init();
  lcd.backlight();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("LCD+Servo+Buzzer");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  delay(1500);

  // Servo setup
  gateServo.attach(SERVO_PIN);
  gateServo.write(servoStartAngle);

  // Buzzer Setup
  pinMode(BUZZER_PIN, OUTPUT);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Ready to Test");
  delay(1500);
}

void loop() {

  // --------------------
  // SERVO MOVEMENT TEST
  // --------------------
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo: Start");
  gateServo.write(servoStartAngle);
  delay(servoMoveDelay);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo: Middle");
  gateServo.write(servoMiddleAngle);
  delay(servoMoveDelay);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo: Full");
  gateServo.write(servoFullAngle);
  delay(servoMoveDelay);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo: Reset");
  gateServo.write(servoStartAngle);
  delay(servoMoveDelay);

  // --------------------
  // BUZZER TEST (HIGH PITCH)
  // --------------------

  // LONG BEEP
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Buzzer: Long");
  tone(BUZZER_PIN, highPitch, longBeepDuration);
  delay(longBeepDuration + 200);

  // SHORT RAPID BEEPS
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Buzzer: Short");
  for (int i = 0; i < shortBeepCount; i++) {
    tone(BUZZER_PIN, highPitch, shortBeepDuration);
    delay(shortBeepDuration + shortBeepPause);
  }

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Cycle Complete");
  delay(1500);
}
