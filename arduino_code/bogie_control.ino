
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <NewPing.h>

// LCD setup
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Motor pins
const int ENA = 5;
const int IN1 = 3;
const int IN2 = 4;
const int ENB = 9;
const int IN3 = 6;
const int IN4 = 7;

// Ultrasonic sensor
const int trigPin = 10;
const int echoPin = 11;
NewPing sonar(trigPin, echoPin, 200);

// Parameters
int motorSpeed = 150;
int stopDistance = 5;

// State from Pi
bool piSaysStop = false;

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600);
  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("System Boot...");
  delay(1000);
  lcd.clear();
}

void loop() {
  // Read from Pi
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    if (msg == "stop") {
      piSaysStop = true;
      Serial.println("PI: STOP");
    } else if (msg == "resume") {
      piSaysStop = false;
      Serial.println("PI: RESUME");
    }
  }

  int distance = sonar.ping_cm();
  Serial.print("Ultrasonic: ");
  Serial.println(distance);

  lcd.setCursor(0, 0);
  lcd.print("Dist:");
  lcd.print(distance);
  lcd.print("cm   ");

  lcd.setCursor(0, 1);
  if (distance > stopDistance && !piSaysStop) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    analogWrite(ENA, motorSpeed);
    analogWrite(ENB, motorSpeed);
    lcd.print("Running       ");
  } else {
    analogWrite(ENA, 0);
    analogWrite(ENB, 0);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    lcd.print("Stopped       ");
  }

  delay(200);
}
