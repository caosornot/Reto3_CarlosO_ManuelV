#include <Servo.h>
 
Servo myservo;  // crea el objeto servo
 
int pos = 90;    // posicion del servo
int sens = 5;
int command = 2;

 
void setup() {
  Serial.begin(9600);
   myservo.attach(9);  // vincula el servo al pin digital 9
   myservo.write(pos);
}
 
void loop() {
   //varia la posicion de 0 a 180, con esperas de 15ms
  if (Serial.available()!= 0){
    command = Serial.parseInt();
    Serial.flush();
  }

  if (command == 0){
    pos = pos - sens;
    if(pos < 0){pos = 0;}
    myservo.write(pos);
  }

   if (command == 1){
    pos = pos + sens;
    if(pos > 180){pos = 180;}
    myservo.write(pos);
  }
  delay(100);
}
