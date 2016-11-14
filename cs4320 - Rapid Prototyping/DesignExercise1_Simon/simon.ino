/*  Luke Goodman (lng26)
  Link to YouTube video of two attempts at three round game:
  https://www.youtube.com/watch?v=uOx_PqL34-A */


const int redLED = D2;
const int greenLED = D1;
const int blueLED = D0;

const int redButton = A2;
const int greenButton = A1;
const int blueButton = A0;

int game_speed = 1000; // time each light shows
int current_round = 0;
int num_rounds = 3;
int time_limit = 2500;

// state of the buttons this iteration
byte current_red = LOW;
int current_green;
int current_blue;
// previous state of the buttons
byte old_red = LOW;
int default_green;
int old_blue;

void setup() {
  pinMode(redLED, OUTPUT);
  pinMode(greenLED, OUTPUT);
  pinMode(blueLED, OUTPUT);

  Serial.begin(9600);
  pinMode(redButton, INPUT);
  pinMode(greenButton, INPUT);
  pinMode(blueButton, INPUT);
}

void loop() {
  char pattern[num_rounds];
  char user_entry[num_rounds];
  bool lost = false;
  default_green = analogRead(greenButton);
  if (Serial.available() > 0) {
    int in_byte = Serial.read();
    lightUp(in_byte);
    if (in_byte == 's') {
      lightUp('o');
      Serial.println("Start Game");
      while(current_round <= num_rounds) {
        int new_color = random(1,100);

        if (new_color <= 33) {
          pattern[current_round] = 'r';
        } else if (new_color <= 66) {
          pattern[current_round] = 'g';
        } else {
          pattern[current_round] = 'b';
        }

        current_round++;
        for (int i = 0; i < current_round; i++) {
          lightUp(pattern[i]);
          delay(game_speed);
          lightUp('o');
          delay(250);
        }

        for (int i = 0; i < current_round; i++) {
          old_blue = analogRead(blueButton);
          char entry = wait_for_button();
          lightUp(entry);
          delay(750);
          lightUp('o');
          if (entry == 'o' || entry != pattern[i]) {
            lost = true;
            lostGame();
            break;
          }
          user_entry[i] = entry;
        }
        if (lost) {
          lightUp('o');
          break; //game over
        }
        if (current_round == num_rounds) {
          wonGame();
          break;
        }
        Serial.print("That's correct! You've completed round ");
        Serial.println(current_round);

        delay(2500);
      }
    }
  }

}

void lightUp(char c) {
  switch(c) {
    case 'r':
      digitalWrite(redLED, HIGH);
      digitalWrite(greenLED, LOW);
      digitalWrite(blueLED, LOW);
      break;
    case 'g':
      digitalWrite(redLED, LOW);
      digitalWrite(greenLED, HIGH);
      digitalWrite(blueLED, LOW);
      break;
    case 'b':
      digitalWrite(redLED, LOW);
      digitalWrite(greenLED, LOW);
      digitalWrite(blueLED, HIGH);
      break;
    case 'o':
      digitalWrite(redLED, LOW);
      digitalWrite(greenLED, LOW);
      digitalWrite(blueLED, LOW);
      break;
  }
}

// Wait for a button to be pressed.
// Returns char of corresponding button if successful, 'n' if timed out
char wait_for_button() {
  long startTime = millis(); // Remember the time we started the this loop
  char button = 'o';
  while ( (millis() - startTime) < time_limit) {// Loop until too much time has passed
    button = checkButtons();
    if (button != 'o') {
      lightUp(button);
      return button;
    }
  }
  lightUp('o');
  return button;
}

// Returns a char corresponding to the pressed button
char checkButtons() {
  current_red = read_button(redButton, old_red);
  current_green = analogRead(greenButton);
  current_blue = analogRead(blueButton);
  // LOW -> HIGH transition
  if ((old_red == 0) && (current_red == 1) || (old_red == 1) && (current_red == 0)) {
    old_red = current_red;
    return 'r';
  } else if (abs(current_green - default_green) > 250) {
    return('g');
  } else if (abs(current_blue - old_blue) > 100) {
    return('b');
  }
  return('o'); // if no button is pressed, return 'o'
}

void lostGame() {
  Serial.println("Sorry, you lose.");
  current_round = 0;
}

void wonGame() {
  Serial.println("You won!!!!");
  int color = 0;
  for (int i=0; i<30; i++) {
    if (color == 0) {
      lightUp('r');
      delay(333);
      color++;
    } else if (color==1) {
      lightUp('g');
      delay(333);
      color++;
    } else {
      lightUp('b');
      delay(333);
      color = 0;
    }
  }
  lightUp('o');
  current_round = 0;
}

byte read_button(byte pin, byte ref_value) {
  // observed the state of the button
  byte current_button = digitalRead(pin);
  // There is a LOW -> HIGH transition
  // or a HIGH -> LOW transition
  if (((ref_value == LOW)
  && (current_button == HIGH))
  || ((ref_value == HIGH)
  && (current_button == LOW))) {
    // wait for a while (5ms)
    delay(5);
    // update the state of the button
    current_button = digitalRead(pin);
  }
  return(current_button);
}
