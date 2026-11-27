#include "weights.h"

#define LED_PIN 2 // Built-in LED on ESP32 WROOM 32
#define BYTES_PER_FRAME 32

// LIF Neuron Parameters
float membrane_potential = 0.0;
const float DECAY_RATE = 0.8;
const float THRESHOLD = -5.0; // Arbitrary threshold for dummy phase

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // Wait until a full mock frame (32 bytes) arrives
  if (Serial.available() >= BYTES_PER_FRAME) {
    uint8_t buffer[BYTES_PER_FRAME];
    Serial.readBytes(buffer, BYTES_PER_FRAME);

    float current_input_sum = 0.0;
    int weight_index = 0;

    // Unpack the 32 bytes back into 256 bits
    for (int i = 0; i < BYTES_PER_FRAME; i++) {
      for (int bit = 7; bit >= 0; bit--) {
        // Extract each bit (0 or 1)
        uint8_t pixel_active = (buffer[i] >> bit) & 0x01; 
        
        // If the pixel "spiked" (is 1), add its corresponding weight
        if (pixel_active) {
          current_input_sum += DUMMY_WEIGHTS[weight_index];
        }
        weight_index++;
      }
    }

    // Leaky Integrate-and-Fire (LIF) Logic
    membrane_potential = (membrane_potential * DECAY_RATE) + current_input_sum;

    // Check for Spike
    if (membrane_potential > THRESHOLD) {
      digitalWrite(LED_PIN, HIGH); // Fire! (Turn LED on)
      membrane_potential = 0.0;    // Reset potential after spiking
      Serial.println("SPIKE!");
    } else {
      digitalWrite(LED_PIN, LOW);  // Don't fire
      Serial.println("Decaying...");
    }
  }
}
