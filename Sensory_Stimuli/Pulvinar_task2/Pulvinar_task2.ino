// =============================================================================
// Pulvinar_task2.ino  —  revised TTL strategy
// =============================================================================
// Trial types (6 total):
//   AUDIO_2K4K  — 2kHz repeated (3-5x) then 4kHz oddball
//   AUDIO_4K2K  — 4kHz repeated (3-5x) then 2kHz oddball
//   AUDIO_6K8K  — 6kHz repeated (3-5x) then 8kHz oddball
//   AUDIO_8K6K  — 8kHz repeated (3-5x) then 6kHz oddball
//   VIS_SINGLE  — single 100ms flash
//   VIS_SEQ     — sequence of N flashes (N = 3, 4, or 5)
//
// TTL STRATEGY (6 BNC cables):
//   trialPin   (pin 8)  — HIGH for 10ms at prestim onset  → marks trial start
//   ttl2kHz    (pin 13) — HIGH for 10ms at every 2kHz tone onset
//   ttl4kHz    (pin 12) — HIGH for 10ms at every 4kHz tone onset
//   ttl6kHz    (pin 11) — HIGH for 10ms at every 6kHz tone onset
//   ttl8kHz    (pin 10) — HIGH for 10ms at every 8kHz tone onset
//   ttlVis     (pin 9)  — HIGH for 10ms at every visual flash onset
//
// What you get in the recording:
//   - Trial identity: which freq pair fires tells you the trial type
//   - Oddball vs standard: position in the pulse train (last = oddball)
//   - Flash count: number of ttlVis pulses per VIS_SEQ trial
//   - Trial start: trialPin lets you epoch independently of stimulus
//
// HARDWARE (Arduino Mega):
//   Speaker    — pin 7
//   LEDpin     — pin 50  (powers the LED)
//   trialPin   — pin 8   (trial start TTL)
//   ttl2kHz    — pin 13
//   ttl4kHz    — pin 12
//   ttl6kHz    — pin 11
//   ttl8kHz    — pin 10
//   ttlVis     — pin 9
// =============================================================================

// ── USER PARAMETERS ──────────────────────────────────────────────────────────
#include <Arduino.h>
const int maxTrialNum  = 80;   // total trials (20 per type × 6 types)
const int prestimDur   = 1000;  // ms before stimulus
const int stimInterval = 1000;  // ms between tones (includes tone duration)
const int toneDur      = 100;   // ms each tone
const int lightDur     = 200;   // ms each flash
const int flashITI     = 300;   // ms between flashes in sequence (gap only)
const int ttlDur       = 10;    // ms TTL pulse width

// Pseudorandom repeat counts (3–5): used for tone repeats AND flash counts
const int repR[] = {
  3,5,4,5,3,4,5,4,3,3,4,5,4,5,3,4,3,5,3,5,
  4,5,3,4,5,4,3,3,4,5,4,5,3,4,3,5,3,5,4,5,
  3,4,5,4,3,3,4,5,4,5,3,4,3,5,3,5,4,5,3,4,
  5,4,3,3,4,5,4,5,3,4,3,5,3,5,4,5,3,4,5,4,
  3,3,4,5,4,5,3,4,3,5,3,5,4,5,3,4,5,4,3,3,
  4,5,4,5,3,4,3,5,3,5,4,5,3,4,5,4,3,3,4,5
};

// ── PIN DEFINITIONS ──────────────────────────────────────────────────────────

const int speakerPin = 7;
const int LEDpin     = 51;

const int trialPin   = 8;   // trial start TTL
const int ttl2kHz    = 13;  // fires at every 2kHz tone onset
const int ttl4kHz    = 12 ;  // fires at every 4kHz tone onset
const int ttl6kHz    = 11;  // fires at every 6kHz tone onset
const int ttl8kHz    = 10;  // fires at every 8kHz tone onset
const int ttlVis     = 9;   // fires at every visual flash onset

// ── TONE FREQUENCIES ─────────────────────────────────────────────────────────

const int FREQ_2K = 2000;
const int FREQ_4K = 4000;
const int FREQ_6K = 3000;
const int FREQ_8K = 5000;

// ── TRIAL TYPES ───────────────────────────────────────────────────────────────

const int AUDIO_2K4K = 0;
const int AUDIO_4K2K = 1;
const int AUDIO_6K8K = 2;
const int AUDIO_8K6K = 3;
const int VIS_SINGLE = 4;
const int VIS_SEQ    = 5;

// Pre-shuffled trial sequence: 20 reps × 6 types = 120 trials

const int trialSequence[80] = {
  5,0,4,1,0,5,1,4,0,1,5,4,1,0,4,5,0,1,4,0,
  1,5,4,0,1,4,0,5,1,4,5,0,4,1,5,0,1,4,0,5,
  4,1,1,5,0,4,1,0,5,4,1,5,0,4,0,1,4,0,5,1,
  0,5,4,1,5,0,1,4,0,5,1,0,5,1,4,0,1,5,1,4
};

// ── STATE ─────────────────────────────────────────────────────────────────────

int trialNum = 0;

// ── HELPERS ───────────────────────────────────────────────────────────────────

const int allTTL[] = { trialPin, ttl2kHz, ttl4kHz, ttl6kHz, ttl8kHz, ttlVis };
const int nTTL = 6;

void pulseTTL(int pin) {
  digitalWrite(pin, HIGH);
  delay(ttlDur);
  digitalWrite(pin, LOW);
}

// Fires TTL and tone simultaneously — TTL pulse overlaps tone onset
void toneWithTTL(int ttlPin, int freq) {
  tone(speakerPin, freq, toneDur);
  pulseTTL(ttlPin);                        // 10ms pulse while tone is playing
}

void flashWithTTL() {
  digitalWrite(LEDpin, HIGH);
  pulseTTL(ttlVis);                        // 10ms pulse while LED is on
  delay(lightDur - ttlDur);
  digitalWrite(LEDpin, LOW);
}

// ── SETUP ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(9600);

  pinMode(LEDpin, OUTPUT);    digitalWrite(LEDpin, LOW);
  pinMode(speakerPin, OUTPUT);

  for (int i = 0; i < nTTL; i++) {
    pinMode(allTTL[i], OUTPUT);
    digitalWrite(allTTL[i], LOW);
  }

  trialNum = 0;
}

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────

void loop() {
  if (trialNum >= maxTrialNum) return;

  int trialType = trialSequence[trialNum];
  int reps      = repR[trialNum];

  // ── Prestim: TTL marks trial start ───────────────────────────────────────
  pulseTTL(trialPin);
  delay(prestimDur - ttlDur);

  // ── Stimulus ─────────────────────────────────────────────────────────────
  switch (trialType) {

    case AUDIO_2K4K:
      // Standards: 2kHz × reps — each gets a TTL on ttl2kHz
      for (int i = 0; i < reps; i++) {
        toneWithTTL(ttl2kHz, FREQ_2K);
        delay(stimInterval - ttlDur);      // ttlDur already elapsed in pulseTTL
      }
      // Oddball: 4kHz — TTL on ttl4kHz
      toneWithTTL(ttl4kHz, FREQ_4K);
      delay(stimInterval - ttlDur);
      break;

    case AUDIO_4K2K:
      for (int i = 0; i < reps; i++) {
        toneWithTTL(ttl4kHz, FREQ_4K);
        delay(stimInterval - ttlDur);
      }
      toneWithTTL(ttl2kHz, FREQ_2K);
      delay(stimInterval - ttlDur);
      break;

    case AUDIO_6K8K:
      for (int i = 0; i < reps; i++) {
        toneWithTTL(ttl6kHz, FREQ_6K);
        delay(stimInterval - ttlDur);
      }
      toneWithTTL(ttl8kHz, FREQ_8K);
      delay(stimInterval - ttlDur);
      break;

    case AUDIO_8K6K:
      for (int i = 0; i < reps; i++) {
        toneWithTTL(ttl8kHz, FREQ_8K);
        delay(stimInterval - ttlDur);
      }
      toneWithTTL(ttl6kHz, FREQ_6K);
      delay(stimInterval - ttlDur);
      break;

    case VIS_SINGLE:
      flashWithTTL();
      delay(stimInterval - lightDur);
      break;

    case VIS_SEQ:
      for (int i = 0; i < reps; i++) {
        flashWithTTL();
        if (i < reps - 1) delay(flashITI);
      }
      delay(stimInterval - lightDur);
      break;
  }

  // ── ITI: 3–5 seconds ─────────────────────────────────────────────────────
  delay(random(3, 6) * 1000);

  // ── Serial log ───────────────────────────────────────────────────────────
  Serial.print(trialNum + 1);
  Serial.print(",");
  Serial.print(trialType);
  Serial.print(",");
  Serial.println(reps);

  trialNum++;
}