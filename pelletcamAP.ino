/*********
  Rui Santos
  Complete project details at https://RandomNerdTutorials.com/esp32-cam-take-photo-display-web-server/
  
  IMPORTANT!!! 
   - Select Board "AI Thinker ESP32-CAM"
   - GPIO 0 must be connected to GND to upload a sketch
   - After connecting GPIO 0 to GND, press the ESP32-CAM on-board RESET button to put your board in flashing mode
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
*********/
#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
 #include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif

#include "WiFi.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"           // Disable brownour problems
#include "soc/rtc_cntl_reg.h"  // Disable brownour problems
#include "driver/rtc_io.h"
#include <ESPAsyncWebServer.h>
#include <StringArray.h>
#include <SPIFFS.h>
#include <FS.h>
#include "SD_MMC.h"


// Which pin on the Arduino is connected to the NeoPixels?
#define PIN        1 // On Trinket or Gemma, suggest changing this to 1
#define NUMPIXELS 8 // Popular NeoPixel ring size
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

  
// Replace with your network credentials
//const char* ssid = "xxxxxx";
//const char* password = "xxxxxx";

const char* ssid = "PELLET2";
const char* password = "OPENBIOLAB";


// Create AsyncWebServer object on port 80
AsyncWebServer server(80);

boolean takeNewPhoto = false;
String typePic = "null";
String pelletLabel = "null";
String concentratie = "null";


// Photo File Name to save in SPIFFS
String FILE_PHOTO  = "/photo3.jpg";

#define FILE_PHOTO2 "/photo.jpg"

// OV2640 camera module pins (CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML><html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { text-align:center; }
    .vert { margin-bottom: 10%; }
    .hori{ margin-bottom: 0%; }
  </style>
</head>
<body>
  <div id="container">
    <h2>ESP32-CAM Last Photo</h2>
    <p>It might take more than 5 seconds to capture a photo.</p>
    <p>
      <button onclick="scanReference()">SCAN REFERENCE</button>
      <button onclick="capturePhoto()">SCAN UNKNOWN</button>
      <button onclick="location.reload();">REFRESH PAGE</button>
    </p>
  </div>
  <div><img src="saved-photo" id="photo" width="70%"></div>
  <div><ul id="result-list"><li>timestamp  R  G  B  </li> </ul></div>
</body>
<script>
  function scanReference() {
    let typePic = "REF";
    let pelletLabel = prompt("Geef hier de dosis/nummer van de referentie:", "getal"); 
    let concentratie = prompt("Geef hier de concentratie van de referentie: (nM)", "nM"); 

    var xhr = new XMLHttpRequest();
    xhr.open('GET', `/capture?typePic=${typePic}&pelletLabel=${pelletLabel}&concentratie=${concentratie}`, true);
    xhr.send();
    setTimeout(() => location.reload(), 5000);
  }

  function capturePhoto() {
    let typePic = "UNK";
    let pelletLabel = prompt("Geef hier het nummer van de ongekende pellet:", "getal"); 
    let concentratie = prompt("Geef hier de concentratie van de referentie: (nM)", "nM");
    var xhr = new XMLHttpRequest();
    xhr.open('GET', `/capture?typePic=${typePic}&pelletLabel=${pelletLabel}&concentratie=${concentratie}`, true);
    xhr.send();
  }
</script>

</html>)rawliteral";

void setup() {

  #if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
    #endif
  
  pixels.begin(); // INITIALIZE NeoPixel strip object (REQUIRED)
  for(int i=0; i<NUMPIXELS; i++) { // For each pixel...
    pixels.setPixelColor(i, pixels.Color(100, 100, 100));
    pixels.show();   // Send the updated pixel colors to the hardware.
    delay(50); // Pause before next pass through loop
  }
  
  // Serial port for debugging purposes
  Serial.begin(115200);

  WiFi.softAP(ssid, password);

  
  if (!SPIFFS.begin(true)) {
    Serial.println("An Error has occurred while mounting SPIFFS");
    ESP.restart();
  }
  else {
    delay(500);
    Serial.println("SPIFFS mounted successfully");
  }

  // Start Micro SD card
  Serial.println("Starting SD Card");
  if(!SD_MMC.begin()){
    Serial.println("SD Card Mount Failed");
    return;
  }
  uint8_t cardType = SD_MMC.cardType();
  if(cardType == CARD_NONE){
    Serial.println("No SD Card attached");
    return;
  }


  // Turn-off the 'brownout detector'
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  // OV2640 camera module
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    ESP.restart();
  }

  // Route for root / web page
  server.on("/", HTTP_GET, [](AsyncWebServerRequest * request) {
    request->send_P(200, "text/html", index_html);
  });

  // server.on("/capture", HTTP_GET, [](AsyncWebServerRequest * request) {
  //   takeNewPhoto = true;
  //   request->send_P(200, "text/plain", "Taking Photo");
  // });
  server.on("/capture", HTTP_GET, [](AsyncWebServerRequest * request) {
    typePic = request->arg("typePic");
    pelletLabel = request->arg("pelletLabel");
    concentratie = request->arg("concentratie");

    // Save the values as needed, for example, print them to Serial
    Serial.println("Type Pic: " + typePic);
    Serial.println("Concentratie: " + concentratie);

    takeNewPhoto = true;
    request->send_P(200, "text/plain", "Taking Photo");
  });

  server.on("/saved-photo", HTTP_GET, [](AsyncWebServerRequest * request) {
    request->send(SPIFFS, FILE_PHOTO2, "image/jpg", false);
  });

  // Start server
  server.begin();
}

void loop() {
  if (takeNewPhoto) {
    capturePhotoSaveSpiffs(); //opslaan voor de site zodat men kan zien wat de laatste foto is 
    capturePhotoSaveSD(); //opslaan op de SD-kaart met speciale URL afhankelijk van REF, label en concentratie
    takeNewPhoto = false;
  }
  delay(1); 
}


// Capture Photo and Save it to SPIFFS
void capturePhotoSaveSD( void ) {
  camera_fb_t * fb = NULL; // pointer

  fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }

  String FILE_PHOTO = "/" + String(typePic) + "_" + String(pelletLabel) + "_" + String(concentratie) +".jpg";
    fs::FS &fs = SD_MMC;
    File file2 = fs.open(FILE_PHOTO,FILE_WRITE);


    // Insert the data in the photo file
    if (!file2) {
      Serial.println("Failed to open file in writing mode");
    }
    else {
      file2.write(fb->buf, fb->len);
    }
    // Close the file
    file2.close();
    esp_camera_fb_return(fb);

}

// Check if photo capture was successful
bool checkPhoto( fs::FS &fs ) {
  File f_pic = fs.open( FILE_PHOTO2 );
  unsigned int pic_sz = f_pic.size();
  return ( pic_sz > 100 );
}

// Capture Photo and Save it to SPIFFS
void capturePhotoSaveSpiffs( void ) {
  camera_fb_t * fb = NULL; // pointer
  bool ok = 0; // Boolean indicating if the picture has been taken correctly

  do {
    // Take a photo with the camera
    Serial.println("Taking a photo...");

    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }

    // Photo file name
    File file = SPIFFS.open(FILE_PHOTO2, FILE_WRITE);

    // Insert the data in the photo file
    if (!file) {
      Serial.println("Failed to open file in writing mode");
    }
    else {
      file.write(fb->buf, fb->len); // payload (image), payload length
    }
    // Close the file
    file.close();
    esp_camera_fb_return(fb);

    // check if file has been correctly saved in SPIFFS
    ok = checkPhoto(SPIFFS);
  } while ( !ok );
}

