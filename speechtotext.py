# speech_to_text.py

import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

print("Speak into your microphone. Press Ctrl+C to stop.\n")

try:
    while True:
        # Use the default microphone as source
        with sr.Microphone() as source:
            print("I am Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)

            try:
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"I think You said: {text}\n")

            except sr.UnknownValueError:
                print("Uh Oh Could not understand audio. Please Try again.\n")

            except sr.RequestError as e:
                print(f"Could not request results. Please check your internet connection. Error: {e}")

except KeyboardInterrupt:
    print("\n Stopped by user.")
