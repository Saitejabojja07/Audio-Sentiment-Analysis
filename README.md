# Audio-Sentiment-Analysis
Audio Sentiment Analysis
This project utilizes various APIs and tools to perform sentiment analysis on audio data. It converts audio to text using the Deepgram API, analyzes the sentiment of the text using the OpenAI API and Langchain, and exposes the results through a RESTful API built with Flask.
Features

    Audio to Text Conversion: Utilizes the Deepgram API to transcribe audio files into text data.
    Sentiment Analysis: Performs sentiment analysis on the transcribed text using the OpenAI API and Langchain, providing insights into the emotional tone of the audio content.
    RESTful API: Implements a RESTful API with Flask, allowing users to interact with the sentiment analysis functionality programmatically.

Requirements

    Python 3.10.0
    Flask
    Deepgram API credentials
    OpenAI API credentials
    Langchain 

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/audio-sentiment-analysis.git

Install dependencies:

bash

    pip install -r requirements.txt

    Set up API credentials:
        Obtain API credentials for Deepgram, OpenAI, and Langchain.
        Add the credentials to the appropriate configuration file or environment variables.

Usage

    Start the Flask server:

    bash

    python app.py

    Access the API endpoints:

        Audio to Text Conversion:
            Endpoint: /transcribe
            Method: POST
            Input: Audio file (e.g., WAV, MP3)
            Output: Transcribed text

        Sentiment Analysis:
            Endpoint: /analyze_audio
            Method: POST
            Input: Text data
            Output: Sentiment analysis results (e.g., positive, negative, neutral) and the content

Contributors

    Sai Teja Bojja
