

import os
import wave
import time
import pickle
import pyaudio
import numpy as np
from scipy.io.wavfile import read
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import butter, lfilter
import python_speech_features as mfcc
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import hashlib
import json
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Voice Authentication System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #2E7D32;  /* Darker green for better contrast */
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #388E3C;  /* Slightly lighter green on hover */
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(46, 125, 50, 0.3);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 1em;
        border: 1px solid #4CAF50;
        background-color: rgba(255, 255, 255, 0.05);  /* Subtle light background */
        color: inherit;  /* Inherits text color from theme */
    }
    
    /* Progress bar styling */
    .stProgress>div>div>div>div {
        background-color: #2E7D32;  /* Darker green for progress bar */
    }
    
    /* Success message styling */
    .success-message {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: rgba(46, 125, 50, 0.1);  /* Semi-transparent green */
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
        color: #4CAF50;  /* Bright green text */
    }
    
    /* Error message styling */
    .error-message {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: rgba(198, 40, 40, 0.1);  /* Semi-transparent red */
        border-left: 5px solid #C62828;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
        color: #EF5350;  /* Bright red text */
    }
    
    /* Status card styling */
    .status-card {
        background-color: rgba(255, 255, 255, 0.05);  /* Very subtle light background */
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);  /* Subtle border */
        color: inherit;  /* Inherits text color from theme */
    }
    
    /* Headings styling */
    h1, h2, h3 {
        color: inherit;  /* Inherits color from theme */
        margin-bottom: 1rem;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(-10px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1rem;
        border-radius: 0.5rem;
        color: inherit;  /* Inherits text color from theme */
        background-color: rgba(255, 255, 255, 0.05);  /* Subtle background */
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(46, 125, 50, 0.1) !important;  /* Selected tab background */
        border-color: #2E7D32 !important;
        color: #4CAF50 !important;  /* Selected tab text color */
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);  /* Slightly darker background for sidebar */
        padding: 1rem;
    }
    
    /* Plot background for better visibility */
    [data-testid="stPlotlyChart"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;  /* Subtle background for plots */
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = "idle"
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0

# Constants
USERS_FILE = "user_data/users.json"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
SILENCE_THRESHOLD = 500
MIN_VOLUME_THRESHOLD = 100

# Ensure required directories exist
REQUIRED_DIRS = ['training_set', 'testing_set', 'trained_models', 'user_data']
for dir_name in REQUIRED_DIRS:
    os.makedirs(dir_name, exist_ok=True)

# Audio Processing Functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            first = max(0, i - j)
            second = min(rows - 1, i + j)
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + 
                     (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

def extract_features(audio, rate):
    # Apply bandpass filter (focus on speech frequencies)
    filtered_audio = butter_bandpass_filter(audio, 300, 3000, rate)
    
    # Extract MFCC features
    mfcc_feat = mfcc.mfcc(filtered_audio, rate, 
                         winlen=0.025, winstep=0.01,
                         numcep=24, nfilt=26, nfft=2048,
                         appendEnergy=True)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
    
    # Calculate delta features
    delta = calculate_delta(mfcc_feat)
    delta2 = calculate_delta(delta)
    
    # Combine all features
    combined = np.hstack((mfcc_feat, delta, delta2))
    return combined

# Visualization Functions
def plot_waveform(audio_file):
    """Plot the waveform of the recorded audio"""
    sr, audio = read(audio_file)
    time_points = np.linspace(0, len(audio)/sr, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=audio, 
        mode='lines', 
        name='Waveform',
        line=dict(color='#4CAF50', width=1)
    ))
    fig.update_layout(
        title={
            'text': "Audio Waveform",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_mfcc_features(audio_file):
    """Plot MFCC features heatmap"""
    sr, audio = read(audio_file)
    features = extract_features(audio, sr)
    
    fig = px.imshow(
        features.T,
        labels=dict(x="Time", y="MFCC Coefficients"),
        title="MFCC Features Heatmap",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_verification_history(username):
    """Plot verification history for a specific user"""
    history_file = f"user_data/{username}_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.line(
            df, 
            x="timestamp", 
            y="score",
            title="Verification Score History",
            labels={"timestamp": "Time", "score": "Verification Score"}
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig
    return None

# User Management Functions
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Audio Recording Function
def record_audio(filename, progress_bar=None):
    try:
        audio = pyaudio.PyAudio()
        default_input = audio.get_default_input_device_info()
        device_index = default_input['index']
        
        stream = audio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          input_device_index=device_index,
                          frames_per_buffer=CHUNK)
        
        st.markdown("""
            <div class="status-card">
                📢 Recording... Speak when ready. Recording will start when voice is detected.
            </div>
        """, unsafe_allow_html=True)
        
        # Voice activity detection
        start_recording = False
        silence_count = 0
        max_silence_counts = int(RATE / CHUNK)
        Recordframes = []
        
        # Pre-recording phase
        detection_timeout = time.time() + 10
        while not start_recording and time.time() < detection_timeout:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            
            if volume > SILENCE_THRESHOLD:
                start_recording = True
                st.markdown("""
                    <div class="success-message">
                        🎤 Voice detected! Recording...
                    </div>
                """, unsafe_allow_html=True)
                Recordframes.append(data)
        
        if not start_recording:
            st.markdown("""
                <div class="error-message">
                    ❌ No voice detected. Please try again.
                </div>
            """, unsafe_allow_html=True)
            return False
        
        # Main recording phase
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                
                if volume < MIN_VOLUME_THRESHOLD:
                    silence_count += 1
                else:
                    silence_count = 0
                    
                Recordframes.append(data)
                if progress_bar:
                    progress_bar.progress((i + 1) / (RATE / CHUNK * RECORD_SECONDS))
                    
            except Exception as e:
                st.error(f"Error during recording: {str(e)}")
                return False
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the recorded audio
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()
        
        return True
            
    except Exception as e:
        st.error(f"Error initializing audio: {str(e)}")
        return False

# def register_user():
#     st.markdown("""
#         <div style='text-align: center; padding: 2rem;'>
#             <h1>🎯 User Registration</h1>
#         </div>
#     """, unsafe_allow_html=True)
    
#     with st.form("registration_form"):
#         col1, col2, col3 = st.columns([1,2,1])
#         with col2:
#             username = st.text_input("Username", placeholder="Choose a username")
#             password = st.text_input("Password", type="password", 
#                                    placeholder="Enter a strong password")
#             confirm_password = st.text_input("Confirm Password", type="password",
#                                            placeholder="Confirm your password")
#             submit_button = st.form_submit_button("Register")
        
#         if submit_button:
#             if not username or not password:
#                 st.error("Please enter both username and password")
#                 return
                
#             if password != confirm_password:
#                 st.error("Passwords do not match!")
#                 return
                
#             users = load_users()
#             if username in users:
#                 st.error("Username already exists!")
#                 return
            
#             st.session_state.recording_state = "recording"
            
#             st.markdown("""
#                 <div class="status-card">
#                     🎤 Please speak for voice registration (5 samples needed)
#                 </div>
#             """, unsafe_allow_html=True)
            
#             successful_recordings = 0
            
#             for count in range(5):
#                 st.write(f"Recording sample {count + 1}/5")
#                 progress_bar = st.progress(0)
                
#                 filename = f"training_set/{username}-sample{count}.wav


#                 if record_audio(filename, progress_bar):
#                     successful_recordings += 1
#                     with open("training_set_addition.txt", 'a') as f:
#                         f.write(f"{username}-sample{count}.wav\n")
                    
#                     # Display waveform after recording
#                     st.plotly_chart(plot_waveform(filename), use_container_width=True)
#                     st.success(f"Sample {count + 1} recorded successfully!")
#                 else:
#                     st.error(f"Failed to record sample {count + 1}")
                
#                 time.sleep(1)
            
#             if successful_recordings == 5:
#                 # Train model for the user
#                 with st.spinner('Training voice model...'):
#                     if train_model():
#                         # Save user credentials
#                         users[username] = {
#                             "password": hash_password(password),
#                             "model_file": f"{username}.gmm",
#                             "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                             "last_login": None
#                         }
#                         save_users(users)
#                         st.markdown("""
#                             <div class="success-message">
#                                 ✅ Registration completed successfully! You can now login.
#                             </div>
#                         """, unsafe_allow_html=True)
#                     else:
#                         st.error("Failed to train voice model. Please try registration again.")
#             else:
#                 st.error("Registration failed. Please try again.")
            
#             st.session_state.recording_state = "idle"
def register_user():
    st.markdown("""
        <div style='text-align: center'>
            <h2>🎯 User Registration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("registration_form"):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username", placeholder="Choose a username")
            password = st.text_input("Password", type="password", 
                                   placeholder="Enter a strong password")
            confirm_password = st.text_input("Confirm Password", type="password",
                                           placeholder="Confirm your password")
            submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
                
            users = load_users()
            if username in users:
                st.error("Username already exists!")
                return
            
            st.session_state.recording_state = "recording"
            
            st.markdown("""
                <div class="status-card">
                    🎤 Please speak for voice registration (5 samples needed)
                </div>
            """, unsafe_allow_html=True)
            
            # Create training_set_addition.txt if it doesn't exist
            if not os.path.exists("training_set_addition.txt"):
                with open("training_set_addition.txt", "w") as f:
                    pass
                    
            successful_recordings = 0
            
            for count in range(5):
                st.write(f"Recording sample {count + 1}/5")
                progress_bar = st.progress(0)
                
                filename = f"training_set/{username}-sample{count}.wav"
                
                try:
                    if record_audio(filename, progress_bar):
                        successful_recordings += 1
                        with open("training_set_addition.txt", 'a') as f:
                            f.write(f"{username}-sample{count}.wav\n")
                        
                        # Display waveform after recording
                        st.plotly_chart(plot_waveform(filename), use_container_width=True)
                        st.success(f"Sample {count + 1} recorded successfully!")
                    else:
                        st.error(f"Failed to record sample {count + 1}")
                except Exception as e:
                    st.error(f"Error during recording: {str(e)}")
                
                time.sleep(1)
            
            if successful_recordings == 5:
                # Train model for the user
                with st.spinner('Training voice model...'):
                    if train_model(username):  # Pass username to train_model
                        # Save user credentials
                        users[username] = {
                            "password": hash_password(password),
                            "model_file": f"{username}.gmm",
                            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "last_login": None
                        }
                        save_users(users)
                        st.markdown("""
                            <div class="success-message">
                                ✅ Registration completed successfully! You can now login.
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to train voice model. Please try registration again.")
            else:
                st.error("Registration failed. Please try again.")
            
            st.session_state.recording_state = "idle"

def train_model(username):
    try:
        source = "training_set"
        dest = "trained_models"
        features = np.asarray(())
        
        # Get the last 5 recordings for this specific user
        recordings = []
        with open("training_set_addition.txt", 'r') as file:
            all_recordings = file.readlines()
            for recording in all_recordings:
                if recording.strip().startswith(f"{username}-"):
                    recordings.append(recording.strip())
        recordings = recordings[-5:]  # Get last 5 recordings
        
        if len(recordings) != 5:
            st.error("Not enough recordings found for training")
            return False
        
        progress_bar = st.progress(0)
        total_steps = len(recordings)
        
        for idx, recording in enumerate(recordings):
            file_path = os.path.join(source, recording)
            if os.path.exists(file_path):
                sr, audio = read(file_path)
                vector = extract_features(audio, sr)
                
                if features.size == 0:
                    features = vector
                else:
                    features = np.vstack((features, vector))
                    
            progress_bar.progress((idx + 1) / total_steps)
        
        if features.size == 0:
            st.error("No features extracted from recordings")
            return False
            
        # Enhanced GMM parameters
        gmm = GaussianMixture(
            n_components=16,
            max_iter=300,
            covariance_type='full',
            n_init=5
        )
        
        # Fit the model
        gmm.fit(features)
        
        # Save the model
        model_file = os.path.join(dest, f"{username}.gmm")
        os.makedirs(dest, exist_ok=True)
        
        with open(model_file, 'wb') as f:
            pickle.dump(gmm, f)
            
        return True
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return False


def verify_speaker(username):
    verification_scores = []
    num_verification_samples = 2
    
    st.markdown("""
        <div class="status-card">
            🎤 Voice Verification Required
            <br>Please speak for verification...
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    for i in range(num_verification_samples):
        with col1:
            progress_bar = st.progress(0)
            filename = f"testing_set/verify_{i}.wav"
            
            if not record_audio(filename, progress_bar):
                continue
            
            # Show waveform and MFCC features
            with col2:
                tab1, tab2 = st.tabs(["Waveform", "MFCC Features"])
                with tab1:
                    st.plotly_chart(plot_waveform(filename), use_container_width=True)
                with tab2:
                    st.plotly_chart(plot_mfcc_features(filename), use_container_width=True)
            
        try:
            sr, audio = read(filename)
            vector = extract_features(audio, sr)
            
            # Load user's model
            model_path = f"trained_models/{username}.gmm"
            if not os.path.exists(model_path):
                st.error("Voice model not found!")
                return False
            
            speaker_model = pickle.load(open(model_path, 'rb'))
            scores = np.array(speaker_model.score(vector))
            score_sum = scores.sum()
            verification_scores.append(score_sum)
            
            # Save verification history
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "score": score_sum
            }
            
            history_file = f"user_data/{username}_history.json"
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(history_entry)
            with open(history_file, "w") as f:
                json.dump(history, f)
            
        except Exception as e:
            st.error(f"Verification error: {str(e)}")
            return False
    
    if not verification_scores:
        return False
    
    avg_score = np.mean(verification_scores)
    score_std = np.std(verification_scores)
    absolute_threshold = -30
    
    if avg_score > absolute_threshold:
        return True
    else:
        st.markdown(f"""
            <div class="error-message">
                ❌ Verification failed. Score: {avg_score:.2f}
            </div>
        """, unsafe_allow_html=True)
        return False

def login():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>🔐 Voice Authentication Login</h1>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if not username or not password:
                st.warning("Please enter both username and password")
                return
            
            users = load_users()
            if username not in users:
                st.session_state.login_attempts += 1
                st.error("Username not found!")
                return
            
            if users[username]["password"] != hash_password(password):
                st.session_state.login_attempts += 1
                st.error("Incorrect password!")
                return
            
            st.info("Password verified. Proceeding to voice verification...")
            
            if verify_speaker(username):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                
                # Update last login
                users[username]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_users(users)
                
                st.markdown("""
                    <div class="success-message">
                        ✅ Voice verified successfully! Redirecting...
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(2)
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= 3:
                    st.markdown("""
                        <div class="error-message">
                            ⚠️ Too many failed attempts. Please try again later.
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(5)
                    st.session_state.login_attempts = 0
def add_custom_styles():
    st.markdown("""
        <style>
        /* Dark theme colors and utility classes */
        :root {
            --bg-dark: #121212;
            --bg-darker: #0a0a0a;
            --bg-card: #1E1E1E;
            --text-primary: #E0E0E0;
            --text-secondary: #A0A0A0;
            --accent-green: #4CAF50;
            --accent-green-dark: #388E3C;
            --border-dark: #333333;
        }

        /* Header */
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: var(--bg-darker);
            border-bottom: 1px solid var(--border-dark);
            z-index: 1000;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        
        .nav-links {
            display: flex;
            gap: 1.5rem;
            align-items: center;
        }
        
        .nav-link {
            color: var(--text-primary);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            transition: all 0.2s ease;
            font-size: 0.95rem;
        }
        
        .nav-link:hover {
            background-color: var(--accent-green);
            color: #ffffff;
        }
        
        /* Footer */
        .footer {
            background-color: var(--bg-darker);
            padding: 2rem 0;
            margin-top: 4rem;
            border-top: 1px solid var(--border-dark);
            color: var(--text-secondary);
        }
        
        .footer-content {
            display: flex;
            justify-content: space-between;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .footer-section {
            margin-right: 4rem;
        }
        
        .footer-links {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 3rem;
        }
        
        .footer-heading {
            color: var(--accent-green);
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        .footer-link {
            color: var(--text-secondary);
            text-decoration: none;
            display: block;
            margin-bottom: 0.5rem;
            transition: color 0.2s ease;
        }
        
        .footer-link:hover {
            color: var(--accent-green);
        }
        
        /* Main content adjustments */
        .main-content {
            margin-top: 5rem;
            padding: 1.5rem;
            background-color: var(--bg-dark);
            min-height: calc(100vh - 5rem);
        }
        
        /* Card styling */
        .card {
            background-color: var(--bg-card);
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-dark);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Fix Streamlit's default padding */
        .stApp {
            margin-top: 60px;
            background-color: var(--bg-dark) !important;
        }
        
        /* Override Streamlit elements */
        .stButton > button {
            background-color: var(--accent-green) !important;
            color: white !important;
            border: none !important;
        }
        
        .stButton > button:hover {
            background-color: var(--accent-green-dark) !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--bg-card) !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 2rem;">
                    <h1 style="color: #4CAF50; font-size: 1.5rem; font-weight: bold;">QuizMaster 🎯</h1>
                    <div class="nav-links">
                        <a href="#" class="nav-link">Home</a>
                        <a href="#" class="nav-link">Quizzes</a>
                        <a href="#" class="nav-link">Leaderboard</a>
                        <a href="#" class="nav-link">Community</a>
                    </div>
                </div>
                <div class="nav-links">
                    <a href="#" class="nav-link">Profile</a>
                </div>
            </div>
        </div>
        <div class="main-content">
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        </div>
        <footer class="footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h3 class="footer-heading">QuizMaster</h3>
                    <p style="color: var(--text-secondary);">Learn, Quiz, Achieve</p>
                    <p style="color: var(--text-secondary); margin-top: 1rem;">
                        Join our global community of learners and unlock your full potential.
                    </p>
                </div>
                <div class="footer-links">
                    <div>
                        <h4 class="footer-heading">Quick Links</h4>
                        <a href="#" class="footer-link">About Us</a>
                        <a href="#" class="footer-link">Features</a>
                        <a href="#" class="footer-link">Community</a>
                        <a href="#" class="footer-link">Contact</a>
                    </div>
                    <div>
                        <h4 class="footer-heading">Resources</h4>
                        <a href="#" class="footer-link">Help Center</a>
                        <a href="#" class="footer-link">Privacy Policy</a>
                        <a href="#" class="footer-link">Terms of Use</a>
                    </div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 2rem; padding-top: 2rem; border-top: 1px solid var(--border-dark);">
                <p style="color: var(--text-secondary);">© 2024 QuizMaster. All rights reserved.</p>
            </div>
        </footer>
    """, unsafe_allow_html=True)
def show_dashboard_sidebar():
    """Move dashboard elements to sidebar"""
    st.sidebar.subheader("📊 Verification History")
    history_plot = plot_verification_history(st.session_state.current_user)
    if history_plot:
        st.sidebar.plotly_chart(history_plot, use_container_width=True)
    
    st.sidebar.subheader("🎤 Voice Profile")
    if st.sidebar.button("Update Voice Model"):
        st.session_state.recording_state = "recording"
        # Add voice model update logic here
        st.session_state.recording_state = "idle"

    st.sidebar.subheader("📝 Recent Activities")
    users = load_users()
    user_data = users[st.session_state.current_user]
    
    st.sidebar.markdown(f"""
        <div class="status-card">
            📅 Registration Date: {user_data['registration_date']}
            <br>🕒 Last Login: {user_data['last_login'] or 'First login'}
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.subheader("🔒 Security Status")
    st.sidebar.markdown("""
        <div class="status-card">
            ✅ Voice Authentication: Active<br>
            ✅ Last Verification: Successful<br>
            ✅ Model Status: Healthy
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Logout", key="logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.recording_state = "idle"
        st.rerun()

def show_homepage():
    # Add custom styles
    add_custom_styles()

    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #4CAF50; font-size: 2.5em;'>Welcome to QuizMaster 🎯</h1>
           
        </div>
    """, unsafe_allow_html=True)

    # Main content in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background-color: rgba(46, 125, 50, 0.1); padding: 2rem; border-radius: 10px; border-left: 5px solid #2E7D32;'>
                <h2 style='color: #4CAF50; margin-bottom: 1rem;'>About QuizMaster</h2>
                <p style='color: #E0E0E0; line-height: 1.6;'>
                    Welcome to QuizMaster, a comprehensive online platform designed to enhance learning through engaging and interactive quizzes. Offering a diverse range of topics—including science, history, technology, and pop culture—QuizMaster caters to students, professionals, and trivia enthusiasts alike.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Features section
        st.markdown("""
            <div style='background-color: rgba(46, 125, 50, 0.1); padding: 2rem; border-radius: 10px; border-left: 5px solid #2E7D32; margin-top: 1rem;'>
                <h3 style='color: #4CAF50; margin-bottom: 1rem;'>Key Features</h3>
                <ul style='color: #E0E0E0; list-style-type: none; padding-left: 0;'>
                    <li style='margin: 0.5rem 0;'>🎯 Multiple-choice, picture-based, and true-or-false quizzes</li>
                    <li style='margin: 0.5rem 0;'>🏆 Badges, leaderboards, and achievements</li>
                    <li style='margin: 0.5rem 0;'>📊 Progress tracking and analytics</li>
                    <li style='margin: 0.5rem 0;'>🔄 Custom quiz creation</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Stats and quick actions
        st.markdown("""
            <div style='background-color: rgba(46, 125, 50, 0.1); padding: 2rem; border-radius: 10px; border-left: 5px solid #2E7D32;'>
                <h3 style='color: #4CAF50; margin-bottom: 1rem;'>Getting Started</h3>
                <p style='color: #E0E0E0; line-height: 1.6;'>
                    Join our global community of learners and challenge yourself to grow both personally and professionally. Our platform provides:
                </p>
                <ul style='color: #E0E0E0; list-style-type: none; padding-left: 0;'>
                    <li style='margin: 0.5rem 0;'>🌟 Personalized learning paths</li>
                    <li style='margin: 0.5rem 0;'>🤝 Community challenges</li>
                    <li style='margin: 0.5rem 0;'>📱 Mobile-friendly interface</li>
                    <li style='margin: 0.5rem 0;'>🎨 Interactive learning experiences</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Quick Stats
        st.markdown("""
            <div style='background-color: rgba(46, 125, 50, 0.1); padding: 2rem; border-radius: 10px; border-left: 5px solid #2E7D32; margin-top: 1rem;'>
                <h3 style='color: #4CAF50; margin-bottom: 1rem;'>Platform Stats</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                    <div style='text-align: center; padding: 1rem; background-color: rgba(46, 125, 50, 0.2); border-radius: 5px;'>
                        <h4 style='color: #4CAF50; margin: 0;'>Active Users</h4>
                        <p style='color: #E0E0E0; font-size: 1.5em; margin: 0.5rem 0;'>10,000+</p>
                    </div>
                    <div style='text-align: center; padding: 1rem; background-color: rgba(46, 125, 50, 0.2); border-radius: 5px;'>
                        <h4 style='color: #4CAF50; margin: 0;'>Available Quizzes</h4>
                        <p style='color: #E0E0E0; font-size: 1.5em; margin: 0.5rem 0;'>500+</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        render_footer()

def main():
    # Your existing main function code remains the same until the authenticated check
    
    if st.session_state.authenticated:
        # Show dashboard in sidebar
        show_dashboard_sidebar()
        # Show homepage in main content area
        show_homepage()
    else:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login()
        with tab2:
            register_user()

if __name__ == "__main__":
    main()