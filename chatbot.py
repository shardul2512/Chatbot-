import streamlit as st

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(
    page_title="TalentScout AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json
from datetime import datetime
from textblob import TextBlob
import plotly.express as px
import pandas as pd

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please make sure your GOOGLE_API_KEY is set correctly. Error: {e}")
    st.stop()

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        color: white;
        margin-right: 2rem;
    }
    
    .translate-btn {
        background: linear-gradient(45deg, #00b894, #00cec9);
        border: none;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .translate-btn:hover {
        transform: scale(1.05);
    }
    
    .sentiment-indicator {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .sentiment-positive {
        background-color: #00b894;
        color: white;
    }
    
    .sentiment-neutral {
        background-color: #fdcb6e;
        color: black;
    }
    
    .sentiment-negative {
        background-color: #e17055;
        color: white;
    }
    
    .info-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Language Support ---
LANGUAGES = {
    "English": "en",
    "Spanish": "es", 
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Arabic": "ar"
}

STAGE_QUESTIONS = {
    "awaiting_name": "Could you please tell me your full name?",
    "awaiting_email": "What's your email address?",
    "awaiting_phone": "Please provide your phone number.",
    "awaiting_experience": "How many years of total experience do you have?",
    "awaiting_position": "What position(s) are you interested in?",
    "awaiting_location": "What's your current location?",
    "awaiting_tech_stack": "Please list your complete tech stack (programming languages, frameworks, databases, tools)."
}

# --- Model and Prompts ---
# Initialize the Gemini Pro model
model = genai.GenerativeModel('gemini-1.5-flash')

# Define the core system prompt that sets the chatbot's persona and rules
SYSTEM_PROMPT = """
You are "HireAssist," a friendly, professional, and highly efficient AI hiring assistant for a recruitment agency named "TalentScout." Your primary purpose is to conduct a structured initial screening of candidates.

Your tasks are:
1. Greet the candidate warmly and briefly explain your role.
2. Collect the following information sequentially, one piece at a time:
   - Full Name
   - Email Address (must be a valid format)
   - Phone Number
   - Total Years of Experience
   - Desired Position(s)
   - Current Location
3. After gathering all details, ask for their complete tech stack (e.g., programming languages, frameworks, databases, tools).
4. Based on their declared tech stack, your *only* next step is to generate exactly 5 relevant technical questions. Present them in a numbered list.
5. After presenting the questions, conclude the conversation politely, thank the candidate, and inform them that the TalentScout team will review their profile and get in touch.

Core Rules:
- **Strictly Adhere to the Flow:** Do not deviate from the sequential information gathering and question generation process.
- **One Question at a Time:** Never ask for multiple pieces of information in a single message.
- **Stay On Topic:** If the user asks an unrelated question (e.g., 'What is the weather like?'), gently guide them back. Use a phrase like: "My focus is on our screening process. Shall we continue with the next question?"
- **Validate Email:** When you receive an email, confirm it's in a valid format. If not, politely ask again.
- **End Conversation Gracefully:** If the user uses a conversation-ending keyword like "bye," "exit," or "quit," respond with a polite closing message and stop.
- **Tone:** Maintain a friendly, encouraging, and professional tone throughout the interaction.
- **Technical Questions:** When generating technical questions based on the tech stack, ensure they are relevant, practical, and suitable for screening purposes. Focus on core concepts, problem-solving, and real-world applications.
"""

def get_initial_prompt():
    """Returns the initial system prompt combined with the start of the conversation."""
    return [
        {'role': 'user', 'parts': [SYSTEM_PROMPT]},
        {'role': 'model', 'parts': ["Hello! I'm HireAssist, an AI assistant from TalentScout. I'm here to help with the initial step of our screening process. To start, could you please tell me your full name?"]}
    ]

# --- Helper Functions ---
def is_valid_email(email):
    """Simple regex for email validation."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def analyze_sentiment(text):
    """Analyze sentiment of user input using TextBlob with enhanced negative detection."""
    try:
        # Enhanced negative keywords and phrases
        negative_indicators = [
            "no", "don't", "dont", "won't", "wont", "can't", "cant", "refuse", "never",
            "not interested", "i dont want", "i don't want", "i wont", "i won't",
            "no way", "absolutely not", "not telling", "not going to", "refuse to",
            "i refuse", "not sharing", "not provide", "not giving", "decline",
            "i decline", "not comfortable", "rather not", "prefer not to"
        ]
        
        # Enhanced positive keywords
        positive_indicators = [
            "yes", "sure", "of course", "definitely", "absolutely", "great", "excellent",
            "wonderful", "fantastic", "amazing", "love", "like", "enjoy", "happy",
            "excited", "thrilled", "perfect", "awesome", "brilliant", "outstanding"
        ]
        
        text_lower = text.lower().strip()
        
        # Check for explicit negative indicators first
        for indicator in negative_indicators:
            if indicator in text_lower:
                return "negative", -0.5
        
        # Check for explicit positive indicators
        for indicator in positive_indicators:
            if indicator in text_lower:
                return "positive", 0.5
        
        # Use TextBlob for more nuanced analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Adjust thresholds to be more sensitive
        if polarity > 0.05:
            return "positive", polarity
        elif polarity < -0.05:
            return "negative", polarity
        else:
            # For very short responses, check context
            if len(text.strip()) <= 3 and text_lower in ["no", "nah", "nope"]:
                return "negative", -0.3
            elif len(text.strip()) <= 3 and text_lower in ["yes", "yep", "yeah", "ok"]:
                return "positive", 0.3
            return "neutral", polarity
            
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return "neutral", 0.0

def translate_text(text, target_language):
    """Translate text using Gemini API."""
    try:
        translation_prompt = f"Translate the following text to {target_language}. Only provide the translation, no additional text: {text}"
        response = model.generate_content(translation_prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def get_personalized_response(user_info, base_response):
    """Generate personalized response based on user information."""
    if not user_info:
        return base_response
    
    personalization_elements = []
    
    if 'name' in user_info:
        personalization_elements.append(f"Hello {user_info['name'].split()[0]}")
    
    if 'experience' in user_info:
        exp = user_info['experience']
        if any(word in exp.lower() for word in ['senior', 'lead', '5+', 'experienced']):
            personalization_elements.append("I can see you have significant experience")
    
    if 'position' in user_info:
        position = user_info['position']
        if any(word in position.lower() for word in ['developer', 'engineer', 'programmer']):
            personalization_elements.append("Great to meet a fellow tech professional")
    
    if personalization_elements:
        return f"{', '.join(personalization_elements)}! {base_response}"
    
    return base_response

def save_user_session(user_info, sentiment_history):
    """Save user session data for future personalization."""
    session_data = {
        'user_info': user_info,
        'sentiment_history': sentiment_history,
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.get('session_id', 'unknown')
    }
    
    # In a real application, you would save this to a database
    # For demo purposes, we'll store in session state
    if 'user_sessions' not in st.session_state:
        st.session_state.user_sessions = []
    
    st.session_state.user_sessions.append(session_data)

def extract_questions_from_response(response_text):
    """Extract numbered questions from LLM response - ensure exactly 5 questions."""
    questions = []
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for numbered questions (1., 2., etc.)
        if re.match(r'^\d+\.', line):
            # Remove the number and clean up
            question = re.sub(r'^\d+\.\s*', '', line).strip()
            if question and len(questions) < 5:  # Ensure exactly 5 questions
                questions.append(question)
    
    # If we don't have exactly 5 questions, return empty list to regenerate
    if len(questions) != 5:
        return []
    
    return questions

def display_progress():
    """Display conversation progress."""
    if 'stage' not in st.session_state:
        return
        
    stages = ["name", "email", "phone", "experience", "position", "location", "tech_stack", "questions", "done"]
    current_stage = st.session_state.stage.replace("awaiting_", "").replace("asking_", "")
    
    if current_stage in stages:
        progress = (stages.index(current_stage) + 1) / len(stages)
        st.progress(progress)
        st.write(f"Progress: {int(progress * 100)}% Complete")

# --- Streamlit UI and State Management ---

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🤖 TalentScout AI Hiring Assistant</h1>
    <p>Advanced AI-powered candidate screening with multilingual support</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables FIRST
if 'messages' not in st.session_state:
    st.session_state.messages = get_initial_prompt()
if 'stage' not in st.session_state:
    st.session_state.stage = "awaiting_name"
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'technical_questions' not in st.session_state:
    st.session_state.technical_questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'technical_answers' not in st.session_state:
    st.session_state.technical_answers = []

# Sidebar for settings and analytics
with st.sidebar:
    st.header("🔧 Settings")
    
    # Language selection
    selected_language = st.selectbox(
        "Preferred Language",
        list(LANGUAGES.keys()),
        index=0
    )
    
    # Display progress
    st.header("📊 Progress")
    display_progress()
    
    # Sentiment tracking
    if 'sentiment_history' in st.session_state and st.session_state.sentiment_history:
        st.header("😊 Sentiment Analysis")
        sentiment_df = pd.DataFrame(st.session_state.sentiment_history)
        if not sentiment_df.empty:
            # Count sentiments by category
            sentiment_counts = sentiment_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            
            # Define colors for each sentiment
            color_map = {
                'positive': '#00b894',  # Green
                'neutral': '#fdcb6e',   # Yellow
                'negative': '#e17055'   # Red
            }
            
            fig = px.bar(sentiment_counts, x='sentiment', y='count', 
                         title='Sentiment Distribution', 
                         color='sentiment',
                         color_discrete_map=color_map)
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
    
    # User info display
    if 'user_info' in st.session_state and st.session_state.user_info:
        st.header("👤 Candidate Info")
        for key, value in st.session_state.user_info.items():
            if value:
                st.write(f"**{key.title()}:** {value}")

# Main chat container
chat_container = st.container()

with chat_container:
    # Display chat messages from history
    for i, message in enumerate(st.session_state.messages[1:], 1):  # Skip the system prompt
        with st.chat_message(message['role']):
            message_text = message['parts'][0]
            st.markdown(message_text)
            
            # Add translate button for bot messages
            if message['role'] == 'model' and selected_language != "English":
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button(f"🌐 Translate", key=f"translate_{i}"):
                        translated_text = translate_text(message_text, selected_language)
                        st.info(f"**{selected_language}:** {translated_text}")
                
                # Store current question for easy access
                if st.session_state.stage in STAGE_QUESTIONS:
                    st.session_state.current_question = message_text

# Main chat logic
if user_input := st.chat_input("Your response..."):
    # Analyze sentiment
    sentiment, polarity = analyze_sentiment(user_input)
    st.session_state.sentiment_history.append({
        'step': len(st.session_state.sentiment_history) + 1,
        'sentiment': sentiment,
        'polarity': polarity,
        'text': user_input[:50] + "..." if len(user_input) > 50 else user_input
    })
    
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'parts': [user_input]})
    with st.chat_message("user"):
        st.markdown(user_input)
        
        # Display sentiment indicator
        sentiment_class = f"sentiment-{sentiment}"
        sentiment_emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😔"
        st.markdown(f'<span class="sentiment-indicator {sentiment_class}">{sentiment_emoji} {sentiment.title()}</span>', 
                   unsafe_allow_html=True)

    # Check for conversation end keywords
    if any(keyword in user_input.lower() for keyword in ["bye", "exit", "quit"]):
        response_text = "Thank you for your time. Have a great day!"
        st.session_state.messages.append({'role': 'model', 'parts': [response_text]})
        with st.chat_message("model"):
            st.markdown(response_text)
        
        # Save session data
        save_user_session(st.session_state.user_info, st.session_state.sentiment_history)
        st.stop()

    # State machine for conversation flow
    current_stage = st.session_state.stage
    response_text = ""

    if current_stage == "awaiting_name":
        st.session_state.user_info['name'] = user_input
        st.session_state.stage = "awaiting_email"
        base_response = f"Nice to meet you, {user_input}! Now I need your email address."
        response_text = get_personalized_response(st.session_state.user_info, base_response)

    elif current_stage == "awaiting_email":
        if is_valid_email(user_input):
            st.session_state.user_info['email'] = user_input
            st.session_state.stage = "awaiting_phone"
            base_response = "Great! Now please provide your phone number."
            response_text = get_personalized_response(st.session_state.user_info, base_response)
        else:
            response_text = "The email provided doesn't seem to be in a valid format. Could you please provide a valid email address? (e.g., yourname@example.com)"

    elif current_stage == "awaiting_phone":
        st.session_state.user_info['phone'] = user_input
        st.session_state.stage = "awaiting_experience"
        base_response = "Perfect! Now, how many years of total professional experience do you have?"
        response_text = get_personalized_response(st.session_state.user_info, base_response)

    elif current_stage == "awaiting_experience":
        st.session_state.user_info['experience'] = user_input
        st.session_state.stage = "awaiting_position"
        base_response = f"Excellent! With {user_input} of experience, what position(s) are you most interested in?"
        response_text = get_personalized_response(st.session_state.user_info, base_response)

    elif current_stage == "awaiting_position":
        st.session_state.user_info['position'] = user_input
        st.session_state.stage = "awaiting_location"
        base_response = f"Wonderful! {user_input} is a great field. What's your current location?"
        response_text = get_personalized_response(st.session_state.user_info, base_response)

    elif current_stage == "awaiting_location":
        st.session_state.user_info['location'] = user_input
        st.session_state.stage = "awaiting_tech_stack"
        base_response = f"Thank you! Now for the technical part - could you please list your complete tech stack? Include programming languages, frameworks, databases, and tools you're proficient with."
        response_text = get_personalized_response(st.session_state.user_info, base_response)

    elif current_stage == "awaiting_tech_stack":
        st.session_state.user_info['tech_stack'] = user_input
        st.session_state.stage = "generating_questions"
        
        # Generate exactly 5 technical questions using the LLM
        try:
            question_prompt = f"""
            Based on this candidate's tech stack: {user_input}
            
            Generate exactly 5 relevant technical screening questions. The questions should be:
            1. Practical and job-relevant
            2. Appropriate for someone with {st.session_state.user_info.get('experience', 'some')} experience
            3. Cover different aspects of their tech stack
            4. Focused on problem-solving and real-world applications
            
            Format your response as:
            1. [First question]
            2. [Second question]
            3. [Third question]
            4. [Fourth question]
            5. [Fifth question]
            
            Only provide the numbered questions, nothing else.
            """
            
            chat_session = model.start_chat(history=[])
            llm_response = chat_session.send_message(question_prompt)
            
            # Extract questions from response
            questions = extract_questions_from_response(llm_response.text)
            
            # If we didn't get exactly 5 questions, try again with a more specific prompt
            if len(questions) != 5:
                fallback_prompt = f"""
                Create exactly 5 technical questions for a candidate with this tech stack: {user_input}
                
                Return only the questions in this exact format:
                1. Question about [main technology]
                2. Question about [framework/library]
                3. Question about [database/storage]
                4. Question about [problem-solving]
                5. Question about [best practices]
                """
                llm_response = chat_session.send_message(fallback_prompt)
                questions = extract_questions_from_response(llm_response.text)
            
            if len(questions) == 5:
                st.session_state.technical_questions = questions
                st.session_state.stage = "asking_questions"
                st.session_state.current_question_index = 0
                
                # Show all questions first, then start asking them one by one
                questions_text = "Perfect! Based on your tech stack, I have 5 technical questions for you:\n\n"
                for i, q in enumerate(questions, 1):
                    questions_text += f"{i}. {q}\n"
                questions_text += f"\nNow let's go through them one by one.\n\n**Question 1 of 5:** {questions[0]}"
                response_text = questions_text
            else:
                # Fallback if question generation fails
                st.session_state.stage = "done"
                response_text = "Thank you for providing your tech stack! The TalentScout team will review your profile and be in touch soon."
                
        except Exception as e:
            st.error(f"Error generating questions: {e}")
            st.session_state.stage = "done"
            response_text = "Thank you for your time! The TalentScout team will review your profile and be in touch soon."

    elif current_stage == "asking_questions":
        # Store the answer to the current question
        current_q_index = st.session_state.current_question_index
        if current_q_index < len(st.session_state.technical_questions):
            st.session_state.technical_answers.append({
                'question': st.session_state.technical_questions[current_q_index],
                'answer': user_input
            })
        
        # Move to next question or finish
        st.session_state.current_question_index += 1
        
        if st.session_state.current_question_index < len(st.session_state.technical_questions):
            # Ask next question
            next_question = st.session_state.technical_questions[st.session_state.current_question_index]
            question_num = st.session_state.current_question_index + 1
            response_text = f"Great answer! **Question {question_num} of 5:** {next_question}"
        else:
            # All questions answered, finish conversation
            st.session_state.stage = "done"
            base_response = "Excellent! You've completed all 5 technical questions. Thank you for your time and detailed responses. The TalentScout team will review your profile and technical answers, and we'll be in touch soon with the next steps. Have a great day!"
            response_text = get_personalized_response(st.session_state.user_info, base_response)

    # Add the response to chat history and display it
    if response_text:
        st.session_state.messages.append({'role': 'model', 'parts': [response_text]})
        with st.chat_message("model"):
            st.markdown(response_text)
            
            # Add translate button for the new response
            if selected_language != "English":
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("🌐 Translate", key=f"translate_new"):
                        translated_text = translate_text(response_text, selected_language)
                        st.info(f"**{selected_language}:** {translated_text}")

    # If the conversation is done, save session and disable input
    if st.session_state.stage == "done":
        save_user_session(st.session_state.user_info, st.session_state.sentiment_history)
        st.success("🎉 Screening completed successfully!")
        
        # Display summary
        with st.expander("📋 Interview Summary"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Candidate Information")
                for key, value in st.session_state.user_info.items():
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                st.subheader("💻 Technical Q&A")
                if st.session_state.technical_answers:
                    for i, qa in enumerate(st.session_state.technical_answers, 1):
                        st.write(f"**Q{i}:** {qa['question']}")
                        st.write(f"**A{i}:** {qa['answer']}")
                        st.write("---")
        
        st.chat_input("The conversation has ended.", disabled=True)
        st.stop()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🚀 Powered by TalentScout AI | Enhanced with Sentiment Analysis & Multilingual Support</p>
</div>
""", unsafe_allow_html=True)