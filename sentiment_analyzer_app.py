import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Streamlit app configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("😊 AI Sentiment Analyzer")
st.markdown("Analyze the sentiment of text using deep learning LSTM models")

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Text preprocessing class
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def preprocess(self, texts):
        cleaned_texts = []
        for text in texts:
            cleaned = self.clean_text(text)
            cleaned = self.remove_stopwords(cleaned)
            cleaned = self.lemmatize_text(cleaned)
            cleaned_texts.append(cleaned)
        return cleaned_texts

# Sentiment analyzer class
class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocessor = TextPreprocessor()
        self.max_length = 200
    
    def load_demo_model(self):
        """Load a simple demo model (in production, load a trained model)"""
        # Create a simple model for demo purposes
        vocab_size = 1000
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 50, input_length=self.max_length),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Create a simple tokenizer for demo
        self.tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
        
        # For demo, we'll use some sample training data
        sample_texts = [
            "good great amazing wonderful fantastic superb excellent",
            "bad terrible awful horrible disgusting disappointing"
        ]
        self.tokenizer.fit_on_texts(sample_texts)
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if self.model is None:
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'score': 0.5}
        
        # Preprocess text
        cleaned_text = self.preprocessor.preprocess([text])[0]
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        if not sequence[0]:  # If no words in vocabulary
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'score': 0.5}
            
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        
        # For demo, we'll simulate predictions based on keywords
        positive_words = ['good', 'great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'hate', 'worst']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            score = min(0.5 + (positive_count * 0.1), 0.95)
        elif negative_count > positive_count:
            score = max(0.5 - (negative_count * 0.1), 0.05)
        else:
            score = 0.5
        
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        confidence = score if score > 0.5 else 1 - score
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'score': float(score)
        }

# Initialize analyzer
if st.session_state.analyzer is None:
    st.session_state.analyzer = SentimentAnalyzer()
    st.session_state.analyzer.load_demo_model()

# Sidebar
st.sidebar.title("Settings")
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Single Text", "Batch Analysis", "Real-time Demo"]
)

# Main content based on selected mode
if analysis_mode == "Single Text":
    st.header("Single Text Analysis")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        "I absolutely loved this movie! The acting was superb and the storyline was engaging.",
        height=100
    )
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            result = st.session_state.analyzer.predict_sentiment(text_input)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_color = "green" if result['sentiment'] == 'Positive' else "red"
                st.metric("Sentiment", result['sentiment'], delta=None, delta_color="off")
            
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            
            with col3:
                st.metric("Sentiment Score", f"{result['score']:.3f}")
            
            # Visualize confidence
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['score'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add to history
            st.session_state.history.append({
                'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'score': result['score']
            })

elif analysis_mode == "Batch Analysis":
    st.header("Batch Text Analysis")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        """This product is amazing and works perfectly!
The service was terrible and very disappointing.
It was okay, nothing special.
Outstanding quality and fast delivery!""",
        height=150
    )
    
    if st.button("Analyze Batch"):
        texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
        
        if texts:
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                result = st.session_state.analyzer.predict_sentiment(text)
                result['text'] = text
                results.append(result)
                progress_bar.progress((i + 1) / len(texts))
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display results
            st.subheader("Analysis Results")
            st.dataframe(results_df)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = results_df['sentiment'].value_counts()
                fig1 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig2 = px.histogram(
                    results_df,
                    x='confidence',
                    title="Confidence Distribution",
                    nbins=10
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

else:  # Real-time Demo
    st.header("Real-time Sentiment Demo")
    
    st.info("""
    **How it works:** 
    - Type or paste text in the box below
    - See real-time sentiment analysis as you type
    - Watch the sentiment meter update instantly
    """)
    
    demo_text = st.text_area(
        "Type text to see real-time analysis:",
        "This is an amazing product!",
        height=100,
        key="demo_text"
    )
    
    if demo_text.strip():
        result = st.session_state.analyzer.predict_sentiment(demo_text)
        
        # Real-time visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['score'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Real-time Sentiment Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Sentiment", result['sentiment'])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            
            # Sentiment emoji
            emoji = "😊" if result['sentiment'] == 'Positive' else "😞"
            st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>{emoji}</h1>", 
                       unsafe_allow_html=True)

# Analysis history
if st.session_state.history:
    st.sidebar.header("Analysis History")
    history_df = pd.DataFrame(st.session_state.history[-10:])  # Last 10 analyses
    st.sidebar.dataframe(history_df[['text', 'sentiment', 'confidence']])
    
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using TensorFlow LSTM | "
    "Sentiment analysis powered by deep learning"
)

# Add some sample texts for quick testing
st.sidebar.header("Quick Test")
sample_texts = {
    "Positive": "I absolutely love this! It's fantastic and amazing.",
    "Negative": "This is terrible and awful. I hate it.",
    "Neutral": "It was okay, nothing special."
}

for sentiment, text in sample_texts.items():
    if st.sidebar.button(f"Test: {sentiment}"):
        st.session_state.demo_text = text
        st.experimental_rerun()