# Insurance RAG Voice Assistant 🏥

A comprehensive voice-based insurance consultation system powered by LiveKit, RAG (Retrieval-Augmented Generation), and multi-agent architecture. The assistant helps users find the best health insurance policies based on their personal requirements using natural voice conversations.

## 🎯 Features

- **Voice-First Experience**: Natural voice conversations using ElevenLabs TTS/STT
- **RAG-Powered Recommendations**: Pinecone vector database with ManipalCigna policy data
- **Multi-Agent System**: Specialized agents for greeting, data collection, and policy advice
- **Intelligent Data Collection**: Collects only essential user information
- **Personalized Recommendations**: AI-powered policy matching based on user profile
- **Console Mode**: Easy testing with `python insurance_agent.py console`

## 🏗️ Architecture

```
User Voice Input → LiveKit → Multi-Agent System → RAG System → Pinecone DB
                                    ↓
              ElevenLabs TTS ← OpenAI LLM ← Policy Recommendations
```

### Agent Workflow:
1. **Greeter Agent**: Welcome and introduction
2. **Data Collector Agent**: Gather essential user information
3. **Policy Advisor Agent**: Generate recommendations and answer questions

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file:
```bash
cp .env.example .env
```

Fill in your API keys:
```env
# LiveKit Configuration
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret  
LIVEKIT_URL=wss://your-project-name.livekit.cloud

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=your-preferred-voice-id

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=insurance-policies
PINECONE_ENVIRONMENT=your-pinecone-environment

# Data Configuration  
INSURANCE_DATA_PATH=Your Insurance file
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Database

Initialize Pinecone with insurance data:
```bash
python setup_database.py
```

### 4. Run the System

**Console Mode (Recommended for testing):**
```bash
python insurance_agent.py console
```

**Full System with Token Server:**
```bash
./run_insurance_system.sh
```

## 📊 Data Collection

The assistant collects essential information for accurate recommendations:

### Required Information:
- **Name**: For personalization
- **Age**: Policy eligibility and pricing
- **Primary Need**: Core requirement selection

### Primary Need Options:
- Critical Illness Coverage
- Maternity & Newborn Care
- OPD (Outpatient) Coverage  
- Family Health Coverage
- Senior Citizen Care
- Basic Hospitalization
- Comprehensive Coverage

### Optional Information:
- City/Location (for network hospitals)
- Family Size (individual vs family coverage)
- Budget Range (premium preferences)
- Existing Medical Conditions (waiting period considerations)

## 🗣️ Example Conversations

### Getting Started:
**User**: "Hello, I need health insurance"
**Agent**: "Hello! I'm your insurance assistant. I'll help you find the perfect health insurance policy. May I have your name?"

### Data Collection:
**User**: "My name is John"
**Agent**: "Nice to meet you, John! To recommend the best policy, could you tell me your age?"
**User**: "I'm 32 years old"
**Agent**: "Got it! What's your primary insurance need? Are you looking for critical illness coverage, maternity care, OPD coverage, or comprehensive family coverage?"

### Policy Questions:
**User**: "What's the waiting period for pre-existing diseases?"
**Agent**: "For pre-existing diseases, the waiting period varies by sum insured. For policies up to ₹5 lakhs, it's 36 months. For ₹7.5 lakhs and above, it's 24 months."

## 🔧 System Components

### 1. RAG System (`rag_system.py`)
- Pinecone vector database integration
- Document chunking and embedding
- Semantic search for policy information
- Context-aware response generation

### 2. Insurance Agent (`insurance_agent.py`)  
- Multi-agent architecture
- Voice conversation handling
- User profile management
- Policy recommendation logic

### 3. Database Setup (`setup_database.py`)
- Pinecone index creation
- Insurance data processing
- Vector embedding generation
- Search functionality testing

### 4. Token Server (`token_server.py`)
- LiveKit authentication
- Session management
- CORS configuration


## 🛠️ Troubleshooting

### Common Issues:

**1. Database Not Found**
```bash
❌ Pinecone index insurance-policies not found!
```
**Solution**: Run `python setup_database.py`

**2. API Key Errors**
```bash
❌ OPENAI_API_KEY environment variable is required
```
**Solution**: Check your `.env` file has all required API keys

**3. Voice Issues**
- Ensure microphone permissions are granted
- Check ElevenLabs API key and voice ID
- Verify audio input/output devices

**4. Connection Issues**
```bash
❌ Cannot connect to token server
```
**Solution**: 
- Check LiveKit credentials in `.env`
- Ensure token server is running on port 8000

### Debug Mode:

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Requirements

### Required Services:
1. **OpenAI**: GPT-4 for conversation and recommendations
2. **ElevenLabs**: Voice synthesis and speech recognition  
3. **Pinecone**: Vector database for RAG
4. **LiveKit**: Real-time voice communication

### Recommended Voice Settings:
- **Voice**: Professional, clear English voice
- **Speed**: 0.9x to 1.1x (natural pace)
- **Stability**: High for consistent output

## 📄 License

This project is for demonstration purposes. Insurance policy data belongs to ManipalCigna Health Insurance Company Limited.

---

**Happy Consulting! 🏥✨**
