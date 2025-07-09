import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Annotated, Dict, Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import elevenlabs, openai, silero

from rag_system import InsuranceRAG

logger = logging.getLogger("insurance-assistant")
logger.setLevel(logging.INFO)

load_dotenv(override=True)

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "bIHbv24MWmeRgasZH58o")  # Default voice
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insurance-policies")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INSURANCE_DATA_PATH = os.getenv("INSURANCE_DATA_PATH")

# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable is required")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")
if not INSURANCE_DATA_PATH:
    raise ValueError("INSURANCE_DATA_PATH environment variable is required")

# Primary needs options
PRIMARY_NEEDS = [
    "Critical Illness Coverage",
    "Maternity & Newborn Care", 
    "OPD (Outpatient) Coverage",
    "Family Health Coverage",
    "Senior Citizen Care",
    "Basic Hospitalization",
    "Comprehensive Coverage"
]

@dataclass
class UserProfile:
    """User profile for insurance recommendations"""
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    family_size: Optional[int] = None
    primary_need: Optional[str] = None
    budget_range: Optional[str] = None
    existing_conditions: Optional[str] = None
    profile_complete: bool = False

@dataclass
class InsuranceData:
    """Insurance agent session data"""
    user_profile: UserProfile = field(default_factory=UserProfile)
    conversation_stage: str = "greeting"  # greeting, data_collection, recommendation, q_and_a
    last_recommendation: Optional[str] = None
    rag_system: Optional[InsuranceRAG] = None
    
    agents: Dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        """Summarize current session state"""
        data = {
            "user_name": self.user_profile.name or "unknown",
            "user_age": self.user_profile.age or "unknown", 
            "conversation_stage": self.conversation_stage,
            "profile_complete": self.user_profile.profile_complete,
            "primary_need": self.user_profile.primary_need or "not selected"
        }
        return yaml.dump(data)

RunContext_T = RunContext[InsuranceData]

# Common functions for all agents

@function_tool()
async def collect_user_name(
    name: Annotated[str, Field(description="The user's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name."""
    insurance_data = context.userdata
    insurance_data.user_profile.name = name.strip().title()
    return f"Nice to meet you, {name}! I'm here to help you find the perfect health insurance policy."

@function_tool()
async def collect_user_age(
    age: Annotated[int, Field(description="The user's age in years")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their age."""
    insurance_data = context.userdata
    if 18 <= age <= 100:
        insurance_data.user_profile.age = age
        return f"Got it! You're {age} years old."
    else:
        return "Please provide a valid age between 18 and 100 years."

@function_tool()
async def collect_user_city(
    city: Annotated[str, Field(description="The user's city or location")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their city."""
    insurance_data = context.userdata
    insurance_data.user_profile.city = city.strip().title()
    return f"Thank you! I've noted that you're from {city}."

@function_tool()
async def collect_family_size(
    family_size: Annotated[int, Field(description="Number of family members to be covered")],
    context: RunContext_T,
) -> str:
    """Called when the user provides family size for coverage."""
    insurance_data = context.userdata
    if 1 <= family_size <= 10:
        insurance_data.user_profile.family_size = family_size
        if family_size == 1:
            return "I understand you need individual coverage."
        else:
            return f"I see you need coverage for {family_size} family members."
    else:
        return "Please provide a valid family size between 1 and 10 members."

@function_tool()
async def collect_primary_need(
    primary_need: Annotated[str, Field(description="User's primary insurance need")],
    context: RunContext_T,
) -> str:
    """Called when the user selects their primary insurance need."""
    insurance_data = context.userdata
    
    # Match user input to available options
    need_lower = primary_need.lower()
    matched_need = None
    
    for need in PRIMARY_NEEDS:
        if any(keyword in need_lower for keyword in need.lower().split()):
            matched_need = need
            break
    
    if matched_need:
        insurance_data.user_profile.primary_need = matched_need
        return f"Perfect! I've noted that {matched_need} is your primary requirement."
    else:
        return f"I understand your primary need is {primary_need}. Let me consider this in my recommendations."

@function_tool()
async def collect_budget_range(
    budget: Annotated[str, Field(description="User's budget range for insurance premium")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their budget range."""
    insurance_data = context.userdata
    insurance_data.user_profile.budget_range = budget
    return f"Thank you! I'll keep your budget preference of {budget} in mind."

@function_tool()
async def collect_existing_conditions(
    conditions: Annotated[str, Field(description="User's existing medical conditions")],
    context: RunContext_T,
) -> str:
    """Called when the user mentions existing medical conditions."""
    insurance_data = context.userdata
    insurance_data.user_profile.existing_conditions = conditions
    return "I've noted your medical history. This will help me recommend policies with appropriate waiting periods."

@function_tool()
async def generate_policy_recommendations(context: RunContext_T) -> str:
    """Generate personalized policy recommendations based on user profile."""
    insurance_data = context.userdata
    user_profile = insurance_data.user_profile
    
    # Mark profile as complete
    user_profile.profile_complete = True
    insurance_data.conversation_stage = "recommendation"
    
    # Create search query based on user profile
    query = f"health insurance policy recommendation for {user_profile.primary_need or 'comprehensive coverage'}"
    
    # Convert user profile to dict for RAG system
    profile_dict = {
        "name": user_profile.name,
        "age": user_profile.age,
        "city": user_profile.city,
        "family_size": user_profile.family_size,
        "primary_need": user_profile.primary_need,
        "budget_range": user_profile.budget_range,
        "existing_conditions": user_profile.existing_conditions
    }
    
    try:
        # Search for relevant policies
        matches = await insurance_data.rag_system.search_policies(query, profile_dict)
        
        # Generate recommendation
        recommendation = await insurance_data.rag_system.generate_recommendation(profile_dict, matches)
        
        insurance_data.last_recommendation = recommendation
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return "I apologize, but I'm having trouble accessing our policy database right now. Let me provide some general guidance based on your needs."

@function_tool()
async def answer_policy_question(
    question: Annotated[str, Field(description="User's question about insurance policies")],
    context: RunContext_T,
) -> str:
    """Answer specific questions about insurance policies."""
    insurance_data = context.userdata
    insurance_data.conversation_stage = "q_and_a"
    
    # Convert user profile to dict
    profile_dict = {
        "name": insurance_data.user_profile.name,
        "age": insurance_data.user_profile.age,
        "city": insurance_data.user_profile.city,
        "family_size": insurance_data.user_profile.family_size,
        "primary_need": insurance_data.user_profile.primary_need,
        "budget_range": insurance_data.user_profile.budget_range,
        "existing_conditions": insurance_data.user_profile.existing_conditions
    }
    
    try:
        answer = await insurance_data.rag_system.answer_question(question, profile_dict)
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return "I'm having trouble finding that information right now. Could you please rephrase your question?"

@function_tool()
async def to_data_collector(context: RunContext_T) -> tuple[Agent, str]:
    """Transfer to data collection agent."""
    curr_agent = context.session.current_agent
    return await curr_agent._transfer_to_agent("data_collector", context)

@function_tool()
async def to_advisor(context: RunContext_T) -> tuple[Agent, str]:
    """Transfer to policy advisor agent."""
    curr_agent = context.session.current_agent
    return await curr_agent._transfer_to_agent("advisor", context)

@function_tool()
async def to_greeter(context: RunContext_T) -> tuple[Agent, str]:
    """Transfer back to greeter agent."""
    curr_agent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)

class BaseAgent(Agent):
    """Base agent class with common functionality"""
    
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name} agent")

        insurance_data: InsuranceData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # Add previous agent's chat history
        if isinstance(insurance_data.prev_agent, Agent):
            truncated_chat_ctx = insurance_data.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=8)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # Add instructions with session context
        chat_ctx.add_message(
            role="system",
            content=f"You are {agent_name} agent. Current session: {insurance_data.summarize()}\n\nIMPORTANT: Be professional, helpful, and empathetic. Speak clearly and ask one question at a time. Keep responses concise and conversational.",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def on_llm_response(self, response) -> None:
        """Log LLM responses for debugging"""
        agent_name = self.__class__.__name__
        logger.info(f"🤖 {agent_name}: {response.content}")
        await super().on_llm_response(response)

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        insurance_data = context.userdata
        current_agent = context.session.current_agent
        next_agent = insurance_data.agents[name]
        insurance_data.prev_agent = current_agent
        return next_agent, ""

class GreeterAgent(BaseAgent):
    """Initial greeting and introduction agent"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly insurance assistant for ManipalCigna Health Insurance! Your role is to:\n"
                "1. Warmly greet new users and introduce yourself\n"
                "2. Explain that you help find the best health insurance policies\n"
                "3. Ask for their name to get started\n"
                "4. Transfer to data collection once they're ready\n\n"
                "Be warm, professional, and reassuring. Let them know you're here to help them find the perfect health insurance coverage."
            ),
            tools=[collect_user_name, to_data_collector],
            llm=openai.LLM(api_key=OPENAI_API_KEY, parallel_tool_calls=False),
            tts=elevenlabs.TTS(api_key=ELEVENLABS_API_KEY, voice_id=ELEVENLABS_VOICE_ID),
        )

class DataCollectorAgent(BaseAgent):
    """Agent responsible for collecting user information"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a data collection specialist for insurance recommendations. Your mission:\n\n"
                "COLLECT ESSENTIAL INFORMATION:\n"
                "1. User's age (required for policy eligibility)\n"
                "2. City/location (for network hospitals)\n"
                "3. Family size for coverage (how many people need coverage)\n"
                "4. Primary insurance need from these options:\n"
                f"   {', '.join(PRIMARY_NEEDS)}\n"
                "5. Budget range (optional but helpful)\n"
                "6. Any existing medical conditions (affects waiting periods)\n\n"
                "GUIDELINES:\n"
                "- Ask ONE question at a time\n"
                "- Be conversational and friendly\n"
                "- Explain why you need each piece of information\n"
                "- Don't rush - let them provide information naturally\n"
                "- Once you have the essential info (name, age, primary need), transfer to advisor\n"
                "- Always confirm what they've told you\n\n"
                "TRANSFER TO ADVISOR when you have enough information to make recommendations."
            ),
            tools=[
                collect_user_age, collect_user_city, collect_family_size, 
                collect_primary_need, collect_budget_range, collect_existing_conditions,
                to_advisor, to_greeter
            ],
            llm=openai.LLM(api_key=OPENAI_API_KEY, parallel_tool_calls=False),
            tts=elevenlabs.TTS(api_key=ELEVENLABS_API_KEY, voice_id=ELEVENLABS_VOICE_ID),
        )

class PolicyAdvisorAgent(BaseAgent):
    """Agent that provides policy recommendations and answers questions"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an expert insurance policy advisor! Your role:\n\n"
                "PROVIDE RECOMMENDATIONS:\n"
                "1. Generate personalized policy recommendations based on user profile\n"
                "2. Explain each recommended plan clearly\n"
                "3. Highlight benefits that match their primary needs\n"
                "4. Mention waiting periods and important considerations\n"
                "5. Compare different plans when asked\n\n"
                "ANSWER QUESTIONS:\n"
                "- Waiting periods for different conditions\n"
                "- Policy exclusions and limitations\n"
                "- Coverage details and benefits\n"
                "- Premium estimates\n"
                "- Claim processes\n\n"
                "COMMUNICATION STYLE:\n"
                "- Be expert yet approachable\n"
                "- Use simple language, avoid jargon\n"
                "- Provide specific, actionable advice\n"
                "- Always be honest about limitations\n"
                "- Encourage questions and clarifications\n\n"
                "Generate recommendations immediately when transferred to you if user profile is complete."
            ),
            tools=[
                generate_policy_recommendations, answer_policy_question,
                collect_budget_range, collect_existing_conditions,
                to_data_collector, to_greeter
            ],
            llm=openai.LLM(api_key=OPENAI_API_KEY, parallel_tool_calls=False),
            tts=elevenlabs.TTS(api_key=ELEVENLABS_API_KEY, voice_id=ELEVENLABS_VOICE_ID),
        )

    async def on_enter(self) -> None:
        await super().on_enter()
        
        # Check if we should generate recommendations immediately
        insurance_data: InsuranceData = self.session.userdata
        user_profile = insurance_data.user_profile
        
        # If we have essential information, generate recommendations
        if user_profile.name and user_profile.age and user_profile.primary_need:
            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(
                role="system",
                content="The user has provided their essential information. Generate policy recommendations immediately using the generate_policy_recommendations function."
            )
            await self.update_chat_ctx(chat_ctx)
            self.session.generate_reply(tool_choice="auto")

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the insurance agent"""
    await ctx.connect()

    # Initialize RAG system
    rag_system = InsuranceRAG(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index_name=PINECONE_INDEX_NAME,
        pinecone_environment=PINECONE_ENVIRONMENT
    )
    
    # Load insurance data if not already loaded
    try:
        await rag_system.load_insurance_data(INSURANCE_DATA_PATH)
        logger.info("Insurance data loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load insurance data: {e}")

    insurance_data = InsuranceData(rag_system=rag_system)
    insurance_data.agents.update({
        "greeter": GreeterAgent(),
        "data_collector": DataCollectorAgent(),
        "advisor": PolicyAdvisorAgent(),
    })
    
    session = AgentSession[InsuranceData](
        userdata=insurance_data,
        stt=elevenlabs.STT(api_key=ELEVENLABS_API_KEY),
        llm=openai.LLM(api_key=OPENAI_API_KEY),
        tts=elevenlabs.TTS(api_key=ELEVENLABS_API_KEY, voice_id=ELEVENLABS_VOICE_ID),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await session.start(
        agent=insurance_data.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

if __name__ == "__main__":
    # Support console mode
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        logging.basicConfig(level=logging.INFO)
        print("🏥 Starting Insurance Assistant in console mode...")
        print("Use 'console' command: python insurance_agent.py console")
        
        # Add console argument for LiveKit CLI
        sys.argv = [sys.argv[0], "console"]
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))