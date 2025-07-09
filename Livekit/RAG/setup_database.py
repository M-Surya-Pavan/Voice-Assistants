#!/usr/bin/env python3
"""
Data processing script to initialize Pinecone database with insurance data
Usage: python setup_database.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from rag_system import InsuranceRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("database-setup")

# Load environment variables
load_dotenv(override=True)

# Get configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insurance-policies")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INSURANCE_DATA_PATH = os.getenv("INSURANCE_DATA_PATH")

async def create_pinecone_index():
    """Create Pinecone index if it doesn't exist"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [index.name for index in existing_indexes]
    
    if PINECONE_INDEX_NAME in index_names:
        logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists")
        return
    
    logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Create index with appropriate configuration
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # OpenAI ada-002 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    logger.info(f"Successfully created index: {PINECONE_INDEX_NAME}")

async def load_insurance_data():
    """Load and process insurance data into Pinecone"""
    logger.info("Initializing RAG system...")
    
    # Initialize RAG system
    rag_system = InsuranceRAG(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index_name=PINECONE_INDEX_NAME,
        pinecone_environment=PINECONE_ENVIRONMENT
    )
    
    # Load insurance data
    logger.info(f"Loading insurance data from: {INSURANCE_DATA_PATH}")
    await rag_system.load_insurance_data(INSURANCE_DATA_PATH)
    
    logger.info("✅ Insurance data loaded successfully!")

async def test_search():
    """Test the search functionality"""
    logger.info("Testing search functionality...")
    
    rag_system = InsuranceRAG(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index_name=PINECONE_INDEX_NAME,
        pinecone_environment=PINECONE_ENVIRONMENT
    )
    
    # Test search
    test_profile = {
        "age": 30,
        "primary_need": "Critical Illness Coverage",
        "family_size": 2
    }
    
    matches = await rag_system.search_policies(
        "critical illness coverage policy", 
        test_profile, 
        top_k=3
    )
    
    logger.info(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches, 1):
        logger.info(f"{i}. Score: {match.score:.3f} | Type: {match.metadata.get('content_type', 'unknown')}")
        logger.info(f"   Content: {match.content[:100]}...")
    
    logger.info("✅ Search test completed!")

async def main():
    """Main setup function"""
    print("🏥 Insurance Database Setup")
    print("=" * 40)
    
    # Validate environment variables
    if not all([OPENAI_API_KEY, PINECONE_API_KEY, INSURANCE_DATA_PATH]):
        print("❌ Missing required environment variables:")
        if not OPENAI_API_KEY:
            print("   - OPENAI_API_KEY")
        if not PINECONE_API_KEY:
            print("   - PINECONE_API_KEY")
        if not INSURANCE_DATA_PATH:
            print("   - INSURANCE_DATA_PATH")
        print("\nPlease check your .env file and try again.")
        return
    
    # Check if data file exists
    if not os.path.exists(INSURANCE_DATA_PATH):
        print(f"❌ Insurance data file not found: {INSURANCE_DATA_PATH}")
        return
    
    try:
        print("🔧 Step 1: Creating Pinecone index...")
        await create_pinecone_index()
        
        print("📚 Step 2: Loading insurance data...")
        await load_insurance_data()
        
        print("🔍 Step 3: Testing search functionality...")
        await test_search()
        
        print("\n✅ Database setup completed successfully!")
        print(f"📊 Index name: {PINECONE_INDEX_NAME}")
        print(f"📁 Data source: {INSURANCE_DATA_PATH}")
        print("\n🚀 You can now run the insurance agent:")
        print("   python insurance_agent.py console")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("Please check the logs above for more details.")

if __name__ == "__main__":
    asyncio.run(main())