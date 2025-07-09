import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger("insurance-rag")

@dataclass
class PolicyMatch:
    """Represents a matched policy section with relevance score"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

class InsuranceRAG:
    """RAG system for insurance policy recommendations using Pinecone"""
    
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
        pinecone_environment: str = "gcp-starter"
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = pinecone_index_name
        
        # Connect to existing index or create new one
        try:
            self.index = self.pc.Index(pinecone_index_name)
            logger.info(f"Connected to existing Pinecone index: {pinecone_index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Token counter
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def load_insurance_data(self, file_path: str) -> None:
        """Load and process insurance data into Pinecone"""
        logger.info(f"Loading insurance data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Process the content to extract structured information
            processed_docs = self._process_insurance_document(content)
            
            # Create embeddings and upload to Pinecone
            await self._upload_to_pinecone(processed_docs)
            
            logger.info(f"Successfully loaded {len(processed_docs)} document chunks to Pinecone")
            
        except Exception as e:
            logger.error(f"Error loading insurance data: {e}")
            raise
    
    def _process_insurance_document(self, content: str) -> List[Document]:
        """Process insurance document into structured chunks"""
        
        # Split content into logical sections
        sections = self._extract_sections(content)
        
        documents = []
        for section_type, section_content in sections.items():
            # Further split large sections
            chunks = self.text_splitter.split_text(section_content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Filter out very small chunks
                    metadata = {
                        "section_type": section_type,
                        "chunk_index": i,
                        "source": "bajaj_prohealth_prime",
                        "content_type": self._classify_content_type(chunk)
                    }
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))
        
        return documents
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract logical sections from insurance document"""
        sections = {
            "plans_overview": "",
            "base_covers": "",
            "optional_packages": "",
            "optional_covers": "",
            "benefits_glance": "",
            "waiting_periods": "",
            "eligibility": "",
            "exclusions": "",
            "discounts": "",
            "wellness_program": ""
        }
        
        lines = content.split('\n')
        current_section = "plans_overview"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Classify line into sections based on keywords
            if any(keyword in line.lower() for keyword in ['optional packages', 'enhance', 'assure', 'freedom']):
                current_section = "optional_packages"
            elif any(keyword in line.lower() for keyword in ['optional covers', 'additional coverages']):
                current_section = "optional_covers"
            elif any(keyword in line.lower() for keyword in ['base covers', 'in-patient', 'pre-hospitalization']):
                current_section = "base_covers"
            elif any(keyword in line.lower() for keyword in ['benefits at a glance', 'type of cover']):
                current_section = "benefits_glance"
            elif any(keyword in line.lower() for keyword in ['waiting periods', 'initial waiting']):
                current_section = "waiting_periods"
            elif any(keyword in line.lower() for keyword in ['eligibility', 'entry age', 'sum insured option']):
                current_section = "eligibility"
            elif any(keyword in line.lower() for keyword in ['exclusions', 'we will not cover']):
                current_section = "exclusions"
            elif any(keyword in line.lower() for keyword in ['discounts', 'family discount', 'wellness discount']):
                current_section = "discounts"
            elif any(keyword in line.lower() for keyword in ['wellness program', 'healthy life management']):
                current_section = "wellness_program"
            
            sections[current_section] += line + "\n"
        
        return sections
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content type for better retrieval"""
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ['sum insured', 'premium', 'cost', 'price', 'rupees', 'lacs']):
            return "pricing"
        elif any(keyword in content_lower for keyword in ['waiting period', 'months', 'days']):
            return "waiting_periods"
        elif any(keyword in content_lower for keyword in ['maternity', 'pregnancy', 'delivery']):
            return "maternity"
        elif any(keyword in content_lower for keyword in ['critical illness', 'cancer', 'heart']):
            return "critical_illness"
        elif any(keyword in content_lower for keyword in ['opd', 'outpatient', 'consultation']):
            return "opd"
        elif any(keyword in content_lower for keyword in ['exclude', 'not cover', 'limitation']):
            return "exclusions"
        elif any(keyword in content_lower for keyword in ['bonus', 'discount', 'reward']):
            return "benefits"
        else:
            return "general"
    
    async def _upload_to_pinecone(self, documents: List[Document]) -> None:
        """Upload document embeddings to Pinecone"""
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Create embeddings
            texts = [doc.page_content for doc in batch]
            embeddings = await asyncio.to_thread(self.embeddings.embed_documents, texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                vector_id = f"doc_{i+j}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        **doc.metadata,
                        "text": doc.page_content[:1000]  # Truncate for metadata limits
                    }
                })
            
            # Upload to Pinecone
            await asyncio.to_thread(self.index.upsert, vectors=vectors)
            logger.info(f"Uploaded batch {i//batch_size + 1} of {(len(documents)-1)//batch_size + 1}")
    
    async def search_policies(
        self, 
        query: str, 
        user_profile: Dict[str, Any],
        top_k: int = 10
    ) -> List[PolicyMatch]:
        """Search for relevant policy information"""
        
        # Create enhanced query with user profile
        enhanced_query = self._create_enhanced_query(query, user_profile)
        
        # Create query embedding
        query_embedding = await asyncio.to_thread(
            self.embeddings.embed_query, enhanced_query
        )
        
        # Search Pinecone
        search_results = await asyncio.to_thread(
            self.index.query,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Convert to PolicyMatch objects
        matches = []
        for match in search_results.matches:
            matches.append(PolicyMatch(
                content=match.metadata.get('text', ''),
                metadata=match.metadata,
                score=match.score,
                source=match.metadata.get('source', 'unknown')
            ))
        
        return matches
    
    def _create_enhanced_query(self, query: str, user_profile: Dict[str, Any]) -> str:
        """Enhance query with user profile information"""
        enhancements = []
        
        if user_profile.get('age'):
            age = user_profile['age']
            if age < 30:
                enhancements.append("young adult coverage")
            elif age < 50:
                enhancements.append("mid-age comprehensive coverage")
            else:
                enhancements.append("senior citizen health insurance")
        
        if user_profile.get('primary_need'):
            enhancements.append(user_profile['primary_need'])
        
        if user_profile.get('family_size', 0) > 1:
            enhancements.append("family floater policy")
        
        enhanced_query = query
        if enhancements:
            enhanced_query += " " + " ".join(enhancements)
        
        return enhanced_query
    
    async def generate_recommendation(
        self, 
        user_profile: Dict[str, Any], 
        policy_matches: List[PolicyMatch]
    ) -> str:
        """Generate personalized policy recommendation"""
        
        # Prepare context from policy matches
        context = "\n\n".join([
            f"Policy Information (Score: {match.score:.2f}):\n{match.content}"
            for match in policy_matches[:5]  # Top 5 matches
        ])
        
        # Create recommendation prompt
        prompt = f"""
Based on the following user profile and policy information, provide a comprehensive insurance policy recommendation:

User Profile:
- Name: {user_profile.get('name', 'N/A')}
- Age: {user_profile.get('age', 'N/A')}
- City: {user_profile.get('city', 'N/A')}
- Family Size: {user_profile.get('family_size', 'N/A')}
- Primary Need: {user_profile.get('primary_need', 'N/A')}
- Budget Range: {user_profile.get('budget_range', 'N/A')}
- Existing Conditions: {user_profile.get('existing_conditions', 'None')}

Policy Information:
{context}

Please provide:
1. Top 2-3 recommended plans with reasons
2. Key features that match the user's needs
3. Waiting periods and important considerations
4. Estimated premium ranges
5. Any exclusions the user should be aware of

Keep the response conversational and easy to understand.
"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert insurance advisor helping customers choose the best health insurance policy."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "I apologize, but I'm having trouble generating a recommendation right now. Please try again."
    
    async def answer_question(
        self, 
        question: str, 
        user_profile: Dict[str, Any]
    ) -> str:
        """Answer specific questions about policies"""
        
        # Search for relevant information
        matches = await self.search_policies(question, user_profile, top_k=5)
        
        if not matches:
            return "I don't have specific information about that. Could you please rephrase your question or ask about our available plans?"
        
        # Create context for answering
        context = "\n\n".join([match.content for match in matches[:3]])
        
        prompt = f"""
Based on the following insurance policy information, answer the user's question accurately and concisely:

Question: {question}

Policy Information:
{context}

User Profile Context:
- Age: {user_profile.get('age', 'N/A')}
- Primary Need: {user_profile.get('primary_need', 'N/A')}

Provide a clear, helpful answer. If the information isn't available in the context, say so clearly.
"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert insurance advisor providing accurate information about health insurance policies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "I'm having trouble finding that information right now. Please try asking again or contact our support team."