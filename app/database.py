from typing import Dict, Any, List, Optional
import logging
import os
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.future import select
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use SQLite for simplicity - can be replaced with a more robust database
DATABASE_URL = "sqlite:///./data/rag_database.db"

# Create tables
Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with answers
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")

class Answer(Base):
    __tablename__ = "answers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_text = Column(Text, nullable=False)
    sources = Column(Text)  # Store as JSON string
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    feedback_score = Column(Float, nullable=True)  # For user feedback
    
    # Relationship with question
    question = relationship("Question", back_populates="answers")

# Create database engine
engine = create_engine(DATABASE_URL)

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Database:
    def __init__(self):
        """Initialize database connection"""
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self):
        """Get a database session"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    async def store_qa(self, question: str, answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a question and answer pair in the database
        
        Args:
            question: User's question
            answer: Generated answer with sources
            
        Returns:
            Stored question and answer data
        """
        # Create session
        session = self.SessionLocal()
        
        try:
            # Create question record
            db_question = Question(question_text=question)
            session.add(db_question)
            session.flush()  # Flush to get the ID
            
            # Create answer record
            sources_str = ",".join(answer.get("sources", []))
            db_answer = Answer(
                question_id=db_question.id,
                answer_text=answer.get("answer", ""),
                sources=sources_str
            )
            session.add(db_answer)
            
            # Commit transaction
            session.commit()
            
            # Return stored data
            return {
                "question_id": db_question.id,
                "question": question,
                "answer_id": db_answer.id,
                "answer": answer.get("answer", ""),
                "sources": answer.get("sources", [])
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing Q&A: {str(e)}")
            raise
        finally:
            session.close()
    
    async def get_qa_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent question-answer pairs
        
        Args:
            limit: Maximum number of QA pairs to return
            
        Returns:
            List of QA pairs
        """
        session = self.SessionLocal()
        
        try:
            # Query for questions with their answers
            questions = session.query(Question).order_by(Question.timestamp.desc()).limit(limit).all()
            
            result = []
            for q in questions:
                # Get the latest answer for each question
                latest_answer = session.query(Answer).filter(
                    Answer.question_id == q.id
                ).order_by(Answer.timestamp.desc()).first()
                
                if latest_answer:
                    result.append({
                        "question_id": q.id,
                        "question": q.question_text,
                        "answer_id": latest_answer.id,
                        "answer": latest_answer.answer_text,
                        "sources": latest_answer.sources.split(",") if latest_answer.sources else [],
                        "timestamp": q.timestamp.isoformat()
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error getting QA history: {str(e)}")
            return []
        finally:
            session.close()
    
    async def add_qa_pair(self, question: str, answer: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        Add a QA pair to improve the system (for the endpoint requirement)
        
        Args:
            question: Question text
            answer: Answer text
            sources: List of sources for the answer
            
        Returns:
            Added QA pair data
        """
        session = self.SessionLocal()
        
        try:
            # Create question record
            db_question = Question(question_text=question)
            session.add(db_question)
            session.flush()
            
            # Create answer record
            sources_str = ",".join(sources) if sources else ""
            db_answer = Answer(
                question_id=db_question.id,
                answer_text=answer,
                sources=sources_str
            )
            session.add(db_answer)
            
            # Commit transaction
            session.commit()
            
            return {
                "question_id": db_question.id,
                "question": question,
                "answer_id": db_answer.id,
                "answer": answer,
                "sources": sources or []
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding QA pair: {str(e)}")
            raise
        finally:
            session.close()
