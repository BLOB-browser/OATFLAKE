from fastapi import APIRouter, HTTPException, Request, status
from scripts.models.schemas import SimpleQuestion, SimpleAnswer, AnswerRequest
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime
from scripts.services.question_generator import generate_questions, save_questions, get_generation_status, get_config_path  # Fix import

router = APIRouter(prefix="/api/questions", tags=["questions"])
logger = logging.getLogger(__name__)

def get_questions_path():
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        config = json.load(f)
    data_path = Path(config.get('data_path', ''))
    return data_path / "questions.csv", data_path / "answers.csv"

@router.get("", 
    responses={
        200: {"description": "List of all questions with their answers"},
        401: {"description": "Authentication required"},
        500: {"description": "Internal server error"}
    }
)
async def list_questions(request: Request, group_id: str = None):
    """List all questions, optionally filtered by group_id"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")

        questions_path, answers_path = get_questions_path()
        
        if not questions_path.exists():
            return {"questions": []}

        # Read questions
        questions_df = pd.read_csv(questions_path)
        
        # Filter by group_id if provided
        if group_id:
            # If we have a group_id column, filter by it
            if 'group_id' in questions_df.columns:
                questions_df = questions_df[questions_df['group_id'] == group_id]
            # If not, return all questions but log a warning
            else:
                logger.warning(f"Group filtering requested but 'group_id' column not found in questions.csv")
        
        # Read answers if they exist
        answers_df = pd.DataFrame()
        if answers_path.exists():
            answers_df = pd.read_csv(answers_path)

        # Format response
        questions_list = []
        for _, q in questions_df.iterrows():
            q_answers = []
            if not answers_df.empty:
                answers = answers_df[answers_df['question_id'] == q['question_id']]
                q_answers = [
                    {
                        "text": row['answer_text'],
                        "answered_by": row['answered_by'],
                        "created_at": row['answered_at']  # Add timestamp
                    }
                    for _, row in answers.iterrows()
                ]
            
            questions_list.append({
                "id": q['question_id'],
                "question": q['question_text'],
                "created_by": q['created_by'],
                "created_at": q['created_at'],  # Add timestamp
                "answers": q_answers,
                "group_id": q.get('group_id', None)  # Include group_id if it exists
            })

        return {"questions": questions_list}

    except Exception as e:
        logger.error(f"Error listing questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/answers",
    responses={
        200: {
            "description": "Answer created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "answer": {
                            "text": "Example answer text",
                            "answered_by": "user123"
                        }
                    }
                }
            }
        },
        401: {"description": "Authentication required"},
        404: {"description": "Question not found"},
        500: {"description": "Internal server error"}
    }
)
async def create_answer(answer_req: AnswerRequest, request: Request):
    """Create a new answer"""
    try:
        # Check authentication
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )

        user_id = request.state.user_id
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found"
            )

        questions_path, answers_path = get_questions_path()
        
        # Verify question exists
        if not questions_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
            
        questions_df = pd.read_csv(questions_path)
        if not any(questions_df['question_id'] == answer_req.question_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )

        # Get user info from request state
        user_id = request.state.user_id
        user_email = request.state.user_email if hasattr(request.state, 'user_email') else user_id

        # Create new answer with complete user info
        answer_data = {
            "answer_text": answer_req.answer,
            "answered_by": user_email,  # Use email for display
            "answered_by_id": user_id,  # Keep ID for reference
            "question_id": answer_req.question_id,
            "answered_at": datetime.now().isoformat()
        }

        # Save to CSV
        df = pd.DataFrame([answer_data])
        if answers_path.exists():
            existing_df = pd.read_csv(answers_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(answers_path, index=False)

        return {
            "success": True,
            "answer": {
                "text": answer_req.answer,
                "answered_by": user_email,
                "created_at": answer_data["answered_at"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/generation/status")
async def get_generation_status_endpoint(request: Request):
    """Get the status of automatic question generation"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")
            
        return get_generation_status()  # Fixed import usage
    except Exception as e:
        logger.error(f"Error getting generation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def trigger_generation(request: Request, num_questions: int = 10):
    """Manually trigger question generation with optional question count"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")
        
        logger.info(f"Manually triggering question generation for {num_questions} questions")
            
        questions = await generate_questions(num_questions=num_questions)
        if not questions:
            logger.warning("Question generation failed or timed out")
            return {
                "status": "error",
                "message": "Question generation failed. The process timed out or no relevant content was found.",
                "questions": []
            }
            
        save_result = await save_questions(questions)
        
        return {
            "status": "success",
            "message": f"Generated {len(questions)} new questions",
            "questions": questions,
            "saved": save_result
        }
    except Exception as e:
        logger.error(f"Error triggering question generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
