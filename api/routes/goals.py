from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from utils.config import BACKEND_CONFIG  # Import BACKEND_CONFIG as used in other modules

router = APIRouter(prefix="/api/goals", tags=["goals"])
logger = logging.getLogger(__name__)

class GoalVote(BaseModel):
    """Model for goal voting requests"""
    goal_id: str
    vote_type: str  # "upvote" or "downvote"

def get_data_path():
    """Get data path from config using BACKEND_CONFIG"""
    try:
        return Path(BACKEND_CONFIG.get('data_path', ''))
    except Exception as e:
        logger.error(f"Error getting data path from BACKEND_CONFIG: {e}")
        
        # Fallback to reading from config.json
        try:
            config_path = Path("config.json")
            if not config_path.exists():
                logger.error("Config file not found.")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            return Path(config.get('data_path', ''))
        except Exception as e:
            logger.error(f"Error getting data path from file: {e}")
            return None

def get_goals_path():
    """Get the paths to the goals CSV and votes CSV"""
    data_path = get_data_path()
    if not data_path:
        logger.error("Could not determine data path")
        return None, None
    
    goals_path = data_path / "goals.csv"
    votes_path = data_path / "goal_votes.csv"
    
    # Log if the goals file exists or not
    if goals_path.exists():
        logger.info(f"Found goals.csv at {goals_path}")
    else:
        logger.warning(f"goals.csv not found at {goals_path}")
        
    return goals_path, votes_path

@router.get("/", 
    responses={
        200: {"description": "List of all goals with their votes"},
        500: {"description": "Internal server error"}
    }
)
async def list_goals(request: Request, topic: Optional[str] = None, group_id: Optional[str] = None):
    """List all goals, optionally filtered by topic. Group ID is accepted but not used."""
    try:
        # Authentication check removed - allow access without authentication
        logger.info("Goals API accessed without authentication check")

        # Log group_id for debugging but don't use it (we're using a single goals file)
        if group_id:
            logger.info(f"Group ID {group_id} provided but not used - using global goals")

        goals_path, votes_path = get_goals_path()
        
        if not goals_path or not goals_path.exists():
            return {"goals": []}

        # Read goals
        goals_df = pd.read_csv(goals_path)
        
        # Filter by topic if provided
        if topic:
            if 'topic' in goals_df.columns:
                # Make case-insensitive topic filter
                goals_df = goals_df[goals_df['topic'].str.lower() == topic.lower()]
            else:
                logger.warning("Topic filtering requested but 'topic' column not found in goals.csv")
        
        # Read votes if they exist
        votes_df = pd.DataFrame()
        if votes_path and votes_path.exists():
            votes_df = pd.read_csv(votes_path)

        # Format response
        goals_list = []
        for idx, goal in goals_df.iterrows():
            # Generate a unique ID for the goal if it doesn't have one
            goal_id = goal.get('goal_id', str(idx))
            
            # Get votes for this goal
            upvotes = 0
            downvotes = 0
            
            if not votes_df.empty and 'goal_id' in votes_df.columns:
                goal_votes = votes_df[votes_df['goal_id'] == goal_id]
                upvotes = len(goal_votes[goal_votes['vote_type'] == 'upvote'])
                downvotes = len(goal_votes[goal_votes['vote_type'] == 'downvote'])
            
            # Format the goal data
            goal_data = {
                "id": goal_id,
                "text": goal['goal_text'],
                "importance": float(goal['importance']) if 'importance' in goal else 5.0,
                "topic": goal.get('topic', 'General'),
                "justification": goal.get('justification', ''),
                "store_type": goal.get('store_type', 'unknown'),
                "extracted_at": goal.get('extracted_at', ''),
                "votes": {
                    "upvotes": upvotes,
                    "downvotes": downvotes,
                    "score": upvotes - downvotes
                }
            }
            goals_list.append(goal_data)
        
        # Sort by vote score (descending) and then by importance (descending)
        goals_list.sort(key=lambda x: (x['votes']['score'], x['importance']), reverse=True)

        return {"goals": goals_list}

    except Exception as e:
        logger.error(f"Error listing goals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vote",
    responses={
        200: {"description": "Vote recorded successfully"},
        404: {"description": "Goal not found"},
        500: {"description": "Internal server error"}
    }
)
async def vote_on_goal(vote: GoalVote, request: Request):
    """Record a vote (upvote or downvote) for a goal"""
    try:
        # Authentication check removed - allow voting without authentication
        logger.info("Goals vote API accessed without authentication check")

        # Get user_id from request state or body, or use a default
        user_id = None
        try:
            # Try to get from request state
            user_id = request.state.user_id
        except AttributeError:
            # If not in state, check if it was sent in the request body
            try:
                body = await request.json()
                user_id = body.get("user_id")
            except:
                pass
        
        # If still no user_id, use a default
        if not user_id:
            logger.warning("No user_id found in request, using 'anonymous' as default")
            user_id = "anonymous"

        # Validate vote type
        if vote.vote_type not in ["upvote", "downvote"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vote type must be 'upvote' or 'downvote'"
            )

        goals_path, votes_path = get_goals_path()
        
        # Verify goal exists
        if not goals_path or not goals_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Goals database not found"
            )
            
        goals_df = pd.read_csv(goals_path)
        
        # Add goal_id column if it doesn't exist
        if 'goal_id' not in goals_df.columns:
            goals_df['goal_id'] = [str(i) for i in range(len(goals_df))]
            goals_df.to_csv(goals_path, index=False)
            logger.info("Added goal_id column to goals.csv")
        
        # Check if the goal exists
        if not any(goals_df['goal_id'] == vote.goal_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Goal not found"
            )

        # Get user info - use default email if not available
        user_email = None
        try:
            user_email = request.state.user_email
        except AttributeError:
            user_email = f"{user_id}@anonymous.user"

        # Create new vote data
        vote_data = {
            "goal_id": vote.goal_id,
            "vote_type": vote.vote_type,
            "voted_by": user_email,
            "voted_by_id": user_id,
            "voted_at": datetime.now().isoformat()
        }

        # Create or update votes CSV
        if votes_path and votes_path.exists():
            votes_df = pd.read_csv(votes_path)
            
            # Check if the user already voted for this goal
            user_previous_votes = votes_df[(votes_df['goal_id'] == vote.goal_id) & 
                                         (votes_df['voted_by_id'] == user_id)]
            
            if not user_previous_votes.empty:
                # Update the existing vote
                votes_df = votes_df.drop(user_previous_votes.index)
                logger.info(f"Removed previous vote by user {user_id} for goal {vote.goal_id}")
                
            # Add the new vote
            votes_df = pd.concat([votes_df, pd.DataFrame([vote_data])], ignore_index=True)
        else:
            # Create new votes file
            votes_df = pd.DataFrame([vote_data])
        
        # Save to CSV
        if votes_path:
            votes_df.to_csv(votes_path, index=False)
            logger.info(f"Saved vote from {user_id} for goal {vote.goal_id}: {vote.vote_type}")

        # Get the current vote counts for the goal
        goal_votes = votes_df[votes_df['goal_id'] == vote.goal_id]
        upvotes = len(goal_votes[goal_votes['vote_type'] == 'upvote'])
        downvotes = len(goal_votes[goal_votes['vote_type'] == 'downvote'])

        return {
            "success": True,
            "message": f"Successfully {vote.vote_type}d the goal",
            "votes": {
                "upvotes": upvotes,
                "downvotes": downvotes,
                "score": upvotes - downvotes
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording vote: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/topics",
    responses={
        200: {"description": "List of all topics in the goals database"},
        500: {"description": "Internal server error"}
    }
)
async def list_topics(request: Request):
    """Get a list of all topics from the goals database"""
    try:
        # Authentication check removed - allow access without authentication
        logger.info("Goals topics API accessed without authentication check")

        goals_path, _ = get_goals_path()
        
        if not goals_path or not goals_path.exists():
            return {"topics": []}

        # Read goals
        goals_df = pd.read_csv(goals_path)
        
        # Extract topics if the column exists
        topics = []
        if 'topic' in goals_df.columns:
            # Get unique topics and count goals per topic
            topic_counts = goals_df['topic'].value_counts().reset_index()
            topic_counts.columns = ['topic', 'count']
            
            # Convert to list of dicts
            topics = [
                {"name": row['topic'], "count": int(row['count'])}
                for _, row in topic_counts.iterrows()
            ]
            
            # Sort by count descending
            topics.sort(key=lambda x: x['count'], reverse=True)
        
        return {"topics": topics}

    except Exception as e:
        logger.error(f"Error listing topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug", 
    responses={
        200: {"description": "Debug information about goals files"}
    }
)
async def goals_debug_info():
    """Get debug information about goals files"""
    goals_path, votes_path = get_goals_path()
    data_path = get_data_path()
    
    # Check if the files exist
    goals_exists = goals_path.exists() if goals_path else False
    votes_exists = votes_path.exists() if votes_path else False
    
    # Check what's in the data directory
    data_files = []
    if data_path and data_path.exists():
        data_files = [str(f.relative_to(data_path)) for f in data_path.glob("*.csv")]
    
    return {
        "data_path": str(data_path) if data_path else None,
        "goals_path": str(goals_path) if goals_path else None,
        "votes_path": str(votes_path) if votes_path else None,
        "goals_exists": goals_exists,
        "votes_exists": votes_exists,
        "data_directory_files": data_files
    }
