from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional
from fastapi import Request, HTTPException, status, Depends
from app.database import SessionLocal, Talent
from app.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, BASE_URL, BASE_URL_BACKEND
from datetime import datetime, date
import json
import time
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

class RateLimiter:
    def __init__(self, limit: int, window: int, error_msg: str):
        self.limit = limit
        self.window = window
        self.error_msg = error_msg
        self.clients = defaultdict(list)

    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < self.window]
        if len(self.clients[client_ip]) >= self.limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=self.error_msg)
        self.clients[client_ip].append(now)
        
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    filters: Dict[str, Any] = {}

class GeneratedJobInfo(BaseModel):
    title: str = Field(..., description="A concise and catchy title for the casting job.")
    description: str = Field(..., description="A one-line summary of the job description.")
    casting_roles: str = Field(..., description="A description of the specific roles or characters needed for this job.")

class ExtractedFilters(BaseModel):
    location: Optional[str] = Field(None, description="Location")
    continent: Optional[str] = Field(None, description="Continent")
    country: Optional[str] = Field(None, description="Country")
    title: Optional[str] = Field(None, description="A concise title for the casting call.")
    description: Optional[str] = Field(None, description="A summary description of the job.")
    casting_roles: Optional[str] = Field(None, description="Descriptions of the specific roles, characters, or parts needed.")
    gender: Optional[str] = Field(None, description="Gender")
    hair_color: Optional[str] = Field(None, description="Hair Color")
    eye_color: Optional[str] = Field(None, description="Eye Color")
    skin_color: Optional[str] = Field(None, description="Skin Color")
    shoot_date: Optional[List[str]] = Field(None, description="Shoot Dates")
    hair_type: Optional[str] = Field(None, description="Hair Type")
    height: Optional[str] = Field(None, description="Height")
    bust: Optional[str] = Field(None, description="Bust")
    waist: Optional[str] = Field(None, description="Waist")
    hips: Optional[str] = Field(None, description="Hips")
    shoe_size: Optional[str] = Field(None, description="Shoe Size")
    dress_size: Optional[str] = Field(None, description="Dress Size")
    skills: Optional[str] = Field(None, description="Specific skills or talents (e.g. dancing, singing, martial arts, swimming).")
    budget: Optional[str] = Field(None, description="Budget")
    job_type: Optional[str] = Field(None, description="The category of the production (e.g. film, TV, commercial, theater, voiceover, modeling).")
    role: Optional[str] = Field(None, description="The specific role for the talent (e.g. model, actor, singer).")
    limit: Optional[int] = Field(None, description="The specific number of talent results requested by the user.")

def extract_information(user_input: str, current_filters: Dict[str, Any], messages: List[BaseMessage] = []) -> Dict[str, Any]:
    """Extracts casting filters from user input using a lightweight LLM call."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(ExtractedFilters)

    history = ""
    # Provide the last few messages for context, excluding the current input to avoid redundancy
    relevant_history = [m for m in messages if getattr(m, 'content', None) != user_input][-5:]
    for m in relevant_history:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history += f"{role}: {m.content}\n"
    
    prompt = f"""
    You are a Casting Assistant. Extract casting requirements from the user's latest message using the conversation history for context.
    Current date: {date.today()}
    Recent conversation:
    {history}

    Current known info: {current_filters}
    User message: "{user_input}"
    
    The mandatory fields are: location, shoot_date, budget, job_type, gender, skin_color.
    If the user input seems to be answering a question about 'job_type' (e.g. 'modeling', 'commercial'), ensure it is mapped to the 'job_type' field.
    If the user mentions a specific number of talents they want to see (e.g., 'show me 5 talents' or 'find 3 models'), extract this into the 'limit' field.
    If multiple genders are mentioned (e.g., "men and women"), extract them as a comma-separated string in the 'gender' field (e.g., "male, female").
    If the user mentions "both" or "all" genders, use "male, female".
    If the user mentions specific roles or character names (e.g., "Seeking a lead dancer and two background actors"), extract this into 'casting_roles'.

    IMPORTANT: 
    1. All 'shoot_date' values MUST be converted to YYYY-MM-DD format. Use the Current Date ({date.today()}) as context to infer the year if it is missing (e.g., if the user says '24 May', convert it to '2026-05-24' assuming the context of the upcoming date).
    2. If the user indicates a field should be cleared or they want "any" (e.g., "any date", "ignore location"), you MUST return that field as null.
    3. If a field is NOT mentioned or the user is just saying "proceed", "yes", "search", "ok", or "hi", do NOT return that field at all (leave it unset).

    Return ONLY the fields that are explicitly mentioned or updated in the user message.
    """
    try:
        result = structured_llm.invoke(prompt)
        # Filter out None values to prevent accidental state erosion during "proceed" commands
        updates = result.model_dump(exclude_unset=True) if hasattr(result, 'model_dump') else result.dict(exclude_unset=True)
        return {k: v for k, v in updates.items() if v not in [None, ""]}
    except Exception as e:
        return {}

def generate_job_details_from_messages(messages: List[str]) -> GeneratedJobInfo:
    """Generates a job title and description from a list of messages."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(GeneratedJobInfo)
    
    conversation_summary = "\n".join(messages)
    
    prompt = f"""
    Based on the following conversation snippets, generate a concise job title and a one-line job description for a casting call.
    Also, generate a 'casting_roles' section describing the specific roles or types of characters being sought.

    Conversation:
    ---
    {conversation_summary}
    ---

    Example Output:
    Title: Runway Model Show
    Description: Seeking experienced runway models for a high-fashion event in Los Angeles.
    Casting Roles: Main runway models and backup talent for a 3-day fashion week activation.

    Generate the title, description, and casting roles.
    NOTE: Never add any physical or facial features in Title, Description, or Casting Roles. You may add gender or job type. 
    """
    
    try:
        result = structured_llm.invoke(prompt)
        return result
    except Exception as e:
        return GeneratedJobInfo(title="Casting Call", description="Casting for a new project.", casting_roles="Casting for specific project roles.")

def generate_ask_response(missing_fields: List[str], user_input: str, is_initial: bool = False) -> str:
    """Generates a polite response to a greeting or asks for missing mandatory fields."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.6, api_key=OPENAI_API_KEY)

    missing_str = ", ".join(f.replace('_', ' ').title() for f in missing_fields)

    prompt = f"""
    You are a Casting Assistant helping a user find talent. 
    User message: "{user_input}"
    Missing criteria: {missing_str}.
    Initial or greeting message: {is_initial}

    Instructions:
    - If 'Initial or greeting message' is True, your response MUST start with: "Hi. To find talents, you need to provide mandatory fields ({missing_str}) and add additional features (like hair color, height etc.) to refine the search."
    - If 'Initial or greeting message' is False, do NOT include the preamble sentence mentioned above. Simply acknowledge the input and ask for: {missing_fields[0]}.

    Keep the response professional and very concise.
    """
    return llm.invoke(prompt).content

@tool
def generate_casting(location: str = None, continent: str = None, country: str = None,
                     gender: str = None, hair_color: str = None, eye_color: str = None, skin_color: str = None, skills: str = None,
                     shoot_date: List[str] = None, role: str = None,
                     height: str = None, bust: str = None, waist: str = None, hips: str = None, 
                     shoe_size: str = None, dress_size: str = None, limit: int = 100):
    """
    Search for talent. Returns ranked recommendations based on matched criteria.
    'limit' is the maximum number of results to return (defaults to 100).
    """
    db = SessionLocal()
    try:
        talents = db.query(Talent).outerjoin(Talent.agent).filter(
            Talent.is_active == True,
            Talent.approval_status == 'approved'
        ).all()
        
        scored_talents = []
        
        for t in talents:
            score = 0
            
            # --- Strict Location/Country Filtering ---
            # If a location or country is specified, talent MUST match it to be considered.
            if location or country:
                location_match = False
                if location and str(location).lower() in str(t.location).lower(): location_match = True
                if country and str(country).lower() in str(t.country).lower(): location_match = True
                if location and not location_match and str(location).lower() in str(t.country).lower(): location_match = True
                if not location_match:
                    continue # Skip talent if they don't match the specified location/country.

            def matches(val, target):
                return str(val).lower() in str(target).lower() if val and target else False

            if gender:
                gender_reqs = [g.strip() for g in re.split(r'[,/]', gender.lower()) if g.strip()]
                if "both" in gender_reqs:
                    gender_reqs = [g for g in gender_reqs if g != "both"]
                    gender_reqs.extend(["male", "female"])
                
                normalized_reqs = []
                for g_req in gender_reqs:
                    if g_req in ["man", "men", "male", "m"]: normalized_reqs.append("male")
                    elif g_req in ["woman", "women", "female", "f"]: normalized_reqs.append("female")
                    elif "non" in g_req: normalized_reqs.append("nonbinary")
                    else: normalized_reqs.append(g_req)
                
                # Normalize talent gender for comparison
                t_gender = t.gender.strip().lower() if t.gender else ""
                if t_gender in ["m", "man", "men"]: t_gender = "male"
                elif t_gender in ["f", "woman", "women"]: t_gender = "female"
                
                if not t_gender or t_gender not in normalized_reqs:
                    continue
            
            score += 10

            if shoot_date and t.available_dates:
                try:
                    today = date.today()
                    user_dates = {datetime.strptime(d.strip(), '%Y-%m-%d').date() for d in shoot_date if d.strip()}
                    valid_future_dates = {d for d in user_dates if d >= today}
                    
                    if valid_future_dates:
                        talent_dates = {ad.available_date for ad in t.available_dates if ad.is_active}
                        if not valid_future_dates.isdisjoint(talent_dates):
                            score += 500  
                except (ValueError, TypeError):
                    pass

            # Scoring for location is still useful for ranking if multiple criteria are met
            if location and matches(location, t.location): score += 4000
            if country and matches(country, t.country): score += 5000
            if continent and matches(continent, t.continent): score += 4000

            
            if role and matches(role, t.role): score += 2000

            if hair_color:
                if matches(hair_color, t.hair_colour): score += 50
            if eye_color:
                if matches(eye_color, t.eye_colour): score += 50
            if skin_color:
                if matches(skin_color, t.skin_color): score += 50
            
            if skills:
                if matches(skills, t.skills): score += 50

            try:
                if height and t.height is not None and float(t.height) == float(height): score += 10
                if bust and t.bust is not None and float(t.bust) == float(bust): score += 10
                if waist and t.waist is not None and float(t.waist) == float(waist): score += 10
                if hips and t.hips is not None and float(t.hips) == float(hips): score += 10
                if shoe_size and t.shoe_size is not None and int(t.shoe_size) == int(shoe_size): score += 10
                if dress_size and t.dress_size is not None and int(t.dress_size) == int(dress_size): score += 10
            except (ValueError, TypeError):
                pass
            
            if score > 0:
                scored_talents.append((score, t))

        scored_talents.sort(key=lambda x: x[0], reverse=True)
        top_results = [item[1] for item in scored_talents[:limit]]
        total_results = len(top_results)
        
        result_list = []
        for t in top_results:
            result_list.append({
                "talent_id": t.talent_id,
                "agent_id": t.agent_id,
                "agent_name": t.agent.full_name,
                "name": t.name,
                "role": t.role,
                "date_of_birth": t.date_of_birth,
                "gender": t.gender,
                "height": t.height,
                "bust": t.bust,
                "waist": t.waist,
                "hips": t.hips,
                "shoe_size": t.shoe_size,
                "dress_size": t.dress_size,
                "eye_color": t.eye_colour,
                "hair_type": t.hair_type,
                "hair_color": t.hair_colour,
                "skin_color": t.skin_color,
                "skills": t.skills,
                "location": t.location,
                "continent": t.continent,
                "country": t.country,
                "portfolio_link": t.portfolio_link if t.portfolio_link else None,
                "instagram_link": t.instagram_link if t.instagram_link else None,
                "rate": t.rate if t.rate else None,
                "is_active": t.is_active,
                "approval_status": t.approval_status,
                "is_available": t.is_available,
                "is_available_on_request": t.is_available_on_request,
                "available_dates": [ad.available_date.isoformat() for ad in t.available_dates if ad.is_active],
                "images": [f"{BASE_URL_BACKEND}/media/{img.image}" for img in sorted(t.images, key=lambda x: x.image_id)] if t.images else [],
            })
        
        return {
            "talents": result_list,
            "total_results": total_results
        }
    finally:
        db.close()

def reasoner_node(state: AgentState):
    SYSTEM_PROMPT = """
    You are an Elite Casting Director. Your primary objective is to provide the most accurate talent matches using the 'generate_casting' tool.

    WORKFLOW & CRITICAL RULES:
    1. MANDATORY COLLECTION: Ensure you have Location, Shoot Date, Budget, Job Type, Gender, and Skin Color.
       CRITICAL: Check the 'Current Filters' object below first. If a field exists there, it is already collected. NEVER ask for information that is already present in 'Current Filters'.
    2. THE REFINEMENT CHECKPOINT: Once all mandatory fields are present in 'Current Filters', if no additional preferences (like hair color, height, skills, etc.) have been provided yet and no search has been performed, you MUST ask the user: "I've noted the mandatory details. Would you like to add any additional features (e.g., hair color, height, eye color, or specific skills) to narrow down the search, or should I proceed with the search now?" Do NOT call 'generate_casting' at this stage unless they've already provided an additional feature or explicitly said to proceed.
    3. SEARCH TRIGGER: You MUST call 'generate_casting' immediately if:
       - All mandatory fields are present in 'Current Filters' and the user provides any additional features/refinements.
       - Says "proceed", "search", "show results", "no", or refuses further refinements.
       - Modifies any existing criteria (even mandatory ones).
    4. PERSISTENCE & RE-SEARCH: If the user modifies any preference or adds a new one after a search has already been shown, you MUST call 'generate_casting' again to refresh results. Do not just acknowledge; call the tool.
    5. TOOL ARGUMENTS: Pass ALL relevant criteria from {filters} to the tool.
    6. PAST DATES: If 'shoot_date' is before {today}, warn the user before proceeding.

    Current date: {today}
    Current Filters: {filters}
    """

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY, max_retries=3)
    tools = [generate_casting]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt_content = SYSTEM_PROMPT.format(filters=state.filters, today=date.today())
    sys_msg = SystemMessage(content=prompt_content)
    
    messages = [sys_msg] + state.messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def custom_tool_node(state: AgentState):
    outputs = []
    for tool_call in state.messages[-1].tool_calls:
        if tool_call["name"] == "generate_casting":
            result = generate_casting.invoke(tool_call["args"])
            
            outputs.append(ToolMessage(
                content=f"Search completed. Found {result['total_results']} talents. The full list has been sent to the user interface.",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                artifact=result
            ))
    return {"messages": outputs}

# Build Graph
graph = StateGraph(AgentState)
graph.add_node("agent", reasoner_node)
graph.add_node("tools", custom_tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda x: "tools" if x.messages[-1].tool_calls else END,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent") 

app_graph = graph.compile()

def time_ago(dt: Optional[datetime]) -> str:
    if not dt:
        return "never"

    if dt.tzinfo:
        now = datetime.now(dt.tzinfo)
    else:
        now = datetime.now()

    delta = now - dt

    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"

    minutes = delta.seconds // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    
    return "just now"

def parse_budget(budget_str):
    if not budget_str: return None, None, "$"
    
    budget_str_upper = str(budget_str).upper()
    
    # Detect currency symbol
    currency = "$"
    if "€" in budget_str_upper or "EUR" in budget_str_upper:
        currency = "€"
    elif "ZAR" in budget_str_upper or "R " in budget_str_upper or budget_str_upper.startswith("R"):
        currency = "R"
    elif "£" in budget_str_upper or "GBP" in budget_str_upper:
        currency = "£"
    elif "$" in budget_str_upper or "USD" in budget_str_upper:
        currency = "$"

    # Clean string for digit extraction by removing all known currency identifiers
    to_remove = [',', '$', '£', '€', 'R', 'USD', 'EUR', 'ZAR', 'GBP']
    clean_str = budget_str_upper
    for item in to_remove:
        clean_str = clean_str.replace(item, '')
        
    nums = re.findall(r'\d+(?:\.\d+)?', clean_str)
    if not nums: return None, None, currency
    
    try:
        vals = [Decimal(n) for n in nums]
    except (InvalidOperation, TypeError):
        return None, None, currency

    if len(vals) == 1:
        return vals[0], vals[0], currency
    
    v1, v2 = vals[0], vals[1]
    return min(v1, v2), max(v1, v2), currency
