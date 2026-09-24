from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import json
from datetime import date, datetime
from decimal import Decimal
from app.config import BASE_URL, BASE_URL_BACKEND

class OptionalDetails(BaseModel):
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = None

class TalentResponse(BaseModel):
    talent_id: int 
    name: str
    images: List[str] = []
    is_active: bool = True
    approval_status: str = "approved"
    is_available: bool = True
    is_available_on_request: bool = False
    role: Optional[str] = None
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    height: Optional[Decimal] = None
    bust: Optional[Decimal] = None
    waist: Optional[Decimal] = None
    hips: Optional[Decimal] = None
    shoe_size: Optional[str] = None
    dress_size: Optional[str] = None
    eye_color: Optional[str] = None
    hair_type: Optional[str] = None
    hair_color: Optional[str] = None
    skin_color: Optional[str] = None
    skills: Optional[str] = None
    location: Optional[str] = None
    continent: Optional[str] = None
    country: Optional[str] = None
    portfolio_link: Optional[str] = None
    instagram_link: Optional[str] = None
    rate: Optional[str] = None
    available_dates: List[date] = []
    status: Optional[str] = None
    tapes: List[str] = []
    polas: List[str] = []
    assigned_roles: List[str] = []

    @model_validator(mode='before')
    @classmethod
    def make_urls_absolute(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'images' in data and data['images']:
                data['images'] = [f"{BASE_URL_BACKEND}{img}" if img and img.startswith('/') else img for img in data['images']]
            if 'tapes' in data and data['tapes']:
                data['tapes'] = [f"{BASE_URL}{tape}" if tape and tape.startswith('/') else tape for tape in data['tapes']]
            if 'polas' in data and data['polas']:
                data['polas'] = [f"{BASE_URL}{pola}" if pola and pola.startswith('/') else pola for pola in data['polas']]
        return data

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = Field(None, alias="shoot_dates")
    budget: Optional[str] = Field(None, alias="budget_range")
    job_type: Optional[str] = None
    gender: Optional[str] = None
    skin_color: Optional[str] = None
    role: Optional[str] = None
    limit: Optional[int] = None
    title:  Optional[str] = None
    description: Optional[str] = None
    casting_roles: Optional[Union[str, List[str]]] = None
    save_as_draft: bool = False

    class Config:
        populate_by_name = True

class GenerateJobRequest(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = Field(None, alias="shoot_dates")
    budget: Optional[str] = Field(None, alias="budget_range")
    job_type: Optional[str] = None
    gender: Optional[str] = None
    skin_color: Optional[str] = None
    casting_roles: Optional[Union[str, List[str]]] = None
    roles: Optional[List[str]] = None

    class Config:
        populate_by_name = True

class UpdateJobRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = Field(None, alias="shoot_dates")
    budget: Optional[str] = Field(None, alias="budget_range")
    currency: Optional[str] = None
    casting_roles: Optional[Union[str, List[str]]] = None
    add_roles: Optional[Union[str, List[str]]] = None
    remove_roles: Optional[Union[str, List[str]]] = None



class SessionRequest(BaseModel):
    session_id: str
    
class RequestTalentJobRequest(BaseModel):
    job_id: Optional[int] = None
    talent_id: int
    session_id: Optional[str] = None

class GenerateCastingRequest(BaseModel):
    session_id: str

class ShortlistTalentRequest(BaseModel):
    job_id: Optional[int] = None
    talent_id: int
    session_id: Optional[str] = None

class BookTalentRequest(BaseModel):
    job_id: Optional[int] = None
    talent_id: int
    session_id: Optional[str] = None
    booking_dates: List[date]

class SelfTapeStatusAction(BaseModel):
    job_id: Optional[int] = None
    talent_id: int
    status: str  # 'accepted' or 'rejected'
    session_id: Optional[str] = None

class JobRoleResponse(BaseModel):
    id: int
    job_role: str
    assign_status: bool
    talent_id: List[int] = []
    session_id: Optional[str] = None

    class Config:
        orm_mode = True

class CreateRoleRequest(BaseModel):
    job_id: Optional[int] = None
    session_id: Optional[str] = None
    job_role: str

    class Config:
        from_attributes = True

class AssignRoleRequest(BaseModel):
    id: int
    talent_id: int

class PolaStatusAction(BaseModel):
    job_id: Optional[int] = None
    talent_id: int
    status: str  # 'accepted' or 'rejected'
    session_id: Optional[str] = None

def _normalize_nested_urls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_nested_urls(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_nested_urls(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        if value.startswith('/media/'):
            return f"{BASE_URL_BACKEND}{value}"
        return f"{BASE_URL}{value}"
    return value

class PolaUploadPageResponse(BaseModel):
    talent_name: str
    talent_role: Optional[str] = None
    job_title: str
    job_budget: Optional[str] = None
    timeline: Optional[str] = None # Shoot dates
    status: str
    existing_images: List[str] = []

    @field_validator('existing_images', mode='before')
    @classmethod
    def make_image_urls_absolute(cls, v):
        if isinstance(v, list):
            return [f"{BASE_URL}{img}" if img and img.startswith('/') else img for img in v]
        return v

class SelfTapeUploadRequest(BaseModel):
    job_id: int
    talent_id: int
    tape_urls: List[str]

class SelfTapeUploadPageResponse(BaseModel):
    talent_name: str
    talent_role: Optional[str] = None
    job_title: str
    job_budget: Optional[str] = None
    timeline: Optional[str] = None # Shoot dates
    status: str
    existing_tapes: List[str] = []

    @field_validator('existing_tapes', mode='before')
    @classmethod
    def make_tape_urls_absolute(cls, v):
        if isinstance(v, list):
            return [f"{BASE_URL}{tape}" if tape and tape.startswith('/') else tape for tape in v]
        return v

class ChatMessageResponse(BaseModel):
    sender: str
    content: str
    timestamp: datetime
    saved_filters: Optional[Dict[str, Any]] = Field(None, alias="saved_filters", validation_alias="filters")

    @field_validator('saved_filters', mode='before')
    @classmethod
    def normalize_saved_filters(cls, v):
        return _normalize_nested_urls(v)

    class Config:
        from_attributes = True
        populate_by_name = True

class JobResponse(BaseModel):
    job_id: int
    job_created_by_id: int
    session_id: Optional[str] = None
    status: str
    title: str
    job_type: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    casting_roles: Optional[Union[str, List[str]]] = None
    job_photo: Optional[str] = None
    budget: Optional[str] = None
    applicants_count: int
    shortlisted_count: int
    selftapes_count: int
    ecastings_count: int
    polas_count: int
    generate_job: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

    @field_validator('casting_roles', mode='before')
    @classmethod
    def parse_casting_roles(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    @field_validator('job_photo', mode='before')
    @classmethod
    def make_photo_url_absolute(cls, v):
        if isinstance(v, str) and v.startswith('/'):
            return f"{BASE_URL}{v}"
        return v

class JobResultResponse(BaseModel):
    job_id: int
    job_created_by_id: int
    session_id: Optional[str] = None
    status: str
    title: str
    job_type: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    casting_roles: Optional[str] = None
    job_photo: Optional[str] = None
    created_at: datetime
    applicants_count: int
    shortlisted_count: int
    selftapes_count: int
    ecastings_count: int
    polas_count: int
    generate_job: bool = True
    shoot_date: Optional[str] = None
    suggested_talents: List[TalentResponse] = []
    requested_selftapes: List[TalentResponse] = []
    requested_ecastings: List[TalentResponse] = []
    requested_polas: List[TalentResponse] = []
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

    @field_validator('job_photo', mode='before')
    @classmethod
    def make_photo_url_absolute(cls, v):
        if isinstance(v, str) and v.startswith('/'):
            return f"{BASE_URL}{v}"
        return v

class DraftResponse(BaseModel):
    draft_id: int
    user_id: int
    session_id: str
    # phase: str
    title: Optional[str] = None
    description: Optional[str] = None
    last_updated: Optional[datetime] = None
    generate_job: bool = False
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

class DraftsSavedFilters(BaseModel):
    job_type: Optional[str] = Field(None, alias="Job type")
    message: Optional[str] = Field(None, alias="Message")
    last_updated_timestamp: Optional[str] = Field(None, alias="Last Updated")

    class Config:
        populate_by_name = True

class UserDraftResponse(BaseModel):
    draft_id: int
    user_id: int
    session_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    saved_filters: DraftsSavedFilters
    updated: str = Field(..., alias="Updated")
    last_updated: Optional[datetime] = None
    generate_job: bool = False

    class Config:
        from_attributes = True
        populate_by_name = True

class ContinueDraftResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    text: str

class PaginationResponse(BaseModel):    
    total_results: int
    page: int
    per_page: int
    has_next: bool

class TalentDataResponse(BaseModel):
    talents: List[TalentResponse] = []

class WrappedChatResponse(BaseModel):
    status_code: int = 200
    status_message: str = "Success"
    data: ChatSessionResponse

class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: int
    created_at: datetime
    messages: List[ChatMessageResponse] = []
    generate_job: bool = False
    job_id: Optional[int] = None

    class Config:
        from_attributes = True

class TalentPreview(BaseModel):
    talent_id: str
    profile_image_url: Optional[str] = None
    is_active: bool = True
    approval_status: str = "approved"
    is_available: bool = True
    is_available_on_request: bool = False

    @field_validator('profile_image_url', mode='before')
    @classmethod
    def make_photo_url_absolute(cls, v):
        if isinstance(v, str) and v.startswith('/'):
            return f"{BASE_URL_BACKEND}{v}"
        return v

class ShortlistSummaryItem(BaseModel):
    job_id: str
    job_title: str
    talent_count: int
    time_remaining_hours: int
    preview_talents: List[TalentPreview]
    extra_talent_count: int

class SummaryPagination(BaseModel):
    page: int
    limit: int
    total: int

class ShortlistSummaryResponse(BaseModel):
    shortlists: List[ShortlistSummaryItem]
    pagination: SummaryPagination
