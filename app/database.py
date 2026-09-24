from sqlalchemy import create_engine, Column, String, Integer, BigInteger, JSON, ForeignKey, Date, Numeric, Boolean, Text, TIMESTAMP, CheckConstraint, UniqueConstraint, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
from app.config import DATABASE_URL

Base = declarative_base()

class UserAuth(Base):
    __tablename__ = "accounts_user"
    
    user_id = Column(BigInteger, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String(255), unique=True, nullable=True) # Matches DB: VARCHAR(255) UNIQUE

class Talent(Base):
    __tablename__ = "talents"
    
    talent_id = Column(BigInteger, primary_key=True)
    name = Column(String, nullable=False) 
    role = Column(String, nullable=True)
    date_of_birth  = Column(Date, nullable=True)
    gender = Column(String(20), nullable=False)
    height = Column(String(50), nullable=True)
    bust = Column(String(50), nullable=True)
    waist = Column(String(50), nullable=True)
    hips = Column(String(50), nullable=True)
    shoe_size = Column(String(50), nullable=True)
    dress_size = Column(String(50), nullable=True)
    eye_colour = Column(String(100), nullable=False)
    hair_type = Column(String(100), nullable=False)
    hair_colour = Column(String(100), nullable=False)
    skin_color = Column(String(100), nullable=False)
    skills = Column(Text, nullable=False)
    location = Column(String(255), nullable=False)
    continent = Column(String(100), nullable=False)
    country = Column(String(100), nullable=False)
    portfolio_link = Column(String(200), nullable=True)
    instagram_link = Column(String(200), nullable=True)
    rate = Column(String(100), nullable=True)
    is_active = Column(Boolean, nullable=False)
    approval_status = Column(String(20), nullable=False)
    is_available = Column(Boolean, nullable=False)
    is_available_on_request = Column(Boolean, nullable=False)
    agent_id = Column(BigInteger, ForeignKey("accounts_user.user_id"), nullable=False)

    agent = relationship("UserAuth", backref="talents")
    images = relationship("TalentImage", back_populates="talent")
    available_dates = relationship("TalentAvailableDate", back_populates="talent")
    assigned_roles = relationship("JobRoleAssignment", back_populates="talent")

class TalentAvailableDate(Base):
    __tablename__ = "talent_available_dates"
    
    availability_id = Column(BigInteger, primary_key=True, index=True)
    available_date = Column(Date, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    note = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)

    talent = relationship("Talent", back_populates="available_dates")

class TalentImage(Base):
    __tablename__ = "talent_images"
    
    image_id = Column(BigInteger, primary_key=True)
    image = Column(String(100), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    talent = relationship("Talent", back_populates="images")

class ShortlistedTalent(Base):
    __tablename__ = "jobs_shortlisted_talents"
    
    shortlisted_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True, index=True)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("accounts_user.user_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    talent = relationship("Talent")
    user = relationship("UserAuth")
    job = relationship("Job")
    session = relationship("ChatSession")

class Booking(Base):
    __tablename__ = "jobs_talent_bookings"
    
    booking_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True, index=True)
    user_id = Column(BigInteger, ForeignKey("accounts_user.user_id"), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=True)
    booking_date = Column(Date, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    talent = relationship("Talent")
    user = relationship("UserAuth")
    job = relationship("Job")
    session = relationship("ChatSession")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    session_id = Column(String, primary_key=True, default=lambda: __import__('uuid').uuid4().hex)
    user_id = Column(BigInteger)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    draft = relationship("Draft", back_populates="session", uselist=False, cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="session", cascade="all, delete-orphan")

    @property
    def saved_filters(self):
        return self.draft.saved_filters if self.draft else {}

    @property
    def generate_job(self):
        return len(self.jobs) > 0

    @property
    def job_id(self):
        if self.jobs:
            return sorted(self.jobs, key=lambda x: x.created_at, reverse=True)[0].job_id
        return None

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"))
    sender = Column(String)
    content = Column(String)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    location = Column(String, nullable=True)
    shoot_date = Column(String, nullable=True)
    budget = Column(String, nullable=True)
    job_type = Column(String, nullable=True)
    filters = Column(JSON, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

class Draft(Base):
    __tablename__ = "jobs_talent_drafts"
    
    draft_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), unique=True, index=True)
    phase = Column(String)
    saved_filters = Column(JSON)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    casting_roles = Column(Text, nullable=True)
    last_updated = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    job_type = Column(String, nullable=True)
    location = Column(String, nullable=True)
    shoot_date = Column(String, nullable=True)
    budget = Column(String, nullable=True)

    session = relationship("ChatSession", back_populates="draft")

    @property
    def generate_job(self):
        return self.session.generate_job if self.session else False

class Job(Base):
    __tablename__ = "jobs_talent_job"   
    
    job_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    location = Column(String(255), nullable=False)
    budget_min = Column(Numeric(10, 2))
    budget_max = Column(Numeric(10, 2))
    currency = Column(String(10), nullable=False, default="$")
    job_type = Column(String(120), nullable=False)
    casting_roles = Column(Text, nullable=False)
    status = Column(String(20), nullable=False)
    applicants_count = Column(Integer, nullable=False, default=0)
    shortlisted_count = Column(Integer, nullable=False, default=0)
    selftapes_count = Column(Integer, nullable=False, default=0)
    ecastings_count = Column(Integer, nullable=False, default=0)
    polas_count = Column(Integer, nullable=False, default=0)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    job_photo = Column(String(255), nullable=True)
    # job_assigned_to_id = Column(
    #     BigInteger,
    #     ForeignKey("accounts_user.user_id"),
    #     nullable=True
    # )
    job_created_by_id = Column(
        BigInteger,
        ForeignKey("accounts_user.user_id"),
        nullable=False
    )

    session = relationship("ChatSession", back_populates="jobs")
    roles = relationship("JobRole", back_populates="job", cascade="all, delete-orphan")
    role_assignments = relationship("JobRoleAssignment", back_populates="job", cascade="all, delete-orphan")

    @property
    def budget(self):
        if self.budget_min is None and self.budget_max is None:
            return None
        
        def fmt(val):
            if val is None: return "0"
            return f"{int(val)}" if val % 1 == 0 else f"{val:.2f}"

        if self.budget_min == self.budget_max:
            return f"{fmt(self.budget_min)}{self.currency}"
        return f"{fmt(self.budget_min)}-{fmt(self.budget_max)}{self.currency}"

    @property
    def generate_job(self):
        return True

    __table_args__ = (
        CheckConstraint('applicants_count >= 0'),
        CheckConstraint('shortlisted_count >= 0'),
        CheckConstraint('selftapes_count >= 0'),
        CheckConstraint('ecastings_count >= 0'),
        CheckConstraint('polas_count >= 0'),
    )

class JobRole(Base):
    __tablename__ = "jobs_jobrole"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=True)
    job_role = Column(String(250), nullable=False)
    assign_status = Column(Boolean, default=False, nullable=False)

    job = relationship("Job", back_populates="roles")
    talent = relationship("Talent")

class JobRoleAssignment(Base):
    __tablename__ = "jobs_jobrole_assignments"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=False)
    job_role_id = Column(Integer, ForeignKey("jobs_jobrole.id"), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    job = relationship("Job", back_populates="role_assignments")
    role = relationship("JobRole")
    talent = relationship("Talent", back_populates="assigned_roles")

    __table_args__ = (
        UniqueConstraint("job_role_id", "talent_id", name="uq_jobrole_assignment_role_talent"),
    )

class JobAIResult(Base):
    __tablename__ = "jobs_ai_results"
    
    result_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=False)
    suggested_talents = Column(JSON)
    requested_selftapes = Column(JSON, nullable=True)
    requested_ecastings = Column(JSON, nullable=True)
    requested_polas = Column(JSON, nullable=True)
    shoot_date = Column(String, nullable=True)

class SelfTapeRequest(Base):
    __tablename__ = "jobs_selftape_requests"
    
    request_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    status = Column(String(50), nullable=False, default="requested") # 'requested', 'accepted', 'rejected', 'responded'
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    tapes = relationship("SelfTapeLink", back_populates="request", cascade="all, delete-orphan")

class SelfTapeLink(Base):
    __tablename__ = "jobs_selftape_links"
    
    link_id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("jobs_selftape_requests.request_id"), nullable=False)
    tape_url = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    request = relationship("SelfTapeRequest", back_populates="tapes")

class PolaRequest(Base):
    __tablename__ = "jobs_pola_requests"
    
    request_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    status = Column(String(50), nullable=False, default="requested") # 'requested', 'accepted', 'rejected', 'responded'
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    images = relationship("PolaLink", back_populates="request", cascade="all, delete-orphan")

class PolaLink(Base):
    __tablename__ = "jobs_pola_links"
    
    link_id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("jobs_pola_requests.request_id"), nullable=False)
    pola_url = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    request = relationship("PolaRequest", back_populates="images")

class Notification(Base):
    __tablename__ = "jobs_notifications"
    id = Column(Integer, primary_key=True, index=True)
    receiver_id = Column(BigInteger, ForeignKey("accounts_user.user_id", ondelete="CASCADE"), nullable=False)
    event = Column(Text, nullable=False)
    sender_id = Column(BigInteger, ForeignKey("accounts_user.user_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=func.now(), server_default=func.now())

# Setup Engine
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database and applies missing schema updates."""
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
