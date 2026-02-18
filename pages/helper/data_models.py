from uuid import uuid4
from datetime import datetime
from sqlmodel import Field, create_engine, SQLModel


class PublicSubmissions(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid4()))
    submitted_by: str = Field(max_length=128, nullable=True)
    face_mesh: str = Field(nullable=False)
    image_path: str = Field(nullable=False)   # 🔥 REAL FILE PATH
    location: str = Field(max_length=128, nullable=True)
    mobile: str = Field(max_length=10, nullable=False)
    email: str = Field(max_length=64, nullable=True)
    color: str = Field(max_length=64, nullable=True)
    height: str = Field(max_length=16, nullable=True)
    status: str = Field(max_length=16, nullable=False)
    birth_marks: str = Field(max_length=512, nullable=True)
    submitted_on: datetime = Field(default_factory=datetime.utcnow)


class RegisteredCases(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid4()))
    submitted_by: str = Field(max_length=64, nullable=False)
    name: str = Field(max_length=128, nullable=False)
    father_name: str = Field(max_length=128, nullable=True)
    age: str = Field(max_length=8, nullable=True)
    color: str = Field(max_length=64, nullable=True)
    height: str = Field(max_length=16, nullable=True)
    complainant_name: str = Field(max_length=128)
    complainant_mobile: str = Field(max_length=10, nullable=True)
    adhaar_card: str = Field(max_length=12)
    last_seen: str = Field(max_length=64)
    address: str = Field(max_length=512)
    face_mesh: str = Field(nullable=False)
    image_path: str = Field(nullable=False)   # 🔥 REAL FILE PATH
    submitted_on: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(max_length=16, nullable=False)
    birth_marks: str = Field(max_length=512)
    matched_with: str = Field(nullable=True)


# ---------------- DB INIT ---------------- #

if __name__ == "__main__":
    sqlite_url = "sqlite:///sqlite_database.db"
    engine = create_engine(sqlite_url)

    SQLModel.metadata.create_all(engine)
    print("✅ Database created with image_path support")
