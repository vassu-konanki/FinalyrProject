from sqlmodel import Session
from pages.helper.db_queries import engine
from pages.helper.data_models import RegisteredCases, PublicSubmissions

with Session(engine) as session:
    session.query(RegisteredCases).delete()
    session.query(PublicSubmissions).delete()
    session.commit()

print("Database cleared successfully.")