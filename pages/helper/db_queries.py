import streamlit as st
from sqlmodel import SQLModel, Session, select
from pages.helper.data_models import RegisteredCases, PublicSubmissions
from db_connection import get_engine

# ---------------- DATABASE CONNECTION ---------------- #

engine = get_engine()


# ---------------- CREATE TABLES ---------------- #

def create_db():
    SQLModel.metadata.create_all(engine)


# ---------------- INSERT ---------------- #

def register_new_case(case_details: RegisteredCases):
    with Session(engine) as session:
        session.add(case_details)
        session.commit()


def new_public_case(public_case_details: PublicSubmissions):
    with Session(engine) as session:
        session.add(public_case_details)
        session.commit()


# ---------------- FETCH ---------------- #

def fetch_registered_cases(submitted_by: str, status: str):
    if status == "All":
        status = ["F", "NF"]
    elif status == "Found":
        status = ["F"]
    elif status == "Not Found":
        status = ["NF"]

    with Session(engine) as session:
        result = session.exec(
            select(RegisteredCases)
            .where(RegisteredCases.submitted_by == submitted_by)
            .where(RegisteredCases.status.in_(status))
        ).all()
        return result


def fetch_public_cases(train_data: bool, status: str):
    with Session(engine) as session:
        if train_data:
            result = session.exec(
                select(
                    PublicSubmissions.id,
                    PublicSubmissions.face_mesh,
                ).where(PublicSubmissions.status == status)
            ).all()
        else:
            result = session.exec(select(PublicSubmissions)).all()
        return result


def get_registered_case_detail(case_id: str):
    with Session(engine) as session:
        result = session.exec(
            select(
                RegisteredCases.name,
                RegisteredCases.complainant_mobile,
                RegisteredCases.age,
                RegisteredCases.last_seen,
                RegisteredCases.birth_marks,
                RegisteredCases.image_path,
            ).where(RegisteredCases.id == case_id)
        ).all()
        return result


def get_public_case_detail(case_id: str):
    with Session(engine) as session:
        result = session.exec(
            select(
                PublicSubmissions.location,
                PublicSubmissions.submitted_by,
                PublicSubmissions.mobile,
                PublicSubmissions.birth_marks,
                PublicSubmissions.image_path,
            ).where(PublicSubmissions.id == case_id)
        ).all()
        return result


# ---------------- NEW FUNCTION FOR MOBILE APP ---------------- #

def get_all_cases():
    """
    Fetch all registered police cases.
    Used by the mobile app.
    """
    with Session(engine) as session:
        result = session.exec(select(RegisteredCases)).all()
        return result


# ---------------- MATCH UPDATE ---------------- #

def update_found_status(register_case_id: str, public_case_id: str):
    with Session(engine) as session:
        registered_case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == register_case_id)
        ).one()

        public_case = session.exec(
            select(PublicSubmissions).where(PublicSubmissions.id == public_case_id)
        ).one()

        registered_case.status = "F"
        registered_case.matched_with = public_case_id
        public_case.status = "F"

        session.add(registered_case)
        session.add(public_case)
        session.commit()


# ---------------- ML SUPPORT ---------------- #

def get_training_data(submitted_by: str):
    with Session(engine) as session:
        result = session.exec(
            select(
                RegisteredCases.id,
                RegisteredCases.face_mesh
            )
            .where(RegisteredCases.submitted_by == submitted_by)
            .where(RegisteredCases.status == "NF")
        ).all()
        return result


# ---------------- UTILS ---------------- #

def update_registered_case_status(case_id: str, status: str):
    with Session(engine) as session:
        case = session.exec(
            select(RegisteredCases).where(RegisteredCases.id == case_id)
        ).first()
        if case:
            case.status = status
            session.add(case)
            session.commit()


def update_public_case_status(case_id: str, status: str):
    with Session(engine) as session:
        case = session.exec(
            select(PublicSubmissions).where(PublicSubmissions.id == case_id)
        ).first()
        if case:
            case.status = status
            session.add(case)
            session.commit()


# ---------------- COUNT ---------------- #

def get_registered_cases_count(submitted_by: str, status: str):
    with Session(engine) as session:
        result = session.exec(
            select(RegisteredCases)
            .where(RegisteredCases.submitted_by == submitted_by)
            .where(RegisteredCases.status == status)
        ).all()
        return result


# ---------------- LINK CASES ---------------- #

def link_cases(registered_case_id: str, public_case_id: str):
    """
    Wrapper used by match_algo.py
    """
    update_found_status(registered_case_id, public_case_id)