import json
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from pages.helper import db_queries
from pages.helper.data_models import RegisteredCases
from pages.helper.db_queries import engine
from sqlmodel import Session, select


# ---------------- LOAD DATA ---------------- #

def get_public_cases_data(status="NF"):
    try:
        result = db_queries.fetch_public_cases(train_data=True, status=status)
        df = pd.DataFrame(result, columns=["id", "face_mesh"])
        df["face_mesh"] = df["face_mesh"].apply(json.loads)
        return df
    except Exception:
        traceback.print_exc()
        return None


def get_registered_cases_data(status="NF"):
    try:
        with Session(engine) as session:
            result = session.exec(
                select(
                    RegisteredCases.id,
                    RegisteredCases.face_mesh,
                    RegisteredCases.status,
                )
            ).all()

        df = pd.DataFrame(result, columns=["id", "face_mesh", "status"])
        df = df[df["status"] == status]
        df["face_mesh"] = df["face_mesh"].apply(json.loads)
        return df
    except Exception:
        traceback.print_exc()
        return None


# ---------------- UTILITIES ---------------- #

def l2_normalize(vec):
    try:
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 1 or vec.size == 0:
            return None

        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return None

        return vec / norm
    except Exception:
        return None


# ---------------- CORE MATCHING ---------------- #

def match(similarity_threshold=0.50):
    """
    Realistic thresholds:
        0.75+ → FaceID / Banking
        0.60+ → Corporate
        0.45–0.55 → Police / Missing persons (YOU)
    """

    matched_images = defaultdict(list)

    public_df = get_public_cases_data(status="NF")
    reg_df = get_registered_cases_data(status="NF")

    if public_df is None or reg_df is None:
        return {"status": False, "message": "Database error"}

    if len(public_df) == 0 or len(reg_df) == 0:
        return {"status": False, "message": "No comparable cases"}

    # ---------------- NORMALIZE ---------------- #

    reg_vectors = []
    reg_ids = []

    for _, row in reg_df.iterrows():
        emb = l2_normalize(row["face_mesh"])
        if emb is not None and emb.shape[0] == 512:
            reg_vectors.append(emb)
            reg_ids.append(row["id"])

    pub_vectors = []
    pub_ids = []

    for _, row in public_df.iterrows():
        emb = l2_normalize(row["face_mesh"])
        if emb is not None and emb.shape[0] == 512:
            pub_vectors.append(emb)
            pub_ids.append(row["id"])

    if not reg_vectors or not pub_vectors:
        return {"status": False, "message": "No valid embeddings"}

    reg_vectors = np.vstack(reg_vectors)
    pub_vectors = np.vstack(pub_vectors)

    # ---------------- SIMILARITY ---------------- #

    similarity_matrix = cosine_similarity(pub_vectors, reg_vectors)

    for i, pub_id in enumerate(pub_ids):
        best_idx = int(np.argmax(similarity_matrix[i]))
        best_score = float(similarity_matrix[i][best_idx])
        reg_id = reg_ids[best_idx]

        print(f"Similarity {pub_id} → {reg_id}: {best_score:.3f}")

        if best_score >= similarity_threshold:
            matched_images[reg_id].append(pub_id)

            # Atomic DB update
            db_queries.link_cases(reg_id, pub_id)

    return {"status": True, "result": matched_images}


if __name__ == "__main__":
    print(match())
