from db_connection import execute_query

# -------------------------------------
# CREATE TABLES
# -------------------------------------

create_missing_persons = """
CREATE TABLE IF NOT EXISTS missing_persons (
    id SERIAL PRIMARY KEY,
    name TEXT,
    age INTEGER,
    gender TEXT,
    location TEXT,
    description TEXT,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_suspected_persons = """
CREATE TABLE IF NOT EXISTS suspected_persons (
    id SERIAL PRIMARY KEY,
    name TEXT,
    location TEXT,
    remarks TEXT,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def migrate():
    execute_query(create_missing_persons)
    execute_query(create_suspected_persons)
    print("Tables created successfully.")


if __name__ == "__main__":
    migrate()