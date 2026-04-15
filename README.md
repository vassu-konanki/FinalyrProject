Finding Missing Persons Using AI
Authors

K. Vasundhara (Y22ACS478)
A. Balaji (Y22ACS412)
CH. Raghavendra (Y22ACS436)
K. Prudhvi (Y22ACS477)

Implementation
AI Recognition Engine & Backend System
Developed using Python, Streamlit, PostgreSQL
Implements MediaPipe Face Mesh AI for facial recognition
Generates 128-dimensional embeddings for accurate face matching
Uses pgvector similarity search for fast retrieval
Includes encryption (AES-256) for data security
Dashboards
1. Police Dashboard
Secure login for authorized personnel
Register missing person cases
Upload and manage photos
Perform real-time face matching
Track investigation progress
2. Public Dashboard
Report missing persons
Submit anonymous tips
Upload sighting images
Track case status using reference ID
Overview

MissingAI is an AI-powered system designed to identify and locate missing persons efficiently using facial recognition technology.

The system processes images, extracts facial features using Face Mesh (468 landmarks), and matches them against a centralized database to identify potential matches in real-time (< 1 second).

It bridges the gap between law enforcement and the public, enabling faster response during critical “golden hours”.

Project Components
1. Facial Recognition Engine
Uses MediaPipe Face Mesh
Extracts 468 facial landmarks
Converts into 128D embeddings
Performs similarity matching (cosine distance)
2. Police Management System
Case registration and tracking
Real-time face matching
Investigation workflow management
Audit logs for all actions
3. Public Interaction System
Missing person reporting
Anonymous tip submission
Case tracking system
4. Database System
PostgreSQL with pgvector
Stores:
Person details
Case data
Images (encrypted)
Face embeddings
5. Security System
AES-256 encryption (data at rest)
TLS for secure communication
Role-based access control
System Architecture

The system consists of:

User Interfaces (Police & Public Dashboards)
Application Layer (Business Logic)
AI Processing Layer (Face Mesh Engine)
Database Layer (PostgreSQL + pgvector)
Infrastructure Layer (Local Server Deployment)
Technologies Used
Backend
Python
Streamlit
OpenCV
MediaPipe
NumPy
Database
PostgreSQL
pgvector
Security
OpenSSL
AES-256 Encryption
Tools
Git
pytest
Locust
OWASP ZAP
Key Features
AI-based facial recognition
Real-time face matching (< 1 sec)
3D landmark-based identification
Case management system
Public participation via dashboard
Secure local data storage
Audit logging system
Research Contribution

This project demonstrates how Artificial Intelligence and Computer Vision can:

Improve missing person identification accuracy
Reduce search time drastically
Enable real-time automated matching
Ensure privacy with local deployment

It replaces manual investigation methods with AI-driven intelligent systems.

Performance Highlights
Accuracy: ~93% recall, ~87% precision
Speed: ~573 ms response time
Real-time processing: 20–30 FPS
Matching time: < 1 second
Future Work
Age progression AI models
CCTV real-time integration
Multi-station federated system
Mobile app (Android/iOS)
Integration with national databases (NCRB, Aadhaar)
Multilingual support
Predictive analytics for missing cases
GitHub Repository (Example Format)


Cloud deploy links

https://finalyrproject-policedashboard.streamlit.app/

https://finalyrproject-public.streamlit.app/
