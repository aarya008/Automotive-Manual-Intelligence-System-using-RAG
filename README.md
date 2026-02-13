🚗 AI-Powered Automotive Owner’s Manual RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed to intelligently answer complex, real-world vehicle troubleshooting and safety queries from large, unstructured automotive PDF manuals.

📌 Problem Statement

Modern vehicle owner manuals:

Contain 300–800+ pages

Are written in dense, technical language

Are difficult to navigate during emergencies

Require manual search across multiple sections

When users experience real-world issues like:

“After driving in heavy rain, my ABS warning light turned on, steering feels stiff, and I hear squeaking while braking. What should I inspect immediately?”

They do not want:

Generic LLM hallucinated advice

Incomplete answers

Unverified internet data

They need:

Precise, manual-grounded guidance

Immediate inspection steps

Warning classification

Driving safety recommendations

🎯 Objective

Build a high-accuracy RAG system that:

Extracts structured content from large automotive manuals

Preserves semantic hierarchy (sections, subsections, warnings)

Retrieves context-aware information

Generates grounded answers only from manual content

Improves retrieval accuracy from ~30% → 77%+

🧠 Solution Overview

This project implements a multi-stage structured RAG pipeline:

PDF → Structured Parsing → Semantic Chunking → Vector Embedding → 
Hybrid Retrieval → Context Ranking → LLM Answer Generation


The key innovation is not just “using RAG” —
but engineering manual-aware document understanding.

🏗 System Architecture
1️⃣ PDF Structure Extraction

Instead of naive text extraction:

Implemented TOC-aware section mapping

Preserved:

Section hierarchy

Headings

Subsections

Safety warnings

Tables

Bullet lists

This prevents semantic fragmentation during chunking.

2️⃣ Semantic Chunking

Traditional chunking splits by fixed token size.

We implemented:

Section-aware chunk boundaries

Context-preserving splits

Semantic continuity enforcement

Overlap optimization

This reduced context dilution significantly.

3️⃣ Embedding & Vector Storage

Dense vector embeddings

FAISS / vector DB integration

Manual-specific similarity search tuning

Reduced top-k noise

4️⃣ Retrieval Strategy

Hybrid retrieval:

Section-prioritized search

Warning-prioritized ranking

Multi-query reformulation

Context score re-ranking

This dramatically improved recall in safety-critical queries.

5️⃣ Controlled Generation

LLM is constrained to:

Only answer using retrieved context

Avoid hallucination

Explicitly classify:

Immediate inspection steps

Safety warnings

Whether driving can continue

📊 Retrieval Accuracy Evolution
Version	Strategy	Accuracy
V1	Naive PDF + fixed chunking	~30%
V2	Section Extractor	~65%
V3	TOC-based structure mapping	~72%
V4	Semantic chunking + reranking	77.34%

Major accuracy gains came from:

Structure-aware parsing

Semantic chunk boundaries

Context ranking optimization

Not from “bigger LLM.”
