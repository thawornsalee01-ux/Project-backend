from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    Float,
)
from sqlalchemy.orm import relationship, Session
from datetime import datetime

from .session import Base



class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    comparisons = relationship("Comparison", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_label = Column(String(100), nullable=False)
    file_path = Column(String(500), nullable=False)
    uploaded_by = Column(String(100), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="versions")

    # ⭐⭐⭐ ต้องมีตรงนี้
    text_content = relationship(
        "DocumentVersionText",
        back_populates="version",
        uselist=False,
        cascade="all, delete-orphan",
    )

    comparisons_as_old = relationship(
        "Comparison",
        back_populates="version_old",
        foreign_keys="Comparison.version_old_id",
    )
    comparisons_as_new = relationship(
        "Comparison",
        back_populates="version_new",
        foreign_keys="Comparison.version_new_id",
    )

class Comparison(Base):
    __tablename__ = "comparisons"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_old_id = Column(Integer, ForeignKey("document_versions.id"), nullable=False)
    version_new_id = Column(Integer, ForeignKey("document_versions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    overall_risk_level = Column(String(20), nullable=True)
    summary_text = Column(Text, nullable=True)
    edit_intensity = Column(String(20), nullable=True)

    # spidergraphใหม่
    risk_comment = Column(Text, nullable=True)
    scope_impact_score = Column(Float, nullable=True)
    timeline_impact_score = Column(Float, nullable=True)
    cost_impact_score = Column(Float, nullable=True)
    resource_impact_score = Column(Float, nullable=True)
    risk_impact_score = Column(Float, nullable=True)
    contract_impact_score = Column(Float, nullable=True)
    stakeholder_impact_score = Column(Float, nullable=True)
    architecture_impact_score = Column(Float, nullable=True)


    document = relationship("Document", back_populates="comparisons")

    version_old = relationship(
        "DocumentVersion",
        foreign_keys=[version_old_id],
        back_populates="comparisons_as_old",
    )
    version_new = relationship(
        "DocumentVersion",
        foreign_keys=[version_new_id],
        back_populates="comparisons_as_new",
    )

    changes = relationship("ChangeItem", back_populates="comparison", cascade="all, delete-orphan")


class ChangeItem(Base):
    __tablename__ = "changes"

    id = Column(Integer, primary_key=True, index=True)
    comparison_id = Column(Integer, ForeignKey("comparisons.id"), nullable=False)
    change_type = Column(String(20), nullable=False)
    section_label = Column(String(255), nullable=True)
    old_text = Column(Text, nullable=True)
    new_text = Column(Text, nullable=True)
    edit_severity = Column(String(20), nullable=True)

    ai_comment = Column(Text, nullable=True)
    ai_suggestion = Column(Text, nullable=True)

    comparison = relationship("Comparison", back_populates="changes")


class DocumentVersionText(Base):
    __tablename__ = "document_version_texts"

    id = Column(Integer, primary_key=True, index=True)

    # FK ไปยัง version
    document_version_id = Column(
        Integer,
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,   # 1 version = 1 full text
    )

    # ชื่อไฟล์ PDF เดิม (optional แต่ useful)
    original_filename = Column(String(255), nullable=True)

    # text เต็มที่ extract + minimal normalize แล้ว
    full_text = Column(Text, nullable=False)

    # metadata เสริม
    extractor = Column(String(100), nullable=True)   # ชื่อ tool ที่ใช้ extract
    language = Column(String(20), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # relationship
    version = relationship("DocumentVersion", back_populates="text_content")


class DocumentPageText(Base):
    __tablename__ = "document_page_texts"

    id = Column(Integer, primary_key=True, index=True)

    # 1. id_document (FK ไปที่ documents.id)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 2. ชื่อ pdf
    pdf_name = Column(String(255), nullable=False)

    version_pdf = Column(Text, nullable=False)

    # 3. หน้า (page number)
    page = Column(Integer, nullable=False)

    # 4. ข้อความของหน้านั้น
    text_page = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # relationship (optional แต่แนะนำให้มี)
    document = relationship("Document", backref="page_texts")
