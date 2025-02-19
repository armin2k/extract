# models.py
import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

# Create the base class for our models
Base = declarative_base()

class BalanceSheet(Base):
    __tablename__ = 'balance_sheets'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    company_name = Column(String(255))  # Field to store the company name (optional)
    cnpj = Column(String(20))           # Field to store the company's CNPJ (optional)
    data = Column(Text, nullable=False) # Stores the extracted analysis data (as JSON text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<BalanceSheet(id={self.id}, filename='{self.filename}', company_name='{self.company_name}', cnpj='{self.cnpj}', created_at='{self.created_at}')>"