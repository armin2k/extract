import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BalanceSheet(Base):
    __tablename__ = 'balance_sheets'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    company_name = Column(String(255))
    cnpj = Column(String(20))
    data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<BalanceSheet(id={self.id}, filename='{self.filename}', company_name='{self.company_name}', cnpj='{self.cnpj}', created_at='{self.created_at}')>"