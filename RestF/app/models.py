from sqlalchemy import Column , Integer , String , BigDecimal 

from .database import Base

#todo Continuar 
class Produto(Base):
    __tablename__ = 'produtos'
    
    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, index=True)
    preco = Column(BigDecimal)
    descricao = Column(String)
    
    #def __repr__(self):
    #    return f'<Produto {self.nome}>'