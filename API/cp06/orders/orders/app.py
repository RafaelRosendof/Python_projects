from pathlib import Path

import yaml
from fastapi import FastAPI

app = FastAPI(debug = True , openapi_url = "/openai/figas/orders.json", docs_url="/docs/orders")

oas_doc = yaml.safe_load((Path(__file__).parent / "../oas.yaml").read_text())

app.openapi = lambda: oas_doc

from api import api

#O erro era só o caminho de compilação o correto é uvicorn app:app --reload e não uvicorn orders.app:app --reload

#/docs para ver o swagger e /openapi/figas/orders.json para ver o json???

'''
código em contrução parando na página 118
'''