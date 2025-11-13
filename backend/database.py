from typing import Any, Dict, Optional, List
import os
import motor.motor_asyncio
from bson import ObjectId
from datetime import datetime, timezone

MONGO_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DATABASE_NAME", "appdb")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

async def create_document(collection_name: str, data: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    data = {**data, "_created_at": now.isoformat(), "_updated_at": now.isoformat()}
    result = await db[collection_name].insert_one(data)
    return str(result.inserted_id)

async def get_documents(collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, limit: int = 50) -> List[Dict[str, Any]]:
    filter_dict = filter_dict or {}
    cursor = db[collection_name].find(filter_dict).limit(limit)
    docs: List[Dict[str, Any]] = []
    async for d in cursor:
        d["_id"] = str(d["_id"]) if isinstance(d.get("_id"), ObjectId) else d.get("_id")
        docs.append(d)
    return docs
