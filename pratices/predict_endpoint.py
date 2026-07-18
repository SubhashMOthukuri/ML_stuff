from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ListingInputs(BaseModel):
    bedrooms: float =1.0
    accommodates : int = 2
    neighbourhood: str = "Brooklyn"

@app.post("/predict")
def predict(listing: ListingInputs) : 
    predicted_price = {}
    predicted_price["price_usd"] = float(listing.bedrooms * 50 + listing.accommodates* 20)
    predicted_price["price_str"] = f"${predicted_price['price_usd']}/night"
    return predicted_price