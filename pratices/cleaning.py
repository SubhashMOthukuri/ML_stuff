#phase 2.5(Code)
""" step 1: parse price
step 2: filter invalid price, 
step 3: encode room type
step 4: fill missing bedrooms. 
"""
import logging as logger
logger.getLogger(__name__)

request = {"price": "$125.00","room_type": "Private room","bedrooms": None}

def clean_listing(listing: dict)-> dict | None:
    result={}
    raw = listing.get("price")
    if raw is None:
        return None
    price = float(raw.replace("$", "").replace(",", ""))

    room_type = listing.get("room_type", "")
    result["room_type_entire_home"]  = room_type == "Entire home/apt"
    result["room_type_private_room"] = room_type == "Private room"
    result["room_type_shared_room"]  = room_type == "Shared room"
    result["room_type_hotel_room"]   = room_type == "Hotel room"

    if price<3 or price>2000:
        return None
    result["price"]= price

    result["bedrooms"] = listing.get("bedrooms") or 1
    return result


def main():
    logger.info("Starting Clenaing Process...")
    print(clean_listing(request))
    logger.info("Completed Cleaning Process...")

if __name__ == "__main__":
    main()

