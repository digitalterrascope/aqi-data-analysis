from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Double, DateTime

Base = declarative_base()

class Entry(Base):
    __tablename__ = "AqiInScrape"

    scrape_id = Column(Integer, primary_key=True, autoincrement=True)

    lat = Column(Double)
    lon = Column(Double)

    locationId = Column(String(50))
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))

    last_updated = Column(DateTime)

    AQI_IN = Column(Integer)
    AQI_US = Column(Integer)

    CO_PPB = Column(Double)
    NO2_PPB = Column(Double)
    O3_PPB = Column(Double)
    SO2_PPB = Column(Double)

    PM1_UGM3 = Column(Double)
    PM2_5_UGM3 = Column(Double)
    PM10_UGM3 = Column(Double)

    H_PERCENT = Column(Double)
    T_C = Column(Double)
    TVOC_PPM = Column(Double)
    Noise_DB = Column(Double)
