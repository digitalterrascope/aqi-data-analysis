from aqi_pkg.models import Entry
from sqlalchemy import distinct, func
from datetime import datetime

def apply_filters(
    q,
    *,
    location: str = None,
    city: str = None,
    state: str = None,
    country: str = None,
    start = None,
    end=None,
):
    if location:
        q = q.filter(Entry.locationId == location)
    if city:
        q = q.filter(Entry.city == city)
    if state:
        q = q.filter(Entry.state == state)
    if country:
        q = q.filter(Entry.country == country)
    if start:
        q = q.filter(Entry.last_updated >= start)
    if end:
        q = q.filter(Entry.last_updated <= end)

    return q


def get_all_locations(session, **filters):
    q = session.query(distinct(Entry.locationId))
    q = apply_filters(q, **filters)
    return [r[0] for r in q.all()]


def get_coverage(session, **filters):
    q = session.query(func.count(distinct(Entry.locationId)))
    q = apply_filters(q, **filters)
    return q.scalar()


def get_measurement_range(session, measurement: str, **filters):
    if not hasattr(Entry, measurement):
        raise ValueError(f"Invalid measurement: {measurement}")

    column = getattr(Entry, measurement)
    q = session.query(func.min(column), func.max(column))
    q = apply_filters(q, **filters)
    return q.one()


def get_measurement_avg(session, measurement: str, **filters):
    if not hasattr(Entry, measurement):
        raise ValueError(f"Invalid measurement: {measurement}")

    column = getattr(Entry, measurement)
    q = session.query(func.avg(column))
    q = apply_filters(q, **filters)
    return q.scalar()


def get_measurement_std(session, measurement: str, **filters):
    if not hasattr(Entry, measurement):
        raise ValueError(f"Invalid measurement: {measurement}")

    column = getattr(Entry, measurement)
    q = session.query(func.stddev(column))
    q = apply_filters(q, **filters)
    return q.scalar()


def get_time_range(session, **filters):
    q = session.query(func.min(Entry.last_updated), func.max(Entry.last_updated))
    q = apply_filters(q, **filters)
    return q.one()


def get_location_coords(session, **filters):
    q = session.query(
        distinct(Entry.locationId),
        Entry.lat,
        Entry.lon,
    )
    q = apply_filters(q, **filters)
    return q.all()


def get_measurements(session, measurement: str, **filters):
    if not hasattr(Entry, measurement):
        raise ValueError(f"Invalid measurement: {measurement}")

    column = getattr(Entry, measurement)
    q = session.query(column)
    q = apply_filters(q, **filters)
    return [r[0] for r in q.all()]


def get_measurements_over_time(session, measurement: str, **filters):
    if not hasattr(Entry, measurement):
        raise ValueError(f"Invalid measurement: {measurement}")

    column = getattr(Entry, measurement)
    q = session.query(Entry.last_updated, column)
    q = apply_filters(q, **filters)
    return q.all()