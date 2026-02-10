from sqlalchemy import text
from tqdm import tqdm

def populate_isduplicate(session, batch_size=50_000):
    print("Populating IsDuplicate table...")

    total = session.execute(text("""
        SELECT COUNT(*)
        FROM AqiInScrape a
        LEFT JOIN IsDuplicate d ON a.scrape_id = d.scrape_id
        WHERE d.scrape_id IS NULL
    """)).scalar()

    print(f"Total entries to process: {total}")

    with tqdm(total=total) as pbar:
        last_id = 0

        while True:
            result = session.execute(text("""
                INSERT INTO IsDuplicate (scrape_id, is_duplicate)
                SELECT
                    a.scrape_id,
                    a.rn > 1
                FROM (
                    SELECT
                        scrape_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY locationId, last_updated
                            ORDER BY scrape_id
                        ) AS rn
                    FROM AqiInScrape
                ) a
                LEFT JOIN IsDuplicate d
                  ON a.scrape_id = d.scrape_id
                WHERE d.scrape_id IS NULL
                  AND a.scrape_id > :last_id
                ORDER BY a.scrape_id
                LIMIT :limit
            """), {"last_id": last_id, "limit": batch_size})

            if result.rowcount == 0:
                break

            last_id = session.execute(text("""
                SELECT MAX(scrape_id)
                FROM IsDuplicate
            """)).scalar()

            pbar.update(result.rowcount)

        session.commit()